from __future__ import annotations

import multiprocessing as mp
from typing import Optional

import numpy as np

from trainable_system import (
    AdamState,
    Config,
    RolloutRunner,
    TerminalProgressBar,
    _ensure_writable_runtime_dirs,
    _log,
    _metrics_summary,
    _queue_put_latest,
    _tree_global_norm,
    build_feature_sequence,
    jax,
    jnp,
)
from trainable_spatial_system import (
    CartPoleTrainingViewer,
    TrainableSpatialWalkingSystem as _ESTrainableSpatialWalkingSystem,
    _require_runtime as _require_spatial_runtime,
    bp,
    bm,
)


_ensure_writable_runtime_dirs()


_BPTT_BASE = bp.DynamicalSystem if bp is not None else object


class _DifferentiableSpatialCartPoleRollout(_BPTT_BASE):
    _TRAINABLE_KEYS = ("w_ee_raw", "w_ei_raw", "w_ie_raw", "w_ii_raw")

    def __init__(self, system: "TrainableSpatialBPTTWalkingSystem"):
        super().__init__()
        self.system = system
        self.cfg = system.cfg
        self.spatial_model = system.spatial_model

        self.n_exc = int(system.n_exc)
        self.n_inh = int(system.n_inh)
        self.n_total = int(system.n_total)
        self.obs_size = int(system.obs_size)
        self.action_size = int(system.action_size)

        self.dt = float(self.cfg.dt_ms) / 1000.0
        self.g = float(self.cfg.gravity)
        self.cart_mass = float(self.cfg.mass)
        self.pole_mass = float(self.cfg.thigh_mass + self.cfg.shank_mass)
        self.total_mass = float(self.cart_mass + self.pole_mass)
        self.pole_length = float(max(1e-3, system.pole_length))
        self.pole_com_length = 0.5 * self.pole_length
        self.polemass_length = self.pole_mass * self.pole_com_length
        self.force_limit = float(system.cart_force_scale)
        self.track_half_width = float(system.track_half_width)
        self.linear_damping = float(getattr(self.cfg, "drag", 0.0))
        self.angular_damping = 0.1 * float(getattr(self.cfg, "angular_drag", 0.0))
        self.cart_track_y = float(system.cart_track_y)
        self.readout_decay = float(system.readout_decay)
        self.input_gain = float(self.cfg.input_gain)
        self.readout_gain = float(self.cfg.readout_gain)

        self.obs_scale = bm.asarray(np.asarray(system.obs_scale, dtype=np.float32))
        self.e_shape = tuple(int(v) for v in np.asarray(self.spatial_model.E.varshape, dtype=int))
        self.i_shape = tuple(int(v) for v in np.asarray(self.spatial_model.I.varshape, dtype=int))

        self.w_in = bm.asarray(np.asarray(system.params["w_in"], dtype=np.float32))
        self.bias_in = bm.asarray(np.asarray(system.params["bias_in"], dtype=np.float32))
        self.w_out = bm.asarray(np.asarray(system.params["w_out"], dtype=np.float32))
        self.bias_out = bm.asarray(np.asarray(system.params["bias_out"], dtype=np.float32))

        self.w_ee_raw = bm.TrainVar(bm.asarray(np.asarray(system.params["w_ee_raw"], dtype=np.float32)))
        self.w_ei_raw = bm.TrainVar(bm.asarray(np.asarray(system.params["w_ei_raw"], dtype=np.float32)))
        self.w_ie_raw = bm.TrainVar(bm.asarray(np.asarray(system.params["w_ie_raw"], dtype=np.float32)))
        self.w_ii_raw = bm.TrainVar(bm.asarray(np.asarray(system.params["w_ii_raw"], dtype=np.float32)))

        self._initial_ext_seed = bm.asarray(np.asarray(self.spatial_model.ext.seed.value))

        self.x = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.x_dot = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.theta = bm.Variable(bm.asarray(float(self.cfg.initial_pitch), dtype=bm.float_))
        self.theta_dot = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.last_force = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.filt = bm.Variable(bm.zeros((self.n_total,), dtype=bm.float_))

        self.reset_state()

    def reset_state(self, batch_or_mode=None, **kwargs):
        bp.reset_state(self.spatial_model)
        if hasattr(self.spatial_model.ext, "seed"):
            self.spatial_model.ext.seed.value = bm.asarray(self._initial_ext_seed)
        self.x.value = bm.asarray(0.0, dtype=bm.float_)
        self.x_dot.value = bm.asarray(0.0, dtype=bm.float_)
        self.theta.value = bm.asarray(float(self.cfg.initial_pitch), dtype=bm.float_)
        self.theta_dot.value = bm.asarray(0.0, dtype=bm.float_)
        self.last_force.value = bm.asarray(0.0, dtype=bm.float_)
        self.filt.value = bm.zeros((self.n_total,), dtype=bm.float_)

    def sync_fixed_params(self, params):
        self.w_in = bm.asarray(np.asarray(params["w_in"], dtype=np.float32))
        self.bias_in = bm.asarray(np.asarray(params["bias_in"], dtype=np.float32))
        self.w_out = bm.asarray(np.asarray(params["w_out"], dtype=np.float32))
        self.bias_out = bm.asarray(np.asarray(params["bias_out"], dtype=np.float32))

    def set_trainable_params(self, params):
        self.w_ee_raw.value = bm.asarray(np.asarray(params["w_ee_raw"], dtype=np.float32))
        self.w_ei_raw.value = bm.asarray(np.asarray(params["w_ei_raw"], dtype=np.float32))
        self.w_ie_raw.value = bm.asarray(np.asarray(params["w_ie_raw"], dtype=np.float32))
        self.w_ii_raw.value = bm.asarray(np.asarray(params["w_ii_raw"], dtype=np.float32))

    def get_trainable_params(self) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(getattr(self, key).value, dtype=np.float32)
            for key in self._TRAINABLE_KEYS
        }

    def sync_recurrent_weights(self):
        self.spatial_model.E2E.proj.comm.weight = bm.asarray(jnp.logaddexp(self.w_ee_raw.value, 0.0))
        self.spatial_model.E2I.proj.comm.weight = bm.asarray(jnp.logaddexp(self.w_ei_raw.value, 0.0))
        self.spatial_model.I2E.proj.comm.weight = bm.asarray(jnp.logaddexp(self.w_ie_raw.value, 0.0))
        self.spatial_model.I2I.proj.comm.weight = bm.asarray(jnp.logaddexp(self.w_ii_raw.value, 0.0))

    def run(self, features):
        features = bm.asarray(features)
        indices = bm.arange(features.shape[0])
        self.sync_recurrent_weights()
        self.reset_state()
        return bm.for_loop(self.step_run, (indices, features), progress_bar=False)

    def _build_observation(self, feature_t):
        theta = self.theta.value
        obs = bm.stack(
            [
                feature_t[0],
                feature_t[1],
                self.x.value,
                self.x_dot.value,
                theta,
                self.theta_dot.value,
                bm.sin(theta),
                bm.cos(theta),
                self.last_force.value,
            ],
            axis=0,
        )
        return obs / self.obs_scale

    def _force_from_action(self, action):
        primary = self.force_limit * action[: self.cfg.n_legs]
        trim = 0.5 * self.force_limit * action[self.cfg.n_legs :]
        return (primary[0] - primary[1]) + (trim[0] - trim[1])

    def _set_input_drive(self, drive):
        e_drive = bm.reshape(drive[: self.n_exc], self.e_shape)
        i_drive = bm.reshape(drive[self.n_exc :], self.i_shape)
        e_target = getattr(self.spatial_model.Ein, "input", self.spatial_model.Ein)
        i_target = getattr(self.spatial_model.Iin, "input", self.spatial_model.Iin)
        if hasattr(e_target, "value"):
            e_target.value = bm.asarray(e_drive)
        else:
            e_target[...] = bm.asarray(e_drive)
        if hasattr(i_target, "value"):
            i_target.value = bm.asarray(i_drive)
        else:
            i_target[...] = bm.asarray(i_drive)

    def _advance_physics(self, force_cmd):
        force = bm.clip(force_cmd, -self.force_limit, self.force_limit)

        sin_theta = bm.sin(self.theta.value)
        cos_theta = bm.cos(self.theta.value)
        temp = (
            force
            + self.polemass_length * self.theta_dot.value * self.theta_dot.value * sin_theta
            - self.linear_damping * self.x_dot.value
        ) / self.total_mass
        denom = self.pole_com_length * (
            4.0 / 3.0 - (self.pole_mass * cos_theta * cos_theta) / self.total_mass
        )
        theta_acc = (
            self.g * sin_theta
            - cos_theta * temp
            - self.angular_damping * self.theta_dot.value
        ) / bm.maximum(1e-6, denom)
        x_acc = temp - (self.polemass_length * theta_acc * cos_theta) / self.total_mass

        x_dot = self.x_dot.value + self.dt * x_acc
        theta_dot = self.theta_dot.value + self.dt * theta_acc
        x = self.x.value + self.dt * x_dot
        theta = self.theta.value + self.dt * theta_dot

        hit_bound = bm.abs(x) > self.track_half_width
        x = bm.where(hit_bound, bm.clip(x, -self.track_half_width, self.track_half_width), x)
        x_dot = bm.where(hit_bound, -0.15 * x_dot, x_dot)

        self.x.value = x
        self.x_dot.value = x_dot
        self.theta.value = theta
        self.theta_dot.value = theta_dot
        self.last_force.value = force
        return force

    def update(self, feature_t):
        obs = self._build_observation(feature_t)
        drive = self.input_gain * (obs @ self.w_in) + self.bias_in
        self._set_input_drive(drive)

        self.spatial_model()

        spike = bm.concatenate(
            [
                bm.reshape(self.spatial_model.E.spike.value, (-1,)),
                bm.reshape(self.spatial_model.I.spike.value, (-1,)),
            ],
            axis=0,
        )
        membrane = bm.concatenate(
            [
                bm.reshape(self.spatial_model.E.V.value, (-1,)),
                bm.reshape(self.spatial_model.I.V.value, (-1,)),
            ],
            axis=0,
        )
        syn_current = bm.concatenate(
            [
                bm.reshape(self.spatial_model.E.input.value, (-1,)),
                bm.reshape(self.spatial_model.I.input.value, (-1,)),
            ],
            axis=0,
        )
        adaptation = bm.concatenate(
            [
                bm.reshape(self.spatial_model.E.g_K.value, (-1,)),
                bm.reshape(self.spatial_model.I.g_K.value, (-1,)),
            ],
            axis=0,
        )

        filt = self.readout_decay * self.filt.value + spike
        self.filt.value = filt
        action = bm.tanh(self.readout_gain * (filt @ self.w_out + self.bias_out))
        force_cmd = self._force_from_action(action)
        force = self._advance_physics(force_cmd)

        pos = bm.stack(
            [
                self.x.value,
                bm.asarray(self.cart_track_y, dtype=bm.float_),
            ],
            axis=0,
        )
        vel = bm.stack(
            [
                self.x_dot.value,
                bm.asarray(0.0, dtype=bm.float_),
            ],
            axis=0,
        )
        force_vec = bm.stack(
            [
                force,
                bm.asarray(0.0, dtype=bm.float_),
            ],
            axis=0,
        )
        return (
            pos,
            vel,
            self.theta.value,
            self.theta_dot.value,
            force_vec,
            obs,
            action,
            membrane,
            syn_current,
            adaptation,
            filt,
            spike,
        )


class TrainableSpatialBPTTWalkingSystem(_ESTrainableSpatialWalkingSystem):
    def __init__(self, cfg: Config):
        _require_spatial_runtime()
        super().__init__(cfg)
        self._enable_spatial_training_mode()
        self.compute_backend = "cartpole+spatial+bptt"
        self.rollout_model = _DifferentiableSpatialCartPoleRollout(self)
        self._build_bptt_functions()

    def _enable_spatial_training_mode(self):
        training_mode = bm.TrainingMode() if hasattr(bm, "TrainingMode") else bm.training_mode
        self.spatial_model._mode = training_mode
        self.spatial_model.E._mode = training_mode
        self.spatial_model.I._mode = training_mode
        self._upgrade_spike_buffer(self.spatial_model.E)
        self._upgrade_spike_buffer(self.spatial_model.I)
        for proj_name in ("E2E", "E2I", "I2E", "I2I", "ext2E", "ext2I"):
            getattr(self.spatial_model, proj_name)._mode = training_mode

    @staticmethod
    def _upgrade_spike_buffer(neuron_group):
        spike_shape = tuple(int(v) for v in np.asarray(neuron_group.spike.value).shape)
        object.__setattr__(
            neuron_group,
            "spike",
            bm.Variable(bm.zeros(spike_shape, dtype=bm.float_)),
        )

    def _build_bptt_functions(self):
        grad_vars = {key: getattr(self.rollout_model, key) for key in self._TRAINABLE_RECURRENT_KEYS}

        @bm.to_object(child_objs=(self.rollout_model,))
        def loss_fun(feature_seq):
            outputs = self.rollout_model.run(feature_seq)
            return self._loss_from_outputs(outputs, feature_seq)

        self._loss_fun = loss_fun
        self._grad_fun = bm.grad(
            loss_fun,
            grad_vars=grad_vars,
            has_aux=True,
            return_value=True,
        )

    @staticmethod
    def _metrics_to_python(metrics) -> dict[str, float]:
        if hasattr(metrics, "items"):
            return {key: float(np.asarray(value)) for key, value in metrics.items()}
        return {}

    def _loss_from_outputs(self, outputs, features):
        pos, vel, angle, omega, force, _obs, action, _v, _syn, _adapt, _filt, _spike = outputs
        target_vx = bm.asarray(features[:, 0])

        distance = pos[-1, 0] - pos[0, 0]
        mean_vx = bm.mean(vel[:, 0])
        speed_tracking = bm.mean((vel[:, 0] - target_vx) ** 2)
        cart_center_error = bm.mean((pos[:, 0] / max(1e-6, self.track_half_width)) ** 2)
        angle_error = bm.mean(angle**2)
        omega_error = bm.mean(omega**2)
        control_effort = bm.mean((force[:, 0] / max(1e-6, self.cart_force_scale)) ** 2)

        if self.num_steps > 1:
            action_rate = bm.mean(bm.sum((action[1:] - action[:-1]) ** 2, axis=1))
        else:
            action_rate = bm.asarray(0.0, dtype=bm.float_)

        reward = self.cfg.reward_distance * distance + self.cfg.reward_speed * mean_vx
        reward = reward - self.cfg.penalty_speed_tracking * speed_tracking
        reward = reward - self.cfg.penalty_pitch * angle_error
        reward = reward - self.cfg.penalty_height * cart_center_error
        reward = reward - 0.25 * omega_error
        reward = reward - self.cfg.penalty_energy * control_effort
        reward = reward - self.cfg.penalty_action_rate * action_rate

        loss = -reward
        metrics = {
            "loss": loss,
            "reward": reward,
            "distance": distance,
            "mean_vx": mean_vx,
            "height_error": cart_center_error,
            "pitch_error": angle_error,
            "speed_tracking": speed_tracking,
        }
        return loss, metrics

    def _sync_rollout_model_from_params(self, params):
        self.rollout_model.sync_fixed_params(params)
        self.rollout_model.set_trainable_params(self._trainable_params(params))

    def _outputs_to_rollout(self, outputs):
        (
            pos,
            vel,
            angle,
            omega,
            force,
            obs,
            action,
            membrane,
            syn_current,
            adaptation,
            filt,
            spike,
        ) = [np.asarray(value, dtype=np.float32) for value in outputs]

        steps = int(pos.shape[0])
        hip_angle = np.zeros((steps, self.cfg.n_legs), dtype=np.float32)
        knee_angle = np.zeros((steps, self.cfg.n_legs), dtype=np.float32)
        joint_torque = np.zeros((steps, self.cfg.n_legs, 2), dtype=np.float32)
        ground_contact = np.zeros((steps, self.cfg.n_legs), dtype=np.float32)

        cart_offsets = np.asarray(
            [
                [-0.25 * self.cart_width, -0.5 * self.cart_height],
                [0.25 * self.cart_width, -0.5 * self.cart_height],
            ],
            dtype=np.float32,
        )
        foot_pos = pos[:, None, :] + cart_offsets[None, :, :]
        foot_vel = np.zeros((steps, self.cfg.n_legs, 2), dtype=np.float32)
        foot_vel[:, :, 0] = vel[:, 0:1]

        return {
            "ts": np.arange(steps, dtype=np.float32) * self.cfg.dt_ms,
            "pos": pos,
            "vel": vel,
            "angle": angle.astype(np.float32),
            "omega": omega.astype(np.float32),
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "foot_pos": foot_pos.astype(np.float32),
            "foot_vel": foot_vel,
            "ground_contact": ground_contact,
            "force": force,
            "joint_torque": joint_torque,
            "obs": obs,
            "action": action,
            "spike": spike,
            "v": membrane,
            "syn": syn_current,
            "adapt": adaptation,
            "filtered_spike": filt,
            "E.spike": spike[:, : self.n_exc],
            "I.spike": spike[:, self.n_exc :],
        }

    def _simulate(self, params, features, progress_label: Optional[str] = None):
        self._sync_rollout_model_from_params(params)
        if progress_label:
            _log(f"{progress_label}: running differentiable spatial rollout")
        outputs = self.rollout_model.run(features)
        rollout = self._outputs_to_rollout(outputs)
        if progress_label:
            _log(f"{progress_label}: rollout complete")
        return rollout

    def _grads_to_numpy(self, grads) -> dict[str, np.ndarray]:
        if hasattr(grads, "items"):
            return {key: np.asarray(value, dtype=np.float32) for key, value in grads.items()}
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in zip(self._TRAINABLE_RECURRENT_KEYS, grads)
        }

    def train_step(
        self,
        features: Optional[np.ndarray] = None,
        progress_prefix: Optional[str] = None,
    ) -> dict[str, float]:
        features = self._coerce_features(features)
        self._sync_rollout_model_from_params(self.params)
        step_prefix = progress_prefix or "bptt step"

        _log(f"{step_prefix}: computing surrogate-gradient BPTT update")
        grads, train_loss, train_metrics = self._grad_fun(bm.asarray(np.asarray(features, dtype=np.float32)))
        grad_accum = self._grads_to_numpy(grads)
        grad_norm = float(_tree_global_norm(grad_accum))

        if grad_norm > self.cfg.gradient_clip > 0.0:
            scale = self.cfg.gradient_clip / (grad_norm + 1e-8)
            grad_accum = jax.tree_util.tree_map(lambda g: g * scale, grad_accum)
            _log(f"{step_prefix}: clipped recurrent gradient to norm {self.cfg.gradient_clip:.4f}")

        pre_metrics = self._metrics_to_python(train_metrics)
        if pre_metrics:
            _log(
                f"{step_prefix}: pre-update "
                f"loss={float(np.asarray(train_loss)):.4f} reward={pre_metrics.get('reward', float('nan')):.4f}"
            )

        updated_trainable = self._adam_update(self._trainable_params(self.params), grad_accum)
        self.params = self._merge_trainable_params(self.params, updated_trainable)
        self._sync_rollout_model_from_params(self.params)

        rollout = self._simulate(self.params, features, progress_label=f"{step_prefix} eval")
        loss, metrics = self._metrics_from_rollout(rollout, features)
        metrics["loss"] = float(loss)
        metrics["grad_norm"] = grad_norm
        _log(f"{step_prefix}: complete {_metrics_summary(metrics)}")
        return metrics


TrainableWalkingSystem = TrainableSpatialBPTTWalkingSystem


def collect_rollout(
    system: TrainableWalkingSystem, features: Optional[np.ndarray] = None
) -> RolloutRunner:
    rollout, _ = system.evaluate(features)
    return RolloutRunner(mon=rollout)


def _training_worker(cfg: Config, features: np.ndarray, message_queue, stop_event):
    try:
        _log("spatial BPTT backend worker booting")
        system = TrainableWalkingSystem(cfg)
        refresh_every = cfg.vis_every if cfg.vis_every > 0 else cfg.eval_every
        train_forever = cfg.train_epochs <= 0
        total_epochs_label = "inf" if train_forever else str(cfg.train_epochs)
        epoch_progress = (
            None if train_forever else TerminalProgressBar(max(1, cfg.train_epochs), "training epochs")
        )
        _log(
            f"spatial BPTT backend ready: epochs={total_epochs_label} "
            f"refresh_every={refresh_every}"
        )

        _queue_put_latest(
            message_queue,
            {
                "type": "status",
                "phase": "evaluating",
                "detail": "running initial rollout",
            },
        )
        _log("starting initial differentiable spatial rollout for viewer bootstrap")
        rollout, metrics = system.evaluate_with_progress(
            features,
            progress_label="initial rollout",
        )
        _log(f"initial differentiable spatial rollout complete {_metrics_summary(metrics)}")
        _queue_put_latest(
            message_queue,
            {
                "type": "snapshot",
                "epoch": 0,
                "rollout": rollout,
                "metrics": metrics,
                "params": system.params,
                "phase": "idle",
                "detail": "ready to train",
            },
        )

        epoch = 0
        while not stop_event.is_set() and (train_forever or epoch < cfg.train_epochs):
            _queue_put_latest(
                message_queue,
                {
                    "type": "status",
                    "phase": "training",
                    "detail": "running BPTT optimizer step",
                },
            )
            _log(f"epoch {epoch + 1}/{total_epochs_label}: starting BPTT optimizer step")
            latest_metrics = system.train_step(
                features,
                progress_prefix=f"epoch {epoch + 1}/{total_epochs_label}",
            )
            epoch += 1
            if epoch_progress is not None:
                epoch_progress.update(epoch)
            _log(
                f"epoch {epoch}/{total_epochs_label}: BPTT optimizer step complete "
                f"{_metrics_summary(latest_metrics)}"
            )

            _queue_put_latest(
                message_queue,
                {
                    "type": "metrics",
                    "epoch": epoch,
                    "metrics": latest_metrics,
                    "phase": "training",
                    "detail": "optimizer step complete",
                },
            )

            should_refresh = (
                epoch == 1
                or epoch % max(1, refresh_every) == 0
                or (not train_forever and epoch >= cfg.train_epochs)
            )
            if should_refresh and not stop_event.is_set():
                _log(f"epoch {epoch}/{total_epochs_label}: refreshing spatial BPTT viewer rollout")
                _queue_put_latest(
                    message_queue,
                    {
                        "type": "status",
                        "phase": "evaluating",
                        "detail": "refreshing rollout for UI",
                    },
                )
                rollout, eval_metrics = system.evaluate_with_progress(
                    features,
                    progress_label="refresh rollout",
                )
                _log(
                    f"epoch {epoch}/{total_epochs_label}: refresh spatial BPTT rollout complete "
                    f"{_metrics_summary(eval_metrics)}"
                )
                _queue_put_latest(
                    message_queue,
                    {
                        "type": "snapshot",
                        "epoch": epoch,
                        "rollout": rollout,
                        "metrics": eval_metrics,
                        "params": system.params,
                        "phase": "idle",
                        "detail": "snapshot ready",
                    },
                )

        if epoch_progress is not None:
            epoch_progress.finish()
        _log(f"spatial BPTT training worker finished at epoch {epoch}/{total_epochs_label}")
        _queue_put_latest(
            message_queue,
            {
                "type": "done",
                "epoch": epoch,
                "phase": "done",
                "detail": "training stopped" if stop_event.is_set() or train_forever else "training complete",
            },
        )
    except Exception as exc:  # pragma: no cover - worker-side runtime path.
        _log(f"spatial BPTT backend worker failed: {exc}")
        _queue_put_latest(
            message_queue,
            {
                "type": "error",
                "phase": "error",
                "detail": str(exc),
            },
        )


def main():
    _require_spatial_runtime()
    print("[trainable_spatial_bptt_system] building system...", flush=True)
    cfg = Config()
    system = TrainableWalkingSystem(cfg)
    print(
        "[trainable_spatial_bptt_system] system ready "
        f"(steps={system.num_steps}, exc={system.n_exc}, inh={system.n_inh}, "
        f"backend={system.compute_backend})",
        flush=True,
    )
    features = build_feature_sequence(
        system.num_steps,
        cfg.dt_ms,
        cfg.target_vx,
        cfg.target_vy,
    )
    ctx = mp.get_context("spawn")
    message_queue = ctx.Queue(maxsize=4)
    stop_event = ctx.Event()
    worker_process = ctx.Process(
        target=_training_worker,
        args=(cfg, features, message_queue, stop_event),
        daemon=True,
    )
    print("[trainable_spatial_bptt_system] starting backend worker...", flush=True)
    worker_process.start()
    print("[trainable_spatial_bptt_system] opening training viewer...", flush=True)
    viewer = CartPoleTrainingViewer(system, features, message_queue, stop_event, worker_process)
    try:
        viewer.show()
    finally:
        stop_event.set()
        worker_process.join(timeout=5.0)
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=2.0)


__all__ = [
    "AdamState",
    "Config",
    "RolloutRunner",
    "TrainableSpatialBPTTWalkingSystem",
    "TrainableWalkingSystem",
    "build_feature_sequence",
    "collect_rollout",
    "main",
]


if __name__ == "__main__":
    main()
