from __future__ import annotations

import multiprocessing as mp
import time
from typing import Optional

import numpy as np

from trainable_system import (
    AdamState,
    Config,
    RolloutRunner,
    TerminalProgressBar,
    _queue_put_latest,
    build_feature_sequence,
)
from trainable_spatial_bptt_system import (
    _DifferentiableSpatialCartPoleRollout as _BaseSpatialRollout,
    _require_spatial_runtime,
    TrainableSpatialBPTTWalkingSystem as _BaseSpatialBPTTWalkingSystem,
    bp,
    bm,
    jax,
)
from trainable_spatial_system import CartPoleTrainingViewer
from trainable_system import _log


def _regulated_metrics_summary(metrics: dict[str, float]) -> str:
    fields = []
    for key in ("loss", "reward", "balance_score", "upright_fraction", "mean_abs_angle", "grad_norm"):
        if key in metrics:
            fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


class _RegulatedSpatialCartPoleRollout(_BaseSpatialRollout):
    def __init__(self, system: "TrainableSpatialBPTTRegulatedWalkingSystem"):
        self._regulation_ready = False
        super().__init__(system)

        self.regulation_decay = 0.97
        self.activity_decay = 0.985
        self.target_filtered_activity = 0.025
        self.activity_feedback_gain = 18.0
        self.activity_gate_floor = 0.18
        self.drive_temperature = 1.35
        self.drive_rms_floor = 0.12
        self.e_drive_limit = 0.080
        self.i_drive_limit = 0.060

        self.e_drive_mean = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.i_drive_mean = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.e_drive_rms = bm.Variable(bm.asarray(1.0, dtype=bm.float_))
        self.i_drive_rms = bm.Variable(bm.asarray(1.0, dtype=bm.float_))
        self.filtered_activity = bm.Variable(bm.asarray(0.0, dtype=bm.float_))
        self.input_gate = bm.Variable(bm.asarray(1.0, dtype=bm.float_))

        self._regulation_ready = True
        self.reset_state()

    def reset_state(self, batch_or_mode=None, **kwargs):
        super().reset_state(batch_or_mode=batch_or_mode, **kwargs)
        if not self._regulation_ready:
            return
        self.e_drive_mean.value = bm.asarray(0.0, dtype=bm.float_)
        self.i_drive_mean.value = bm.asarray(0.0, dtype=bm.float_)
        self.e_drive_rms.value = bm.asarray(1.0, dtype=bm.float_)
        self.i_drive_rms.value = bm.asarray(1.0, dtype=bm.float_)
        self.filtered_activity.value = bm.asarray(0.0, dtype=bm.float_)
        self.input_gate.value = bm.asarray(1.0, dtype=bm.float_)

    def _regulated_drive(self, raw_drive):
        e_raw = raw_drive[: self.n_exc]
        i_raw = raw_drive[self.n_exc :]

        e_mean_now = bm.mean(e_raw)
        i_mean_now = bm.mean(i_raw)
        e_centered = e_raw - e_mean_now
        i_centered = i_raw - i_mean_now
        e_rms_now = bm.sqrt(bm.mean(e_centered * e_centered) + 1e-6)
        i_rms_now = bm.sqrt(bm.mean(i_centered * i_centered) + 1e-6)

        self.e_drive_mean.value = (
            self.regulation_decay * self.e_drive_mean.value + (1.0 - self.regulation_decay) * e_mean_now
        )
        self.i_drive_mean.value = (
            self.regulation_decay * self.i_drive_mean.value + (1.0 - self.regulation_decay) * i_mean_now
        )
        self.e_drive_rms.value = (
            self.regulation_decay * self.e_drive_rms.value + (1.0 - self.regulation_decay) * e_rms_now
        )
        self.i_drive_rms.value = (
            self.regulation_decay * self.i_drive_rms.value + (1.0 - self.regulation_decay) * i_rms_now
        )

        prev_activity = bm.mean(self.filt.value)
        self.filtered_activity.value = (
            self.activity_decay * self.filtered_activity.value + (1.0 - self.activity_decay) * prev_activity
        )
        activity_excess = bm.maximum(0.0, self.filtered_activity.value - self.target_filtered_activity)
        gate = 1.0 / (1.0 + self.activity_feedback_gain * activity_excess)
        gate = bm.clip(gate, self.activity_gate_floor, 1.0)
        self.input_gate.value = gate

        e_norm = (e_raw - self.e_drive_mean.value) / bm.maximum(self.e_drive_rms.value, self.drive_rms_floor)
        i_norm = (i_raw - self.i_drive_mean.value) / bm.maximum(self.i_drive_rms.value, self.drive_rms_floor)

        e_drive = gate * self.e_drive_limit * bm.tanh(e_norm / self.drive_temperature)
        i_drive = gate * self.i_drive_limit * bm.tanh(i_norm / self.drive_temperature)
        return bm.concatenate([e_drive, i_drive], axis=0)

    def update(self, feature_t):
        obs = self._build_observation(feature_t)
        raw_drive = self.input_gain * (obs @ self.w_in) + self.bias_in
        drive = self._regulated_drive(raw_drive)
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
            [self.x.value, bm.asarray(self.cart_track_y, dtype=bm.float_)],
            axis=0,
        )
        vel = bm.stack(
            [self.x_dot.value, bm.asarray(0.0, dtype=bm.float_)],
            axis=0,
        )
        force_vec = bm.stack(
            [force, bm.asarray(0.0, dtype=bm.float_)],
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


class RegulatedCartPoleTrainingViewer(CartPoleTrainingViewer):
    def __init__(self, *args, **kwargs):
        self.history_balance = []
        self.history_upright = []
        super().__init__(*args, **kwargs)

    def _select_top_recurrent_edges(self, weights: np.ndarray):
        if getattr(self, "max_recurrent_edges", 0) <= 0:
            return (
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )
        return super()._select_top_recurrent_edges(weights)

    def _init_network_panel(self):
        self.max_recurrent_edges = 0
        self.max_input_edges_per_channel = 1
        self.max_output_edges_per_command = 4
        super()._init_network_panel()
        self.rec_lines.set_segments([])
        self.rec_lines.set_alpha(0.0)
        self.exc_spike_rings.set_alpha(0.0)
        self.inh_spike_rings.set_alpha(0.0)
        self.ax_net.set_title("Spatial Controller Activity", fontsize=11, color="#435569")

    def _init_metric_panel(self):
        self.balance_line, = self.ax_metric.plot([], [], color="tab:blue", lw=2.0, label="balance score")
        self.upright_line, = self.ax_metric.plot([], [], color="tab:orange", lw=2.0, label="upright fraction")
        self.ax_metric.set_title("Inverted Balance Progress")
        self.ax_metric.set_xlabel("epoch")
        self.ax_metric.set_ylabel("score")
        self.ax_metric.set_ylim(0.0, 1.02)
        self.ax_metric.legend(loc="lower right")
        self.ax_metric.grid(alpha=0.25)

    def _record_metrics(self, epoch: int, metrics: dict[str, float]):
        if epoch <= self.last_epoch_seen:
            return
        self.last_epoch_seen = epoch
        self.epoch = epoch
        self.latest_metrics = metrics
        self.history_steps.append(epoch)
        self.history_balance.append(float(metrics.get("balance_score", np.nan)))
        self.history_upright.append(float(metrics.get("upright_fraction", np.nan)))
        self.history_loss.append(float(metrics.get("loss", np.nan)))

    def _refresh_metric_plot(self):
        if not self.history_steps:
            return
        x = np.asarray(self.history_steps, dtype=float)
        self.balance_line.set_data(x, np.asarray(self.history_balance, dtype=float))
        self.upright_line.set_data(x, np.asarray(self.history_upright, dtype=float))
        self.ax_metric.set_xlim(0.0, max(1.0, x[-1]))
        self.ax_metric.set_ylim(0.0, 1.02)

    def _on_train_tick(self):
        if self.closed:
            return
        changed = self._drain_backend_messages()
        if not self.worker_process.is_alive() and not self.worker_done:
            self.worker_done = True
            if self.current_phase not in {"done", "error"}:
                self._set_phase("done", "backend exited")

        if self.worker_done and self.current_phase != "error":
            self.train_timer.stop()

        if changed:
            now = time.perf_counter()
            if self.latest_metrics is not None and (now - self.last_log_time) >= 0.5:
                total_epochs_label = "inf" if self.cfg.train_epochs <= 0 else str(self.cfg.train_epochs)
                print(
                    "[trainable_system] "
                    f"epoch={self.epoch}/{total_epochs_label} "
                    f"phase={self.current_phase} "
                    f"reward={self.latest_metrics.get('reward', float('nan')):.4f} "
                    f"balance={self.latest_metrics.get('balance_score', float('nan')):.4f} "
                    f"upright={self.latest_metrics.get('upright_fraction', float('nan')):.4f}",
                    flush=True,
                )
                self.last_log_time = now

        self._refresh_metric_plot()
        self.fig.canvas.draw_idle()


class TrainableSpatialBPTTRegulatedWalkingSystem(_BaseSpatialBPTTWalkingSystem):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.compute_backend = "cartpole+spatial+bptt+regulated"
        self.rollout_model = _RegulatedSpatialCartPoleRollout(self)
        self._build_bptt_functions()

    def _soft_balance_score(self, pos_x, vel_x, angle, omega):
        angle_scale = 0.18
        omega_scale = 1.2
        x_scale = max(0.25, 0.35 * self.track_half_width)
        vx_scale = 0.9
        state_cost = (
            (angle / angle_scale) ** 2
            + 0.22 * (omega / omega_scale) ** 2
            + 0.10 * (pos_x / x_scale) ** 2
            + 0.05 * (vel_x / vx_scale) ** 2
        )
        return bm.exp(-state_cost)

    def _soft_upright_fraction(self, angle, omega):
        angle_ok = jax.nn.sigmoid(18.0 * (0.12 - bm.abs(angle)))
        omega_ok = jax.nn.sigmoid(10.0 * (0.9 - bm.abs(omega)))
        return angle_ok * omega_ok

    def _loss_from_outputs(self, outputs, features):
        del features
        pos, vel, angle, omega, force, _obs, action, _v, _syn, _adapt, _filt, _spike = outputs

        balance_trace = self._soft_balance_score(pos[:, 0], vel[:, 0], angle, omega)
        upright_trace = self._soft_upright_fraction(angle, omega)
        balance_score = bm.mean(balance_trace)
        upright_fraction = bm.mean(upright_trace)
        mean_abs_angle = bm.mean(bm.abs(angle))
        control_effort = bm.mean((force[:, 0] / max(1e-6, self.cart_force_scale)) ** 2)
        cart_drift = bm.mean((pos[:, 0] / max(1e-6, self.track_half_width)) ** 2)
        if self.num_steps > 1:
            action_rate = bm.mean(bm.sum((action[1:] - action[:-1]) ** 2, axis=1))
        else:
            action_rate = bm.asarray(0.0, dtype=bm.float_)

        reward = 0.70 * balance_score + 0.30 * upright_fraction
        reward = reward - 0.035 * control_effort
        reward = reward - 0.025 * action_rate
        reward = reward - 0.030 * cart_drift

        loss = -reward
        metrics = {
            "loss": loss,
            "reward": reward,
            "balance_score": balance_score,
            "upright_fraction": upright_fraction,
            "mean_abs_angle": mean_abs_angle,
            "distance": pos[-1, 0] - pos[0, 0],
            "mean_vx": bm.mean(vel[:, 0]),
        }
        return loss, metrics

    def _metrics_from_rollout(self, rollout, features):
        del features
        pos = np.asarray(rollout["pos"], dtype=np.float32)
        vel = np.asarray(rollout["vel"], dtype=np.float32)
        angle = np.asarray(rollout["angle"], dtype=np.float32)
        omega = np.asarray(rollout["omega"], dtype=np.float32)
        force = np.asarray(rollout["force"], dtype=np.float32)
        action = np.asarray(rollout["action"], dtype=np.float32)

        angle_scale = 0.18
        omega_scale = 1.2
        x_scale = max(0.25, 0.35 * self.track_half_width)
        vx_scale = 0.9
        state_cost = (
            np.square(angle / angle_scale)
            + 0.22 * np.square(omega / omega_scale)
            + 0.10 * np.square(pos[:, 0] / x_scale)
            + 0.05 * np.square(vel[:, 0] / vx_scale)
        )
        balance_score = float(np.mean(np.exp(-state_cost)))

        angle_ok = 1.0 / (1.0 + np.exp(-18.0 * (0.12 - np.abs(angle))))
        omega_ok = 1.0 / (1.0 + np.exp(-10.0 * (0.9 - np.abs(omega))))
        upright_fraction = float(np.mean(angle_ok * omega_ok))

        mean_abs_angle = float(np.mean(np.abs(angle)))
        control_effort = float(np.mean(np.square(force[:, 0] / max(1e-6, self.cart_force_scale))))
        cart_drift = float(np.mean(np.square(pos[:, 0] / max(1e-6, self.track_half_width))))
        if action.shape[0] > 1:
            action_rate = float(np.mean(np.sum(np.square(action[1:] - action[:-1]), axis=1)))
        else:
            action_rate = 0.0

        reward = 0.70 * balance_score + 0.30 * upright_fraction
        reward = reward - 0.035 * control_effort
        reward = reward - 0.025 * action_rate
        reward = reward - 0.030 * cart_drift

        loss = -reward
        return loss, {
            "loss": float(loss),
            "reward": float(reward),
            "balance_score": float(balance_score),
            "upright_fraction": float(upright_fraction),
            "mean_abs_angle": float(mean_abs_angle),
            "distance": float(pos[-1, 0] - pos[0, 0]),
            "mean_vx": float(np.mean(vel[:, 0])),
            "height_error": float(cart_drift),
            "pitch_error": float(np.mean(np.square(angle))),
            "speed_tracking": float(np.mean(np.square(vel[:, 0]))),
        }


TrainableWalkingSystem = TrainableSpatialBPTTRegulatedWalkingSystem


def collect_rollout(
    system: TrainableWalkingSystem, features: Optional[np.ndarray] = None
) -> RolloutRunner:
    rollout, _ = system.evaluate(features)
    return RolloutRunner(mon=rollout)


def _training_worker(cfg: Config, features: np.ndarray, message_queue, stop_event):
    try:
        _log("regulated spatial BPTT backend worker booting")
        system = TrainableWalkingSystem(cfg)
        refresh_every = cfg.vis_every if cfg.vis_every > 0 else cfg.eval_every
        train_forever = cfg.train_epochs <= 0
        total_epochs_label = "inf" if train_forever else str(cfg.train_epochs)
        epoch_progress = (
            None if train_forever else TerminalProgressBar(max(1, cfg.train_epochs), "training epochs")
        )
        _log(
            f"regulated spatial BPTT backend ready: epochs={total_epochs_label} "
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
        _log("starting initial regulated spatial rollout for viewer bootstrap")
        rollout, metrics = system.evaluate_with_progress(features, progress_label="initial rollout")
        _log(f"initial regulated spatial rollout complete {_regulated_metrics_summary(metrics)}")
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
                    "detail": "running regulated BPTT optimizer step",
                },
            )
            _log(f"epoch {epoch + 1}/{total_epochs_label}: starting regulated BPTT optimizer step")
            latest_metrics = system.train_step(features, progress_prefix=f"epoch {epoch + 1}/{total_epochs_label}")
            epoch += 1
            if epoch_progress is not None:
                epoch_progress.update(epoch)
            _log(
                f"epoch {epoch}/{total_epochs_label}: regulated BPTT optimizer step complete "
                f"{_regulated_metrics_summary(latest_metrics)}"
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
                _log(f"epoch {epoch}/{total_epochs_label}: refreshing regulated viewer rollout")
                _queue_put_latest(
                    message_queue,
                    {
                        "type": "status",
                        "phase": "evaluating",
                        "detail": "refreshing rollout for UI",
                    },
                )
                rollout, eval_metrics = system.evaluate_with_progress(features, progress_label="refresh rollout")
                _log(
                    f"epoch {epoch}/{total_epochs_label}: refresh regulated rollout complete "
                    f"{_regulated_metrics_summary(eval_metrics)}"
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
        _log(f"regulated spatial BPTT worker finished at epoch {epoch}/{total_epochs_label}")
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
        _log(f"regulated spatial BPTT backend worker failed: {exc}")
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
    print("[trainable_spatial_bptt_regulated_system] building system...", flush=True)
    cfg = Config()
    system = TrainableWalkingSystem(cfg)
    print(
        "[trainable_spatial_bptt_regulated_system] system ready "
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
    print("[trainable_spatial_bptt_regulated_system] starting backend worker...", flush=True)
    worker_process.start()
    print("[trainable_spatial_bptt_regulated_system] opening training viewer...", flush=True)
    viewer = RegulatedCartPoleTrainingViewer(system, features, message_queue, stop_event, worker_process)
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
    "TrainableSpatialBPTTRegulatedWalkingSystem",
    "TrainableWalkingSystem",
    "build_feature_sequence",
    "collect_rollout",
    "main",
]


if __name__ == "__main__":
    main()
