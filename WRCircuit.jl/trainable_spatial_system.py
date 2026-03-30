from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Optional

import numpy as np

from trainable_system import (
    AdamState,
    Config,
    RolloutRunner,
    TerminalProgressBar,
    TrainingViewer,
    TrainableWalkingSystem as _BaseTrainableWalkingSystem,
    _ensure_writable_runtime_dirs,
    _log,
    _metrics_summary,
    _queue_put_latest,
    _require_runtime as _require_base_runtime,
    _tree_global_norm,
    build_feature_sequence,
    jax,
    jnp,
)


_ensure_writable_runtime_dirs()


_SPATIAL_IMPORT_ERROR = None

try:
    import brainpy as bp
    import brainpy.math as bm

    from src.models.Spatial import Spatial
except Exception as exc:  # pragma: no cover - exercised only when runtime is missing.
    bp = None
    bm = None
    Spatial = None
    _SPATIAL_IMPORT_ERROR = exc


def _require_runtime():
    _require_base_runtime()
    if bp is None or bm is None or Spatial is None:
        raise ImportError(
            "trainable_spatial_system.py requires the spatial BrainPy runtime. "
            "Install at least: brainpy, jax, jaxlib, numpy, matplotlib. "
            f"Missing runtime pieces: brainpy/spatial ({_SPATIAL_IMPORT_ERROR})"
        )


_RUN_SIMULATION_SPATIAL_SEED = 42
_RUN_SIMULATION_SPATIAL_RHO = 10000
_RUN_SIMULATION_SPATIAL_DX_MM = 0.1


@dataclass(frozen=True)
class CartPoleObservation:
    pos: np.ndarray
    vel: np.ndarray
    angle: float
    omega: float
    last_force: float


class CartPolePhysics:
    def __init__(self, cfg: Config, cart_width: float, cart_height: float, pole_length: float, force_limit: float):
        self.cfg = cfg
        self.dt = float(cfg.dt_ms) / 1000.0
        self.g = float(cfg.gravity)
        self.cart_mass = float(cfg.mass)
        self.pole_mass = float(cfg.thigh_mass + cfg.shank_mass)
        self.total_mass = self.cart_mass + self.pole_mass
        self.cart_width = float(cart_width)
        self.cart_height = float(cart_height)
        self.cart_y = 0.5 * self.cart_height + 0.04
        self.pole_length = float(max(1e-3, pole_length))
        self.pole_com_length = 0.5 * self.pole_length
        self.polemass_length = self.pole_mass * self.pole_com_length
        self.force_limit = float(force_limit)
        self.track_half_width = float(max(1.5, 6.0 * self.cart_width))
        self.linear_damping = float(getattr(cfg, "drag", 0.0))
        self.angular_damping = 0.1 * float(getattr(cfg, "angular_drag", 0.0))

        self.x = 0.0
        self.x_dot = 0.0
        self.theta = float(cfg.initial_pitch)
        self.theta_dot = 0.0
        self.last_force = 0.0
        self.last_joint_torque = np.zeros((cfg.n_legs, 2), dtype=np.float32)

    def observe(self) -> CartPoleObservation:
        return CartPoleObservation(
            pos=np.asarray([self.x, self.cart_y], dtype=np.float32),
            vel=np.asarray([self.x_dot, 0.0], dtype=np.float32),
            angle=float(self.theta),
            omega=float(self.theta_dot),
            last_force=float(self.last_force),
        )

    def step(self, force_cmd: float):
        force = float(np.clip(force_cmd, -self.force_limit, self.force_limit))
        self.last_force = force

        sin_theta = float(np.sin(self.theta))
        cos_theta = float(np.cos(self.theta))
        temp = (
            force
            + self.polemass_length * self.theta_dot * self.theta_dot * sin_theta
            - self.linear_damping * self.x_dot
        ) / self.total_mass
        denom = self.pole_com_length * (
            4.0 / 3.0 - (self.pole_mass * cos_theta * cos_theta) / self.total_mass
        )
        theta_acc = (
            self.g * sin_theta
            - cos_theta * temp
            - self.angular_damping * self.theta_dot
        ) / max(1e-6, denom)
        x_acc = temp - (self.polemass_length * theta_acc * cos_theta) / self.total_mass

        self.x_dot += self.dt * x_acc
        self.theta_dot += self.dt * theta_acc
        self.x += self.dt * self.x_dot
        self.theta += self.dt * self.theta_dot

        if abs(self.x) > self.track_half_width:
            self.x = float(np.clip(self.x, -self.track_half_width, self.track_half_width))
            self.x_dot *= -0.15


class CartPoleTrainingViewer(TrainingViewer):
    def _init_robot_panel(self):
        self.ax_robot.set_aspect("equal", adjustable="box")
        self.ax_robot.set_xlabel("x")
        self.ax_robot.set_ylabel("y")
        self.ax_robot.axhline(self.system.cart_track_y - 0.5 * self.system.cart_height, color="0.65", lw=1.0)

        self.path_line, = self.ax_robot.plot([], [], color="0.75", lw=1.5)
        self.body_box, = self.ax_robot.plot([], [], "-", color="black", lw=2)
        self.body_com, = self.ax_robot.plot([], [], "o", color="black", ms=4)
        self.pole_line, = self.ax_robot.plot([], [], "-", color="tab:orange", lw=3)
        self.force_line, = self.ax_robot.plot([], [], "-", color="tab:red", lw=2)

    def _update_neural_canvas(
        self,
        t_ms: float,
        obs: np.ndarray,
        action: np.ndarray,
        membrane: np.ndarray,
        filtered: np.ndarray,
        adapt: np.ndarray,
        spike: np.ndarray,
    ):
        super()._update_neural_canvas(t_ms, obs, action, membrane, filtered, adapt, spike)
        primary_force, trim_force = self.system._decode_targets(action.astype(np.float32))
        target_values = np.concatenate([primary_force, trim_force], axis=0)
        for idx, text in enumerate(self.output_target_texts):
            text.set_text(f"{target_values[idx]:+.2f} N")

    def _set_rollout(self, rollout, metrics):
        self.latest_rollout = rollout
        self.latest_metrics = metrics

        ts_ms = np.asarray(rollout["ts"], dtype=float)
        frame_stride = max(1, int(round((1000.0 / self.cfg.animation_fps) / self.cfg.dt_ms)))
        frame_indices = np.arange(0, len(ts_ms), frame_stride, dtype=int)
        if len(frame_indices) == 0 or frame_indices[-1] != len(ts_ms) - 1:
            frame_indices = np.append(frame_indices, len(ts_ms) - 1)
        self.frame_indices = frame_indices
        self.frame_times = ts_ms[frame_indices]
        self.frame_ptr = 0

        self.rollout_cache = {
            key: np.asarray(rollout[key], dtype=float)
            for key in ("pos", "vel", "angle", "force", "obs", "action", "v", "filtered_spike", "adapt", "spike")
        }

        membrane = self.rollout_cache["v"]
        filtered = self.rollout_cache["filtered_spike"]
        adapt = self.rollout_cache["adapt"]
        v_lo = float(min(0.0, np.percentile(membrane, 1.0)))
        v_hi = float(max(self.cfg.v_th, np.percentile(membrane, 99.0)))
        self.exc_nodes.set_clim(v_lo, v_hi)
        self.inh_nodes.set_clim(v_lo, v_hi)
        self.filtered_node_scale = float(max(1.0, np.percentile(filtered, 99.0)))
        self.adapt_node_scale = float(max(1.0, np.percentile(adapt, 99.0)))

        pos = self.rollout_cache["pos"]
        self.robot_y_limits = (
            float(self.system.cart_track_y - 0.75 * self.system.cart_height),
            float(self.system.cart_track_y + self.system.pole_length + 0.15),
        )
        self.camera_half_width = float(max(1.5, min(self.system.track_half_width + 0.2, 2.4)))
        self.ax_robot.set_ylim(*self.robot_y_limits)
        self.force_scale = 0.30 * self.system.pole_length / max(
            1.0, float(np.max(np.abs(self.rollout_cache["force"][:, 0])))
        )
        self._draw_frame(0)

    def _draw_frame(self, frame_ptr: int):
        if self.latest_rollout is None:
            return
        frame_ptr = int(np.clip(frame_ptr, 0, len(self.frame_indices) - 1))
        self.frame_ptr = frame_ptr
        i = int(self.frame_indices[frame_ptr])

        pos = self.rollout_cache["pos"]
        vel = self.rollout_cache["vel"]
        angle = self.rollout_cache["angle"]
        force = self.rollout_cache["force"]
        obs = self.rollout_cache["obs"]
        action = self.rollout_cache["action"]
        membrane = self.rollout_cache["v"]
        filtered = self.rollout_cache["filtered_spike"]
        adapt = self.rollout_cache["adapt"]
        spike = self.rollout_cache["spike"]

        self._update_neural_canvas(
            self.frame_times[frame_ptr],
            obs[i],
            action[i],
            membrane[i],
            filtered[i],
            adapt[i],
            spike[i],
        )

        p = pos[i]
        theta = float(angle[i])
        half_w = 0.5 * self.system.cart_width
        half_h = 0.5 * self.system.cart_height
        cart_outline = np.array(
            [
                [p[0] - half_w, p[1] + half_h],
                [p[0] + half_w, p[1] + half_h],
                [p[0] + half_w, p[1] - half_h],
                [p[0] - half_w, p[1] - half_h],
                [p[0] - half_w, p[1] + half_h],
            ],
            dtype=float,
        )
        pivot = np.array([p[0], p[1] + half_h], dtype=float)
        tip = pivot + self.system.pole_length * np.array([np.sin(theta), np.cos(theta)], dtype=float)

        self.path_line.set_data(pos[: i + 1, 0], pos[: i + 1, 1])
        self.body_box.set_data(cart_outline[:, 0], cart_outline[:, 1])
        self.body_com.set_data([p[0]], [p[1]])
        self.pole_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])

        f = force[i]
        self.force_line.set_data(
            [p[0], p[0] + f[0] * self.force_scale],
            [p[1], p[1]],
        )
        self.ax_robot.set_xlim(p[0] - self.camera_half_width, p[0] + self.camera_half_width)
        self.ax_robot.set_title(
            "Cart-Pole Rollout | "
            f"epoch={self.epoch} | x={p[0]:.3f} vx={vel[i, 0]:.3f} theta={theta:.3f}"
        )


class TrainableSpatialWalkingSystem(_BaseTrainableWalkingSystem):
    _TRAINABLE_RECURRENT_KEYS = ("w_ee_raw", "w_ei_raw", "w_ie_raw", "w_ii_raw")

    def _init_params(self, key):
        k1, k2 = jax.random.split(key, 2)
        scale_in = 1.0 / np.sqrt(max(1, self.obs_size))
        scale_out = 1.0 / np.sqrt(max(1, self.n_total))
        return {
            "w_in": jax.random.normal(k1, (self.obs_size, self.n_total)) * scale_in,
            "bias_in": jnp.zeros((self.n_total,), dtype=jnp.float32),
            "w_out": jax.random.normal(k2, (self.n_total, self.action_size)) * scale_out,
            "bias_out": jnp.zeros((self.action_size,), dtype=jnp.float32),
            "w_ee_raw": jnp.zeros((1,), dtype=jnp.float32),
            "w_ei_raw": jnp.zeros((1,), dtype=jnp.float32),
            "w_ie_raw": jnp.zeros((1,), dtype=jnp.float32),
            "w_ii_raw": jnp.zeros((1,), dtype=jnp.float32),
        }

    def __init__(self, cfg: Config):
        _require_runtime()
        super().__init__(cfg)
        self.compute_backend = "cartpole+spatial"
        self.obs_size = 9
        self.action_size = 2 * cfg.n_legs
        self.observation_labels = [
            "target vx",
            "target vy",
            "cart x",
            "cart vx",
            "pole angle",
            "pole omega",
            "sin theta",
            "cos theta",
            "last force",
        ]
        self.action_labels = ["push +", "push -", "trim +", "trim -"]
        self.cart_width = float(cfg.body_length)
        self.cart_height = float(cfg.body_height)
        self.cart_track_y = 0.5 * self.cart_height + 0.04
        self.pole_length = float(cfg.thigh_length + cfg.shank_length)
        self.cart_force_scale = 0.5 * float(cfg.hip_torque_limit + cfg.knee_torque_limit)
        self.track_half_width = float(max(1.5, 6.0 * self.cart_width))
        self.obs_scale = np.asarray(
            [
                max(1.0, abs(cfg.target_vx)),
                max(1.0, abs(cfg.target_vy)),
                self.track_half_width,
                1.5,
                1.0,
                2.5,
                1.0,
                1.0,
                max(1.0, self.cart_force_scale),
            ],
            dtype=np.float32,
        )

        bm.set_dt(cfg.dt_ms)

        spatial_key = jax.random.PRNGKey(_RUN_SIMULATION_SPATIAL_SEED)
        self.spatial_model = Spatial(
            key=spatial_key,
            rho=_RUN_SIMULATION_SPATIAL_RHO,
            dx=_RUN_SIMULATION_SPATIAL_DX_MM,
        )
        self.spatial_connectivity = {
            "K_ee": int(self.spatial_model.K_ee),
            "K_ei": int(self.spatial_model.K_ei),
            "K_ie": int(self.spatial_model.K_ie),
            "K_ii": int(self.spatial_model.K_ii),
        }
        self.cfg.rho = _RUN_SIMULATION_SPATIAL_RHO
        self.cfg.controller_dx_mm = _RUN_SIMULATION_SPATIAL_DX_MM
        self.cfg.gamma = float(self.spatial_model.gamma)

        self.exc_side = int(np.asarray(self.spatial_model.E.size, dtype=int)[0])
        self.n_exc = int(np.prod(np.asarray(self.spatial_model.E.size, dtype=int)))
        self.n_inh = int(np.prod(np.asarray(self.spatial_model.I.size, dtype=int)))
        self.n_total = self.n_exc + self.n_inh

        self.E = self.spatial_model.E
        self.I = self.spatial_model.I
        self.exc_positions = np.asarray(self.E.positions, dtype=np.float32).reshape(self.n_exc, 2)
        self.inh_positions = np.asarray(self.I.positions, dtype=np.float32).reshape(self.n_inh, 2)
        self.all_positions = jnp.asarray(
            np.concatenate([self.exc_positions, self.inh_positions], axis=0),
            dtype=jnp.float32,
        )

        self.exc_mask = np.concatenate(
            [np.ones((self.n_exc,), dtype=np.float32), np.zeros((self.n_inh,), dtype=np.float32)],
            axis=0,
        )
        self.recurrent_specs = self._build_recurrent_specs()
        self.params = self._init_spatial_params(cfg.random_seed)
        self.opt_state = self._adam_init(self._trainable_params(self.params))
        self._apply_recurrent_params(self.params)

        if self.params["w_in"].shape[1] != self.n_total:
            raise ValueError(
                "Spatial controller size mismatch. "
                f"Expected params width {self.n_total}, got {self.params['w_in'].shape[1]}."
            )

    @staticmethod
    def _value_to_numpy(value) -> np.ndarray:
        if hasattr(value, "value"):
            value = value.value
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @staticmethod
    def _softplus(value) -> np.ndarray:
        return np.logaddexp(np.asarray(value, dtype=np.float32), 0.0).astype(np.float32)

    @staticmethod
    def _inverse_softplus(value) -> np.ndarray:
        value = np.maximum(np.asarray(value, dtype=np.float32), 1e-6)
        large = value > 20.0
        raw = np.empty_like(value, dtype=np.float32)
        raw[large] = value[large]
        raw[~large] = np.log(np.expm1(value[~large])).astype(np.float32)
        return raw

    @staticmethod
    def _row_sparse_matrix(
        rng: np.random.Generator,
        num_rows: int,
        num_cols: int,
        density: float,
        scale: float,
    ) -> np.ndarray:
        count = min(num_cols, max(1, int(round(float(np.clip(density, 0.0, 1.0)) * num_cols))))
        weights = np.zeros((num_rows, num_cols), dtype=np.float32)
        for row_idx in range(num_rows):
            cols = rng.choice(num_cols, size=count, replace=False)
            weights[row_idx, cols] = scale * rng.standard_normal(count).astype(np.float32)
        return weights

    @staticmethod
    def _column_sparse_matrix(
        rng: np.random.Generator,
        num_rows: int,
        num_cols: int,
        density: float,
        scale: float,
    ) -> np.ndarray:
        count = min(num_rows, max(1, int(round(float(np.clip(density, 0.0, 1.0)) * num_rows))))
        weights = np.zeros((num_rows, num_cols), dtype=np.float32)
        for col_idx in range(num_cols):
            rows = rng.choice(num_rows, size=count, replace=False)
            weights[rows, col_idx] = scale * rng.standard_normal(count).astype(np.float32)
        return weights

    @staticmethod
    def _assign_input_var(input_var, value):
        target = getattr(input_var, "input", input_var)
        if hasattr(target, "value"):
            target_value = target.value
        else:
            target_value = target

        target_shape = np.asarray(target_value).shape
        value = np.asarray(value, dtype=np.float32).reshape(target_shape)
        if hasattr(target, "value"):
            target.value = bm.asarray(value)
        else:
            target[...] = bm.asarray(value)

    @staticmethod
    def _csr_to_dense(indices, indptr, weights, num_pre: int, num_post: int) -> np.ndarray:
        dense = np.zeros((num_pre, num_post), dtype=np.float32)
        idx = np.asarray(indices, dtype=int)
        ptr = np.asarray(indptr, dtype=int)
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        for pre_idx in range(num_pre):
            start = ptr[pre_idx]
            stop = ptr[pre_idx + 1]
            dense[pre_idx, idx[start:stop]] = w[start:stop]
        return dense

    def _recurrent_spec(self, proj, num_pre: int, num_post: int, sign: float) -> dict[str, object]:
        return {
            "proj": proj,
            "indices": np.asarray(proj.proj.comm.indices, dtype=int),
            "indptr": np.asarray(proj.proj.comm.indptr, dtype=int),
            "weights": np.asarray(proj.proj.comm.weight, dtype=np.float32).reshape(-1),
            "num_pre": int(num_pre),
            "num_post": int(num_post),
            "sign": float(sign),
        }

    def _build_recurrent_specs(self) -> dict[str, dict[str, object]]:
        return {
            "w_ee_raw": self._recurrent_spec(
                self.spatial_model.E2E,
                num_pre=self.n_exc,
                num_post=self.n_exc,
                sign=1.0,
            ),
            "w_ei_raw": self._recurrent_spec(
                self.spatial_model.E2I,
                num_pre=self.n_exc,
                num_post=self.n_inh,
                sign=1.0,
            ),
            "w_ie_raw": self._recurrent_spec(
                self.spatial_model.I2E,
                num_pre=self.n_inh,
                num_post=self.n_exc,
                sign=-1.0,
            ),
            "w_ii_raw": self._recurrent_spec(
                self.spatial_model.I2I,
                num_pre=self.n_inh,
                num_post=self.n_inh,
                sign=-1.0,
            ),
        }

    def _init_spatial_params(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed + 101)
        density = float(np.clip(self.cfg.conn_sparsity, 0.0, 1.0))
        scale_in = 1.0 / np.sqrt(max(1, self.obs_size))
        scale_out = 1.0 / np.sqrt(max(1, self.n_total))
        params = {
            "w_in": self._row_sparse_matrix(
                rng,
                num_rows=self.obs_size,
                num_cols=self.n_total,
                density=density,
                scale=scale_in,
            ),
            "bias_in": np.zeros((self.n_total,), dtype=np.float32),
            "w_out": self._column_sparse_matrix(
                rng,
                num_rows=self.n_total,
                num_cols=self.action_size,
                density=density,
                scale=scale_out,
            ),
            "bias_out": np.zeros((self.action_size,), dtype=np.float32),
        }
        for key, spec in self.recurrent_specs.items():
            params[key] = self._inverse_softplus(spec["weights"])
        return params

    def _trainable_params(self, params) -> dict[str, np.ndarray]:
        return {key: np.asarray(params[key], dtype=np.float32) for key in self._TRAINABLE_RECURRENT_KEYS}

    def _merge_trainable_params(self, base_params, trainable_params) -> dict[str, np.ndarray]:
        merged = {key: np.asarray(value, dtype=np.float32) for key, value in base_params.items()}
        for key, value in trainable_params.items():
            merged[key] = np.asarray(value, dtype=np.float32)
        return merged

    def _dense_projection_from_params(self, params, key: str) -> np.ndarray:
        spec = self.recurrent_specs[key]
        weights = self._softplus(params[key])
        return spec["sign"] * self._csr_to_dense(
            spec["indices"],
            spec["indptr"],
            weights,
            num_pre=spec["num_pre"],
            num_post=spec["num_post"],
        )

    def _apply_recurrent_params(self, params):
        for key, spec in self.recurrent_specs.items():
            spec["proj"].proj.comm.weight = bm.asarray(self._softplus(params[key]))

    def _effective_recurrent_weights(self, params):
        weights = np.zeros((self.n_total, self.n_total), dtype=np.float32)
        weights[: self.n_exc, : self.n_exc] = self._dense_projection_from_params(params, "w_ee_raw")
        weights[: self.n_exc, self.n_exc :] = self._dense_projection_from_params(params, "w_ei_raw")
        weights[self.n_exc :, : self.n_exc] = self._dense_projection_from_params(params, "w_ie_raw")
        weights[self.n_exc :, self.n_exc :] = self._dense_projection_from_params(params, "w_ii_raw")
        return weights

    def _spatial_snapshot(self, filt: np.ndarray, step: int) -> dict[str, np.ndarray]:
        spike = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.spike),
                self._value_to_numpy(self.spatial_model.I.spike),
            ],
            axis=0,
        )
        membrane = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.V),
                self._value_to_numpy(self.spatial_model.I.V),
            ],
            axis=0,
        )
        syn_current = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.input),
                self._value_to_numpy(self.spatial_model.I.input),
            ],
            axis=0,
        )
        adaptation = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.g_K),
                self._value_to_numpy(self.spatial_model.I.g_K),
            ],
            axis=0,
        )
        return {
            "step": int(step),
            "v": membrane.astype(np.float32),
            "syn": syn_current.astype(np.float32),
            "adapt": adaptation.astype(np.float32),
            "spike": spike.astype(np.float32),
            "filt": np.asarray(filt, dtype=np.float32),
        }

    def _controller_initial_state(self):
        bp.reset_state(self.spatial_model)
        filt = np.zeros((self.n_total,), dtype=np.float32)
        return self._spatial_snapshot(filt=filt, step=0)

    def _make_physics(self):
        return CartPolePhysics(
            self.cfg,
            cart_width=self.cart_width,
            cart_height=self.cart_height,
            pole_length=self.pole_length,
            force_limit=self.cart_force_scale,
        )

    def _build_observation(self, state_like, feature_t):
        theta = float(state_like.angle)
        obs = np.asarray(
            [
                feature_t[0],
                feature_t[1],
                state_like.pos[0],
                state_like.vel[0],
                theta,
                state_like.omega,
                np.sin(theta),
                np.cos(theta),
                state_like.last_force,
            ],
            dtype=np.float32,
        )
        return obs / self.obs_scale

    def _decode_targets(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_size)
        primary = self.cart_force_scale * action[: self.cfg.n_legs]
        trim = 0.5 * self.cart_force_scale * action[self.cfg.n_legs :]
        return primary.astype(np.float32), trim.astype(np.float32)

    def _force_from_action(self, action) -> float:
        primary_force, trim_force = self._decode_targets(action)
        return float((primary_force[0] - primary_force[1]) + (trim_force[0] - trim_force[1]))

    def _simulate(self, params, features, progress_label: Optional[str] = None):
        self._apply_recurrent_params(params)
        physics = self._make_physics()
        sensed = physics.observe()
        ctrl_state = self._controller_initial_state()
        progress = None
        progress_stride = None
        if progress_label:
            progress = TerminalProgressBar(self.num_steps, progress_label)
            progress_stride = max(1, self.num_steps // 40)

        pos = np.zeros((self.num_steps, 2), dtype=np.float32)
        vel = np.zeros((self.num_steps, 2), dtype=np.float32)
        angle = np.zeros((self.num_steps,), dtype=np.float32)
        omega = np.zeros((self.num_steps,), dtype=np.float32)
        hip_angle = np.zeros((self.num_steps, self.cfg.n_legs), dtype=np.float32)
        knee_angle = np.zeros((self.num_steps, self.cfg.n_legs), dtype=np.float32)
        foot_pos = np.zeros((self.num_steps, self.cfg.n_legs, 2), dtype=np.float32)
        foot_vel = np.zeros((self.num_steps, self.cfg.n_legs, 2), dtype=np.float32)
        ground_contact = np.zeros((self.num_steps, self.cfg.n_legs), dtype=np.float32)
        force = np.zeros((self.num_steps, 2), dtype=np.float32)
        joint_torque = np.zeros((self.num_steps, self.cfg.n_legs, 2), dtype=np.float32)
        obs_hist = np.zeros((self.num_steps, self.obs_size), dtype=np.float32)
        action_hist = np.zeros((self.num_steps, self.action_size), dtype=np.float32)
        spike_hist = np.zeros((self.num_steps, self.n_total), dtype=np.float32)
        v_hist = np.zeros((self.num_steps, self.n_total), dtype=np.float32)
        syn_hist = np.zeros((self.num_steps, self.n_total), dtype=np.float32)
        adapt_hist = np.zeros((self.num_steps, self.n_total), dtype=np.float32)
        filt_hist = np.zeros((self.num_steps, self.n_total), dtype=np.float32)

        cart_offsets = np.asarray(
            [
                [-0.25 * self.cart_width, -0.5 * self.cart_height],
                [0.25 * self.cart_width, -0.5 * self.cart_height],
            ],
            dtype=np.float32,
        )

        for t in range(self.num_steps):
            obs = self._build_observation(sensed, features[t])
            ctrl_state, action = self._controller_step(params, ctrl_state, obs)
            force_cmd = self._force_from_action(action)
            physics.step(force_cmd)
            next_sensed = physics.observe()

            pos[t] = next_sensed.pos
            vel[t] = next_sensed.vel
            angle[t] = np.float32(next_sensed.angle)
            omega[t] = np.float32(next_sensed.omega)
            foot_pos[t] = next_sensed.pos[None, :] + cart_offsets
            foot_vel[t, :, 0] = next_sensed.vel[0]
            force[t, 0] = np.float32(force_cmd)
            obs_hist[t] = obs
            action_hist[t] = action
            spike_hist[t] = ctrl_state["spike"]
            v_hist[t] = ctrl_state["v"]
            syn_hist[t] = ctrl_state["syn"]
            adapt_hist[t] = ctrl_state["adapt"]
            filt_hist[t] = ctrl_state["filt"]
            sensed = next_sensed

            if progress is not None and ((t + 1) % progress_stride == 0 or (t + 1) == self.num_steps):
                progress.update(t + 1)

        if progress is not None:
            progress.finish()

        return {
            "ts": np.arange(self.num_steps, dtype=np.float32) * self.cfg.dt_ms,
            "pos": pos,
            "vel": vel,
            "angle": angle,
            "omega": omega,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "foot_pos": foot_pos,
            "foot_vel": foot_vel,
            "ground_contact": ground_contact,
            "force": force,
            "joint_torque": joint_torque,
            "obs": obs_hist,
            "action": action_hist,
            "spike": spike_hist,
            "v": v_hist,
            "syn": syn_hist,
            "adapt": adapt_hist,
            "filtered_spike": filt_hist,
            "E.spike": spike_hist[:, : self.n_exc],
            "I.spike": spike_hist[:, self.n_exc :],
        }

    def _controller_step(self, params, ctrl_state, obs):
        obs = np.asarray(obs, dtype=np.float32)
        drive = self.cfg.input_gain * (obs @ params["w_in"]) + params["bias_in"]
        e_drive = drive[: self.n_exc]
        i_drive = drive[self.n_exc :]

        self._assign_input_var(self.spatial_model.Ein, e_drive.astype(np.float32))
        self._assign_input_var(self.spatial_model.Iin, i_drive.astype(np.float32))
        self.spatial_model.step_run(int(ctrl_state["step"]))

        spike = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.spike),
                self._value_to_numpy(self.spatial_model.I.spike),
            ],
            axis=0,
        ).astype(np.float32)
        filt = self.readout_decay * np.asarray(ctrl_state["filt"], dtype=np.float32) + spike
        action = np.tanh(self.cfg.readout_gain * (filt @ params["w_out"] + params["bias_out"]))

        next_state = self._spatial_snapshot(filt=filt, step=int(ctrl_state["step"]) + 1)
        return next_state, action.astype(np.float32)

    def _metrics_from_rollout(self, rollout, features):
        pos = np.asarray(rollout["pos"], dtype=np.float32)
        vel = np.asarray(rollout["vel"], dtype=np.float32)
        angle = np.asarray(rollout["angle"], dtype=np.float32)
        omega = np.asarray(rollout["omega"], dtype=np.float32)
        force = np.asarray(rollout["force"], dtype=np.float32)
        action = np.asarray(rollout["action"], dtype=np.float32)
        target_vx = np.asarray(features[:, 0], dtype=np.float32)

        distance = float(pos[-1, 0] - pos[0, 0])
        mean_vx = float(np.mean(vel[:, 0]))
        speed_tracking = float(np.mean(np.square(vel[:, 0] - target_vx)))
        cart_center_error = float(np.mean(np.square(pos[:, 0] / max(1e-6, self.track_half_width))))
        angle_error = float(np.mean(np.square(angle)))
        omega_error = float(np.mean(np.square(omega)))
        control_effort = float(np.mean(np.square(force[:, 0] / max(1e-6, self.cart_force_scale))))
        if action.shape[0] > 1:
            action_rate = float(np.mean(np.sum(np.square(action[1:] - action[:-1]), axis=1)))
        else:
            action_rate = 0.0

        reward = self.cfg.reward_distance * distance + self.cfg.reward_speed * mean_vx
        reward = reward - self.cfg.penalty_speed_tracking * speed_tracking
        reward = reward - self.cfg.penalty_pitch * angle_error
        reward = reward - self.cfg.penalty_height * cart_center_error
        reward = reward - 0.25 * omega_error
        reward = reward - self.cfg.penalty_energy * control_effort
        reward = reward - self.cfg.penalty_action_rate * action_rate

        loss = -reward
        metrics = {
            "loss": float(loss),
            "reward": float(reward),
            "distance": float(distance),
            "mean_vx": float(mean_vx),
            "height_error": float(cart_center_error),
            "pitch_error": float(angle_error),
            "speed_tracking": float(speed_tracking),
        }
        return loss, metrics

    def train_step(
        self,
        features: Optional[np.ndarray] = None,
        progress_prefix: Optional[str] = None,
    ) -> dict[str, float]:
        features = self._coerce_features(features)
        sigma = float(self.cfg.es_noise_std)
        population = max(1, int(self.cfg.es_population))
        trainable_params = self._trainable_params(self.params)
        grad_accum = jax.tree_util.tree_map(
            lambda p: np.zeros_like(p, dtype=np.float32),
            trainable_params,
        )
        step_prefix = progress_prefix or "optimizer step"

        _log(f"{step_prefix}: estimating recurrent-only gradient with {population} perturbation pairs")

        for sample_idx in range(population):
            sample_tag = f"{step_prefix} pair {sample_idx + 1}/{population}"
            noise = jax.tree_util.tree_map(
                lambda p: self.rng.standard_normal(p.shape).astype(np.float32),
                trainable_params,
            )
            trainable_plus = jax.tree_util.tree_map(
                lambda p, n: p + sigma * n,
                trainable_params,
                noise,
            )
            trainable_minus = jax.tree_util.tree_map(
                lambda p, n: p - sigma * n,
                trainable_params,
                noise,
            )
            params_plus = self._merge_trainable_params(self.params, trainable_plus)
            params_minus = self._merge_trainable_params(self.params, trainable_minus)
            rollout_plus = self._simulate(params_plus, features, progress_label=f"{sample_tag} (+)")
            reward_plus = self._metrics_from_rollout(rollout_plus, features)[1]["reward"]
            rollout_minus = self._simulate(params_minus, features, progress_label=f"{sample_tag} (-)")
            reward_minus = self._metrics_from_rollout(rollout_minus, features)[1]["reward"]
            coeff = self.cfg.es_reward_scale * (reward_plus - reward_minus) / (2.0 * sigma * population)
            grad_accum = jax.tree_util.tree_map(
                lambda g, n: g + coeff * n,
                grad_accum,
                noise,
            )
            _log(
                f"{sample_tag}: reward_plus={reward_plus:.4f} "
                f"reward_minus={reward_minus:.4f} coeff={coeff:.4f}"
            )

        grad_norm = float(_tree_global_norm(grad_accum))
        if grad_norm > self.cfg.gradient_clip > 0.0:
            scale = self.cfg.gradient_clip / (grad_norm + 1e-8)
            grad_accum = jax.tree_util.tree_map(lambda g: g * scale, grad_accum)
            _log(f"{step_prefix}: clipped recurrent gradient to norm {self.cfg.gradient_clip:.4f}")

        _log(f"{step_prefix}: applying Adam update to recurrent weights")
        updated_trainable = self._adam_update(trainable_params, grad_accum)
        self.params = self._merge_trainable_params(self.params, updated_trainable)
        rollout = self._simulate(self.params, features, progress_label=f"{step_prefix} eval")
        loss, metrics = self._metrics_from_rollout(rollout, features)
        metrics["loss"] = float(loss)
        metrics["grad_norm"] = grad_norm
        _log(f"{step_prefix}: complete {_metrics_summary(metrics)}")
        return metrics


TrainableWalkingSystem = TrainableSpatialWalkingSystem


def collect_rollout(
    system: TrainableWalkingSystem, features: Optional[np.ndarray] = None
) -> RolloutRunner:
    rollout, _ = system.evaluate(features)
    return RolloutRunner(mon=rollout)


def _training_worker(cfg: Config, features: np.ndarray, message_queue, stop_event):
    try:
        _log("spatial backend worker booting")
        system = TrainableWalkingSystem(cfg)
        refresh_every = cfg.vis_every if cfg.vis_every > 0 else cfg.eval_every
        train_forever = cfg.train_epochs <= 0
        total_epochs_label = "inf" if train_forever else str(cfg.train_epochs)
        epoch_progress = (
            None if train_forever else TerminalProgressBar(max(1, cfg.train_epochs), "training epochs")
        )
        _log(
            f"spatial backend ready: epochs={total_epochs_label} refresh_every={refresh_every} "
            f"es_population={cfg.es_population}"
        )

        _queue_put_latest(
            message_queue,
            {
                "type": "status",
                "phase": "evaluating",
                "detail": "running initial rollout",
            },
        )
        _log("starting initial spatial rollout for viewer bootstrap")
        rollout, metrics = system.evaluate_with_progress(
            features,
            progress_label="initial rollout",
        )
        _log(f"initial spatial rollout complete {_metrics_summary(metrics)}")
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
                    "detail": "running optimizer step",
                },
            )
            _log(f"epoch {epoch + 1}/{total_epochs_label}: starting spatial optimizer step")
            latest_metrics = system.train_step(
                features,
                progress_prefix=f"epoch {epoch + 1}/{total_epochs_label}",
            )
            epoch += 1
            if epoch_progress is not None:
                epoch_progress.update(epoch)
            _log(
                f"epoch {epoch}/{total_epochs_label}: spatial optimizer step complete "
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
                _log(f"epoch {epoch}/{total_epochs_label}: refreshing spatial viewer rollout")
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
                    f"epoch {epoch}/{total_epochs_label}: refresh spatial rollout complete "
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
        _log(f"spatial training worker finished at epoch {epoch}/{total_epochs_label}")
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
        _log(f"spatial backend worker failed: {exc}")
        _queue_put_latest(
            message_queue,
            {
                "type": "error",
                "phase": "error",
                "detail": str(exc),
            },
        )


def main():
    _require_runtime()
    print("[trainable_spatial_system] building system...", flush=True)
    cfg = Config()
    system = TrainableWalkingSystem(cfg)
    print(
        "[trainable_spatial_system] system ready "
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
    print("[trainable_spatial_system] starting backend worker...", flush=True)
    worker_process.start()
    print("[trainable_spatial_system] opening training viewer...", flush=True)
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
    "TrainableSpatialWalkingSystem",
    "TrainableWalkingSystem",
    "build_feature_sequence",
    "collect_rollout",
    "main",
]


if __name__ == "__main__":
    main()
