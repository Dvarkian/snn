from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import queue
import time
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np


def _ensure_writable_runtime_dirs():
    runtime_home = os.path.expanduser("~")
    try:
        probe_path = os.path.join(runtime_home, ".wrcircuit_write_probe")
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write("")
        os.remove(probe_path)
    except OSError:
        runtime_home = os.path.join("/tmp", "wrcircuit-home")
        os.makedirs(runtime_home, exist_ok=True)
        os.environ["HOME"] = runtime_home

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = os.path.join(runtime_home, ".config", "matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir


_ensure_writable_runtime_dirs()

import matplotlib.pyplot as plt


_JAX_IMPORT_ERROR = None

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - exercised only when runtime is missing.
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc


def _require_runtime():
    missing = []
    if jax is None:
        missing.append(f"jax ({_JAX_IMPORT_ERROR})")
    if missing:
        raise ImportError(
            "trainable_system.py requires pip-installable Python packages. "
            "Install at least: numpy matplotlib jax jaxlib. "
            f"Missing runtime pieces: {', '.join(missing)}"
        )


def _import_pymunk_walker():
    try:
        from walking_physics_pymunk import (
            PymunkPassiveWalker,
            initial_joint_configuration,
            make_initial_state,
        )
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        raise ImportError(
            "Unable to import walking_physics_pymunk. Install pymunk first."
        ) from exc
    return PymunkPassiveWalker, initial_joint_configuration, make_initial_state


if jax is not None:

    @jax.custom_jvp
    def surrogate_spike(x):
        return (x > 0.0).astype(x.dtype)


    @surrogate_spike.defjvp
    def _surrogate_spike_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        y = surrogate_spike(x)
        grad = 1.0 / (1.0 + jnp.abs(x)) ** 2
        return y, grad * x_dot

else:

    def surrogate_spike(x):  # pragma: no cover - runtime guard catches this path.
        raise ImportError("JAX is required for surrogate_spike().")


@dataclass
class Config:
    random_seed: int = 7

    # Compact trainable spatial controller.
    rho: int = 6000
    controller_dx_mm: float = 0.20
    gamma: float = 4.0
    sigma_mm: float = 0.050
    conn_sparsity: float = 0.55
    membrane_tau_ms: float = 18.0
    synapse_tau_ms: float = 12.0
    adapt_tau_ms: float = 80.0
    adapt_strength: float = 0.35
    v_th: float = 1.0
    v_reset: float = 0.0
    input_gain: float = 0.35
    recurrent_gain: float = 0.18
    readout_tau_ms: float = 24.0
    readout_gain: float = 0.22
    hip_action_scale: float = 0.42
    knee_action_scale: float = 0.55

    # Task / training.
    target_vx: float = 0.40
    target_vy: float = 0.0
    episode_ms: float = 1500.0
    dt_ms: float = 4.0
    train_epochs: int = 300
    learning_rate: float = 2e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    eval_every: int = 5
    vis_every: int = 5
    ui_interval_ms: int = 40
    optim_steps_per_tick: int = 1
    animation_fps: float = 30.0
    window_size_ms: float = 40.0
    es_population: int = 4
    es_noise_std: float = 0.03
    es_reward_scale: float = 1.0

    # Reward shaping.
    reward_distance: float = 6.0
    reward_speed: float = 2.5
    penalty_speed_tracking: float = 4.0
    penalty_height: float = 6.0
    penalty_pitch: float = 4.0
    penalty_energy: float = 0.01
    penalty_action_rate: float = 0.03
    penalty_slip: float = 0.10
    penalty_collapse: float = 8.0
    penalty_joint_limit: float = 0.5

    # Exact Pymunk walker configuration. These mirror run_walking_physics.py.
    n_legs: int = 2
    mass: float = 2.0
    thigh_mass: float = 0.25
    shank_mass: float = 0.18
    drag: float = 0.12
    angular_drag: float = 1.2
    joint_drag: float = 0.18
    gravity: float = 9.81
    ground_k: float = 1500.0
    ground_c: float = 35.0
    ground_tangent_damping: float = 80.0
    friction_mu: float = 0.9
    body_length: float = 0.45
    body_height: float = 0.18
    hip_x_offset: float = 0.16
    thigh_length: float = 0.22
    shank_length: float = 0.22
    leg_radius: float = 0.022
    foot_radius: float = 0.028
    body_corner_radius: float = 0.012
    height_target: float = 0.48
    hip_limit: float = 0.95
    knee_min: float = 0.05
    knee_max: float = 1.45
    hip_kp: float = 55.0
    hip_kd: float = 6.0
    knee_kp: float = 85.0
    knee_kd: float = 8.0
    hip_torque_limit: float = 18.0
    knee_torque_limit: float = 24.0
    physics_substeps: int = 10
    solver_iterations: int = 50
    collision_slop: float = 1e-3
    contact_epsilon: float = 2e-3
    drop_height: float = 0.16
    initial_pitch: float = 0.22
    initial_omega: float = 0.0
    initial_body_x: float = 0.0
    initial_body_vx: float = 0.0
    initial_body_vy: float = 0.0
    hip_offsets: tuple[float, float] = (0.18, -0.10)
    knee_offsets: tuple[float, float] = (-0.10, 0.06)


@dataclass
class RolloutRunner:
    mon: Dict[str, np.ndarray]


@dataclass
class AdamState:
    step: int
    m: Any
    v: Any


def build_feature_sequence(
    num_steps: int,
    dt_ms: float,
    target_vx: float,
    target_vy: float,
) -> np.ndarray:
    return np.stack(
        [
            np.full((num_steps,), target_vx, dtype=np.float32),
            np.full((num_steps,), target_vy, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _grid_positions(side: int, extent_mm: float) -> np.ndarray:
    xs = np.linspace(0.0, extent_mm, side, dtype=float)
    ys = np.linspace(0.0, extent_mm, side, dtype=float)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def _tree_global_norm(tree) -> Any:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return jnp.sqrt(total + 1e-12)


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _python_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


def _queue_put_latest(message_queue, message):
    try:
        message_queue.put_nowait(message)
        return
    except queue.Full:
        pass

    try:
        while True:
            message_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        message_queue.put_nowait(message)
    except queue.Full:
        pass


def _log(message: str):
    print(f"[trainable_system] {message}", flush=True)


def _metrics_summary(metrics: Dict[str, float]) -> str:
    fields = []
    for key in ("loss", "reward", "distance", "mean_vx", "height_error", "pitch_error"):
        if key in metrics:
            fields.append(f"{key}={metrics[key]:.4f}")
    if "grad_norm" in metrics:
        fields.append(f"grad_norm={metrics['grad_norm']:.4f}")
    return " ".join(fields)


class TerminalProgressBar:
    def __init__(self, total: int, label: str, width: int = 28):
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.current = 0
        self.start = time.perf_counter()
        self._draw(0)

    def _draw(self, current: int):
        current = int(np.clip(current, 0, self.total))
        frac = current / float(self.total)
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.start
        print(
            f"\r[trainable_system] {self.label:<24} [{bar}] "
            f"{100.0 * frac:6.2f}% ({current:>4d}/{self.total:<4d}) "
            f"{elapsed:6.1f}s",
            end="",
            flush=True,
        )

    def update(self, current: int):
        if current <= self.current and current < self.total:
            return
        self.current = min(int(current), self.total)
        self._draw(self.current)

    def finish(self):
        self.current = self.total
        self._draw(self.total)
        print("", flush=True)


class TrainableWalkingSystem:
    def __init__(self, cfg: Config):
        _require_runtime()
        if cfg.n_legs != 2:
            raise ValueError("TrainableWalkingSystem currently expects n_legs == 2.")

        self.cfg = cfg
        self.num_steps = int(round(cfg.episode_ms / cfg.dt_ms))
        if self.num_steps < 1:
            raise ValueError("episode_ms must produce at least one simulation step.")
        self.compute_backend = "pymunk"
        self.rng = np.random.default_rng(cfg.random_seed)

        (
            self.PymunkPassiveWalker,
            self.passive_initial_joint_configuration,
            self.make_initial_state,
        ) = _import_pymunk_walker()

        self.exc_side = max(4, int(round(np.sqrt(cfg.rho * cfg.controller_dx_mm**2))))
        self.n_exc = self.exc_side * self.exc_side
        self.n_inh = max(1, int(round(self.n_exc / cfg.gamma)))
        self.n_total = self.n_exc + self.n_inh
        self.obs_size = 21
        self.action_size = 2 * cfg.n_legs
        self.observation_labels = [
            "target vx",
            "target vy",
            "body y",
            "vel x",
            "vel y",
            "pitch",
            "omega",
            "hip 0",
            "hip 1",
            "knee 0",
            "knee 1",
            "hip w0",
            "hip w1",
            "knee w0",
            "knee w1",
            "foot y0",
            "foot y1",
            "foot vx0",
            "foot vx1",
            "contact 0",
            "contact 1",
        ]
        self.action_labels = ["hip 0", "hip 1", "knee 0", "knee 1"]

        self.exc_positions = _grid_positions(self.exc_side, cfg.controller_dx_mm)
        key = jax.random.PRNGKey(cfg.random_seed)
        key, inh_key, mask_key, param_key = jax.random.split(key, 4)
        inh_positions = np.asarray(
            jax.random.uniform(
                inh_key,
                shape=(self.n_inh, 2),
                minval=0.0,
                maxval=cfg.controller_dx_mm,
            )
        )
        self.all_positions = jnp.asarray(
            np.concatenate([self.exc_positions, inh_positions], axis=0), dtype=jnp.float32
        )

        self.E = SimpleNamespace(
            positions=self.exc_positions,
            embedding=SimpleNamespace(
                domain=np.asarray([cfg.controller_dx_mm, cfg.controller_dx_mm], dtype=float)
            ),
            size=np.asarray([self.exc_side, self.exc_side], dtype=int),
            num=self.n_exc,
        )
        self.I = SimpleNamespace(
            positions=inh_positions,
            embedding=SimpleNamespace(
                domain=np.asarray([cfg.controller_dx_mm, cfg.controller_dx_mm], dtype=float)
            ),
            size=np.asarray([self.n_inh], dtype=int),
            num=self.n_inh,
        )

        self.exc_mask = np.concatenate(
            [np.ones((self.n_exc,), dtype=np.float32), np.zeros((self.n_inh,), dtype=np.float32)],
            axis=0,
        )
        self.pre_sign = np.concatenate(
            [np.ones((self.n_exc,), dtype=np.float32), -np.ones((self.n_inh,), dtype=np.float32)],
            axis=0,
        )

        all_positions_np = np.asarray(self.all_positions, dtype=np.float32)
        distance = all_positions_np[:, None, :] - all_positions_np[None, :, :]
        dist_sq = np.sum(distance**2, axis=-1)
        sigma_sq = max(cfg.sigma_mm**2, 1e-6)
        self.distance_kernel = np.exp(-0.5 * dist_sq / sigma_sq).astype(np.float32)
        self.distance_kernel = self.distance_kernel * (
            1.0 - np.eye(self.n_total, dtype=np.float32)
        )

        keep_prob = np.clip(cfg.conn_sparsity * self.distance_kernel, 0.0, 1.0)
        self.recurrent_mask = (
            self.rng.uniform(size=(self.n_total, self.n_total)) < keep_prob
        ).astype(np.float32)
        self.recurrent_mask = self.recurrent_mask * (
            1.0 - np.eye(self.n_total, dtype=np.float32)
        )

        self.membrane_decay = float(np.exp(-cfg.dt_ms / cfg.membrane_tau_ms))
        self.synapse_decay = float(np.exp(-cfg.dt_ms / cfg.synapse_tau_ms))
        self.adapt_decay = float(np.exp(-cfg.dt_ms / cfg.adapt_tau_ms))
        self.readout_decay = float(np.exp(-cfg.dt_ms / cfg.readout_tau_ms))

        nominal_hip, nominal_knee = self.passive_initial_joint_configuration(cfg)
        self.base_hip = np.asarray([nominal_hip, nominal_hip], dtype=np.float32)
        self.base_knee = np.asarray([nominal_knee, nominal_knee], dtype=np.float32)

        self.obs_scale = np.asarray(
            [
                max(1.0, abs(cfg.target_vx)),
                max(1.0, abs(cfg.target_vy)),
                cfg.height_target,
                1.0,
                1.0,
                0.7,
                2.0,
                cfg.hip_limit,
                cfg.hip_limit,
                cfg.knee_max,
                cfg.knee_max,
                2.0,
                2.0,
                2.0,
                2.0,
                0.6,
                0.6,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        self.params = _tree_to_numpy(self._init_params(param_key))
        self.opt_state = self._adam_init(self.params)

    def _init_params(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        scale_in = 1.0 / np.sqrt(max(1, self.obs_size))
        scale_rec = 1.0 / np.sqrt(max(1, self.n_total))
        scale_out = 1.0 / np.sqrt(max(1, self.n_total))
        return {
            "w_in": jax.random.normal(k1, (self.obs_size, self.n_total)) * scale_in,
            "w_rec_raw": jax.random.normal(k2, (self.n_total, self.n_total)) * scale_rec,
            "bias_rec": jnp.zeros((self.n_total,), dtype=jnp.float32),
            "w_out": jax.random.normal(k3, (self.n_total, self.action_size)) * scale_out,
            "bias_out": jnp.zeros((self.action_size,), dtype=jnp.float32),
        }

    def _adam_init(self, params) -> AdamState:
        zeros = jax.tree_util.tree_map(lambda x: np.zeros_like(x, dtype=np.float32), params)
        return AdamState(step=0, m=zeros, v=zeros)

    def _adam_update(self, params, grads):
        cfg = self.cfg
        state = self.opt_state
        step = state.step + 1
        m = jax.tree_util.tree_map(
            lambda mm, gg: cfg.adam_beta1 * mm + (1.0 - cfg.adam_beta1) * gg.astype(np.float32),
            state.m,
            grads,
        )
        v = jax.tree_util.tree_map(
            lambda vv, gg: cfg.adam_beta2 * vv + (1.0 - cfg.adam_beta2) * np.square(gg),
            state.v,
            grads,
        )
        m_hat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - cfg.adam_beta1**step), m)
        v_hat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - cfg.adam_beta2**step), v)
        params = jax.tree_util.tree_map(
            lambda pp, mm, vv: (1.0 - cfg.learning_rate * cfg.weight_decay) * pp
            - cfg.learning_rate * mm / (np.sqrt(vv) + cfg.adam_eps),
            params,
            m_hat,
            v_hat,
        )
        self.opt_state = AdamState(step=step, m=m, v=v)
        return params

    def _controller_initial_state(self):
        zeros = np.zeros((self.n_total,), dtype=np.float32)
        return {
            "v": zeros,
            "syn": zeros,
            "adapt": zeros,
            "spike": zeros,
            "filt": zeros,
        }

    def _make_physics(self):
        initial_state = self.make_initial_state(
            self.cfg,
            hip_offsets=np.asarray(self.cfg.hip_offsets, dtype=float),
            knee_offsets=np.asarray(self.cfg.knee_offsets, dtype=float),
        )
        return self.PymunkPassiveWalker(self.cfg, initial_state)

    def _effective_recurrent_weights(self, params):
        magnitude = np.log1p(np.exp(params["w_rec_raw"]))
        sign = self.pre_sign[:, None]
        return (
            self.cfg.recurrent_gain
            * self.recurrent_mask
            * self.distance_kernel
            * sign
            * magnitude
        )

    def _controller_step(self, params, ctrl_state, obs):
        recurrent = ctrl_state["spike"] @ self._effective_recurrent_weights(params)
        current = (
            self.cfg.input_gain * (obs @ params["w_in"])
            + recurrent
            + params["bias_rec"]
        )
        syn = self.synapse_decay * ctrl_state["syn"] + current
        v_candidate = self.membrane_decay * ctrl_state["v"] + (1.0 - self.membrane_decay) * syn
        v_candidate = v_candidate - self.cfg.adapt_strength * ctrl_state["adapt"] * self.exc_mask
        spike = (v_candidate > self.cfg.v_th).astype(np.float32)
        v_new = np.where(spike > 0.0, self.cfg.v_reset, v_candidate)
        adapt = self.adapt_decay * ctrl_state["adapt"] + self.exc_mask * spike
        filt = self.readout_decay * ctrl_state["filt"] + spike
        action = np.tanh(self.cfg.readout_gain * (filt @ params["w_out"] + params["bias_out"]))
        return {
            "v": v_new,
            "syn": syn,
            "adapt": adapt,
            "spike": spike,
            "filt": filt,
        }, action

    def _build_observation(self, state_like, feature_t):
        obs = np.concatenate(
            [
                feature_t,
                np.asarray([state_like.pos[1]], dtype=np.float32),
                np.asarray(state_like.vel, dtype=np.float32),
                np.asarray([state_like.angle, state_like.omega], dtype=np.float32),
                np.asarray(state_like.hip_angle, dtype=np.float32),
                np.asarray(state_like.knee_angle, dtype=np.float32),
                np.asarray(state_like.hip_omega, dtype=np.float32),
                np.asarray(state_like.knee_omega, dtype=np.float32),
                np.asarray(state_like.foot_pos[:, 1], dtype=np.float32),
                np.asarray(state_like.foot_vel[:, 0], dtype=np.float32),
                np.asarray(state_like.ground_contact, dtype=np.float32),
            ],
            axis=0,
        )
        return obs / self.obs_scale

    def _decode_targets(self, action):
        hip = np.clip(
            self.base_hip + self.cfg.hip_action_scale * action[: self.cfg.n_legs],
            -self.cfg.hip_limit,
            self.cfg.hip_limit,
        )
        knee = np.clip(
            self.base_knee + self.cfg.knee_action_scale * action[self.cfg.n_legs :],
            self.cfg.knee_min,
            self.cfg.knee_max,
        )
        return hip, knee

    def _coerce_features(self, features: Optional[np.ndarray]) -> Any:
        if features is None:
            features = build_feature_sequence(
                self.num_steps,
                self.cfg.dt_ms,
                self.cfg.target_vx,
                self.cfg.target_vy,
            )
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.num_steps, 2):
            raise ValueError(
                f"Expected features with shape {(self.num_steps, 2)}, got {features.shape}."
            )
        return features

    def _simulate(self, params, features, progress_label: Optional[str] = None):
        physics = self._make_physics()
        sensed = physics.observe()
        ctrl_state = self._controller_initial_state()
        progress = None
        progress_stride = None
        if progress_label:
            progress = TerminalProgressBar(self.num_steps, progress_label)
            progress_stride = max(1, self.num_steps // 40)

        pos = []
        vel = []
        angle = []
        omega = []
        hip_angle = []
        knee_angle = []
        foot_pos = []
        foot_vel = []
        ground_contact = []
        force = []
        joint_torque = []
        obs_hist = []
        action_hist = []
        spike_hist = []
        v_hist = []
        syn_hist = []
        adapt_hist = []
        filt_hist = []

        for t in range(self.num_steps):
            obs = self._build_observation(sensed, features[t])
            ctrl_state, action = self._controller_step(params, ctrl_state, obs)
            hip_target, knee_target = self._decode_targets(action)
            physics.step(hip_target=hip_target, knee_target=knee_target)
            sensed = physics.observe()

            pos.append(np.asarray(sensed.pos, dtype=np.float32))
            vel.append(np.asarray(sensed.vel, dtype=np.float32))
            angle.append(np.asarray(sensed.angle, dtype=np.float32))
            omega.append(np.asarray(sensed.omega, dtype=np.float32))
            hip_angle.append(np.asarray(sensed.hip_angle, dtype=np.float32))
            knee_angle.append(np.asarray(sensed.knee_angle, dtype=np.float32))
            foot_pos.append(np.asarray(sensed.foot_pos, dtype=np.float32))
            foot_vel.append(np.asarray(sensed.foot_vel, dtype=np.float32))
            ground_contact.append(np.asarray(sensed.ground_contact, dtype=np.float32))
            force.append(np.asarray(sensed.total_ground_force, dtype=np.float32))
            joint_torque.append(np.asarray(physics.last_joint_torque, dtype=np.float32))
            obs_hist.append(obs)
            action_hist.append(action)
            spike_hist.append(ctrl_state["spike"])
            v_hist.append(ctrl_state["v"])
            syn_hist.append(ctrl_state["syn"])
            adapt_hist.append(ctrl_state["adapt"])
            filt_hist.append(ctrl_state["filt"])
            if progress is not None and ((t + 1) % progress_stride == 0 or (t + 1) == self.num_steps):
                progress.update(t + 1)

        spikes = np.stack(spike_hist, axis=0).astype(np.float32)
        if progress is not None:
            progress.finish()
        return {
            "ts": np.arange(self.num_steps, dtype=np.float32) * self.cfg.dt_ms,
            "pos": np.stack(pos, axis=0).astype(np.float32),
            "vel": np.stack(vel, axis=0).astype(np.float32),
            "angle": np.stack(angle, axis=0).astype(np.float32),
            "omega": np.stack(omega, axis=0).astype(np.float32),
            "hip_angle": np.stack(hip_angle, axis=0).astype(np.float32),
            "knee_angle": np.stack(knee_angle, axis=0).astype(np.float32),
            "foot_pos": np.stack(foot_pos, axis=0).astype(np.float32),
            "foot_vel": np.stack(foot_vel, axis=0).astype(np.float32),
            "ground_contact": np.stack(ground_contact, axis=0).astype(np.float32),
            "force": np.stack(force, axis=0).astype(np.float32),
            "joint_torque": np.stack(joint_torque, axis=0).astype(np.float32),
            "obs": np.stack(obs_hist, axis=0).astype(np.float32),
            "action": np.stack(action_hist, axis=0).astype(np.float32),
            "spike": spikes,
            "v": np.stack(v_hist, axis=0).astype(np.float32),
            "syn": np.stack(syn_hist, axis=0).astype(np.float32),
            "adapt": np.stack(adapt_hist, axis=0).astype(np.float32),
            "filtered_spike": np.stack(filt_hist, axis=0).astype(np.float32),
            "E.spike": spikes[:, : self.n_exc],
            "I.spike": spikes[:, self.n_exc :],
        }

    def _metrics_from_rollout(self, rollout, features):
        cfg = self.cfg
        pos = rollout["pos"]
        vel = rollout["vel"]
        angle = rollout["angle"]
        contact = rollout["ground_contact"]
        foot_vel = rollout["foot_vel"]
        action = rollout["action"]
        joint_torque = rollout["joint_torque"]
        hip_angle = rollout["hip_angle"]
        knee_angle = rollout["knee_angle"]

        distance = pos[-1, 0] - pos[0, 0]
        mean_vx = float(np.mean(vel[:, 0]))
        speed_tracking = float(np.mean(np.square(vel[:, 0] - cfg.target_vx)))
        speed_tracking = speed_tracking + 0.25 * float(np.mean(np.square(vel[:, 1] - cfg.target_vy)))
        height_error = float(np.mean(np.square(pos[:, 1] - cfg.height_target)))
        pitch_error = float(np.mean(np.square(angle)))
        energy = float(np.mean(np.sum(np.square(joint_torque), axis=(1, 2))))
        if action.shape[0] > 1:
            action_rate = float(np.mean(np.sum(np.square(action[1:] - action[:-1]), axis=1)))
        else:
            action_rate = 0.0
        slip = float(np.mean(np.sum(contact * np.square(foot_vel[:, :, 0]), axis=1)))
        collapse = float(np.mean(np.square(np.maximum(0.30 - pos[:, 1], 0.0))))
        hip_limit_penalty = float(np.mean(
            np.square(np.maximum(np.abs(hip_angle) - 0.95 * cfg.hip_limit, 0.0))
        ))
        knee_limit_penalty = float(np.mean(
            np.square(np.maximum(cfg.knee_min + 0.02 - knee_angle, 0.0))
            + np.square(np.maximum(knee_angle - (cfg.knee_max - 0.02), 0.0))
        ))
        joint_limit_penalty = hip_limit_penalty + knee_limit_penalty

        reward = cfg.reward_distance * distance + cfg.reward_speed * mean_vx
        reward = reward - cfg.penalty_speed_tracking * speed_tracking
        reward = reward - cfg.penalty_height * height_error
        reward = reward - cfg.penalty_pitch * pitch_error
        reward = reward - cfg.penalty_energy * energy
        reward = reward - cfg.penalty_action_rate * action_rate
        reward = reward - cfg.penalty_slip * slip
        reward = reward - cfg.penalty_collapse * collapse
        reward = reward - cfg.penalty_joint_limit * joint_limit_penalty

        loss = -reward
        metrics = {
            "loss": float(loss),
            "reward": float(reward),
            "distance": float(distance),
            "mean_vx": float(mean_vx),
            "height_error": float(height_error),
            "pitch_error": float(pitch_error),
            "speed_tracking": float(speed_tracking),
        }
        return loss, metrics

    def train_step(
        self,
        features: Optional[np.ndarray] = None,
        progress_prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        features = self._coerce_features(features)
        sigma = float(self.cfg.es_noise_std)
        population = max(1, int(self.cfg.es_population))
        grad_accum = jax.tree_util.tree_map(lambda p: np.zeros_like(p, dtype=np.float32), self.params)
        step_prefix = progress_prefix or "optimizer step"

        _log(f"{step_prefix}: estimating gradient with {population} perturbation pairs")

        for sample_idx in range(population):
            sample_tag = f"{step_prefix} pair {sample_idx + 1}/{population}"
            noise = jax.tree_util.tree_map(
                lambda p: self.rng.standard_normal(p.shape).astype(np.float32),
                self.params,
            )
            params_plus = jax.tree_util.tree_map(lambda p, n: p + sigma * n, self.params, noise)
            params_minus = jax.tree_util.tree_map(lambda p, n: p - sigma * n, self.params, noise)
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
            _log(f"{step_prefix}: clipped gradient to norm {self.cfg.gradient_clip:.4f}")

        _log(f"{step_prefix}: applying Adam update")
        self.params = self._adam_update(self.params, grad_accum)
        rollout = self._simulate(self.params, features, progress_label=f"{step_prefix} eval")
        loss, metrics = self._metrics_from_rollout(rollout, features)
        metrics["loss"] = float(loss)
        metrics["grad_norm"] = grad_norm
        _log(f"{step_prefix}: complete {_metrics_summary(metrics)}")
        return metrics

    def evaluate(self, features: Optional[np.ndarray] = None):
        features = self._coerce_features(features)
        rollout = self._simulate(self.params, features)
        _, metrics = self._metrics_from_rollout(rollout, features)
        return rollout, metrics

    def evaluate_with_progress(
        self, features: Optional[np.ndarray] = None, progress_label: str = "rollout"
    ):
        features = self._coerce_features(features)
        rollout = self._simulate(self.params, features, progress_label=progress_label)
        _, metrics = self._metrics_from_rollout(rollout, features)
        return rollout, metrics


def collect_rollout(
    system: TrainableWalkingSystem, features: Optional[np.ndarray] = None
) -> RolloutRunner:
    rollout, _ = system.evaluate(features)
    return RolloutRunner(mon=rollout)


def prepare_spike_histograms_for_times(
    system: TrainableWalkingSystem,
    runner: RolloutRunner,
    frame_times: np.ndarray,
    window_size_ms: float,
    progress_label: Optional[str] = None,
):
    ts = np.asarray(runner.mon["ts"], dtype=float)
    e_spikes = np.asarray(runner.mon["E.spike"], dtype=float)
    e_positions = np.asarray(system.E.positions, dtype=float)
    domain = np.asarray(system.E.embedding.domain, dtype=float)
    grid_size = np.asarray(system.E.size, dtype=int)

    x_edges = np.linspace(0.0, domain[0], grid_size[0] + 1)
    y_edges = np.linspace(0.0, domain[1], grid_size[1] + 1)
    histograms = np.zeros((len(frame_times), grid_size[0], grid_size[1]), dtype=float)
    progress = None
    progress_stride = None
    if progress_label:
        progress = TerminalProgressBar(len(frame_times), progress_label)
        progress_stride = max(1, len(frame_times) // 40)

    for i, frame_t in enumerate(frame_times):
        win_start_t = frame_t - window_size_ms
        idx_start = np.searchsorted(ts, win_start_t, side="left")
        idx_end = np.searchsorted(ts, frame_t, side="right")
        spike_counts = np.sum(e_spikes[idx_start:idx_end, :], axis=0)
        hist, _, _ = np.histogram2d(
            e_positions[:, 0],
            e_positions[:, 1],
            bins=[x_edges, y_edges],
            weights=spike_counts,
        )
        histograms[i] = hist
        if progress is not None and ((i + 1) % progress_stride == 0 or (i + 1) == len(frame_times)):
            progress.update(i + 1)

    if progress is not None:
        progress.finish()
    return histograms, domain


def _training_worker(cfg: Config, features: np.ndarray, message_queue, stop_event):
    try:
        _log("backend worker booting")
        system = TrainableWalkingSystem(cfg)
        refresh_every = cfg.vis_every if cfg.vis_every > 0 else cfg.eval_every
        epoch_progress = TerminalProgressBar(max(1, cfg.train_epochs), "training epochs")
        _log(
            f"backend ready: epochs={cfg.train_epochs} refresh_every={refresh_every} "
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
        _log("starting initial rollout for viewer bootstrap")
        rollout, metrics = system.evaluate_with_progress(
            features,
            progress_label="initial rollout",
        )
        _log(f"initial rollout complete {_metrics_summary(metrics)}")
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
        while epoch < cfg.train_epochs and not stop_event.is_set():
            _queue_put_latest(
                message_queue,
                {
                    "type": "status",
                    "phase": "training",
                    "detail": "running optimizer step",
                },
            )
            _log(f"epoch {epoch + 1}/{cfg.train_epochs}: starting optimizer step")
            latest_metrics = system.train_step(
                features,
                progress_prefix=f"epoch {epoch + 1}/{cfg.train_epochs}",
            )
            epoch += 1
            epoch_progress.update(epoch)
            _log(f"epoch {epoch}/{cfg.train_epochs}: optimizer step complete {_metrics_summary(latest_metrics)}")

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

            should_refresh = epoch == 1 or epoch % max(1, refresh_every) == 0 or epoch >= cfg.train_epochs
            if should_refresh and not stop_event.is_set():
                _log(f"epoch {epoch}/{cfg.train_epochs}: refreshing viewer rollout")
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
                _log(f"epoch {epoch}/{cfg.train_epochs}: refresh rollout complete {_metrics_summary(eval_metrics)}")
                _queue_put_latest(
                    message_queue,
                    {
                        "type": "snapshot",
                        "epoch": epoch,
                        "rollout": rollout,
                        "metrics": eval_metrics,
                        "params": system.params,
                        "phase": "idle" if epoch < cfg.train_epochs else "done",
                        "detail": "snapshot ready",
                    },
                )

        epoch_progress.finish()
        _log(f"training worker finished at epoch {epoch}/{cfg.train_epochs}")
        _queue_put_latest(
            message_queue,
            {
                "type": "done",
                "epoch": epoch,
                "phase": "done",
                "detail": "training complete",
            },
        )
    except Exception as exc:  # pragma: no cover - worker-side runtime path.
        _log(f"backend worker failed: {exc}")
        _queue_put_latest(
            message_queue,
            {
                "type": "error",
                "phase": "error",
                "detail": str(exc),
            },
        )


class TrainingViewer:
    def __init__(
        self,
        system: TrainableWalkingSystem,
        features: np.ndarray,
        message_queue,
        stop_event,
        worker_process,
    ):
        self.system = system
        self.cfg = system.cfg
        self.features = features
        self.message_queue = message_queue
        self.stop_event = stop_event
        self.worker_process = worker_process
        self.closed = False
        self.epoch = 0
        self.last_epoch_seen = -1

        self.history_steps = []
        self.history_distance = []
        self.history_speed = []
        self.history_loss = []

        self.latest_rollout = None
        self.latest_metrics = None
        self.frame_indices = np.asarray([0], dtype=int)
        self.frame_times = np.asarray([0.0], dtype=float)
        self.frame_ptr = 0
        self.histograms = np.zeros((1, self.system.exc_side, self.system.exc_side), dtype=float)
        self.current_phase = "starting"
        self.phase_detail = "initializing viewer"
        self.last_phase_seconds = 0.0
        self.train_tick_count = 0
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.worker_done = False
        self.robot_y_limits = (0.0, 1.0)
        self.camera_half_width = max(
            0.5,
            self.cfg.body_length + self.cfg.thigh_length + self.cfg.shank_length + 0.2,
        )
        self.latest_params = _tree_to_numpy(system.params)

        self.fig = plt.figure(figsize=(19, 10))
        outer = self.fig.add_gridspec(
            2,
            3,
            width_ratios=[1.15, 1.15, 0.95],
            height_ratios=[1.0, 1.0],
            hspace=0.24,
            wspace=0.26,
        )
        left = outer[:, 0].subgridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.38, wspace=0.30)
        weights = outer[:, 1].subgridspec(3, 1, height_ratios=[1.0, 1.2, 0.95], hspace=0.38)
        bottom_weights = weights[2, 0].subgridspec(1, 2, wspace=0.35)
        right = outer[:, 2].subgridspec(3, 1, height_ratios=[1.6, 0.9, 0.7], hspace=0.35)
        self.ax_spike = self.fig.add_subplot(left[0, 0])
        self.ax_membrane = self.fig.add_subplot(left[0, 1])
        self.ax_filtered = self.fig.add_subplot(left[1, 0])
        self.ax_adapt = self.fig.add_subplot(left[1, 1])
        self.ax_obs = self.fig.add_subplot(left[2, 0])
        self.ax_action = self.fig.add_subplot(left[2, 1])
        self.ax_w_in = self.fig.add_subplot(weights[0, 0])
        self.ax_w_rec = self.fig.add_subplot(weights[1, 0])
        self.ax_w_out = self.fig.add_subplot(bottom_weights[0, 0])
        self.ax_rec_summary = self.fig.add_subplot(bottom_weights[0, 1])
        self.ax_robot = self.fig.add_subplot(right[0, 0])
        self.ax_metric = self.fig.add_subplot(right[1, 0])
        self.ax_status = self.fig.add_subplot(right[2, 0])

        self._init_network_panel()
        self._init_weight_panel()
        self._init_robot_panel()
        self._init_metric_panel()

        self._init_status_panel()
        self.status_text = self.ax_status.text(
            0.0,
            0.74,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )

        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._set_phase("starting", "viewer ready, waiting for backend")
        self._refresh_status()

        self.anim_timer = self.fig.canvas.new_timer(
            interval=max(1, int(round(1000.0 / self.cfg.animation_fps)))
        )
        self.anim_timer.add_callback(self._on_anim_tick)

        self.train_timer = self.fig.canvas.new_timer(interval=max(1, self.cfg.ui_interval_ms))
        self.train_timer.add_callback(self._on_train_tick)

    def _init_network_panel(self):
        zero_grid = np.zeros((self.system.exc_side, self.system.exc_side), dtype=float)
        self.spike_im = self.ax_spike.imshow(
            zero_grid.T,
            origin="lower",
            extent=[0.0, self.cfg.controller_dx_mm, 0.0, self.cfg.controller_dx_mm],
            interpolation="nearest",
            aspect="equal",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
        )
        self.membrane_im = self.ax_membrane.imshow(
            zero_grid.T,
            origin="lower",
            extent=[0.0, self.cfg.controller_dx_mm, 0.0, self.cfg.controller_dx_mm],
            interpolation="nearest",
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=max(1.0, self.cfg.v_th),
        )
        self.filtered_im = self.ax_filtered.imshow(
            zero_grid.T,
            origin="lower",
            extent=[0.0, self.cfg.controller_dx_mm, 0.0, self.cfg.controller_dx_mm],
            interpolation="nearest",
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        self.adapt_im = self.ax_adapt.imshow(
            self.histograms[0].T,
            origin="lower",
            extent=[0.0, self.cfg.controller_dx_mm, 0.0, self.cfg.controller_dx_mm],
            interpolation="nearest",
            aspect="equal",
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
        )
        for ax in (self.ax_spike, self.ax_membrane, self.ax_filtered, self.ax_adapt):
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_spike.set_title("Spike Density")
        self.ax_membrane.set_title("Membrane Potential")
        self.ax_filtered.set_title("Filtered Activity")
        self.ax_adapt.set_title("Adaptation")

        obs_zero = np.zeros((self.system.obs_size,), dtype=float)
        action_zero = np.zeros((self.system.action_size,), dtype=float)
        self.obs_bar_y = np.arange(self.system.obs_size)
        self.action_bar_y = np.arange(self.system.action_size)
        self.obs_bars = self.ax_obs.barh(self.obs_bar_y, obs_zero, color="0.5", edgecolor="none")
        self.action_bars = self.ax_action.barh(
            self.action_bar_y, action_zero, color="0.5", edgecolor="none"
        )
        self.ax_obs.axvline(0.0, color="0.5", lw=1.0)
        self.ax_action.axvline(0.0, color="0.5", lw=1.0)
        self.ax_obs.set_title("Current Input Channels")
        self.ax_action.set_title("Current Motor Commands")
        self.ax_obs.set_yticks(self.obs_bar_y, labels=self.system.observation_labels, fontsize=7)
        self.ax_action.set_yticks(self.action_bar_y, labels=self.system.action_labels, fontsize=8)
        self.ax_obs.invert_yaxis()
        self.ax_action.invert_yaxis()
        self.ax_obs.set_xlim(-2.2, 2.2)
        self.ax_action.set_xlim(-1.05, 1.05)
        self.ax_obs.grid(alpha=0.20, axis="x")
        self.ax_action.grid(alpha=0.20, axis="x")

    def _init_weight_panel(self):
        self.w_in_im = self.ax_w_in.imshow(
            np.zeros((self.system.obs_size, self.system.n_total), dtype=float),
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        self.ax_w_in.set_title("Input Weights Into SNN")
        self.ax_w_in.set_ylabel("input channel")
        self.ax_w_in.set_xticks([])
        self.ax_w_in.set_yticks(np.arange(self.system.obs_size), labels=self.system.observation_labels, fontsize=7)

        self.w_rec_im = self.ax_w_rec.imshow(
            np.zeros((self.system.n_total, self.system.n_total), dtype=float),
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        self.ax_w_rec.set_title("Recurrent Weights Within SNN")
        self.ax_w_rec.set_xlabel("post neuron")
        self.ax_w_rec.set_ylabel("pre neuron")
        self.ax_w_rec.set_xticks([])
        self.ax_w_rec.set_yticks([])

        self.w_out_im = self.ax_w_out.imshow(
            np.zeros((self.system.action_size, self.system.n_total), dtype=float),
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        self.ax_w_out.set_title("Output Readout Weights")
        self.ax_w_out.set_xlabel("neuron")
        self.ax_w_out.set_yticks(np.arange(self.system.action_size), labels=self.system.action_labels, fontsize=8)
        self.ax_w_out.set_xticks([])

        self.rec_summary_im = self.ax_rec_summary.imshow(
            np.zeros((self.system.exc_side, self.system.exc_side), dtype=float),
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        self.ax_rec_summary.set_aspect("equal", adjustable="box")
        self.ax_rec_summary.set_box_aspect(1.0)
        self.ax_rec_summary.set_title("Incoming |W_rec| to E")
        self.ax_rec_summary.set_xticks([])
        self.ax_rec_summary.set_yticks([])
        self._update_weight_views()

    def _init_robot_panel(self):
        self.ax_robot.set_aspect("equal", adjustable="box")
        self.ax_robot.set_xlabel("x")
        self.ax_robot.set_ylabel("y")
        self.ax_robot.axhline(0.0, color="0.65", lw=1.0)

        self.path_line, = self.ax_robot.plot([], [], color="0.75", lw=1.5)
        self.body_box, = self.ax_robot.plot([], [], "-", color="black", lw=2)
        self.body_com, = self.ax_robot.plot([], [], "o", color="black", ms=4)
        self.thighs = [self.ax_robot.plot([], [], "-", lw=2)[0] for _ in range(self.cfg.n_legs)]
        self.shanks = [self.ax_robot.plot([], [], "-", lw=2)[0] for _ in range(self.cfg.n_legs)]
        self.knees = [
            self.ax_robot.plot([], [], "o", color="tab:orange", ms=3)[0]
            for _ in range(self.cfg.n_legs)
        ]
        self.feet = [
            self.ax_robot.plot([], [], "o", color="tab:blue", ms=4)[0]
            for _ in range(self.cfg.n_legs)
        ]
        self.force_line, = self.ax_robot.plot([], [], "-", color="tab:red", lw=2)

    def _init_metric_panel(self):
        self.distance_line, = self.ax_metric.plot([], [], color="tab:blue", label="distance")
        self.speed_line, = self.ax_metric.plot([], [], color="tab:green", label="mean vx")
        self.ax_metric.set_title("Training Progress")
        self.ax_metric.set_xlabel("epoch")
        self.ax_metric.legend(loc="upper left")
        self.ax_metric.grid(alpha=0.25)

    def _init_status_panel(self):
        self.ax_status.set_xlim(0.0, 1.0)
        self.ax_status.set_ylim(0.0, 1.0)
        self.ax_status.axis("off")
        self.ax_status.text(0.0, 0.98, "Status", va="top", ha="left", fontsize=10, fontweight="bold")
        self.phase_text = self.ax_status.text(
            0.0, 0.88, "", va="top", ha="left", family="monospace", fontsize=9
        )

    def _exc_grid(self, values: np.ndarray) -> np.ndarray:
        exc = np.asarray(values[: self.system.n_exc], dtype=float)
        return exc.reshape(self.system.exc_side, self.system.exc_side)

    def _bar_colors(self, values: np.ndarray, limit: float):
        cmap = plt.get_cmap("coolwarm")
        scale = max(1e-6, float(limit))
        normalized = np.clip(0.5 + 0.5 * (values / scale), 0.0, 1.0)
        return cmap(normalized)

    def _update_bar_plot(self, bars, values: np.ndarray, limit: float):
        values = np.asarray(values, dtype=float)
        colors = self._bar_colors(values, limit)
        for idx, bar in enumerate(bars):
            bar.set_width(float(values[idx]))
            bar.set_facecolor(colors[idx])

    def _update_weight_views(self):
        params = _tree_to_numpy(self.latest_params)
        w_in = np.asarray(params["w_in"], dtype=float)
        w_out = np.asarray(params["w_out"], dtype=float).T
        w_rec = np.asarray(self.system._effective_recurrent_weights(params), dtype=float)
        incoming_exc = np.sum(np.abs(w_rec[:, : self.system.n_exc]), axis=0)
        incoming_exc = incoming_exc.reshape(self.system.exc_side, self.system.exc_side)

        def _set_signed_image(image, values):
            vmax = max(1e-6, float(np.percentile(np.abs(values), 99.0)))
            image.set_data(values)
            image.set_clim(-vmax, vmax)

        _set_signed_image(self.w_in_im, w_in)
        _set_signed_image(self.w_rec_im, w_rec)
        _set_signed_image(self.w_out_im, w_out)

        rec_vmax = max(1e-6, float(np.percentile(incoming_exc, 99.0)))
        self.rec_summary_im.set_data(incoming_exc.T)
        self.rec_summary_im.set_clim(0.0, rec_vmax)

    def _set_phase(self, phase: str, detail: str):
        self.current_phase = phase
        self.phase_detail = detail

    def _on_close(self, _event):
        self.closed = True
        self.stop_event.set()
        self.anim_timer.stop()
        self.train_timer.stop()

    def _record_metrics(self, epoch: int, metrics: Dict[str, float]):
        if epoch <= self.last_epoch_seen:
            return
        self.last_epoch_seen = epoch
        self.epoch = epoch
        self.latest_metrics = metrics
        self.history_steps.append(epoch)
        self.history_distance.append(metrics["distance"])
        self.history_speed.append(metrics["mean_vx"])
        self.history_loss.append(metrics["loss"])

    def _drain_backend_messages(self):
        received = False
        while True:
            try:
                message = self.message_queue.get_nowait()
            except queue.Empty:
                break

            received = True
            msg_type = message.get("type")
            phase = message.get("phase")
            detail = message.get("detail")
            if phase is not None and detail is not None:
                self._set_phase(phase, detail)

            if msg_type == "status":
                continue

            if msg_type == "metrics":
                self._record_metrics(int(message["epoch"]), message["metrics"])
                continue

            if msg_type == "snapshot":
                self._record_metrics(int(message["epoch"]), message["metrics"])
                if "params" in message and message["params"] is not None:
                    self.latest_params = _tree_to_numpy(message["params"])
                    self.system.params = self.latest_params
                    self._update_weight_views()
                self._set_rollout(message["rollout"], message["metrics"])
                continue

            if msg_type == "done":
                self.worker_done = True
                self.epoch = max(self.epoch, int(message.get("epoch", self.epoch)))
                self._set_phase("done", message.get("detail", "training complete"))
                continue

            if msg_type == "error":
                self.worker_done = True
                self._set_phase("error", message.get("detail", "backend failed"))
                print(
                    "[trainable_system] backend error: "
                    f"{message.get('detail', 'unknown error')}",
                    flush=True,
                )
        return received

    def _rotation_matrix(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _world_points(self, local_points: np.ndarray, body_pos: np.ndarray, theta: float) -> np.ndarray:
        return local_points @ self._rotation_matrix(theta).T + body_pos[None, :]

    def _segment_dir(self, joint_angle: float) -> np.ndarray:
        return np.array([np.sin(joint_angle), -np.cos(joint_angle)], dtype=float)

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

        runner = RolloutRunner(mon=rollout)
        _log(f"viewer: preparing spike maps for {len(self.frame_times)} frames")
        self.histograms, _ = prepare_spike_histograms_for_times(
            self.system,
            runner,
            self.frame_times,
            self.cfg.window_size_ms,
            progress_label="preparing spike maps",
        )
        _log("viewer: spike maps ready")
        spike_vmax = max(1.0, float(np.max(self.histograms)))
        self.spike_im.set_clim(0.0, spike_vmax)

        exc_v = np.asarray(rollout["v"][:, : self.system.n_exc], dtype=float)
        exc_filt = np.asarray(rollout["filtered_spike"][:, : self.system.n_exc], dtype=float)
        exc_adapt = np.asarray(rollout["adapt"][:, : self.system.n_exc], dtype=float)
        v_lo = float(min(0.0, np.percentile(exc_v, 1.0)))
        v_hi = float(max(self.cfg.v_th, np.percentile(exc_v, 99.0)))
        filt_hi = float(max(1.0, np.percentile(exc_filt, 99.0)))
        adapt_hi = float(max(1.0, np.percentile(exc_adapt, 99.0)))
        self.membrane_im.set_clim(v_lo, v_hi)
        self.filtered_im.set_clim(0.0, filt_hi)
        self.adapt_im.set_clim(0.0, adapt_hi)

        pos = np.asarray(rollout["pos"], dtype=float)
        foot_pos = np.asarray(rollout["foot_pos"], dtype=float)
        min_y = float(
            min(
                np.min(pos[:, 1]) - self.cfg.body_height,
                np.min(foot_pos[:, :, 1]) - 0.05,
                0.0,
            )
        )
        max_y = float(
            max(
                np.max(pos[:, 1]) + self.cfg.body_height,
                np.max(foot_pos[:, :, 1]) + 0.15,
                self.cfg.height_target + 0.08,
            )
        )
        self.robot_y_limits = (min_y, max_y)
        self.camera_half_width = max(
            0.5,
            self.cfg.body_length + self.cfg.thigh_length + self.cfg.shank_length + 0.2,
        )
        self.ax_robot.set_ylim(*self.robot_y_limits)
        self.force_scale = 0.35 * self.cfg.body_length / max(
            1.0, float(np.max(np.linalg.norm(np.asarray(rollout["force"], dtype=float), axis=1)))
        )
        self._draw_frame(0)

    def _draw_frame(self, frame_ptr: int):
        if self.latest_rollout is None:
            return
        frame_ptr = int(np.clip(frame_ptr, 0, len(self.frame_indices) - 1))
        self.frame_ptr = frame_ptr
        i = int(self.frame_indices[frame_ptr])

        rollout = self.latest_rollout
        pos = np.asarray(rollout["pos"], dtype=float)
        angle = np.asarray(rollout["angle"], dtype=float)
        hip_angle = np.asarray(rollout["hip_angle"], dtype=float)
        knee_angle = np.asarray(rollout["knee_angle"], dtype=float)
        foot_pos = np.asarray(rollout["foot_pos"], dtype=float)
        force = np.asarray(rollout["force"], dtype=float)
        obs = np.asarray(rollout["obs"], dtype=float)
        action = np.asarray(rollout["action"], dtype=float)
        membrane = np.asarray(rollout["v"], dtype=float)
        filtered = np.asarray(rollout["filtered_spike"], dtype=float)
        adapt = np.asarray(rollout["adapt"], dtype=float)

        self.spike_im.set_data(self.histograms[frame_ptr].T)
        self.membrane_im.set_data(self._exc_grid(membrane[i]).T)
        self.filtered_im.set_data(self._exc_grid(filtered[i]).T)
        self.adapt_im.set_data(self._exc_grid(adapt[i]).T)
        self.ax_spike.set_title(f"Spike Density | t={self.frame_times[frame_ptr]:.0f} ms")
        self.ax_membrane.set_title(
            f"Membrane Potential | mean={np.mean(membrane[i, : self.system.n_exc]):.3f}"
        )
        self.ax_filtered.set_title(
            f"Filtered Activity | mean={np.mean(filtered[i, : self.system.n_exc]):.3f}"
        )
        self.ax_adapt.set_title(
            f"Adaptation | mean={np.mean(adapt[i, : self.system.n_exc]):.3f}"
        )
        self._update_bar_plot(self.obs_bars, obs[i], limit=2.0)
        self._update_bar_plot(self.action_bars, action[i], limit=1.0)

        p = pos[i]
        theta = float(angle[i])
        hip_local = np.array(
            [
                [self.cfg.hip_x_offset, -0.5 * self.cfg.body_height],
                [-self.cfg.hip_x_offset, -0.5 * self.cfg.body_height],
            ],
            dtype=float,
        )
        body_outline_local = np.array(
            [
                [0.5 * self.cfg.body_length, 0.5 * self.cfg.body_height],
                [0.5 * self.cfg.body_length, -0.5 * self.cfg.body_height],
                [-0.5 * self.cfg.body_length, -0.5 * self.cfg.body_height],
                [-0.5 * self.cfg.body_length, 0.5 * self.cfg.body_height],
                [0.5 * self.cfg.body_length, 0.5 * self.cfg.body_height],
            ],
            dtype=float,
        )
        hip_pos = self._world_points(hip_local, p, theta)
        body_outline = self._world_points(body_outline_local, p, theta)

        self.path_line.set_data(pos[: i + 1, 0], pos[: i + 1, 1])
        self.body_box.set_data(body_outline[:, 0], body_outline[:, 1])
        self.body_com.set_data([p[0]], [p[1]])

        for leg_idx in range(self.cfg.n_legs):
            hip = hip_pos[leg_idx]
            thigh_abs = theta + float(hip_angle[i, leg_idx])
            knee_abs = thigh_abs + float(knee_angle[i, leg_idx])
            knee_pos = hip + self.cfg.thigh_length * self._segment_dir(thigh_abs)
            foot = foot_pos[i, leg_idx]
            self.thighs[leg_idx].set_data([hip[0], knee_pos[0]], [hip[1], knee_pos[1]])
            self.shanks[leg_idx].set_data([knee_pos[0], foot[0]], [knee_pos[1], foot[1]])
            self.knees[leg_idx].set_data([knee_pos[0]], [knee_pos[1]])
            self.feet[leg_idx].set_data([foot[0]], [max(0.0, foot[1])])

        f = force[i]
        self.force_line.set_data(
            [p[0], p[0] + f[0] * self.force_scale],
            [p[1], p[1] + f[1] * self.force_scale],
        )
        self.ax_robot.set_xlim(p[0] - self.camera_half_width, p[0] + self.camera_half_width)
        self.ax_robot.set_title(
            "Walker Rollout | "
            f"epoch={self.epoch} | x={p[0]:.3f} y={p[1]:.3f} pitch={theta:.3f}"
        )

    def _refresh_status(self):
        elapsed = time.perf_counter() - self.start_time
        self.phase_text.set_text(f"{self.current_phase:<10} {self.phase_detail}")
        playback_frame = 0 if len(self.frame_indices) <= 1 else self.frame_ptr + 1
        playback_total = max(1, len(self.frame_indices))
        metric_lines = (
            [
                f"loss           : {self.history_loss[-1]: .4f}" if self.history_loss else "loss           : n/a",
                f"distance       : {self.latest_metrics['distance']: .4f}",
                f"mean vx        : {self.latest_metrics['mean_vx']: .4f}",
                f"reward         : {self.latest_metrics['reward']: .4f}",
                f"height error   : {self.latest_metrics['height_error']: .4f}",
                f"pitch error    : {self.latest_metrics['pitch_error']: .4f}",
            ]
            if self.latest_metrics is not None
            else [
                "loss           : compiling / waiting",
                "distance       : n/a",
                "mean vx        : n/a",
                "reward         : n/a",
                "height error   : n/a",
                "pitch error    : n/a",
            ]
        )
        self.status_text.set_text(
            "\n".join(
                [
                    f"epoch          : {self.epoch:4d} / {self.cfg.train_epochs}",
                    *metric_lines,
                    f"playback frame : {playback_frame:4d} / {playback_total}",
                    f"tick count     : {self.train_tick_count:6d}",
                    f"elapsed        : {elapsed:6.1f} s",
                ]
            )
        )

    def _refresh_metric_plot(self):
        if not self.history_steps:
            return
        x = np.asarray(self.history_steps, dtype=float)
        self.distance_line.set_data(x, np.asarray(self.history_distance, dtype=float))
        self.speed_line.set_data(x, np.asarray(self.history_speed, dtype=float))
        self.ax_metric.set_xlim(0.0, max(1.0, x[-1]))

        y_values = np.concatenate(
            [
                np.asarray(self.history_distance, dtype=float),
                np.asarray(self.history_speed, dtype=float),
            ]
        )
        ymin = float(np.min(y_values))
        ymax = float(np.max(y_values))
        if abs(ymax - ymin) < 1e-6:
            ymax = ymin + 1.0
        pad = 0.1 * (ymax - ymin)
        self.ax_metric.set_ylim(ymin - pad, ymax + pad)

    def _on_anim_tick(self):
        if self.closed or self.latest_rollout is None:
            return
        next_ptr = (self.frame_ptr + 1) % len(self.frame_indices)
        self._draw_frame(next_ptr)
        if self.current_phase == "idle":
            self._refresh_status()
        self.fig.canvas.draw_idle()

    def _on_train_tick(self):
        if self.closed:
            return
        self.train_tick_count += 1
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
                print(
                    "[trainable_system] "
                    f"epoch={self.epoch}/{self.cfg.train_epochs} "
                    f"phase={self.current_phase} "
                    f"loss={self.latest_metrics['loss']:.4f} "
                    f"distance={self.latest_metrics['distance']:.4f} "
                    f"vx={self.latest_metrics['mean_vx']:.4f}",
                    flush=True,
                )
                self.last_log_time = now
        self._refresh_metric_plot()
        self._refresh_status()
        self.fig.canvas.draw_idle()

    def show(self):
        self.anim_timer.start()
        self.train_timer.start()
        plt.show()


def main():
    _require_runtime()
    print("[trainable_system] building system...", flush=True)
    cfg = Config()
    system = TrainableWalkingSystem(cfg)
    print(
        "[trainable_system] system ready "
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
    print("[trainable_system] starting backend worker...", flush=True)
    worker_process.start()
    print("[trainable_system] opening training viewer...", flush=True)
    viewer = TrainingViewer(system, features, message_queue, stop_event, worker_process)
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
    "TrainableWalkingSystem",
    "build_feature_sequence",
    "collect_rollout",
    "main",
]


if __name__ == "__main__":
    main()
