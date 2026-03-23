"""
Trainable spiking control system for a 2D articulated walker.

This script builds a Spatial SNN (same model used in run_simulation.py),
injects low-dimensional inputs through fixed spatial patterns, and trains
readout weights with surrogate gradients + BPTT to make a rigid rectangular
body with front and back two-link legs walk forward on level ground.

The controller still trains against the existing differentiable JAX walker
proxy, but rollouts, evaluation, and visualization now use the same Pymunk
contact simulation as run_walking_physics.py.

It also visualizes:
- training curves
- surrogate gradient shape
- spike raster and spatial activity heatmap
- walking robot trajectory animation
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import threading
import traceback
import warnings
import math
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

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

# Prefer CPU by default to avoid CUDA/cuDNN init failures on systems without GPU setup.
# Set SNN_USE_GPU=1 to allow GPU usage.
if os.environ.get("SNN_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import brainpy as bp
import brainpy.math as bm
from brainpy._src.helpers import clear_input

from walking_physics import (
    WalkerPhysics,
    WalkerSensors,
    WalkerState,
    WalkerStepResult,
)
from walking_physics_pymunk import PymunkPassiveWalker, make_initial_state

spatial_mod = importlib.import_module("src.models.Spatial")
from src.models.Spatial import Spatial
neurons_mod = importlib.import_module("src.neurons")
linear_mod = importlib.import_module("brainpy._src.dnn.linear")


def _patch_spatial_reinit_weights():
    """Patch Spatial.reinit_weights to tolerate scalar CSRLinear weights.

    Some BrainPy versions construct CSRLinear weights as scalars when g_max
    is a float. The original Spatial.reinit_weights assigns a per-connection
    vector, which then triggers a shape mismatch. This patch detects scalar
    weights and assigns scalar means instead, while keeping the vector path
    for cases where the weight storage supports per-connection arrays.
    """

    if getattr(spatial_mod.Spatial, "_safe_reinit_patched", False):
        return

    def _safe_reinit_weights(self, delta=None, J_e=None):
        if delta is not None:
            self.delta = delta
        if J_e is not None:
            self.J_ee = J_e[0]
            self.J_ei = J_e[1]

        self.J_ie = self.J_ee * self.delta
        self.J_ii = self.J_ei * self.delta

        def _weight_shape(w):
            try:
                return tuple(w.value.shape)
            except Exception:
                try:
                    return tuple(w.shape)
                except Exception:
                    return ()

        def _assign(proj, target_j, N):
            w = proj.comm.weight
            if not hasattr(w, "value"):
                # Plain Python/JAX scalar storage.
                proj.comm.weight = target_j
                return
            shape = _weight_shape(w)
            if shape in [(), (1,)]:
                # Scalar storage: use a single mean weight.
                w.value = target_j
                return
            # Vector storage: use the original correlation logic.
            self.key, subkey = jax.random.split(self.key)
            new_w = spatial_mod.correlate_weights(proj, target_j, N, subkey)
            w.value = new_w

        _assign(self.E2E.proj, self.J_ee, self.N_e)
        _assign(self.E2I.proj, self.J_ei, self.N_i)
        _assign(self.I2E.proj, self.J_ie, self.N_e)
        _assign(self.I2I.proj, self.J_ii, self.N_i)
        _assign(self.ext2E.proj, self.J_ee, self.N_e)
        _assign(self.ext2I.proj, self.J_ei, self.N_i)
        # Reset can fail in some BrainPy/JAX setups (batch axis mismatch).
        # Defer reset to the caller if needed.
        try:
            self.reset_state()
        except Exception as exc:
            warnings.warn(f"Spatial.reset_state skipped during init: {exc}")

    spatial_mod.Spatial._orig_reinit_weights = spatial_mod.Spatial.reinit_weights
    spatial_mod.Spatial.reinit_weights = _safe_reinit_weights
    spatial_mod.Spatial._safe_reinit_patched = True


_patch_spatial_reinit_weights()


def _patch_fns_reset_state():
    """Patch FNSNeuron.reset_state to tolerate scalar initializers."""

    if getattr(neurons_mod.FNSNeuron, "_safe_reset_patched", False):
        return

    orig = neurons_mod.FNSNeuron.reset_state

    def _expand(val, shape):
        arr = bm.asarray(val)
        if arr.shape == ():
            arr = bm.full(shape, float(arr))
        return arr

    def _safe_reset_state(self, batch_size=None, **kwargs):
        # Pre-emptively force float spikes for surrogate gradients
        try:
            self.spk_dtype = jnp.float32
        except Exception:
            pass
        try:
            ret = orig(self, batch_size=batch_size, **kwargs)
        except Exception as exc:
            warnings.warn(f"FNSNeuron.reset_state fallback: {exc}")

            base = self.varshape
            if isinstance(base, int):
                base_shape = (base,)
            else:
                base_shape = tuple(base)
            shape = base_shape if batch_size is None else (batch_size,) + base_shape

            V = self.init_variable(self._V_initializer, batch_size)
            V = _expand(V, shape)
            g_K = self.init_variable(self._g_K_initializer, batch_size)
            g_K = _expand(g_K, shape)

            spike = bm.zeros(shape, dtype=jnp.float32)

            t_last_spike = bm.ones(shape) * (-1e8)

            self.V = V
            self.g_K = g_K
            self.spike = bm.Variable(spike)
            self.t_last_spike = t_last_spike
            self.input = bm.Variable(bm.zeros(shape))
            ret = None

        # Ensure spike dtype is float even when the original reset_state succeeds
        try:
            if self.spike.value.dtype != jnp.float32:
                shape = self.spike.value.shape
                self.spike = bm.Variable(bm.zeros(shape, dtype=jnp.float32))
        except Exception:
            pass

        return ret

    neurons_mod.FNSNeuron.reset_state = _safe_reset_state
    neurons_mod.FNSNeuron._safe_reset_patched = True


_patch_fns_reset_state()


def _patch_fns_init():
    """Ensure FNSNeuron creates float spike variables from the start."""

    if getattr(neurons_mod.FNSNeuron, "_safe_init_patched", False):
        return

    orig_init = neurons_mod.FNSNeuron.__init__

    def _safe_init(self, *args, **kwargs):
        if "spk_dtype" not in kwargs or kwargs["spk_dtype"] is None:
            kwargs["spk_dtype"] = jnp.float32
        return orig_init(self, *args, **kwargs)

    neurons_mod.FNSNeuron.__init__ = _safe_init
    neurons_mod.FNSNeuron._safe_init_patched = True


_patch_fns_init()

# Ensure stop_gradient exists in neurons module (used in FNSNeuron.update)
if not hasattr(neurons_mod, "stop_gradient"):
    neurons_mod.stop_gradient = jax.lax.stop_gradient


def _patch_csrlinear_update():
    """Avoid Taichi CSR matvec failures by using JAX sparse BCOO matvec."""

    if getattr(linear_mod.CSRLinear, "_safe_bcoo_update_patched", False):
        return

    def _csr_rows(indptr):
        indptr_np = np.asarray(indptr, dtype=np.int32)
        counts = np.diff(indptr_np)
        return np.repeat(np.arange(counts.shape[0], dtype=np.int32), counts)

    def _safe_update(self, x):
        rows = _csr_rows(self.indptr)
        indices_np = np.asarray(self.indices, dtype=np.int32)
        bcoo_indices = jnp.asarray(np.stack([rows, indices_np], axis=1), dtype=jnp.int32)

        data = jnp.asarray(self.weight)
        if data.shape in [(), (1,)]:
            data = jnp.broadcast_to(jnp.reshape(data, (1,)), (self.indices.shape[0],))
        else:
            data = jnp.reshape(data, (self.indices.shape[0],))

        mat = jsparse.BCOO(
            (data, bcoo_indices),
            shape=(self.conn.pre_num, self.conn.post_num),
        )

        if x.ndim == 1:
            return mat.T @ x if self.transpose else mat @ x
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = x @ mat if self.transpose else x @ mat.T
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    linear_mod.CSRLinear._orig_update = linear_mod.CSRLinear.update
    linear_mod.CSRLinear.update = _safe_update
    linear_mod.CSRLinear._safe_bcoo_update_patched = True


_patch_csrlinear_update()


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class Config:
    # Spatial model parameters.
    # The training dashboard uses lighter defaults than run_simulation.py so
    # initialization and BPTT stay interactive on CPU.
    rho: int = 900
    dx: float = 1.0
    seed: int = 42
    gamma: int = 4
    K_ee: int = 80
    K_ei: int = 100
    K_ie: int = 70
    K_ii: int = 80

    # Robot parameters
    n_legs: int = 2
    mass: float = 2.0
    thigh_mass: float = 0.25
    shank_mass: float = 0.18
    drag: float = 0.15
    angular_drag: float = 1.5
    joint_drag: float = 0.2
    gravity: float = 9.81
    ground_k: float = 1500.0
    ground_c: float = 35.0
    ground_tangent_damping: float = 80.0
    friction_mu: float = 0.9
    body_length: float = 0.45
    body_height: float = 0.18
    hip_x_offset: float = 0.16
    leg_radius: float = 0.022
    foot_radius: float = 0.028
    body_corner_radius: float = 0.012
    thigh_length: float = 0.22
    shank_length: float = 0.22
    height_target: float = 0.48
    hip_limit: float = 0.95
    knee_min: float = 0.05
    knee_max: float = 1.45
    hip_kp: float = 18.0
    hip_kd: float = 1.8
    knee_kp: float = 24.0
    knee_kd: float = 2.2
    hip_torque_limit: float = 12.0
    knee_torque_limit: float = 14.0

    # Simulation parameters
    dt_ms: float = 1.0
    episode_ms: float = 300.0
    rollout_ms: Optional[float] = 1200.0
    base_freq_hz: float = 1.5
    target_vx: float = 0.4
    target_vy: float = 0.0
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
    hip_offsets: Tuple[float, float] = (0.18, -0.10)
    knee_offsets: Tuple[float, float] = (-0.10, 0.06)

    # Input injection
    feature_dim: int = 7  # [1, target_vx, target_vy, sin, cos, sin2, cos2]
    input_rank: int = 6
    input_scale: float = 0.6
    input_init_scale: float = 0.05

    # Readout
    out_per_leg: int = 2  # hip target angle, knee target angle
    readout_rank: Optional[int] = 64  # None means full readout
    readout_init_scale: float = 0.02
    rate_decay: float = 0.95

    # Training
    train_epochs: int = 60
    lr: float = 3e-3
    eval_every: int = 5
    train_core: bool = False  # if True, also train core synaptic weights (heavy)
    vis_every: int = 5  # visualize progress every N epochs (0 disables)

    # Loss weights
    w_forward: float = 1.0
    w_height: float = 3.0
    w_pitch: float = 2.0
    w_omega: float = 0.1
    w_energy: float = 0.002
    w_spike: float = 0.01
    w_smooth: float = 0.004
    w_contact: float = 0.05
    w_slip: float = 0.2
    w_gait_balance: float = 0.2
    desired_contact: float = 0.5

    # Visualization
    raster_neurons: int = 200
    heatmap_window_ms: float = 10.0
    heatmap_frame_ms: float = 2.0
    animation_fps: float = 30.0


# -----------------------------
# Utility functions
# -----------------------------


def make_spatial_patterns(
    positions: np.ndarray, num_patterns: int, seed: int = 0
) -> np.ndarray:
    """Generate smooth spatial patterns to inject low-rank inputs."""
    rng = np.random.default_rng(seed)
    pos = positions.astype(np.float32)
    pos = pos - pos.min(axis=0, keepdims=True)
    denom = np.maximum(pos.max(axis=0, keepdims=True), 1e-6)
    pos = pos / denom

    k = rng.normal(size=(num_patterns, 2))
    k = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-6)
    k = k * rng.uniform(0.5, 2.5, size=(num_patterns, 1)) * (2 * np.pi)
    phase = rng.uniform(0.0, 2 * np.pi, size=(num_patterns, 1))

    patterns = np.sin(pos @ k.T + phase.T)
    patterns = (patterns - patterns.mean(axis=0, keepdims=True)) / (
        patterns.std(axis=0, keepdims=True) + 1e-6
    )
    # return shape (num_patterns, num_neurons)
    return patterns.T.astype(np.float32)


def build_feature_sequence(
    num_steps: int,
    dt_ms: float,
    target_vx: float,
    target_vy: float,
    base_freq_hz: float,
) -> bm.Array:
    """Build a time series of low-dimensional input features."""
    t = bm.arange(num_steps) * dt_ms / 1000.0
    phase = 2 * math.pi * base_freq_hz * t
    feats = bm.stack(
        [
            bm.ones_like(t),
            bm.ones_like(t) * target_vx,
            bm.ones_like(t) * target_vy,
            bm.sin(phase),
            bm.cos(phase),
            bm.sin(2 * phase),
            bm.cos(2 * phase),
        ],
        axis=1,
    )
    return feats


def build_controller_features(
    cfg: Config, duration_ms: Optional[float] = None
) -> bm.Array:
    if duration_ms is None:
        duration_ms = cfg.episode_ms
    num_steps = max(1, int(round(float(duration_ms) / float(cfg.dt_ms))))
    return build_feature_sequence(
        num_steps,
        cfg.dt_ms,
        cfg.target_vx,
        cfg.target_vy,
        cfg.base_freq_hz,
    )


def global_norm(grads: Dict[str, bm.Array]) -> bm.Array:
    total = 0.0
    for g in grads.values():
        total += bm.sum(g * g)
    return bm.sqrt(total + 1e-12)


# -----------------------------
# Physics backends
# -----------------------------


class ControlledPymunkWalker(PymunkPassiveWalker):
    """Pymunk walker with explicit joint-angle targets."""

    def __init__(self, cfg, initial_state):
        self.hip_targets = np.asarray(initial_state.hip_angle, dtype=float)
        self.knee_targets = np.asarray(initial_state.knee_angle, dtype=float)
        self.last_joint_torque = np.zeros((cfg.n_legs, 2), dtype=float)
        super().__init__(cfg, initial_state)

    def set_targets(self, hip_target, knee_target):
        self.hip_targets = np.asarray(hip_target, dtype=float).reshape(self.cfg.n_legs)
        self.knee_targets = np.asarray(knee_target, dtype=float).reshape(self.cfg.n_legs)

    def _apply_joint_torques(self):
        cfg = self.cfg
        joint_torque = np.zeros((cfg.n_legs, 2), dtype=float)

        for leg_idx in range(cfg.n_legs):
            thigh = self.thigh_bodies[leg_idx]
            shank = self.shank_bodies[leg_idx]

            hip_angle = float(thigh.angle - self.body.angle)
            knee_angle = float(shank.angle - thigh.angle)
            hip_omega = float(thigh.angular_velocity - self.body.angular_velocity)
            knee_omega = float(shank.angular_velocity - thigh.angular_velocity)

            hip_tau_cmd = cfg.hip_kp * (self.hip_targets[leg_idx] - hip_angle)
            hip_tau_cmd = hip_tau_cmd - cfg.hip_kd * hip_omega
            hip_tau_cmd = float(
                np.clip(hip_tau_cmd, -cfg.hip_torque_limit, cfg.hip_torque_limit)
            )

            knee_tau_cmd = cfg.knee_kp * (self.knee_targets[leg_idx] - knee_angle)
            knee_tau_cmd = knee_tau_cmd - cfg.knee_kd * knee_omega
            knee_tau_cmd = float(
                np.clip(knee_tau_cmd, -cfg.knee_torque_limit, cfg.knee_torque_limit)
            )

            joint_torque[leg_idx, 0] = hip_tau_cmd
            joint_torque[leg_idx, 1] = knee_tau_cmd

            hip_tau = hip_tau_cmd - cfg.joint_drag * hip_omega
            knee_tau = knee_tau_cmd - cfg.joint_drag * knee_omega

            self.body.torque -= hip_tau
            thigh.torque += hip_tau - knee_tau
            shank.torque += knee_tau

        self.last_joint_torque = joint_torque


class PymunkWalkerPhysicsAdapter:
    """Match the differentiable walker API with the Pymunk backend."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._walker: Optional[ControlledPymunkWalker] = None

    def _hip_offsets(self) -> np.ndarray:
        return np.asarray(getattr(self.cfg, "hip_offsets", (0.0, 0.0)), dtype=float)

    def _knee_offsets(self) -> np.ndarray:
        return np.asarray(getattr(self.cfg, "knee_offsets", (0.0, 0.0)), dtype=float)

    def _initial_passive_state(self):
        return make_initial_state(
            self.cfg,
            hip_offsets=self._hip_offsets(),
            knee_offsets=self._knee_offsets(),
        )

    def _walker_state_to_passive(self, state: WalkerState):
        return self._initial_passive_state().__class__(
            pos=np.asarray(state.pos, dtype=float),
            vel=np.asarray(state.vel, dtype=float),
            angle=float(state.angle),
            omega=float(state.omega),
            hip_angle=np.asarray(state.hip_angle, dtype=float),
            knee_angle=np.asarray(state.knee_angle, dtype=float),
            hip_omega=np.asarray(state.hip_omega, dtype=float),
            knee_omega=np.asarray(state.knee_omega, dtype=float),
        )

    def _to_walker_state(self, state) -> WalkerState:
        return WalkerState(
            pos=bm.asarray(state.pos),
            vel=bm.asarray(state.vel),
            angle=bm.asarray(state.angle),
            omega=bm.asarray(state.omega),
            hip_angle=bm.asarray(state.hip_angle),
            knee_angle=bm.asarray(state.knee_angle),
            hip_omega=bm.asarray(state.hip_omega),
            knee_omega=bm.asarray(state.knee_omega),
        )

    def _rebuild(self, state: Optional[WalkerState] = None):
        passive_state = (
            self._initial_passive_state()
            if state is None
            else self._walker_state_to_passive(state)
        )
        self._walker = ControlledPymunkWalker(self.cfg, passive_state)
        self._walker.set_targets(passive_state.hip_angle, passive_state.knee_angle)
        return passive_state

    def _ensure_walker(self, state: Optional[WalkerState] = None):
        if self._walker is None:
            return self._rebuild(state)
        return None

    def initial_state(self) -> WalkerState:
        passive_state = self._rebuild()
        return self._to_walker_state(passive_state)

    def sense(self, state: WalkerState) -> WalkerSensors:
        self._ensure_walker(state)
        sensed = self._walker.observe()
        return WalkerSensors(
            foot_pos=bm.asarray(sensed.foot_pos),
            foot_vel=bm.asarray(sensed.foot_vel),
            ground_contact=bm.asarray(sensed.ground_contact),
        )

    def step(self, state: WalkerState, hip_target, knee_target) -> WalkerStepResult:
        self._ensure_walker(state)
        self._walker.set_targets(hip_target, knee_target)
        self._walker.step()
        sensed = self._walker.observe()
        return WalkerStepResult(
            state=self._to_walker_state(sensed),
            foot_pos=bm.asarray(sensed.foot_pos),
            foot_vel=bm.asarray(sensed.foot_vel),
            ground_contact=bm.asarray(sensed.ground_contact),
            total_ground_force=bm.asarray(sensed.total_ground_force),
            joint_torque=bm.asarray(self._walker.last_joint_torque),
        )


# -----------------------------
# Trainable system
# -----------------------------


class TrainableWalkingSystem(bp.DynamicalSystem):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.n_legs != 2:
            raise ValueError(
                "TrainableWalkingSystem currently implements a front-leg/back-leg walker "
                "and expects n_legs == 2."
            )

        self.n_legs = cfg.n_legs
        self.out_per_leg = cfg.out_per_leg
        self.out_dim = self.n_legs * self.out_per_leg
        self.diff_physics = WalkerPhysics(cfg)
        self.pymunk_physics = PymunkWalkerPhysicsAdapter(cfg)
        self.physics_backend = "differentiable"

        # Core SNN. This is still the same Spatial model family used in
        # run_simulation.py, but with training-oriented connectivity defaults
        # taken from Config.
        self.core = Spatial(
            key=cfg.seed,
            rho=cfg.rho,
            dx=cfg.dx,
            gamma=cfg.gamma,
            K_ee=cfg.K_ee,
            K_ei=cfg.K_ei,
            K_ie=cfg.K_ie,
            K_ii=cfg.K_ii,
        )

        # Fixed spatial input patterns
        self.E_patterns = bm.asarray(
            make_spatial_patterns(
                np.asarray(self.core.E.positions), cfg.input_rank, seed=cfg.seed + 1
            )
        )
        self.I_patterns = bm.asarray(
            make_spatial_patterns(
                np.asarray(self.core.I.positions), cfg.input_rank, seed=cfg.seed + 2
            )
        )

        # Trainable input weights (feature -> low-rank spatial coefficients)
        rng = jax.random.PRNGKey(cfg.seed + 10)
        rng, sk = jax.random.split(rng)
        self.W_in_E = bm.TrainVar(
            jax.random.normal(sk, (cfg.feature_dim, cfg.input_rank))
            * cfg.input_init_scale
        )
        rng, sk = jax.random.split(rng)
        self.W_in_I = bm.TrainVar(
            jax.random.normal(sk, (cfg.feature_dim, cfg.input_rank))
            * cfg.input_init_scale
        )

        # Readout projection (optional)
        if cfg.readout_rank is None:
            self.readout_proj = None
            readout_in = self.core.E.num
        else:
            rng, sk = jax.random.split(rng)
            proj = (
                jax.random.normal(sk, (self.core.E.num, cfg.readout_rank))
                / math.sqrt(self.core.E.num)
            )
            self.readout_proj = bm.asarray(proj)
            readout_in = cfg.readout_rank

        # Trainable readout weights
        rng, sk = jax.random.split(rng)
        self.W_out = bm.TrainVar(
            jax.random.normal(sk, (readout_in, self.out_dim))
            * cfg.readout_init_scale
        )
        self.b_out = bm.TrainVar(bm.zeros((self.out_dim,)))

        # Optionally train core synaptic weights (heavy)
        if cfg.train_core:
            self.core.E2E.proj.comm.weight = bm.TrainVar(
                self.core.E2E.proj.comm.weight
            )
            self.core.E2I.proj.comm.weight = bm.TrainVar(
                self.core.E2I.proj.comm.weight
            )
            self.core.I2E.proj.comm.weight = bm.TrainVar(
                self.core.I2E.proj.comm.weight
            )
            self.core.I2I.proj.comm.weight = bm.TrainVar(
                self.core.I2I.proj.comm.weight
            )

        # State variables
        self.rate = bm.Variable(bm.zeros(self.core.E.varshape))
        self.pos = bm.Variable(bm.zeros((2,)))
        self.vel = bm.Variable(bm.zeros((2,)))
        self.angle = bm.Variable(bm.asarray(0.0))
        self.omega = bm.Variable(bm.asarray(0.0))
        self.hip_angle = bm.Variable(bm.zeros((self.n_legs,)))
        self.knee_angle = bm.Variable(bm.zeros((self.n_legs,)))
        self.hip_omega = bm.Variable(bm.zeros((self.n_legs,)))
        self.knee_omega = bm.Variable(bm.zeros((self.n_legs,)))
        self.last_foot_pos = bm.Variable(bm.zeros((self.n_legs, 2)))
        self.last_foot_vel = bm.Variable(bm.zeros((self.n_legs, 2)))
        self.last_force = bm.Variable(bm.zeros((2,)))
        self.last_joint_torque = bm.Variable(bm.zeros((self.n_legs, 2)))
        self.last_contact = bm.Variable(bm.zeros((self.n_legs,)))
        self.last_hip_target = bm.Variable(bm.zeros((self.n_legs,)))
        self.last_knee_target = bm.Variable(bm.zeros((self.n_legs,)))

    def set_physics_backend(self, backend: str):
        if backend not in {"differentiable", "pymunk"}:
            raise ValueError(f"Unsupported physics backend: {backend}")
        self.physics_backend = backend

    def _active_physics(self):
        if self.physics_backend == "pymunk":
            return self.pymunk_physics
        return self.diff_physics

    def _initial_state(self) -> WalkerState:
        passive_state = make_initial_state(
            self.cfg,
            hip_offsets=np.asarray(self.cfg.hip_offsets, dtype=float),
            knee_offsets=np.asarray(self.cfg.knee_offsets, dtype=float),
        )
        return WalkerState(
            pos=bm.asarray(passive_state.pos),
            vel=bm.asarray(passive_state.vel),
            angle=bm.asarray(passive_state.angle),
            omega=bm.asarray(passive_state.omega),
            hip_angle=bm.asarray(passive_state.hip_angle),
            knee_angle=bm.asarray(passive_state.knee_angle),
            hip_omega=bm.asarray(passive_state.hip_omega),
            knee_omega=bm.asarray(passive_state.knee_omega),
        )

    def _physics_state(self) -> WalkerState:
        return WalkerState(
            pos=self.pos.value,
            vel=self.vel.value,
            angle=self.angle.value,
            omega=self.omega.value,
            hip_angle=self.hip_angle.value,
            knee_angle=self.knee_angle.value,
            hip_omega=self.hip_omega.value,
            knee_omega=self.knee_omega.value,
        )

    def _set_physics_state(self, state: WalkerState):
        self.pos.value = state.pos
        self.vel.value = state.vel
        self.angle.value = state.angle
        self.omega.value = state.omega
        self.hip_angle.value = state.hip_angle
        self.knee_angle.value = state.knee_angle
        self.hip_omega.value = state.hip_omega
        self.knee_omega.value = state.knee_omega

    def _joint_targets(self, motor_raw):
        hip_target = bm.tanh(motor_raw[:, 0]) * self.cfg.hip_limit
        knee_target = self.cfg.knee_min + bm.sigmoid(motor_raw[:, 1]) * (
            self.cfg.knee_max - self.cfg.knee_min
        )
        return hip_target, knee_target

    def reset_state(self, batch_size=None, **kwargs):
        self.core.reset_state(batch_size)
        self.rate.value = bm.zeros_like(self.rate.value)
        physics = self._active_physics()
        state = physics.initial_state() if self.physics_backend == "pymunk" else self._initial_state()
        self._set_physics_state(state)
        sensed = physics.sense(state)
        self.last_foot_pos.value = sensed.foot_pos
        self.last_foot_vel.value = sensed.foot_vel
        self.last_force.value = bm.zeros_like(self.last_force.value)
        self.last_joint_torque.value = bm.zeros_like(self.last_joint_torque.value)
        self.last_contact.value = sensed.ground_contact
        self.last_hip_target.value = state.hip_angle
        self.last_knee_target.value = state.knee_angle

    def _inject_inputs(self, features: bm.Array):
        coeffs_E = features @ self.W_in_E
        coeffs_I = features @ self.W_in_I

        E_in_flat = coeffs_E @ self.E_patterns
        I_in_flat = coeffs_I @ self.I_patterns

        E_in = bm.reshape(E_in_flat, self.core.E.varshape)
        I_in = bm.reshape(I_in_flat, self.core.I.varshape)

        self.core.Ein.value = self.cfg.input_scale * E_in
        self.core.Iin.value = self.cfg.input_scale * I_in

    def _readout(self) -> Tuple[bm.Array, bm.Array]:
        # Low-pass filter spikes for smoother control
        spikes = self.core.E.spike
        self.rate.value = self.cfg.rate_decay * self.rate.value + (
            1.0 - self.cfg.rate_decay
        ) * spikes

        flat_rate = bm.reshape(self.rate.value, (self.core.E.num,))
        if self.readout_proj is not None:
            features = flat_rate @ self.readout_proj
        else:
            features = flat_rate

        motor_raw = features @ self.W_out + self.b_out
        motor_raw = bm.reshape(motor_raw, (self.n_legs, self.out_per_leg))
        return self._joint_targets(motor_raw)

    def update(self, features: bm.Array):
        # Inject inputs
        self._inject_inputs(features)

        # Run core network
        self.core()

        hip_target, knee_target = self._readout()
        step = self._active_physics().step(self._physics_state(), hip_target, knee_target)
        self._set_physics_state(step.state)
        self.last_foot_pos.value = step.foot_pos
        self.last_foot_vel.value = step.foot_vel
        self.last_force.value = step.total_ground_force
        self.last_joint_torque.value = step.joint_torque
        self.last_contact.value = step.ground_contact
        self.last_hip_target.value = hip_target
        self.last_knee_target.value = knee_target

        mean_rate = bm.mean(self.rate.value)
        return (
            self.pos.value,
            self.vel.value,
            self.angle.value,
            self.omega.value,
            self.last_joint_torque.value,
            mean_rate,
            self.last_contact.value,
            self.last_foot_pos.value,
            self.last_foot_vel.value,
        )


# -----------------------------
# Training
# -----------------------------


def enable_training_mode(system: TrainableWalkingSystem):
    """Force training mode on core components to enable surrogate gradients."""
    if not hasattr(bm, "TrainingMode"):
        warnings.warn("TrainingMode not available; surrogate gradients may be disabled.")
        return
    mode = bm.TrainingMode
    if isinstance(mode, type):
        mode = mode()

    system.mode = mode
    system.core.mode = mode
    system.core.E.mode = mode
    system.core.I.mode = mode
    system.core.ext.mode = mode

    for proj in [
        system.core.E2E,
        system.core.E2I,
        system.core.I2E,
        system.core.I2I,
        system.core.ext2E,
        system.core.ext2I,
    ]:
        try:
            proj.mode = mode
        except Exception:
            pass

    # Ensure spike variables use float dtype for surrogate gradients
    def _ensure_float_spike(neuron):
        try:
            neuron.spk_dtype = jnp.float32
        except Exception:
            pass
        try:
            shape = neuron.spike.value.shape
        except Exception:
            shape = neuron.varshape
        try:
            neuron.spike = bm.Variable(bm.zeros(shape, dtype=jnp.float32))
        except Exception:
            pass

    _ensure_float_spike(system.core.E)
    _ensure_float_spike(system.core.I)


def compute_loss(outputs: Tuple[bm.Array, ...], cfg: Config) -> bm.Array:
    pos, _vel, angle, omega, joint_torque, rate, ground_contact, _foot_pos, foot_vel = outputs

    forward = pos[-1, 0]
    height = bm.mean((pos[:, 1] - cfg.height_target) ** 2)
    pitch = bm.mean(angle**2)
    angular_rate = bm.mean(omega**2)
    energy = bm.mean(bm.sum(joint_torque**2, axis=(-1, -2)))
    spike_cost = bm.mean(rate)
    smooth = bm.mean(
        bm.sum((joint_torque[1:] - joint_torque[:-1]) ** 2, axis=(-1, -2))
    )
    contact_mean = bm.mean(ground_contact, axis=1)
    contact_cost = bm.mean((contact_mean - cfg.desired_contact) ** 2)
    slip = bm.mean(bm.sum(ground_contact * (foot_vel[:, :, 0] ** 2), axis=1))
    if cfg.n_legs >= 2:
        gait_balance = bm.mean((ground_contact[:, 0] - ground_contact[:, 1]) ** 2)
    else:
        gait_balance = 0.0

    loss = (
        -cfg.w_forward * forward
        + cfg.w_height * height
        + cfg.w_pitch * pitch
        + cfg.w_omega * angular_rate
        + cfg.w_energy * energy
        + cfg.w_spike * spike_cost
        + cfg.w_smooth * smooth
        + cfg.w_contact * contact_cost
        + cfg.w_slip * slip
        - cfg.w_gait_balance * gait_balance
    )
    return loss


@dataclass(frozen=True)
class RolloutSnapshot:
    epoch: int
    ts_ms: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    angle: np.ndarray
    omega: np.ndarray
    force: np.ndarray
    hip_angle: np.ndarray
    knee_angle: np.ndarray
    hip_target: np.ndarray
    knee_target: np.ndarray
    ground_contact: np.ndarray
    foot_pos: np.ndarray
    joint_torque: np.ndarray
    heatmaps: np.ndarray
    heatmap_frame_times_ms: np.ndarray
    heatmap_domain: np.ndarray
    raster_t_ms: np.ndarray
    raster_neuron_idx: np.ndarray
    forward_end: float
    height_end: float

    @property
    def duration_ms(self) -> float:
        if self.ts_ms.size < 2:
            return 1.0
        step_ms = float(np.median(np.diff(self.ts_ms)))
        return max(1.0, float(self.ts_ms[-1] - self.ts_ms[0]) + step_ms)


class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.history = {
            "loss": [],
            "grad_norm": [],
            "forward": [],
            "timestamp": [],
        }
        self.history_version = 0
        self.rollouts: List[RolloutSnapshot] = []
        self.rollout_version = 0
        self.current_epoch = 0
        self.status = "Waiting for training worker..."
        self.complete = False
        self.error: Optional[str] = None

    def set_status(self, status: str):
        with self._lock:
            self.status = status

    def append_epoch_stats(
        self, epoch: int, loss_value: float, grad_norm: float, timestamp: float
    ):
        with self._lock:
            self.current_epoch = epoch
            self.history["loss"].append(loss_value)
            self.history["grad_norm"].append(grad_norm)
            self.history["timestamp"].append(timestamp)
            self.status = f"Training epoch {epoch}"
            self.history_version += 1

    def append_forward(self, epoch: int, forward: float):
        with self._lock:
            self.history["forward"].append((epoch, forward))
            self.history_version += 1

    def publish_rollout(self, rollout: RolloutSnapshot):
        with self._lock:
            if self.rollouts and self.rollouts[-1].epoch == rollout.epoch:
                self.rollouts[-1] = rollout
            else:
                self.rollouts.append(rollout)
            self.rollout_version += 1
            self.current_epoch = max(self.current_epoch, rollout.epoch)
            self.status = f"Showing rollout from epoch {rollout.epoch}"

    def mark_complete(self, status: str):
        with self._lock:
            self.complete = True
            self.status = status

    def mark_error(self, error: str):
        with self._lock:
            self.complete = True
            self.error = error
            self.status = "Training failed"

    def snapshot(self):
        with self._lock:
            history = {
                "loss": list(self.history["loss"]),
                "grad_norm": list(self.history["grad_norm"]),
                "forward": list(self.history["forward"]),
                "timestamp": list(self.history["timestamp"]),
            }
            return {
                "history": history,
                "history_version": self.history_version,
                "rollouts": tuple(self.rollouts),
                "rollout": self.rollouts[-1] if self.rollouts else None,
                "rollout_version": self.rollout_version,
                "epoch": self.current_epoch,
                "status": self.status,
                "complete": self.complete,
                "error": self.error,
            }


def train_system(cfg: Config, dashboard_state: Optional[DashboardState] = None):
    bm.set_dt(cfg.dt_ms)
    system = TrainableWalkingSystem(cfg)
    enable_training_mode(system)

    features = build_controller_features(cfg, cfg.episode_ms)
    rollout_ms = cfg.rollout_ms if cfg.rollout_ms is not None else cfg.episode_ms
    rollout_features = (
        features
        if math.isclose(float(rollout_ms), float(cfg.episode_ms))
        else build_controller_features(cfg, rollout_ms)
    )
    indices = bm.arange(features.shape[0])

    def loss_fn():
        system.set_physics_backend("differentiable")
        system.reset_state()
        outputs = bm.for_loop(system.step_run, (indices, features))
        return compute_loss(outputs, cfg)

    opt = bp.optim.Adam(lr=cfg.lr)
    opt.register_train_vars(system.train_vars().unique())
    f_grad = bm.grad(loss_fn, grad_vars=opt.vars_to_train, return_value=True)

    history = {
        "loss": [],
        "grad_norm": [],
        "forward": [],
        "timestamp": [],
    }

    if dashboard_state is not None:
        dashboard_state.set_status("Preparing initial rollout...")
        initial_snapshot = build_rollout_snapshot(
            system, rollout_features, cfg, epoch=0
        )
        history["forward"].append((0, initial_snapshot.forward_end))
        dashboard_state.append_forward(0, initial_snapshot.forward_end)
        dashboard_state.publish_rollout(initial_snapshot)

    for epoch in range(cfg.train_epochs):
        if dashboard_state is not None and dashboard_state.stop_requested.is_set():
            break

        if dashboard_state is not None:
            dashboard_state.set_status(
                f"Training epoch {epoch + 1}/{cfg.train_epochs}..."
            )

        grads, loss = f_grad()
        opt.update(grads)
        gnorm = global_norm(grads)

        loss_value = float(bm.as_numpy(loss))
        grad_value = float(bm.as_numpy(gnorm))
        timestamp = time.time()

        history["loss"].append(loss_value)
        history["grad_norm"].append(grad_value)
        history["timestamp"].append(timestamp)

        if dashboard_state is not None:
            dashboard_state.append_epoch_stats(
                epoch + 1, loss_value, grad_value, timestamp
            )

        eval_due = (epoch + 1) % cfg.eval_every == 0 or epoch == 0
        snapshot_due = cfg.vis_every > 0 and (epoch + 1) % cfg.vis_every == 0
        final_due = (epoch + 1) == cfg.train_epochs

        snapshot = None
        if dashboard_state is not None and (eval_due or snapshot_due or final_due):
            dashboard_state.set_status(
                f"Collecting rollout for epoch {epoch + 1}/{cfg.train_epochs}..."
            )
            snapshot = build_rollout_snapshot(
                system, rollout_features, cfg, epoch=epoch + 1
            )

        if eval_due:
            if snapshot is not None:
                forward = snapshot.forward_end
            else:
                forward = float(
                    bm.as_numpy(evaluate_forward(system, rollout_features, cfg))
                )
            history["forward"].append((epoch + 1, forward))
            if dashboard_state is not None:
                dashboard_state.append_forward(epoch + 1, forward)
            print(
                f"Epoch {epoch+1:4d} | Loss {loss_value:.4f} | "
                f"Forward {forward:.3f} | GradNorm {grad_value:.4f}"
            )

        if snapshot is not None and dashboard_state is not None:
            dashboard_state.publish_rollout(snapshot)

    return system, features, history


def evaluate_forward(system: TrainableWalkingSystem, features: bm.Array, cfg: Config):
    runner = collect_rollout(system, features, backend="pymunk")
    pos = np.asarray(runner.mon["pos"], dtype=float)
    forward = float(pos[-1, 0]) if pos.size else 0.0
    return bm.asarray(forward)


# -----------------------------
# Visualization helpers
# -----------------------------


class LiveDashboard:
    def __init__(self, cfg: Config, state: DashboardState):
        self.cfg = cfg
        self.state = state
        self.rollout_sequence: List[RolloutSnapshot] = []
        self.current_rollout_idx = -1
        self.current_rollout: Optional[RolloutSnapshot] = None
        self.last_history_version = -1
        self.last_rollout_version = -1
        self.last_title = ""
        self.clip_started_at = time.perf_counter()

        plt.ion()

        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(4, 4, hspace=0.65, wspace=0.5)

        self.ax_robot = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_heatmap = self.fig.add_subplot(gs[0:2, 2:4])
        self.ax_raster = self.fig.add_subplot(gs[2, 0:2])
        self.ax_training = self.fig.add_subplot(gs[2, 2:4])
        self.ax_contact = self.fig.add_subplot(gs[3, 0:2])
        self.ax_body = self.fig.add_subplot(gs[3, 2:4])
        self.ax_training_right = self.ax_training.twinx()

        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._setup_robot_panel()
        self._setup_heatmap_panel()
        self._setup_raster_panel()
        self._setup_training_panel()
        self._setup_contact_panel()
        self._setup_body_panel()

        self.fig.suptitle("Starting live training dashboard...")
        plt.show(block=False)

    def _on_close(self, _event):
        self.state.stop_requested.set()

    def _setup_robot_panel(self):
        self.ax_robot.set_title("Walking Animation")
        self.ax_robot.set_xlabel("x")
        self.ax_robot.set_ylabel("y")
        self.ax_robot.set_aspect("equal", adjustable="box")
        self.ax_robot.axhline(0.0, color="0.6", lw=1.0)

        self.robot_path, = self.ax_robot.plot([], [], color="0.75", lw=1.5)
        self.body_box, = self.ax_robot.plot([], [], "-", color="black", lw=2)
        self.body_com, = self.ax_robot.plot([], [], "o", color="black", markersize=4)
        self.thighs = [
            self.ax_robot.plot([], [], "-", lw=2)[0] for _ in range(self.cfg.n_legs)
        ]
        self.shanks = [
            self.ax_robot.plot([], [], "-", lw=2)[0] for _ in range(self.cfg.n_legs)
        ]
        self.knees = [
            self.ax_robot.plot([], [], "o", color="tab:orange", ms=3)[0]
            for _ in range(self.cfg.n_legs)
        ]
        self.feet = [
            self.ax_robot.plot([], [], "o", color="tab:blue", ms=4)[0]
            for _ in range(self.cfg.n_legs)
        ]
        self.force_line, = self.ax_robot.plot([], [], "-", color="tab:red", lw=2)
        self.hip_local = np.array(
            [
                [self.cfg.hip_x_offset, -0.5 * self.cfg.body_height],
                [-self.cfg.hip_x_offset, -0.5 * self.cfg.body_height],
            ],
            dtype=float,
        )
        self.body_outline_local = np.array(
            [
                [0.5 * self.cfg.body_length, 0.5 * self.cfg.body_height],
                [0.5 * self.cfg.body_length, -0.5 * self.cfg.body_height],
                [-0.5 * self.cfg.body_length, -0.5 * self.cfg.body_height],
                [-0.5 * self.cfg.body_length, 0.5 * self.cfg.body_height],
                [0.5 * self.cfg.body_length, 0.5 * self.cfg.body_height],
            ],
            dtype=float,
        )
        self.force_reference = 1.0

    def _setup_heatmap_panel(self):
        self.ax_heatmap.set_title("Spatial Firing")
        self.ax_heatmap.set_xlabel("x")
        self.ax_heatmap.set_ylabel("y")
        self.im = self.ax_heatmap.imshow(
            np.zeros((2, 2)),
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_heatmap)
        self.cbar.set_label("Spike Count")

    def _setup_raster_panel(self):
        self.ax_raster.set_title("Spike Raster")
        self.ax_raster.set_xlabel("time (ms)")
        self.ax_raster.set_ylabel("sampled neuron")
        self.raster_scatter = self.ax_raster.scatter([], [], s=3, color="black")
        self.raster_cursor = self.ax_raster.axvline(
            0.0, color="tab:red", lw=1.0, ls="--", alpha=0.7
        )

    def _setup_training_panel(self):
        self.ax_training.set_title("Training Progress")
        self.ax_training.set_xlabel("epoch")
        self.ax_training.set_ylabel("loss / grad")
        self.ax_training_right.set_ylabel("forward")

        self.line_loss, = self.ax_training.plot([], [], color="tab:blue", label="Loss")
        self.line_grad, = self.ax_training.plot(
            [], [], color="tab:orange", label="Grad Norm"
        )
        self.line_forward, = self.ax_training_right.plot(
            [], [], "o-", color="tab:green", ms=3, label="Forward"
        )
        handles = [self.line_loss, self.line_grad, self.line_forward]
        self.ax_training.legend(handles, [h.get_label() for h in handles], loc="upper left")
        self.training_text = self.ax_training.text(
            0.02,
            0.05,
            "",
            transform=self.ax_training.transAxes,
            va="bottom",
            ha="left",
        )

    def _setup_contact_panel(self):
        self.ax_contact.set_title("Ground Contact")
        self.ax_contact.set_xlabel("time (ms)")
        self.ax_contact.set_ylabel("contact")
        self.ax_contact.set_ylim(-0.05, 1.05)
        colors = ["tab:blue", "tab:orange"]
        self.ground_lines = []
        for i in range(self.cfg.n_legs):
            color = colors[i % len(colors)]
            self.ground_lines.append(
                self.ax_contact.plot(
                    [], [], lw=1.5, color=color, label=f"Leg {i + 1}"
                )[0]
            )
        self.contact_cursor = self.ax_contact.axvline(
            0.0, color="black", lw=1.0, ls="--", alpha=0.6
        )
        self.ax_contact.legend(loc="upper left")

    def _setup_body_panel(self):
        self.ax_body.set_title("Body Pose")
        self.ax_body.set_xlabel("time (ms)")
        self.ax_body.set_ylabel("state")
        self.line_body_height, = self.ax_body.plot(
            [], [], color="tab:purple", label="height"
        )
        self.line_body_pitch, = self.ax_body.plot(
            [], [], color="tab:brown", label="pitch"
        )
        self.body_cursor = self.ax_body.axvline(
            0.0, color="black", lw=1.0, ls="--", alpha=0.6
        )
        self.ax_body.legend(loc="upper left")

    def _rotation_matrix_np(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _world_points_np(
        self, local_points: np.ndarray, pos: np.ndarray, theta: float
    ) -> np.ndarray:
        rot = self._rotation_matrix_np(theta)
        return local_points @ rot.T + pos[None, :]

    def _segment_dir_np(self, angle: float) -> np.ndarray:
        return np.array([np.sin(angle), -np.cos(angle)], dtype=float)

    def _lookup_index(self, times: np.ndarray, target_time: float) -> int:
        if times.size <= 1:
            return 0
        idx = int(np.searchsorted(times, target_time, side="right") - 1)
        return int(np.clip(idx, 0, len(times) - 1))

    def _set_current_rollout_idx(self, idx: int, reset_clock: bool) -> None:
        if not self.rollout_sequence:
            self.current_rollout_idx = -1
            self.current_rollout = None
            return

        idx = int(idx) % len(self.rollout_sequence)
        rollout = self.rollout_sequence[idx]
        rollout_changed = (
            self.current_rollout_idx != idx or self.current_rollout is not rollout
        )
        self.current_rollout_idx = idx
        self.current_rollout = rollout

        if rollout_changed:
            self._apply_rollout(rollout)
        if reset_clock:
            self.clip_started_at = time.perf_counter()

    def _apply_rollouts(self, rollouts: Tuple[RolloutSnapshot, ...]) -> None:
        previous_epoch = self.current_rollout.epoch if self.current_rollout else None
        self.rollout_sequence = list(rollouts)

        if not self.rollout_sequence:
            self.current_rollout_idx = -1
            self.current_rollout = None
            return

        if previous_epoch is None:
            self._set_current_rollout_idx(0, reset_clock=True)
            return

        for idx, rollout in enumerate(self.rollout_sequence):
            if rollout.epoch == previous_epoch:
                self._set_current_rollout_idx(idx, reset_clock=False)
                return

        self._set_current_rollout_idx(len(self.rollout_sequence) - 1, reset_clock=True)

    def _advance_rollout_if_needed(self) -> None:
        if self.current_rollout is None or not self.rollout_sequence:
            return

        elapsed_ms = (time.perf_counter() - self.clip_started_at) * 1000.0
        while elapsed_ms >= self.current_rollout.duration_ms:
            elapsed_ms -= self.current_rollout.duration_ms
            next_idx = (self.current_rollout_idx + 1) % len(self.rollout_sequence)
            self._set_current_rollout_idx(next_idx, reset_clock=False)
            self.clip_started_at = time.perf_counter() - elapsed_ms / 1000.0

    def _playback_time_ms(self) -> float:
        rollout = self.current_rollout
        if rollout is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self.clip_started_at) * 1000.0
        span_ms = max(0.0, float(rollout.ts_ms[-1] - rollout.ts_ms[0]))
        return float(rollout.ts_ms[0]) + min(elapsed_ms, span_ms)

    def _apply_history(self, view):
        history = view["history"]
        epochs = np.arange(1, len(history["loss"]) + 1)
        self.line_loss.set_data(epochs, history["loss"])
        self.line_grad.set_data(epochs, history["grad_norm"])

        if history["forward"]:
            f_epochs, f_vals = zip(*history["forward"])
            self.line_forward.set_data(f_epochs, f_vals)
        else:
            self.line_forward.set_data([], [])

        self.ax_training.relim()
        self.ax_training.autoscale_view()
        self.ax_training_right.relim()
        self.ax_training_right.autoscale_view()

        latest_bits = [f"epoch {view['epoch']}/{self.cfg.train_epochs}"]
        if history["loss"]:
            latest_bits.append(f"loss {history['loss'][-1]:.3f}")
        if history["grad_norm"]:
            latest_bits.append(f"grad {history['grad_norm'][-1]:.3f}")
        if history["forward"]:
            latest_bits.append(f"forward {history['forward'][-1][1]:.3f}")
        self.training_text.set_text(" | ".join(latest_bits))

    def _apply_rollout(self, rollout: RolloutSnapshot):
        body_x = rollout.pos[:, 0]
        body_y = rollout.pos[:, 1]
        foot_x = rollout.foot_pos[:, :, 0].reshape(-1)
        foot_y = rollout.foot_pos[:, :, 1].reshape(-1)

        min_x = float(min(np.min(body_x) - self.cfg.body_length, np.min(foot_x) - 0.1))
        max_x = float(max(np.max(body_x) + self.cfg.body_length, np.max(foot_x) + 0.1))
        min_y = float(min(np.min(body_y) - self.cfg.body_height, np.min(foot_y) - 0.05, 0.0))
        max_y = float(
            max(
                np.max(body_y) + self.cfg.body_height,
                np.max(foot_y) + 0.15,
                self.cfg.height_target + 0.25,
            )
        )
        self.ax_robot.set_xlim(min_x, max_x)
        self.ax_robot.set_ylim(min_y, max_y)
        self.force_reference = max(1.0, float(np.max(np.linalg.norm(rollout.force, axis=1))))

        vmax = max(1.0, float(np.max(rollout.heatmaps)))
        self.im.set_data(rollout.heatmaps[0].T)
        self.im.set_extent(
            [
                0.0,
                float(rollout.heatmap_domain[0]),
                0.0,
                float(rollout.heatmap_domain[1]),
            ]
        )
        self.im.set_clim(vmin=0.0, vmax=vmax)
        self.cbar.update_normal(self.im)

        if rollout.raster_t_ms.size:
            offsets = np.column_stack([rollout.raster_t_ms, rollout.raster_neuron_idx])
        else:
            offsets = np.empty((0, 2))
        self.raster_scatter.set_offsets(offsets)
        self.ax_raster.set_xlim(float(rollout.ts_ms[0]), float(rollout.ts_ms[-1]))
        raster_max = (
            int(np.max(rollout.raster_neuron_idx)) + 1
            if rollout.raster_neuron_idx.size
            else max(1, self.cfg.raster_neurons)
        )
        self.ax_raster.set_ylim(-1, max(1, raster_max))

        time_axis = rollout.ts_ms
        for leg_idx, line in enumerate(self.ground_lines):
            line.set_data(time_axis, rollout.ground_contact[:, leg_idx])
        self.ax_contact.set_xlim(float(time_axis[0]), float(time_axis[-1]))

        self.line_body_height.set_data(time_axis, rollout.pos[:, 1])
        self.line_body_pitch.set_data(time_axis, rollout.angle)
        self.ax_body.set_xlim(float(time_axis[0]), float(time_axis[-1]))
        self.ax_body.relim()
        self.ax_body.autoscale_view()

    def _update_robot(self, rollout: RolloutSnapshot, idx: int, current_time_ms: float):
        p = rollout.pos[idx]
        theta = float(rollout.angle[idx])
        x, y = float(p[0]), float(p[1])
        self.robot_path.set_data(rollout.pos[: idx + 1, 0], rollout.pos[: idx + 1, 1])
        body_outline = self._world_points_np(
            self.body_outline_local, np.asarray([x, y], dtype=float), theta
        )
        hip_pos = self._world_points_np(
            self.hip_local, np.asarray([x, y], dtype=float), theta
        )
        self.body_box.set_data(body_outline[:, 0], body_outline[:, 1])
        self.body_com.set_data([x], [y])

        for leg_idx in range(self.cfg.n_legs):
            hip = hip_pos[leg_idx]
            thigh_abs = theta + float(rollout.hip_angle[idx, leg_idx])
            shank_abs = thigh_abs + float(rollout.knee_angle[idx, leg_idx])
            knee = hip + self.cfg.thigh_length * self._segment_dir_np(thigh_abs)
            foot = rollout.foot_pos[idx, leg_idx]
            self.thighs[leg_idx].set_data([hip[0], knee[0]], [hip[1], knee[1]])
            self.shanks[leg_idx].set_data([knee[0], foot[0]], [knee[1], foot[1]])
            self.knees[leg_idx].set_data([knee[0]], [knee[1]])
            self.feet[leg_idx].set_data([foot[0]], [max(0.0, foot[1])])

        force = np.asarray(rollout.force[idx], dtype=float)
        force_scale = 0.35 * self.cfg.body_length / max(1e-6, self.force_reference)
        self.force_line.set_data(
            [x, x + force[0] * force_scale],
            [y, y + force[1] * force_scale],
        )

        t_rel = current_time_ms - float(rollout.ts_ms[0])
        clip_label = f"snapshot {self.current_rollout_idx + 1}/{len(self.rollout_sequence)}"
        self.ax_robot.set_title(
            f"Rigid Walker | {clip_label} | epoch {rollout.epoch} | t={t_rel:.0f} ms | "
            f"x={x:.3f} y={y:.3f} pitch={theta:.3f}"
        )

    def _update_playback(self):
        self._advance_rollout_if_needed()
        rollout = self.current_rollout
        if rollout is None:
            return

        current_time_ms = self._playback_time_ms()
        state_idx = self._lookup_index(rollout.ts_ms, current_time_ms)
        heatmap_idx = self._lookup_index(rollout.heatmap_frame_times_ms, current_time_ms)

        self._update_robot(rollout, state_idx, current_time_ms)
        self.im.set_data(rollout.heatmaps[heatmap_idx].T)
        t_rel = current_time_ms - float(rollout.ts_ms[0])
        self.ax_heatmap.set_title(
            f"Spatial Firing | snapshot {self.current_rollout_idx + 1}/{len(self.rollout_sequence)} | "
            f"epoch {rollout.epoch} | t={t_rel:.0f} ms"
        )

        cursor_x = float(rollout.ts_ms[state_idx])
        self.raster_cursor.set_xdata([cursor_x, cursor_x])
        self.contact_cursor.set_xdata([cursor_x, cursor_x])
        self.body_cursor.set_xdata([cursor_x, cursor_x])

    def refresh(self) -> bool:
        if not plt.fignum_exists(self.fig.number):
            self.state.stop_requested.set()
            return False

        view = self.state.snapshot()
        if view["history_version"] != self.last_history_version:
            self._apply_history(view)
            self.last_history_version = view["history_version"]

        if view["rollouts"] and view["rollout_version"] != self.last_rollout_version:
            self._apply_rollouts(view["rollouts"])
            self.last_rollout_version = view["rollout_version"]

        self._update_playback()

        if view["error"]:
            title = "Training failed. See terminal for traceback."
        elif view["complete"]:
            title = f"{view['status']} | close the window to exit"
        else:
            title = view["status"]

        if title != self.last_title:
            self.fig.suptitle(title)
            self.last_title = title

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return True


class TrainingWorker(threading.Thread):
    def __init__(self, cfg: Config, state: DashboardState):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.state = state
        self.result = None
        self.error: Optional[str] = None

    def run(self):
        try:
            self.result = train_system(self.cfg, dashboard_state=self.state)
            if self.state.stop_requested.is_set():
                self.state.mark_complete("Training stopped")
            else:
                trained_epochs = 0
                if self.result is not None:
                    trained_epochs = len(self.result[2]["loss"])
                self.state.mark_complete(
                    f"Training complete after {trained_epochs} epochs"
                )
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)
            self.state.mark_error(self.error)


def plot_training_history(history: Dict[str, List[float]], title: Optional[str] = None):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    ax[0].plot(history["loss"], label="Loss")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history["grad_norm"], label="Grad Norm")
    if history["forward"]:
        epochs, fwds = zip(*history["forward"])
        ax[1].plot(epochs, fwds, "o-", label="Forward (eval)")
    ax[1].set_ylabel("Grad Norm / Forward")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_surrogate_gradient(spk_fun, title: Optional[str] = None):
    xs = bm.linspace(-3, 3, 400)
    try:
        ys = spk_fun(xs)
        grads = bm.vector_grad(spk_fun)(xs)
        xs_np = bm.as_numpy(xs)
        ys_np = bm.as_numpy(ys)
        grads_np = bm.as_numpy(grads)
    except Exception as exc:
        print(f"Surrogate gradient plot failed: {exc}")
        return

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(xs_np, ys_np)
    ax[0].set_title("Surrogate Spike")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("spk(x)")

    ax[1].plot(xs_np, grads_np)
    ax[1].set_title("Surrogate Gradient")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("d spk / dx")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_spike_raster(spikes: np.ndarray, sample: int = 200, title: Optional[str] = None):
    if spikes.ndim != 2:
        return
    num_neurons = spikes.shape[1]
    if num_neurons > sample:
        idx = np.random.choice(num_neurons, sample, replace=False)
        spikes = spikes[:, idx]
    ts, ns = np.where(spikes > 0)
    plt.figure(figsize=(8, 4))
    plt.scatter(ts, ns, s=2)
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title(title or "Spike Raster (sampled)")
    plt.tight_layout()
    plt.show()


def prepare_spike_histograms(
    positions: np.ndarray,
    domain: np.ndarray,
    grid_size: Tuple[int, int],
    spikes: np.ndarray,
    ts: np.ndarray,
    window_ms: float,
    frame_step_ms: float,
):
    frame_times = np.arange(ts[0], ts[-1] + frame_step_ms, frame_step_ms)
    x_edges = np.linspace(0, domain[0], grid_size[0] + 1)
    y_edges = np.linspace(0, domain[1], grid_size[1] + 1)

    histograms = np.zeros((len(frame_times), grid_size[0], grid_size[1]), dtype=float)
    for i, frame_t in enumerate(frame_times):
        win_start = frame_t - window_ms
        idx_start = np.searchsorted(ts, win_start, side="left")
        idx_end = np.searchsorted(ts, frame_t, side="right")
        spike_counts = np.sum(spikes[idx_start:idx_end, :], axis=0)
        hist, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=[x_edges, y_edges],
            weights=spike_counts,
        )
        histograms[i] = hist
    return histograms, frame_times


def animate_spike_heatmap(
    histograms: np.ndarray,
    frame_times: np.ndarray,
    domain,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = max(1.0, float(np.max(histograms)))
    im = ax.imshow(
        histograms[0].T,
        origin="lower",
        extent=[0, domain[0], 0, domain[1]],
        cmap="hot",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    title_artist = ax.set_title("")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(i):
        im.set_data(histograms[i].T)
        if title:
            title_artist.set_text(f"{title} | t = {frame_times[i]:.1f} ms")
        else:
            title_artist.set_text(f"t = {frame_times[i]:.1f} ms")
        return (im, title_artist)

    ani = FuncAnimation(fig, update, frames=len(frame_times), interval=50, blit=False)
    plt.show()
    return ani


def animate_robot(
    pos: np.ndarray,
    angle: np.ndarray,
    hip_angle: np.ndarray,
    knee_angle: np.ndarray,
    foot_pos: np.ndarray,
    forces: np.ndarray,
    cfg: Config,
    title: Optional[str] = None,
):
    hip_local = np.array(
        [
            [cfg.hip_x_offset, -0.5 * cfg.body_height],
            [-cfg.hip_x_offset, -0.5 * cfg.body_height],
        ],
        dtype=float,
    )
    body_outline_local = np.array(
        [
            [0.5 * cfg.body_length, 0.5 * cfg.body_height],
            [0.5 * cfg.body_length, -0.5 * cfg.body_height],
            [-0.5 * cfg.body_length, -0.5 * cfg.body_height],
            [-0.5 * cfg.body_length, 0.5 * cfg.body_height],
            [0.5 * cfg.body_length, 0.5 * cfg.body_height],
        ],
        dtype=float,
    )

    def rotation_matrix(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def world_points(local_points: np.ndarray, body_pos: np.ndarray, theta: float) -> np.ndarray:
        rot = rotation_matrix(theta)
        return local_points @ rot.T + body_pos[None, :]

    def segment_dir(joint_angle: float) -> np.ndarray:
        return np.array([np.sin(joint_angle), -np.cos(joint_angle)], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or "Rigid Walker")
    ax.axhline(0.0, color="0.6", lw=1.0)

    path_line, = ax.plot([], [], color="0.75", lw=1.5)
    body_box, = ax.plot([], [], "-", color="black", lw=2)
    body_com, = ax.plot([], [], "o", color="black", markersize=4)
    thighs = [ax.plot([], [], "-", lw=2)[0] for _ in range(cfg.n_legs)]
    shanks = [ax.plot([], [], "-", lw=2)[0] for _ in range(cfg.n_legs)]
    knees = [
        ax.plot([], [], "o", color="tab:orange", ms=3)[0] for _ in range(cfg.n_legs)
    ]
    feet = [
        ax.plot([], [], "o", color="tab:blue", ms=4)[0] for _ in range(cfg.n_legs)
    ]
    force_line, = ax.plot([], [], "-", color="tab:red", lw=2)

    min_x = float(
        min(np.min(pos[:, 0]) - cfg.body_length, np.min(foot_pos[:, :, 0]) - 0.1)
    )
    max_x = float(
        max(np.max(pos[:, 0]) + cfg.body_length, np.max(foot_pos[:, :, 0]) + 0.1)
    )
    min_y = float(
        min(
            np.min(pos[:, 1]) - cfg.body_height,
            np.min(foot_pos[:, :, 1]) - 0.05,
            0.0,
        )
    )
    max_y = float(
        max(
            np.max(pos[:, 1]) + cfg.body_height,
            np.max(foot_pos[:, :, 1]) + 0.15,
            cfg.height_target + 0.25,
        )
    )
    force_reference = max(1.0, float(np.max(np.linalg.norm(forces, axis=1))))
    force_scale = 0.35 * cfg.body_length / force_reference

    def init():
        path_line.set_data([], [])
        body_box.set_data([], [])
        body_com.set_data([], [])
        force_line.set_data([], [])
        for artist in thighs + shanks + knees + feet:
            artist.set_data([], [])
        return [path_line, body_box, body_com, force_line] + thighs + shanks + knees + feet

    def update(i):
        p = np.asarray(pos[i], dtype=float)
        theta = float(angle[i])
        path_line.set_data(pos[: i + 1, 0], pos[: i + 1, 1])
        body_outline = world_points(body_outline_local, p, theta)
        hip_pos = world_points(hip_local, p, theta)
        body_box.set_data(body_outline[:, 0], body_outline[:, 1])
        body_com.set_data([p[0]], [p[1]])

        for j in range(cfg.n_legs):
            hip = hip_pos[j]
            thigh_abs = theta + float(hip_angle[i, j])
            knee_pos = hip + cfg.thigh_length * segment_dir(thigh_abs)
            foot = np.asarray(foot_pos[i, j], dtype=float)

            thighs[j].set_data([hip[0], knee_pos[0]], [hip[1], knee_pos[1]])
            shanks[j].set_data([knee_pos[0], foot[0]], [knee_pos[1], foot[1]])
            knees[j].set_data([knee_pos[0]], [knee_pos[1]])
            feet[j].set_data([foot[0]], [max(0.0, foot[1])])

        f = forces[i]
        force_line.set_data(
            [p[0], p[0] + f[0] * force_scale],
            [p[1], p[1] + f[1] * force_scale],
        )
        t_ms = float(i * cfg.dt_ms)
        ax.set_title(
            f"{title or 'Rigid Walker'} | t={t_ms:.0f} ms | "
            f"x={p[0]:.3f} y={p[1]:.3f} pitch={theta:.3f}"
        )
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        return [path_line, body_box, body_com, force_line] + thighs + shanks + knees + feet

    ani = FuncAnimation(
        fig,
        update,
        frames=len(pos),
        init_func=init,
        interval=max(1, int(round(1000.0 / max(1.0, cfg.animation_fps)))),
        blit=False,
    )
    plt.show()
    return ani


def collect_rollout(
    system: TrainableWalkingSystem,
    features: bm.Array,
    backend: str = "pymunk",
):
    previous_backend = system.physics_backend
    system.set_physics_backend(backend)
    try:
        if backend == "pymunk":
            return _collect_rollout_python(system, features)

        runner = bp.DSRunner(
            system,
            monitors={
                "E.spike": system.core.E.spike,
                "pos": system.pos,
                "vel": system.vel,
                "angle": system.angle,
                "omega": system.omega,
                "force": system.last_force,
                "hip_angle": system.hip_angle,
                "knee_angle": system.knee_angle,
                "hip_target": system.last_hip_target,
                "knee_target": system.last_knee_target,
                "ground_contact": system.last_contact,
                "foot_pos": system.last_foot_pos,
                "joint_torque": system.last_joint_torque,
            },
            data_first_axis="T",
        )
        # Reset explicitly without batch sizing to avoid shape mismatches.
        system.reset_state()
        runner.run(inputs=features, reset_state=False)
        return runner
    finally:
        system.set_physics_backend(previous_backend)


def _collect_rollout_python(system: TrainableWalkingSystem, features: bm.Array):
    system.reset_state()

    num_steps = int(features.shape[0])
    dt = float(bm.get_dt())
    mon = {
        "E.spike": [],
        "pos": [],
        "vel": [],
        "angle": [],
        "omega": [],
        "force": [],
        "hip_angle": [],
        "knee_angle": [],
        "hip_target": [],
        "knee_target": [],
        "ground_contact": [],
        "foot_pos": [],
        "joint_torque": [],
        "ts": [],
    }

    for i in range(num_steps):
        bp.share.save(t=i * dt, i=i, dt=dt)
        system(features[i])

        mon["E.spike"].append(np.asarray(system.core.E.spike))
        mon["pos"].append(np.asarray(system.pos.value, dtype=float))
        mon["vel"].append(np.asarray(system.vel.value, dtype=float))
        mon["angle"].append(float(system.angle.value))
        mon["omega"].append(float(system.omega.value))
        mon["force"].append(np.asarray(system.last_force.value, dtype=float))
        mon["hip_angle"].append(np.asarray(system.hip_angle.value, dtype=float))
        mon["knee_angle"].append(np.asarray(system.knee_angle.value, dtype=float))
        mon["hip_target"].append(np.asarray(system.last_hip_target.value, dtype=float))
        mon["knee_target"].append(np.asarray(system.last_knee_target.value, dtype=float))
        mon["ground_contact"].append(np.asarray(system.last_contact.value, dtype=float))
        mon["foot_pos"].append(np.asarray(system.last_foot_pos.value, dtype=float))
        mon["joint_torque"].append(np.asarray(system.last_joint_torque.value, dtype=float))
        mon["ts"].append(i * dt)

        clear_input(system)

    stacked = {key: np.asarray(value) for key, value in mon.items()}
    return SimpleNamespace(mon=stacked)


def select_raster_indices(num_neurons: int, sample: int) -> np.ndarray:
    if num_neurons <= sample:
        return np.arange(num_neurons, dtype=int)
    return np.linspace(0, num_neurons - 1, sample, dtype=int)


def prepare_raster_points(
    spikes: np.ndarray, ts_ms: np.ndarray, sample: int
) -> Tuple[np.ndarray, np.ndarray]:
    if spikes.ndim != 2 or spikes.shape[0] == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    sample_idx = select_raster_indices(spikes.shape[1], sample)
    sampled_spikes = spikes[:, sample_idx]
    spike_rows, spike_cols = np.where(sampled_spikes > 0)
    if spike_rows.size == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    return ts_ms[spike_rows], spike_cols.astype(float)


def build_rollout_snapshot(
    system: TrainableWalkingSystem,
    features: bm.Array,
    cfg: Config,
    epoch: int,
) -> RolloutSnapshot:
    runner = collect_rollout(system, features)

    spikes = np.asarray(runner.mon["E.spike"])
    pos = np.asarray(runner.mon["pos"], dtype=float)
    vel = np.asarray(runner.mon["vel"], dtype=float)
    angle = np.asarray(runner.mon["angle"], dtype=float)
    omega = np.asarray(runner.mon["omega"], dtype=float)
    force = np.asarray(runner.mon["force"], dtype=float)
    hip_angle = np.asarray(runner.mon["hip_angle"], dtype=float)
    knee_angle = np.asarray(runner.mon["knee_angle"], dtype=float)
    hip_target = np.asarray(runner.mon["hip_target"], dtype=float)
    knee_target = np.asarray(runner.mon["knee_target"], dtype=float)
    ground_contact = np.asarray(runner.mon["ground_contact"], dtype=float)
    foot_pos = np.asarray(runner.mon["foot_pos"], dtype=float)
    joint_torque = np.asarray(runner.mon["joint_torque"], dtype=float)
    ts_ms = np.asarray(runner.mon["ts"], dtype=float)

    domain = np.asarray(system.core.E.embedding.domain, dtype=float)
    grid_size = tuple(int(v) for v in np.asarray(system.core.E.size).ravel())
    heatmap_step_ms = max(cfg.heatmap_frame_ms, 1000.0 / max(1.0, cfg.animation_fps))
    heatmaps, heatmap_frame_times_ms = prepare_spike_histograms(
        positions=np.asarray(system.core.E.positions, dtype=float),
        domain=domain,
        grid_size=grid_size,
        spikes=spikes,
        ts=ts_ms,
        window_ms=cfg.heatmap_window_ms,
        frame_step_ms=heatmap_step_ms,
    )

    raster_t_ms, raster_neuron_idx = prepare_raster_points(
        spikes, ts_ms, cfg.raster_neurons
    )

    forward_end = float(pos[-1, 0]) if pos.size else 0.0
    height_end = float(pos[-1, 1]) if pos.size else 0.0

    return RolloutSnapshot(
        epoch=epoch,
        ts_ms=ts_ms,
        pos=pos,
        vel=vel,
        angle=angle,
        omega=omega,
        force=force,
        hip_angle=hip_angle,
        knee_angle=knee_angle,
        hip_target=hip_target,
        knee_target=knee_target,
        ground_contact=ground_contact,
        foot_pos=foot_pos,
        joint_torque=joint_torque,
        heatmaps=heatmaps,
        heatmap_frame_times_ms=heatmap_frame_times_ms,
        heatmap_domain=domain,
        raster_t_ms=raster_t_ms,
        raster_neuron_idx=raster_neuron_idx,
        forward_end=forward_end,
        height_end=height_end,
    )


def _supports_live_dashboard() -> bool:
    backend = plt.get_backend().lower()
    non_interactive = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }
    return backend not in non_interactive


def run_live_training_dashboard(cfg: Config):
    state = DashboardState()
    dashboard = LiveDashboard(cfg, state)
    worker = TrainingWorker(cfg, state)
    worker.start()

    try:
        while dashboard.refresh():
            plt.pause(1.0 / max(1.0, cfg.animation_fps))
    finally:
        state.stop_requested.set()
        worker.join(timeout=1.0)

    if worker.error is not None:
        raise RuntimeError(worker.error)
    return worker.result


# -----------------------------
# Main
# -----------------------------


def main():
    cfg = Config()
    if _supports_live_dashboard():
        run_live_training_dashboard(cfg)
    else:
        train_system(cfg)


if __name__ == "__main__":
    main()
