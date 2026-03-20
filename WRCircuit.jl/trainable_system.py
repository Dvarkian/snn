"""
Trainable spiking control system for a 2D walking robot.

This script builds a Spatial SNN (same model used in run_simulation.py),
injects low-dimensional inputs through fixed spatial patterns, and trains
readout weights with surrogate gradients + BPTT to make a simple 2D robot
walk forward.

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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import brainpy as bp
import brainpy.math as bm

spatial_mod = importlib.import_module("src.models.Spatial")
from src.models.Spatial import Spatial
neurons_mod = importlib.import_module("src.neurons")


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


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class Config:
    # Spatial model parameters (same defaults as run_simulation.py)
    rho: int = 10000
    dx: float = 1.0
    seed: int = 42

    # Robot parameters
    n_legs: int = 2
    mass: float = 1.0
    drag: float = 0.2
    gravity: float = 1.2  # [force units / mass], downward acceleration in side-view
    ground_k: float = 25.0  # soft ground penalty strength (keeps y >= 0)
    height_target: float = 0.35  # desired body height in side-view
    max_force: float = 1.5
    leg_radius: float = 0.15

    # Simulation parameters
    dt_ms: float = 1.0
    episode_ms: float = 2000.0
    base_freq_hz: float = 1.5
    target_vx: float = 0.4
    target_vy: float = 0.0

    # Input injection
    feature_dim: int = 7  # [1, target_vx, target_vy, sin, cos, sin2, cos2]
    input_rank: int = 8
    input_scale: float = 0.6
    input_init_scale: float = 0.05

    # Readout
    out_per_leg: int = 3  # contact, fx, fy
    readout_rank: Optional[int] = 128  # None means full readout
    readout_init_scale: float = 0.02
    rate_decay: float = 0.95

    # Training
    train_epochs: int = 80
    lr: float = 3e-3
    eval_every: int = 10
    train_core: bool = False  # if True, also train core synaptic weights (heavy)
    vis_every: int = 5  # visualize progress every N epochs (0 disables)

    # Loss weights
    w_forward: float = 1.0
    # In side-view this is the vertical "uprightness" term.
    w_lateral: float = 0.0
    w_height: float = 0.35
    w_energy: float = 0.02
    w_spike: float = 0.01
    w_smooth: float = 0.02
    w_contact: float = 0.02
    w_gait_balance: float = 0.08  # encourage alternating left/right stance
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


def global_norm(grads: Dict[str, bm.Array]) -> bm.Array:
    total = 0.0
    for g in grads.values():
        total += bm.sum(g * g)
    return bm.sqrt(total + 1e-12)


# -----------------------------
# Trainable system
# -----------------------------


class TrainableWalkingSystem(bp.DynamicalSystem):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.n_legs = cfg.n_legs
        self.out_per_leg = cfg.out_per_leg
        self.out_dim = self.n_legs * self.out_per_leg

        # Core SNN (same model as run_simulation.py)
        self.core = Spatial(key=cfg.seed, rho=cfg.rho, dx=cfg.dx)

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
        self.last_force = bm.Variable(bm.zeros((2,)))
        self.last_contact = bm.Variable(bm.zeros((self.n_legs,)))

    def reset_state(self, batch_size=None, **kwargs):
        self.core.reset_state(batch_size)
        self.rate.value = bm.zeros_like(self.rate.value)
        self.pos.value = bm.asarray([0.0, self.cfg.height_target])
        self.vel.value = bm.zeros_like(self.vel.value)
        self.last_force.value = bm.zeros_like(self.last_force.value)
        self.last_contact.value = bm.zeros_like(self.last_contact.value)

    def _inject_inputs(self, features: bm.Array):
        coeffs_E = features @ self.W_in_E
        coeffs_I = features @ self.W_in_I

        E_in_flat = coeffs_E @ self.E_patterns
        I_in_flat = coeffs_I @ self.I_patterns

        E_in = bm.reshape(E_in_flat, self.core.E.varshape)
        I_in = bm.reshape(I_in_flat, self.core.I.varshape)

        self.core.Ein.value = self.cfg.input_scale * E_in
        self.core.Iin.value = self.cfg.input_scale * I_in

    def _readout(self) -> Tuple[bm.Array, bm.Array, bm.Array]:
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

        contact = bm.sigmoid(motor_raw[:, 0])
        fx = bm.tanh(motor_raw[:, 1]) * self.cfg.max_force
        fy = bm.tanh(motor_raw[:, 2]) * self.cfg.max_force
        return contact, fx, fy

    def update(self, features: bm.Array):
        # Inject inputs
        self._inject_inputs(features)

        # Run core network
        self.core()

        # Readout to motor commands
        contact, fx, fy = self._readout()

        forces = bm.stack([fx, fy], axis=1)
        total_force = bm.sum(forces * contact[:, None], axis=0)

        # Robot dynamics (2D point-mass)
        dt = bm.get_dt() / 1000.0
        # Side-view dynamics: x forward, y vertical with gravity and a soft ground constraint.
        # Forces are applied to the body; stance contact modulates which legs contribute.
        vx, vy = self.vel.value[0], self.vel.value[1]
        ay_ground = self.cfg.ground_k * bm.relu(-self.pos.value[1])
        ax = total_force[0] / self.cfg.mass - self.cfg.drag * vx
        ay = total_force[1] / self.cfg.mass - self.cfg.drag * vy - self.cfg.gravity + ay_ground / self.cfg.mass

        acc = bm.stack([ax, ay])
        self.vel.value = self.vel.value + acc * dt
        self.pos.value = self.pos.value + self.vel.value * dt

        self.last_force.value = total_force
        self.last_contact.value = contact

        mean_rate = bm.mean(self.rate.value)
        return self.pos.value, self.vel.value, total_force, mean_rate, contact


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
    pos, vel, force, rate, contact = outputs

    forward = pos[-1, 0]
    # Uprightness / height keeping (side-view).
    height = bm.mean((pos[:, 1] - cfg.height_target) ** 2)
    energy = bm.mean(bm.sum(force**2, axis=-1))
    spike_cost = bm.mean(rate)
    smooth = bm.mean(bm.sum((force[1:] - force[:-1]) ** 2, axis=-1))
    contact_mean = bm.mean(contact, axis=1)
    contact_cost = bm.mean((contact_mean - cfg.desired_contact) ** 2)
    # Encourage alternating stance between legs (mainly helps 2-leg clarity in animation).
    if cfg.n_legs >= 2:
        gait_balance = bm.mean((contact[:, 0] - contact[:, 1]) ** 2)
    else:
        gait_balance = 0.0

    loss = (
        -cfg.w_forward * forward
        + cfg.w_height * height
        + cfg.w_energy * energy
        + cfg.w_spike * spike_cost
        + cfg.w_smooth * smooth
        + cfg.w_contact * contact_cost
        - cfg.w_gait_balance * gait_balance
    )
    return loss


@dataclass(frozen=True)
class RolloutSnapshot:
    epoch: int
    ts_ms: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    force: np.ndarray
    contact: np.ndarray
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
        return max(1.0, float(self.ts_ms[-1] - self.ts_ms[0]))


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
        self.rollout: Optional[RolloutSnapshot] = None
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
            self.rollout = rollout
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
                "rollout": self.rollout,
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

    num_steps = int(cfg.episode_ms / cfg.dt_ms)
    features = build_feature_sequence(
        num_steps, cfg.dt_ms, cfg.target_vx, cfg.target_vy, cfg.base_freq_hz
    )
    indices = bm.arange(num_steps)

    def loss_fn():
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
        initial_snapshot = build_rollout_snapshot(system, features, cfg, epoch=0)
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
            snapshot = build_rollout_snapshot(system, features, cfg, epoch=epoch + 1)

        if eval_due:
            if snapshot is not None:
                forward = snapshot.forward_end
            else:
                forward = float(bm.as_numpy(evaluate_forward(system, features, cfg)))
            history["forward"].append((epoch + 1, forward))
            if dashboard_state is not None:
                dashboard_state.append_forward(epoch + 1, forward)
            print(
                f"Epoch {epoch+1:4d} | Loss {loss_value:.4f} | "
                f"Forward {forward:.3f} | GradNorm {grad_value:.4f}"
            )

        if snapshot is not None and (snapshot_due or final_due):
            dashboard_state.publish_rollout(snapshot)

    return system, features, history


def evaluate_forward(system: TrainableWalkingSystem, features: bm.Array, cfg: Config):
    num_steps = features.shape[0]
    indices = bm.arange(num_steps)
    system.reset_state()
    outputs = bm.for_loop(system.step_run, (indices, features))
    pos = outputs[0]
    return pos[-1, 0]


# -----------------------------
# Visualization helpers
# -----------------------------


class LiveDashboard:
    def __init__(self, cfg: Config, state: DashboardState):
        self.cfg = cfg
        self.state = state
        self.current_rollout: Optional[RolloutSnapshot] = None
        self.last_history_version = -1
        self.last_rollout_version = -1
        self.last_title = ""
        self.playback_started_at = time.perf_counter()

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
        self.body, = self.ax_robot.plot([], [], "o", color="black", markersize=8)
        self.legs = [
            self.ax_robot.plot([], [], "-", lw=2)[0] for _ in range(self.cfg.n_legs)
        ]
        self.feet = [
            self.ax_robot.plot([], [], "o", color="tab:blue", ms=4)[0]
            for _ in range(self.cfg.n_legs)
        ]
        self.force_line, = self.ax_robot.plot([], [], "-", color="tab:red", lw=2)

        if self.cfg.n_legs > 1:
            self.leg_base_offsets = np.linspace(
                -self.cfg.leg_radius, self.cfg.leg_radius, self.cfg.n_legs
            )
        else:
            self.leg_base_offsets = np.array([0.0])
        self.leg_phase_offsets = np.linspace(
            0.0, 2.0 * np.pi, self.cfg.n_legs, endpoint=False
        )

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
        self.ax_contact.set_title("Leg Contact")
        self.ax_contact.set_xlabel("time (ms)")
        self.ax_contact.set_ylabel("contact")
        self.ax_contact.set_ylim(-0.05, 1.05)
        self.contact_lines = [
            self.ax_contact.plot([], [], lw=1.5, label=f"Leg {i + 1}")[0]
            for i in range(self.cfg.n_legs)
        ]
        self.contact_cursor = self.ax_contact.axvline(
            0.0, color="black", lw=1.0, ls="--", alpha=0.6
        )
        if self.contact_lines:
            self.ax_contact.legend(loc="upper left", ncol=min(2, self.cfg.n_legs))

    def _setup_body_panel(self):
        self.ax_body.set_title("Body Position")
        self.ax_body.set_xlabel("time (ms)")
        self.ax_body.set_ylabel("state")
        self.line_body_x, = self.ax_body.plot([], [], color="tab:purple", label="x")
        self.line_body_y, = self.ax_body.plot([], [], color="tab:brown", label="y")
        self.body_cursor = self.ax_body.axvline(
            0.0, color="black", lw=1.0, ls="--", alpha=0.6
        )
        self.ax_body.legend(loc="upper left")

    def _lookup_index(self, times: np.ndarray, target_time: float) -> int:
        if times.size <= 1:
            return 0
        idx = int(np.searchsorted(times, target_time, side="right") - 1)
        return int(np.clip(idx, 0, len(times) - 1))

    def _playback_time_ms(self) -> float:
        rollout = self.current_rollout
        if rollout is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self.playback_started_at) * 1000.0
        return float(rollout.ts_ms[0]) + (elapsed_ms % rollout.duration_ms)

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
        self.current_rollout = rollout
        self.playback_started_at = time.perf_counter()

        min_x = float(np.min(rollout.pos[:, 0])) if rollout.pos.size else 0.0
        max_x = float(np.max(rollout.pos[:, 0])) if rollout.pos.size else 1.0
        min_y = float(np.min(rollout.pos[:, 1])) if rollout.pos.size else 0.0
        max_y = float(np.max(rollout.pos[:, 1])) if rollout.pos.size else self.cfg.height_target
        self.ax_robot.set_xlim(min_x - 0.35, max_x + 0.35)
        self.ax_robot.set_ylim(min(min_y, 0.0) - 0.15, max(max_y, self.cfg.height_target) + 0.35)

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
        for leg_idx, line in enumerate(self.contact_lines):
            line.set_data(time_axis, rollout.contact[:, leg_idx])
        self.ax_contact.set_xlim(float(time_axis[0]), float(time_axis[-1]))

        self.line_body_x.set_data(time_axis, rollout.pos[:, 0])
        self.line_body_y.set_data(time_axis, rollout.pos[:, 1])
        self.ax_body.set_xlim(float(time_axis[0]), float(time_axis[-1]))
        self.ax_body.relim()
        self.ax_body.autoscale_view()

    def _update_robot(self, rollout: RolloutSnapshot, idx: int, current_time_ms: float):
        p = rollout.pos[idx]
        x, y = float(p[0]), float(p[1])
        self.robot_path.set_data(rollout.pos[: idx + 1, 0], rollout.pos[: idx + 1, 1])
        self.body.set_data([x], [y])

        for leg_idx in range(self.cfg.n_legs):
            contact = float(rollout.contact[idx, leg_idx])
            base_x = x + float(self.leg_base_offsets[leg_idx])
            base_y = y

            phase = 2.0 * np.pi * float(self.cfg.base_freq_hz) * (current_time_ms / 1000.0)
            swing = 0.35 * self.cfg.leg_radius * np.sin(
                phase + float(self.leg_phase_offsets[leg_idx])
            )

            foot_y = (1.0 - contact) * self.cfg.leg_radius
            foot_x = base_x + (1.0 - contact) * swing

            self.legs[leg_idx].set_data([base_x, foot_x], [base_y, foot_y])
            self.feet[leg_idx].set_data([foot_x], [max(0.0, foot_y)])

        force = np.asarray(rollout.force[idx], dtype=float)
        force_scale = 0.35 * self.cfg.leg_radius / max(1e-6, self.cfg.max_force)
        self.force_line.set_data(
            [x, x + force[0] * force_scale],
            [y, y + force[1] * force_scale],
        )

        t_rel = current_time_ms - float(rollout.ts_ms[0])
        self.ax_robot.set_title(
            f"Walking Animation | epoch {rollout.epoch} | t={t_rel:.0f} ms | "
            f"x={x:.3f} y={y:.3f}"
        )

    def _update_playback(self):
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
            f"Spatial Firing | epoch {rollout.epoch} | t={t_rel:.0f} ms"
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

        if (
            view["rollout"] is not None
            and view["rollout_version"] != self.last_rollout_version
        ):
            self._apply_rollout(view["rollout"])
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
    forces: np.ndarray,
    contact: np.ndarray,
    cfg: Config,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or "Walking Robot")

    leg_angles = np.linspace(0, 2 * np.pi, cfg.n_legs, endpoint=False)
    leg_offsets = np.stack(
        [np.cos(leg_angles), np.sin(leg_angles)], axis=1
    ) * cfg.leg_radius

    body, = ax.plot([], [], "o", color="black", markersize=8)
    legs = [ax.plot([], [], "-", lw=2)[0] for _ in range(cfg.n_legs)]
    force_arrow = ax.arrow(0, 0, 0, 0, color="tab:red", width=0.01)

    def init():
        body.set_data([], [])
        for leg in legs:
            leg.set_data([], [])
        return [body] + legs

    def update(i):
        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title or "Walking Robot")

        p = pos[i]
        ax.plot(p[0], p[1], "o", color="black", markersize=8)

        for j in range(cfg.n_legs):
            base = p + leg_offsets[j]
            direction = np.array([1.0, 0.0])
            leg_len = cfg.leg_radius * (0.5 + 0.5 * contact[i, j])
            tip = base + direction * leg_len
            ax.plot([base[0], tip[0]], [base[1], tip[1]], "-", lw=2)

        f = forces[i]
        ax.arrow(
            p[0],
            p[1],
            f[0],
            f[1],
            color="tab:red",
            width=0.01,
            length_includes_head=True,
        )

        ax.set_xlim(np.min(pos[:, 0]) - 0.5, np.max(pos[:, 0]) + 0.5)
        ax.set_ylim(np.min(pos[:, 1]) - 0.5, np.max(pos[:, 1]) + 0.5)
        return []

    ani = FuncAnimation(fig, update, frames=len(pos), init_func=init, interval=50)
    plt.show()
    return ani


def collect_rollout(system: TrainableWalkingSystem, features: bm.Array):
    runner = bp.DSRunner(
        system,
        monitors={
            "E.spike": system.core.E.spike,
            "pos": system.pos,
            "vel": system.vel,
            "force": system.last_force,
            "contact": system.last_contact,
        },
        data_first_axis="T",
    )
    # Reset explicitly without batch sizing to avoid shape mismatches.
    system.reset_state()
    runner.run(inputs=features, reset_state=False)
    return runner


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
    force = np.asarray(runner.mon["force"], dtype=float)
    contact = np.asarray(runner.mon["contact"], dtype=float)
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
        force=force,
        contact=contact,
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
