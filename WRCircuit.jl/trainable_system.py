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
import math
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
# Prefer CPU by default to avoid CUDA/cuDNN init failures on systems without GPU setup.
# Set SNN_USE_GPU=1 to allow GPU usage.
if os.environ.get("SNN_USE_GPU", "0") != "1":
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import brainpy as bp
import brainpy.math as bm

spatial_mod = importlib.import_module("src.models.Spatial")
from src.models.Spatial import Spatial


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
        self.reset_state()

    spatial_mod.Spatial._orig_reinit_weights = spatial_mod.Spatial.reinit_weights
    spatial_mod.Spatial.reinit_weights = _safe_reinit_weights
    spatial_mod.Spatial._safe_reinit_patched = True


_patch_spatial_reinit_weights()


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
    n_legs: int = 4
    mass: float = 1.0
    drag: float = 0.2
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

    # Loss weights
    w_forward: float = 1.0
    w_lateral: float = 0.3
    w_energy: float = 0.02
    w_spike: float = 0.01
    w_smooth: float = 0.02
    w_contact: float = 0.02
    desired_contact: float = 0.5

    # Visualization
    raster_neurons: int = 200
    heatmap_window_ms: float = 10.0
    heatmap_frame_ms: float = 2.0


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
        self.pos.value = bm.zeros_like(self.pos.value)
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
        acc = total_force / self.cfg.mass - self.cfg.drag * self.vel.value
        self.vel.value = self.vel.value + acc * dt
        self.pos.value = self.pos.value + self.vel.value * dt

        self.last_force.value = total_force
        self.last_contact.value = contact

        mean_rate = bm.mean(self.rate.value)
        return self.pos.value, self.vel.value, total_force, mean_rate, contact


# -----------------------------
# Training
# -----------------------------


def compute_loss(outputs: Tuple[bm.Array, ...], cfg: Config) -> bm.Array:
    pos, vel, force, rate, contact = outputs

    forward = pos[-1, 0]
    lateral = bm.mean(pos[:, 1] ** 2)
    energy = bm.mean(bm.sum(force**2, axis=-1))
    spike_cost = bm.mean(rate)
    smooth = bm.mean(bm.sum((force[1:] - force[:-1]) ** 2, axis=-1))
    contact_mean = bm.mean(contact, axis=1)
    contact_cost = bm.mean((contact_mean - cfg.desired_contact) ** 2)

    loss = (
        -cfg.w_forward * forward
        + cfg.w_lateral * lateral
        + cfg.w_energy * energy
        + cfg.w_spike * spike_cost
        + cfg.w_smooth * smooth
        + cfg.w_contact * contact_cost
    )
    return loss


def train_system(cfg: Config):
    bm.set_dt(cfg.dt_ms)

    with bm.training_environment():
        system = TrainableWalkingSystem(cfg)

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

    for epoch in range(cfg.train_epochs):
        grads, loss = f_grad()
        opt.update(grads)
        gnorm = global_norm(grads)

        history["loss"].append(float(bm.as_numpy(loss)))
        history["grad_norm"].append(float(bm.as_numpy(gnorm)))
        history["timestamp"].append(time.time())

        if (epoch + 1) % cfg.eval_every == 0 or epoch == 0:
            forward = float(bm.as_numpy(evaluate_forward(system, features, cfg)))
            history["forward"].append((epoch + 1, forward))
            print(
                f"Epoch {epoch+1:4d} | Loss {history['loss'][-1]:.4f} | "
                f"Forward {forward:.3f} | GradNorm {history['grad_norm'][-1]:.4f}"
            )

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


def plot_training_history(history: Dict[str, List[float]]):
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

    plt.tight_layout()
    plt.show()


def plot_surrogate_gradient(spk_fun):
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
    plt.tight_layout()
    plt.show()


def plot_spike_raster(spikes: np.ndarray, sample: int = 200):
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
    plt.title("Spike Raster (sampled)")
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


def animate_spike_heatmap(histograms: np.ndarray, frame_times: np.ndarray, domain):
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
    title = ax.set_title("")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(i):
        im.set_data(histograms[i].T)
        title.set_text(f"t = {frame_times[i]:.1f} ms")
        return (im, title)

    ani = FuncAnimation(fig, update, frames=len(frame_times), interval=50, blit=False)
    plt.show()
    return ani


def animate_robot(
    pos: np.ndarray,
    forces: np.ndarray,
    contact: np.ndarray,
    cfg: Config,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Walking Robot")

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
        ax.set_title("Walking Robot")

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
    runner.run(inputs=features, reset_state=True)
    return runner


# -----------------------------
# Main
# -----------------------------


def main():
    cfg = Config()

    system, features, history = train_system(cfg)

    # Training curves
    plot_training_history(history)

    # Surrogate gradient visualization
    plot_surrogate_gradient(system.core.E.spk_fun)

    # Rollout for visualization
    runner = collect_rollout(system, features)
    spikes = np.asarray(runner.mon["E.spike"])
    pos = np.asarray(runner.mon["pos"])
    force = np.asarray(runner.mon["force"])
    contact = np.asarray(runner.mon["contact"])
    ts = np.asarray(runner.mon["ts"])

    # Spike raster
    plot_spike_raster(spikes, sample=cfg.raster_neurons)

    # Spatial heatmap
    histograms, frame_times = prepare_spike_histograms(
        positions=np.asarray(system.core.E.positions),
        domain=np.asarray(system.core.E.embedding.domain, dtype=float),
        grid_size=tuple(system.core.E.size),
        spikes=spikes,
        ts=ts,
        window_ms=cfg.heatmap_window_ms,
        frame_step_ms=cfg.heatmap_frame_ms,
    )
    animate_spike_heatmap(
        histograms, frame_times, np.asarray(system.core.E.embedding.domain, dtype=float)
    )

    # Robot animation
    animate_robot(pos, force, contact, cfg)


if __name__ == "__main__":
    main()
