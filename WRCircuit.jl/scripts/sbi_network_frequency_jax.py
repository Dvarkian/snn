#### python
# filepath: /home/brendan/OneDrive/Masters/Code/Vortices/Julia/WRCircuit.jl/scripts/sbi_network_frequency_jax.py
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jr

import brainpy as bp
import brainpy.math as bm
from scipy import signal
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from sbijax import NLE
from sbijax.nn import make_maf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.Nonspatial import FNS

# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
bp.math.set_dt(0.05)
TARGET_FREQUENCY = 55.0
DURATION = 100.0
DISCARD = 20.0
N_e = 4000
J_e = 0.0008
nu = 10
n_ext = 80


# ------------------------------------------------------------------------------
# SIMULATION UTIL
# ------------------------------------------------------------------------------
def calculate_peak_frequency(lfp, dt=bp.share.load("dt")):
    fs = 1000.0 / dt
    freqs, power = signal.welch(lfp, fs, nperseg=min(8192, len(lfp)))
    mask = freqs > 1.0
    peak_idx = np.argmax(power[mask]) + np.sum(~mask)
    return freqs[peak_idx]


def simulate_network(params, seed=None):
    """Run one FNS simulation and return the peak frequency."""
    if seed is None:
        seed = np.random.randint(0, 2**32)
    if isinstance(seed, jnp.ndarray) and seed.shape == (2,):
        rng = seed
    else:
        rng = jax.random.PRNGKey(seed)

    model = FNS(
        N_e=N_e,
        omega_ee=params["omega_ee"],
        omega_ie=params["omega_ie"],
        omega_ei=params["omega_ei"],
        omega_ii=params["omega_ii"],
        delta=params["delta"],
        gamma=4,
        nu=nu,
        n_ext=n_ext,
        J_e=J_e,
        method="exp_auto",
        key=rng,
    )
    runner = bp.DSRunner(
        model, monitors=["E.input", "I.input"], numpy_mon_after_run=True
    )
    runner.run(duration=DURATION)
    discard_steps = int(DISCARD / bp.math.get_dt())

    e_input = runner.mon["E.input"]
    i_input = runner.mon["I.input"]
    lfp = jnp.sum(e_input, axis=1) - jnp.sum(i_input, axis=1)
    lfp = lfp[discard_steps:]

    return calculate_peak_frequency(lfp)


# ------------------------------------------------------------------------------
# PRIOR & SIMULATOR (SBI-JAX COMPATIBLE)
# ------------------------------------------------------------------------------
def prior_fn():
    """
    Single key "theta": a 5D vector within the given Uniform bounds.
    """
    # theta = [omega_ee, omega_ie, omega_ei, omega_ii, delta]
    lower = jnp.array([0.1, 0.2, 0.2, 0.3, 2.0])
    upper = jnp.array([0.5, 0.7, 0.5, 0.8, 6.0])
    return tfd.JointDistributionNamed(
        dict(theta=tfd.Uniform(lower, upper)), batch_ndims=0
    )


def simulator_fn(seed, theta):
    """
    Called by SBI-JAX. `theta` is a dict with a single key "theta".
    This key has shape either (5,) for a single sample or (batch_size, 5) for batch mode.
    We return a 2D array with shape (batch_size, data_dim).
    """
    theta_array = theta["theta"]

    # Single-sample case (theta_array.shape == (5,))
    if theta_array.ndim == 1:
        params = {
            "omega_ee": float(theta_array[0]),
            "omega_ie": float(theta_array[1]),
            "omega_ei": float(theta_array[2]),
            "omega_ii": float(theta_array[3]),
            "delta": float(theta_array[4]),
        }
        freq = simulate_network(params, seed=seed)
        return jnp.array([[freq]], dtype=jnp.float32)

    # Batched case (theta_array.shape == (batch_size, 5))
    batch_size = theta_array.shape[0]

    # If the given seed is a JAX key, split it. Otherwise, generate new keys.
    if isinstance(seed, jnp.ndarray) and seed.shape == (2,):
        subkeys = jax.random.split(seed, batch_size)
    else:
        subkeys = [
            jax.random.PRNGKey(np.random.randint(0, 2**32)) for _ in range(batch_size)
        ]

    results = []
    for i in range(batch_size):
        params = {
            "omega_ee": float(theta_array[i, 0]),
            "omega_ie": float(theta_array[i, 1]),
            "omega_ei": float(theta_array[i, 2]),
            "omega_ii": float(theta_array[i, 3]),
            "delta": float(theta_array[i, 4]),
        }
        freq = simulate_network(params, seed=subkeys[i])
        results.append(freq)

    return jnp.array(results, dtype=jnp.float32)[:, None]


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # Build an NLE model with the provided prior & simulator
    fns = (prior_fn, simulator_fn)
    model = NLE(fns, make_maf(1))

    # Simulate some data from the prior
    data, _ = model.simulate_data(jr.PRNGKey(1), n_simulations=3)

    # Fit the neural likelihood
    params, info = model.fit(jr.PRNGKey(2), data=data, n_epochs=300)

    # Suppose we have an observed frequency
    y_observed = jnp.array([TARGET_FREQUENCY])[None]  # shape (1,)
    posterior_samples, _ = model.sample_posterior(
        jr.PRNGKey(3), params, y_observed, n_samples=1000
    )
    print("Posterior samples:", posterior_samples)


if __name__ == "__main__":
    main()
