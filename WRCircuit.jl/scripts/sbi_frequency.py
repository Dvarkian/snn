#### python
# filepath: /home/brendan/OneDrive/Masters/Code/Vortices/Julia/WRCircuit.jl/scripts/sbi_network_frequency.py

import os
import sys
import numpy as np
import torch

# import torch

# We'll still import jax for the BrainPy-based simulation
import jax
import jax.numpy as jnp
import brainpy as bp
import brainpy.math as bm

import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from tqdm import tqdm

# The main sbi imports
from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils import BoxUniform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.Nonspatial import FNS

# --------------------------------------------------------------------------------
# Globals / Constants
# --------------------------------------------------------------------------------
bp.math.set_dt(0.1)
TARGET_FREQUENCY = 55.0
N_e = 4000
J_e = 0.0008
nu = 10
n_ext = 80


# --------------------------------------------------------------------------------
# Simulation (BrainPy + JAX)
# --------------------------------------------------------------------------------
def calculate_peak_frequency(lfp, dt=bp.share.load("dt")):
    """Compute the frequency spectrum and return the largest peak above 1 Hz."""
    fs = 1000.0 / dt  # convert dt (ms) to sampling frequency (Hz)
    freqs, power = signal.welch(lfp, fs, nperseg=min(8192, len(lfp)))
    mask = freqs > 1.0
    peak_idx = np.argmax(power[mask]) + np.sum(~mask)
    peak_freq = freqs[peak_idx]
    return peak_freq, power, freqs


def simulate_network(params, duration=10.0, discard_time=2.0, seed=None):
    """Simulate the FNS network for `duration` ms and return the peak LFP frequency."""
    # Set up the JAX random key
    # If `seed` is a scalar, we create a PRNGKey; if it’s already a PRNGKey, just use it
    if seed is None:
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    elif (isinstance(seed, jnp.ndarray) and seed.shape == (2,)) or (
        isinstance(seed, np.ndarray) and seed.shape == (2,)
    ):
        rng = seed
    else:
        rng = jax.random.PRNGKey(seed)

    # Unpack the parameters
    omega_ee = params["omega_ee"]
    omega_ie = params["omega_ie"]
    omega_ei = params["omega_ei"]
    omega_ii = params["omega_ii"]
    delta = params["delta"]

    # Initialize the BrainPy FNS model
    model = FNS(
        N_e=N_e,
        omega_ee=omega_ee,
        omega_ie=omega_ie,
        omega_ei=omega_ei,
        omega_ii=omega_ii,
        delta=delta,
        gamma=4,
        nu=nu,
        n_ext=n_ext,
        J_e=J_e,
        method="exp_auto",
        key=rng,
    )
    monitors = ["E.spike"]
    runner = bp.DSRunner(model, monitors=monitors, numpy_mon_after_run=True)
    runner.run(duration=duration)

    # Discard initial transient
    discard_steps = int(discard_time / bp.math.get_dt())
    spike_monitor = runner.mon["E.spike"][:, discard_steps:]
    times = runner.mon["ts"][discard_steps:].view()

    # Compute the LFP from excitatory spikes
    lfp = bp.measure.unitary_LFP(times, spike_monitor.view())
    peak_freq, _, _ = calculate_peak_frequency(lfp)
    return peak_freq


# --------------------------------------------------------------------------------
# SBI Setup
# --------------------------------------------------------------------------------
def build_sbi_prior():
    """
    Build a uniform prior over [omega_ee, omega_ie, omega_ei, omega_ii, delta]
    with the ranges specified in your script:
      omega_ee in [0.1, 0.5]
      omega_ie in [0.2, 0.7]
      omega_ei in [0.2, 0.5]
      omega_ii in [0.3, 0.8]
      delta     in [2.0, 6.0]
    Using sbi's BoxUniform -> a PyTorch distribution.
    """
    low = torch.tensor([0.1, 0.2, 0.2, 0.3, 2.0], dtype=torch.float32)
    high = torch.tensor([0.5, 0.7, 0.5, 0.8, 6.0], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high, device="cpu")
    return prior


def sbi_simulator(params: torch.Tensor) -> np.ndarray:
    """
    The simulator function for sbi. Receives a single param vector of shape (5,),
    returns an observation of shape (1,). We'll treat the peak frequency as (1,).

    1) Convert the param vector (PyTorch) to a Python dict of floats.
    2) Run `simulate_network`.
    3) Return the peak frequency in a 1D array.
    """
    params_dict = {
        "omega_ee": float(params[0].item()),
        "omega_ie": float(params[1].item()),
        "omega_ei": float(params[2].item()),
        "omega_ii": float(params[3].item()),
        "delta": float(params[4].item()),
    }
    # Single-run simulation, returns a float
    freq = simulate_network(params_dict)
    # Return as a 1D array (or torch tensor). sbi expects shape (#observations,).
    return np.array([freq], dtype=np.float32)


def run_sbi(num_simulations=500, num_rounds=3):
    """
    Run multi-round Sequential Neural Posterior Estimation (SNPE) from sbi.
    We'll treat the "observed data" as x = [TARGET_FREQUENCY].
    """
    prior = build_sbi_prior()

    # Initialize the SNPE inference object
    inference = SNPE(prior=prior, density_estimator="maf")

    # The 'observed data' for sbi is a 1D array or tensor. We'll say x = [55.]
    x_o = torch.tensor([TARGET_FREQUENCY], dtype=torch.float32)

    # Run the inference for multiple rounds in one go. This automatically:
    # 1) Proposes parameters from the prior/posterior
    # 2) Simulates using `sbi_simulator`
    # 3) Trains a density estimator
    # 4) Repeats for `num_rounds`
    posterior = inference(
        simulator=sbi_simulator,
        x=x_o,
        num_simulations=num_simulations,
        num_rounds=num_rounds,
    )

    return posterior


# --------------------------------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------------------------------
def plot_posterior_samples(posterior, n_samples=1000):
    """
    Sample from the posterior at x=55, make a pairplot and violin plots.
    Returns a DataFrame with columns [omega_ee, omega_ie, omega_ei, omega_ii, delta].
    """
    with torch.no_grad():
        samples = posterior.sample(
            (n_samples,), x=torch.tensor([TARGET_FREQUENCY], dtype=torch.float32)
        )
    # 'samples' is shape (n_samples, 5) in torch
    df = pd.DataFrame(
        samples.numpy(),
        columns=["omega_ee", "omega_ie", "omega_ei", "omega_ii", "delta"],
    )

    plt.figure(figsize=(15, 15))
    sns.pairplot(df, corner=True)
    plt.suptitle(
        f"Posterior samples for target frequency {TARGET_FREQUENCY} Hz", y=1.02
    )
    plt.tight_layout()
    plt.savefig("posterior_pairplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    for i, param in enumerate(
        ["omega_ee", "omega_ie", "omega_ei", "omega_ii", "delta"]
    ):
        plt.subplot(1, 5, i + 1)
        sns.violinplot(y=df[param])
        plt.title(param)
    plt.tight_layout()
    plt.savefig("posterior_violins.png", dpi=300)
    plt.close()

    return df


def plot_frequency_vs_parameters(posterior, n_samples=100):
    """
    Sample from the posterior, then for each sample simulate the network
    and measure the frequency, finally scatter-plot frequency vs param.
    """
    with torch.no_grad():
        samples = posterior.sample(
            (n_samples,), x=torch.tensor([TARGET_FREQUENCY], dtype=torch.float32)
        )
    samples_np = samples.numpy()

    frequencies = []
    for i in tqdm(range(n_samples), desc="Simulating for frequency-vs-params plot"):
        params_dict = {
            "omega_ee": samples_np[i, 0],
            "omega_ie": samples_np[i, 1],
            "omega_ei": samples_np[i, 2],
            "omega_ii": samples_np[i, 3],
            "delta": samples_np[i, 4],
        }
        freq = simulate_network(params_dict, seed=None)
        frequencies.append(freq)

    df = pd.DataFrame(
        {
            "omega_ee": samples_np[:, 0],
            "omega_ie": samples_np[:, 1],
            "omega_ei": samples_np[:, 2],
            "omega_ii": samples_np[:, 3],
            "delta": samples_np[:, 4],
            "frequency": frequencies,
        }
    )

    plt.figure(figsize=(15, 10))
    for i, param in enumerate(
        ["omega_ee", "omega_ie", "omega_ei", "omega_ii", "delta"]
    ):
        plt.subplot(2, 3, i + 1)
        plt.scatter(df[param], df["frequency"], alpha=0.7)
        plt.xlabel(param)
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Frequency vs {param}")
        plt.axhline(y=TARGET_FREQUENCY, color="r", linestyle="--")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("frequency_vs_parameters.png", dpi=300)
    plt.close()

    return df


def plot_example_simulation(params, duration=500.0, discard_time=100.0):
    """
    Run one example simulation with the given parameter dict and plot the results.
    """
    peak_freq = simulate_network(params, seed=None)
    print(f"Example simulation peak frequency: {peak_freq:.2f} Hz")

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
        key=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )
    monitors = ["E.spike", "I.spike"]
    runner = bp.DSRunner(model, monitors=monitors, numpy_mon_after_run=True)
    runner.run(duration=duration)
    discard_steps = int(discard_time / bp.math.get_dt())

    e_spikes = runner.mon["E.spike"][:, discard_steps:]
    i_spikes = runner.mon["I.spike"][:, discard_steps:]
    times = runner.mon["ts"][discard_steps:].view()
    lfp = bp.measure.unitary_LFP(times, e_spikes.view())

    _, power, freqs = calculate_peak_frequency(lfp)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time = np.arange(0, duration - discard_time, bp.math.get_dt())
    e_indices, e_times = np.where(e_spikes[:100])
    axes[0].scatter(time[e_times], e_indices, s=1, alpha=0.5)
    axes[0].set_ylabel("E Neuron Index")
    axes[0].set_title("Spike Raster Plot")

    i_indices, i_times = np.where(i_spikes)
    axes[1].scatter(time[i_times], i_indices, s=1, alpha=0.5)
    axes[1].set_ylabel("I Neuron Index")

    axes[2].plot(time, lfp)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("LFP")
    plt.tight_layout()
    plt.savefig("example_simulation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, power)
    plt.axvline(x=peak_freq, linestyle="--", label=f"Peak: {peak_freq:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Power Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("power_spectrum.png", dpi=300)
    plt.close()

    return peak_freq


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    print(
        f"Running SBI (SNPE) to find parameters for target frequency: {TARGET_FREQUENCY} Hz."
    )

    # 1) Run multi-round inference with sbi
    posterior = run_sbi(num_simulations=500, num_rounds=3)

    # 2) Sample from posterior, do pairplot, etc.
    posterior_df = plot_posterior_samples(posterior, n_samples=1000)

    # 3) Plot frequency vs parameters by simulating from posterior samples
    freq_params_df = plot_frequency_vs_parameters(posterior, n_samples=100)

    # 4) Pick the sample whose frequency is closest to the target
    freq_array = freq_params_df["frequency"].values
    closest_idx = np.argmin(np.abs(freq_array - TARGET_FREQUENCY))
    best_params = {
        "omega_ee": freq_params_df["omega_ee"].iloc[closest_idx],
        "omega_ie": freq_params_df["omega_ie"].iloc[closest_idx],
        "omega_ei": freq_params_df["omega_ei"].iloc[closest_idx],
        "omega_ii": freq_params_df["omega_ii"].iloc[closest_idx],
        "delta": freq_params_df["delta"].iloc[closest_idx],
    }
    print("\nBest parameters found (closest to 55 Hz):")
    for param, value in best_params.items():
        print(f"  {param}: {value:.4f}")

    # 5) Plot an example simulation with these parameters
    print("\nRunning example simulation with best parameters...")
    peak_freq = plot_example_simulation(best_params)
    print(f"Peak frequency: {peak_freq:.2f} Hz")

    print("\nAll plots have been saved to the current directory.")


if __name__ == "__main__":
    main()
