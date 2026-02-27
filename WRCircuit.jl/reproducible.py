#!/usr/bin/env python
"""
Minimal script to verify that Spatial model simulations are deterministic.
"""
import jax
import numpy as np
import brainpy as bp
from src.models.Spatial import Spatial

# Set up a fixed seed
seed = 42
key = jax.random.PRNGKey(seed)

# Create two identical models with the same seed
print("Creating model 1...")
model1 = Spatial(key=key, rho=10000, dx=0.5)  # Smaller network for speed

# Run both models for a short duration
duration = 50.0  # ms, short for quick testing
print(f"\nRunning both models for {duration} ms...")

runner1 = bp.DSRunner(model1, monitors=["E.spike", "I.spike"])
runner1.run(duration)

print("Creating model 2...")
model2 = Spatial(key=key, rho=10000, dx=0.5)

runner2 = bp.DSRunner(model2, monitors=["E.spike", "I.spike"])
runner2.run(duration)

# Compare outputs
e_spikes_1 = np.array(runner1.mon["E.spike"])
i_spikes_1 = np.array(runner1.mon["I.spike"])
e_spikes_2 = np.array(runner2.mon["E.spike"])
i_spikes_2 = np.array(runner2.mon["I.spike"])

# Check if outputs are identical
e_match = np.array_equal(e_spikes_1, e_spikes_2)
i_match = np.array_equal(i_spikes_1, i_spikes_2)

# Check that outputs are non-zero
assert np.any(e_spikes_1), "No excitatory spikes recorded in model 1!"
assert np.any(i_spikes_1), "No inhibitory spikes recorded in model 1!"

print("\n" + "=" * 50)
print("REPRODUCIBILITY TEST RESULTS")
print("=" * 50)
print(f"Excitatory spikes match: {e_match}")
print(f"Inhibitory spikes match: {i_match}")
print(f"\nE spike shape: {e_spikes_1.shape}")
print(f"I spike shape: {i_spikes_1.shape}")

if e_match and i_match:
    print("\n✓ SUCCESS: Simulations are deterministic!")
else:
    print("\n✗ FAILURE: Simulations differ!")
    print(f"  E spikes differ at {np.sum(e_spikes_1 != e_spikes_2)} positions")
    print(f"  I spikes differ at {np.sum(i_spikes_1 != i_spikes_2)} positions")
