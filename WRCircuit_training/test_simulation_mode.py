"""
Test script to check if the network produces spikes in simulation mode.
"""

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np
from src.models.Spatial import Spatial

# Set simulation mode
bm.set_environment(mode=bm.training_mode, dt=1.0)

print("=" * 60)
print("Testing Spatial Model in Simulation Mode")
print("=" * 60)

# Network configuration
rho = 5000
dx = 0.5

print(f"\nNetwork Configuration:")
print(f"  - rho: {rho}")
print(f"  - dx: {dx}")

# Initialize model in simulation mode (training_mode=False)
print("\nInitializing model in simulation mode...")
key = jax.random.PRNGKey(42)
model = Spatial(rho=rho, dx=dx, key=key, training_mode=False)
print(f"✓ Model initialized successfully")
print(f"  - Network size: {model.N_e} excitatory, {model.N_i} inhibitory")

# Run simulation using DSRunner
print("\nRunning simulation...")
runner = bp.DSRunner(model, monitors={'E.spike': model.E.spike, 'E.V': model.E.V})
runner.run(duration=1000)

# Check spikes
spikes = runner.mon['E.spike']
total_spikes = bm.sum(spikes)
duration_ms = spikes.shape[0] * bm.get_dt()
actual_rate = total_spikes / duration_ms * 1000  # Convert to Hz

print(f"\nSimulation Results:")
print(f"  - Duration: {duration_ms} ms")
print(f"  - Total spikes: {total_spikes}")
print(f"  - Firing rate: {actual_rate} Hz")
print(f"  - Spike array shape: {spikes.shape}")
print(f"  - Number of neurons that spiked: {bm.sum(bm.any(spikes, axis=0))}")

# Check membrane potential
V = runner.mon['E.V']
print(f"\nMembrane Potential Statistics:")
print(f"  - Mean V: {bm.mean(V)}")
print(f"  - Max V: {bm.max(V)}")
print(f"  - Min V: {bm.min(V)}")
print(f"  - Std V: {bm.std(V)}")
