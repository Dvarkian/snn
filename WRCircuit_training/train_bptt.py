"""
BPTT Training Script for Spatial.py Model

This script implements backpropagation through time (BPTT) training for the Spatial.py
spiking neural network using rate matching as the training objective.
"""

import brainpy as bp
import brainpy.math as bm
from brainpy._src.context import share
import jax
import jax.numpy as jnp
import numpy as np
from src.models.Spatial import Spatial

# Set training mode
bm.set_environment(mode=bm.training_mode, dt=1.0)

print("=" * 60)
print("BPTT Training for Spatial.py Model")
print("=" * 60)

# Network configuration
rho = 5000
dx = 0.5
target_rate = 5.0  # Hz

print(f"\nNetwork Configuration:")
print(f"  - rho: {rho}")
print(f"  - dx: {dx}")
print(f"  - Target rate: {target_rate} Hz")

# Initialize model in training mode
print("\nInitializing model in training mode...")
key = jax.random.PRNGKey(42)
model = Spatial(rho=rho, dx=dx, key=key, training_mode=True)
print(f"✓ Model initialized successfully")
print(f"  - Network size: {model.N_e} excitatory, {model.N_i} inhibitory")

# Check trainable variables
print("\nChecking trainable variables...")
trainable_vars = model.train_vars().unique()
print(f"✓ Trainable variables accessible")
print(f"  - Number of trainable variables: {len(trainable_vars)}")
print(f"  - Variable names:")
for var in trainable_vars:
    print(f"    - {var}")

# Check E2E weight
print("\nChecking E2E weight...")
print(f"  - Trainable E2E weight shape: {model.w_ee_trainable.shape}")
print(f"  - Trainable E2E weight type: {type(model.w_ee_trainable)}")
print(f"  - Trainable E2E weight mean: {bm.mean(model.w_ee_trainable.value)}")
print(f"  - Actual E2E weight mean: {bm.mean(model.E2E.proj.comm._weight.value)}")
print(f"  - Actual E2E weight min: {bm.min(model.E2E.proj.comm._weight.value)}")
print(f"  - Actual E2E weight max: {bm.max(model.E2E.proj.comm._weight.value)}")

# Define loss function using BrainPy's BPTT infrastructure
def loss_fun():
    """
    Compute rate matching loss using BrainPy's BPTT infrastructure.
    """
    # Reset time counter at start of each loss computation
    model.E._time_counter.value = bm.asarray(0.0)
    model.I._time_counter.value = bm.asarray(0.0)
    
    # Use BrainPy's for_loop to run the model
    def step(i):
        # Set shared context variables required by BrainPy's built-in synapses
        share.save('t', i * bm.get_dt(), 'dt', bm.get_dt())
        model()
        return bm.sum(model.E.spike), bm.sum(model.ext.spike)
    
    results = bm.for_loop(step, bm.arange(100))
    e_spikes = bm.sum(jnp.stack([r[0] for r in results]))
    ext_spikes = bm.sum(jnp.stack([r[1] for r in results]))
    
    # Calculate actual firing rate
    duration_ms = 100 * bm.get_dt()
    actual_rate = e_spikes / duration_ms * 1000  # Convert to Hz
    
    # Target rate
    loss = bm.square(actual_rate - target_rate)
    
    return loss, actual_rate, ext_spikes

# Test forward pass
print("\nTesting forward pass...")
try:
    loss, actual_rate, ext_spikes = loss_fun()
    print(f"✓ Forward pass successful")
    print(f"  - Loss: {loss}")
    print(f"  - Actual rate: {actual_rate} Hz")
    print(f"  - External spikes: {ext_spikes}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Training loop
print("\n" + "=" * 60)
print("Starting Training Loop")
print("=" * 60)

# Training parameters
n_epochs = 100
learning_rate = 1e-5

# Initialize optimizer with all recurrent weights
optimizer = bp.optim.Adam(lr=learning_rate, train_vars={
    'w_ee': model.w_ee_trainable,
    'w_ei': model.w_ei_trainable,
    'w_ie': model.w_ie_trainable,
    'w_ii': model.w_ii_trainable
})

print(f"\nTraining parameters:")
print(f"  - Epochs: {n_epochs}")
print(f"  - Learning rate: {learning_rate}")

losses = []
rates = []

for epoch in range(n_epochs):
    # Compute loss and gradients
    def compute_loss():
        return loss_fun()[0]
    
    # Compute gradients for all recurrent weights in one call
    grads = bm.grad(compute_loss, grad_vars=[
        model.w_ee_trainable,
        model.w_ei_trainable,
        model.w_ie_trainable,
        model.w_ii_trainable
    ])()
    
    # Apply gradient clipping to stabilize training
    grad_norm = bm.sqrt(sum([bm.sum(g**2) for g in grads]))
    clip_norm = 1.0
    if grad_norm > clip_norm:
        grads = [g * (clip_norm / grad_norm) for g in grads]
    
    # Update weights
    optimizer.update({
        'w_ee': grads[0],
        'w_ei': grads[1],
        'w_ie': grads[2],
        'w_ii': grads[3]
    })
    
    # Compute loss and monitoring metrics
    loss_val, actual_rate, ext_spikes = loss_fun()
    losses.append(float(loss_val))
    rates.append(float(actual_rate))
    
    print(f"Epoch {epoch+1}/{n_epochs}: Loss = {loss_val:.4f}, Rate = {actual_rate:.2f} Hz, Ext spikes = {ext_spikes}")

print("\n" + "=" * 60)
print("Training Complete")
print("=" * 60)
print(f"\nFinal loss: {losses[-1]:.4f}")
print(f"Final rate: {rates[-1]:.2f} Hz")
print(f"Target rate: {target_rate} Hz")
