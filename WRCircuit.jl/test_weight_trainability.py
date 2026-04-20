#!/usr/bin/env python3
"""
Test script to verify weight trainability in Spatial.py model.

This script tests:
1. Whether model.E2E.proj.comm.weight is accessible
2. Whether weights are registered as trainable BrainPy variables
3. Whether gradient computation works on E2E weights
4. Whether the model works in training mode
"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

import brainpy as bp
import brainpy.math as bm

# Direct import from file
import importlib.util
spatial_path = os.path.join(src_dir, 'models', 'Spatial.py')
spec = importlib.util.spec_from_file_location("Spatial", spatial_path)
Spatial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Spatial)

print("=" * 60)
print("TEST 1: Model Initialization in Training Mode")
print("=" * 60)

try:
    # Set training mode
    bm.set_environment(mode=bm.training_mode, dt=1.0)
    print("✓ Training mode set successfully")
except Exception as e:
    print(f"✗ Failed to set training mode: {e}")
    sys.exit(1)

try:
    # Create model with reduced scale
    model = Spatial(rho=5000, dx=0.5)
    print("✓ Model initialized successfully")
    print(f"  - Network size: {model.N_e} excitatory, {model.N_i} inhibitory")
except Exception as e:
    print(f"✗ Failed to initialize model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 2: Check Trainable Variables")
print("=" * 60)

try:
    trainable_vars = model.train_vars().unique()
    print(f"✓ Trainable variables accessible")
    print(f"  - Number of trainable variables: {len(trainable_vars)}")
    print(f"  - Variable names:")
    for var in trainable_vars:
        print(f"    - {var}")
except Exception as e:
    print(f"✗ Failed to get trainable variables: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 3: Check E2E Weight Accessibility")
print("=" * 60)

try:
    e2e_weight = model.E2E.proj.comm.weight
    print(f"✓ E2E weight accessible")
    print(f"  - Shape: {e2e_weight.shape}")
    print(f"  - Type: {type(e2e_weight)}")
    print(f"  - Dtype: {e2e_weight.dtype}")
except Exception as e:
    print(f"✗ Failed to access E2E weight: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 4: Check if E2E Weight is in Trainable Variables")
print("=" * 60)

e2e_in_trainable = False
for var in trainable_vars:
    if 'E2E' in str(var) or 'comm' in str(var) or 'weight' in str(var):
        print(f"  Found trainable variable: {var}")
        e2e_in_trainable = True

if e2e_in_trainable:
    print("✓ E2E weight appears to be in trainable variables")
else:
    print("✗ E2E weight NOT found in trainable variables")
    print("  This suggests weights may not be automatically trainable")

print("\n" + "=" * 60)
print("TEST 5: Test Forward Pass in Training Mode")
print("=" * 60)

try:
    runner = bp.DSRunner(model, monitors={'E.spike': model.E.spike, 'I.spike': model.I.spike})
    outputs = runner.run(duration=100)
    print("✓ Forward pass successful in training mode")
    print(f"  - Output shape: {outputs.shape}")
    print(f"  - E spikes shape: {runner.mon['E.spike'].shape}")
    print(f"  - I spikes shape: {runner.mon['I.spike'].shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 6: Test Gradient Computation on E2E Weights")
print("=" * 60)

try:
    def dummy_loss():
        return bm.sum(model.E2E.proj.comm.weight)
    
    grad_fn = bm.grad(dummy_loss, grad_vars=model.E2E.proj.comm.weight)
    grads = grad_fn()
    print("✓ Gradient computation successful")
    print(f"  - Gradient shape: {grads.shape}")
    print(f"  - Gradient type: {type(grads)}")
    print(f"  - Gradient dtype: {grads.dtype}")
except Exception as e:
    print(f"✗ Gradient computation failed: {e}")
    print("  This suggests E2E weights may not be differentiable/trainable")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 7: Test Gradient Computation via Trainable Vars")
print("=" * 60)

try:
    def loss_with_trainable_vars():
        return bm.sum(trainable_vars[0])  # Use first trainable variable
    
    grad_fn = bm.grad(loss_with_trainable_vars, grad_vars=trainable_vars)
    grads = grad_fn()
    print("✓ Gradient computation via trainable_vars successful")
    print(f"  - Number of gradients: {len(grads)}")
except Exception as e:
    print(f"✗ Gradient computation via trainable_vars failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 8: Check if Weights Can Be Modified Directly")
print("=" * 60)

try:
    original_weight = model.E2E.proj.comm.weight.copy()
    # Try to modify weight
    model.E2E.proj.comm.weight = model.E2E.proj.comm.weight * 1.1
    modified_weight = model.E2E.proj.comm.weight
    print("✓ Weights can be modified directly")
    print(f"  - Original sum: {bm.sum(original_weight)}")
    print(f"  - Modified sum: {bm.sum(modified_weight)}")
    # Restore original
    model.E2E.proj.comm.weight = original_weight
except Exception as e:
    print(f"✗ Direct weight modification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Key findings:")
print("1. Model initializes in training mode: ✓")
print("2. Trainable variables accessible: ✓")
print("3. E2E weight accessible: ✓")
print("4. E2E weight in trainable vars:", "✓" if e2e_in_trainable else "✗")
print("5. Forward pass works: ✓")
print("6. Gradient on E2E weight works: (see above)")
print("7. Gradient via trainable_vars works: (see above)")
print("8. Direct weight modification works: (see above)")

print("\nRecommendation:")
if e2e_in_trainable:
    print("E2E weights appear to be trainable. Proceed with BPTT implementation.")
else:
    print("E2E weights may not be automatically trainable. May need explicit registration.")
    print("Consider using manual gradient approach or modifying Spatial.py.")

print("\n" + "=" * 60)
