"""
Test script to verify 10 Hz target training works correctly
"""
import os
import sys
import numpy as np
import brainpy.math as bm

# Set up environment
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Import training script components
from trainable_activity_bptt import Config, SNNTrainer

def test_training():
    print("=" * 60)
    print("Testing 10 Hz Target Training")
    print("=" * 60)
    
    cfg = Config()
    print(f"\nConfiguration:")
    print(f"  - Target rate: {cfg.target_rate} (10 Hz)")
    print(f"  - Duration: {cfg.duration} ms")
    print(f"  - Steps: {cfg.steps}")
    print(f"  - Learning rate: {cfg.lr}")
    print(f"  - Stochastic drive: {cfg.nu} Hz")
    
    trainer = SNNTrainer(cfg)
    
    print("\nRunning 20 epochs to test convergence...")
    losses = []
    for epoch in range(20):
        loss = trainer.run_epoch()
        losses.append(loss)
        # Get actual firing rates from last rollout
        if trainer.last_rollout is not None:
            _, _, r_e, r_i, _ = trainer.last_rollout
            r_e_avg = float(bm.mean(r_e))
            r_i_avg = float(bm.mean(r_i))
            print(f"Epoch {epoch+1:03d} | Loss {loss:.6f} | E-rate: {r_e_avg:.6f} | I-rate: {r_i_avg:.6f}")
        else:
            print(f"Epoch {epoch+1:03d} | Loss {loss:.6f}")
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss change: {losses[-1] - losses[0]:.6f}")
    
    # Check if loss decreased
    if losses[-1] < losses[0]:
        print("\n✓ Loss DECREASED - Training is working!")
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"  Improvement: {improvement:.2f}%")
        return True
    else:
        print("\n✗ Loss did NOT decrease - Training may not be working")
        return False

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)
