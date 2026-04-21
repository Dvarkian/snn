"""
Modify original trainable_activity_bptt.py to print gradient norms
"""
import os
import sys
sys.path.insert(0, os.getcwd())

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from src.neurons import FNSNeuron
from src.models.Spatial import Spatial
import src.neurons as _neurons

# Monkey-patch FNSNeuron to FORCE float32 spikes via spk_dtype argument.
_orig_fns_init = FNSNeuron.__init__
def _patched_fns_init(self, *args, **kwargs):
    kwargs['spk_dtype'] = jnp.float32
    return _orig_fns_init(self, *args, **kwargs)
FNSNeuron.__init__ = _patched_fns_init

# Monkey-patch missing stop_gradient dependency
_neurons.stop_gradient = jax.lax.stop_gradient

class Config:
    def __init__(self):
        self.rho = 6000
        self.dx = 0.5
        self.dt = 0.5
        self.duration = 400.0
        self.steps = int(self.duration / self.dt)
        self.lr = 2e-3
        self.target_rate = 0.005  # 10 Hz
        self.smooth_tau = 12.0
        self.nu = 10.0

cfg = Config()

class DifferentiableSNN(bp.DynamicalSystem):
    def __init__(self, cfg, spatial_model):
        super().__init__()
        self.cfg = cfg
        self.model = spatial_model
        self.model.reinit_nu(cfg.nu)
        
        self.w_ee = bm.TrainVar(self.model.E2E.proj.comm.weight)
        self.w_ei = bm.TrainVar(self.model.E2I.proj.comm.weight)
        self.w_ie = bm.TrainVar(self.model.I2E.proj.comm.weight)
        self.w_ii = bm.TrainVar(self.model.I2I.proj.comm.weight)
        self.trainable_vars = {'EE': self.w_ee, 'EI': self.w_ei, 'IE': self.w_ie, 'II': self.w_ii}
        
        self.alpha = float(np.exp(-cfg.dt / cfg.smooth_tau))
        self.smooth_e = bm.Variable(bm.zeros(1))
        self.smooth_i = bm.Variable(bm.zeros(1))
    
    def reset_state(self, batch_size=None):
        self.model.reset_state(batch_size)
        self.smooth_e.value = bm.zeros(1)
        self.smooth_i.value = bm.zeros(1)
    
    def rollout(self, target_rates):
        indices = bm.arange(self.cfg.steps)
        def step(i, target_at_t):
            bp.share.save(i=i, t=i * self.cfg.dt)
            self.model.update()
            se = self.model.E.spike.value
            si = self.model.I.spike.value
            re = bm.mean(se)
            ri = bm.mean(si)
            self.smooth_e.value = self.alpha * self.smooth_e.value + (1 - self.alpha) * re
            self.smooth_i.value = self.alpha * self.smooth_i.value + (1 - self.alpha) * ri
            fit_loss = bm.square(self.smooth_e.value[0] - target_at_t[0]) + bm.square(self.smooth_i.value[0] - target_at_t[1])
            silence_p = 1.0 * (bm.exp(-40.0 * re) + bm.exp(-40.0 * ri))
            return (se, si, self.smooth_e.value[0], self.smooth_i.value[0], fit_loss + silence_p)
        return bm.for_loop(step, (indices, target_rates), progress_bar=False)

class SNNTrainer(bp.DynamicalSystem):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spatial = Spatial(rho=cfg.rho, dx=cfg.dx, key=42, nu=cfg.nu)
        
        self.spatial.mode = bm.TrainingMode()
        self.spatial.E.mode = bm.TrainingMode()
        self.spatial.I.mode = bm.TrainingMode()
        
        self.diff_snn = DifferentiableSNN(cfg, self.spatial)
        self.opt = bp.optim.Adam(lr=cfg.lr, train_vars=self.diff_snn.trainable_vars)

        ts = np.linspace(0, cfg.duration, cfg.steps)
        target_constant = np.full_like(ts, cfg.target_rate)
        self.target_rates = bm.asarray(np.stack([target_constant, target_constant], axis=1))
        self.loss_history = []
        self.epoch = 0

    def train_step(self):
        def _loss_fn():
            self.diff_snn.reset_state()
            outs = self.diff_snn.rollout(self.target_rates)
            return bm.mean(outs[-1]), outs
        
        @bm.jit
        def _update():
            grads, loss, outs = bm.grad(_loss_fn, grad_vars=self.diff_snn.trainable_vars, has_aux=True, return_value=True)()
            self.opt.update(grads)
            return grads, loss, outs
        
        grads, loss, outs = _update()
        
        # Print gradient norms (outside of @bm.jit)
        grad_norms = {k: float(jnp.sqrt(jnp.sum(g**2))) for k, g in grads.items()}
        non_zero = sum(1 for v in grad_norms.values() if v > 1e-6)
        print(f"  Gradient norms: {grad_norms}")
        print(f"  Non-zero gradients: {non_zero}/{len(grad_norms)}")
        
        return loss

    def run_epoch(self):
        self.epoch += 1
        print(f"Epoch {self.epoch:03d} - Running training step...")
        loss, outputs = self.train_step()
        self.loss_history.append(float(loss))
        print(f"Epoch {self.epoch:03d} | Loss {loss:.6f}")
        return loss

def main():
    cfg = Config()
    trainer = SNNTrainer(cfg)
    print(f"SNN BPTT Active. Target: 10 Hz, Stochastic Drive: Spatial.nu={cfg.nu}Hz")
    print("=" * 60)
    
    for i in range(5):
        trainer.run_epoch()
    
    print("=" * 60)
    print("Training completed!")

if __name__ == "__main__":
    main()
