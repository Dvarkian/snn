import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple

# Set up environment
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Import existing model components
try:
    from src.neurons import FNSNeuron
    from src.models.Spatial import Spatial
    import src.neurons as _neurons
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from src.neurons import FNSNeuron
    from src.models.Spatial import Spatial
    import src.neurons as _neurons

# Monkey-patch FNSNeuron to FORCE float32 spikes via spk_dtype argument.
# This ensures that during initial setup (NormalMode), the spike Variable is created
# with float32 dtype, making it compatible with BPTT surrogate gradients later.
_orig_fns_init = FNSNeuron.__init__
def _patched_fns_init(self, *args, **kwargs):
    # Force spk_dtype to float32 regardless of other settings
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
        self.target_rate_exc = 0.025
        self.target_rate_inh = 0.020
        self.smooth_tau = 12.0
        self.nu = 25.0
        self.ui_update_interval = 2
        self.dark_mode = True
        self.colors = {'exc': '#FF2E63', 'inh': '#08D9D6', 'bg': '#1A1A1D', 'text': '#EAEAEA'}

class DifferentiableSNN(bp.DynamicalSystem):
    def __init__(self, cfg: Config, spatial_model: Spatial):
        super().__init__()
        self.cfg = cfg
        self.model = spatial_model
        # Use built-in Stochastic Background Population
        self.model.reinit_nu(cfg.nu)
        
        # Trainable recurrent weights
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
            se = self.model.E.spike.value; si = self.model.I.spike.value
            re = bm.mean(se); ri = bm.mean(si)
            self.smooth_e.value = self.alpha * self.smooth_e.value + (1 - self.alpha) * re
            self.smooth_i.value = self.alpha * self.smooth_i.value + (1 - self.alpha) * ri
            fit_loss = bm.square(self.smooth_e.value[0] - target_at_t[0]) + bm.square(self.smooth_i.value[0] - target_at_t[1])
            silence_p = 2.0 * (bm.exp(-40.0 * re) + bm.exp(-40.0 * ri))
            reg = 5.0 * (bm.maximum(0.0, re - 0.15)**2 + bm.maximum(0.0, ri - 0.15)**2)
            return (se, si, self.smooth_e.value[0], self.smooth_i.value[0], fit_loss + silence_p + reg)
        return bm.for_loop(step, (indices, target_rates), progress_bar=False)

class SNNTrainer(bp.DynamicalSystem):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Step 1: Initialize Spatial model (NormalMode) with forced float32 spikes
        self.spatial = Spatial(rho=cfg.rho, dx=cfg.dx, key=42, nu=cfg.nu)
        
        # Step 2: Now enable TrainingMode
        self.spatial.mode = bm.TrainingMode()
        self.spatial.E.mode = bm.TrainingMode()
        self.spatial.I.mode = bm.TrainingMode()
        
        self.diff_snn = DifferentiableSNN(cfg, self.spatial)
        self.opt = bp.optim.Adam(lr=cfg.lr, train_vars=self.diff_snn.trainable_vars)
        
        ts = np.linspace(0, cfg.duration, cfg.steps)
        target_e = cfg.target_rate_exc * (1.0 + 0.6 * np.sin(2 * np.pi * 0.005 * ts))
        target_i = cfg.target_rate_inh * (1.0 + 0.4 * np.cos(2 * np.pi * 0.005 * ts))
        self.target_rates = bm.asarray(np.stack([target_e, target_i], axis=1))
        self.loss_history = []; self.last_rollout = None; self.epoch = 0

    def train_step(self):
        def _loss_fn():
            self.diff_snn.reset_state()
            outs = self.diff_snn.rollout(self.target_rates)
            return bm.mean(outs[-1]), outs
        @bm.jit
        def _update():
            # Unpack (grads, loss, aux)
            grads, loss, outs = bm.grad(_loss_fn, grad_vars=self.diff_snn.trainable_vars, has_aux=True, return_value=True)()
            self.opt.update(grads)
            return loss, outs
        return _update()

    def run_epoch(self):
        self.epoch += 1
        loss, outputs = self.train_step()
        self.loss_history.append(float(loss))
        self.last_rollout = outputs
        return loss

class UI:
    def __init__(self, trainer: SNNTrainer):
        self.trainer = trainer; self.cfg = trainer.cfg
        if self.cfg.dark_mode: plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.tight_layout(pad=4.5)
    def update(self, loss):
        s_e, s_i, r_e, r_i, _ = [np.asarray(x) for x in self.trainer.last_rollout]
        ts = np.linspace(0, self.cfg.duration, self.cfg.steps)
        targets = np.asarray(self.trainer.target_rates)
        ax = self.axs[0, 0]; ax.clear()
        se_sel = np.linspace(0, s_e.shape[1]-1, min(s_e.shape[1], 40), dtype=int)
        for i, idx in enumerate(se_sel):
            t = ts[s_e[:, idx] > 0]
            ax.scatter(t, np.ones_like(t)*i, s=2, color=self.cfg.colors['exc'])
        ax.set_title(f"Raster Plot - Epoch {self.trainer.epoch}")
        ax = self.axs[1, 0]; ax.clear()
        ax.plot(ts, targets[:, 0], '--', alpha=0.15); ax.plot(ts, r_e, color=self.cfg.colors['exc'], lw=2); ax.plot(ts, r_i, color=self.cfg.colors['inh'], lw=2)
        ax.set_title("Target vs Observed Rates")
        ax = self.axs[0, 1]; ax.clear(); ax.plot(self.trainer.loss_history, color='yellow'); ax.set_yscale('log'); ax.set_title("Training Loss")
        ax = self.axs[1, 1]; ax.clear()
        w_vals = [np.mean(np.abs(np.asarray(v))) for v in self.trainer.diff_snn.trainable_vars.values()]
        ax.bar(['EE', 'EI', 'IE', 'II'], w_vals, color=['#FF2E63']*2 + ['#08D9D6']*2)
        ax.set_title("Mean Weights")
        plt.draw(); plt.pause(0.01)

def main():
    cfg = Config(); trainer = SNNTrainer(cfg); ui = UI(trainer)
    print(f"SNN BPTT Active. stochastic Drive: Spatial.nu={cfg.nu}Hz")
    try:
        while True:
            loss = trainer.run_epoch()
            if trainer.epoch % cfg.ui_update_interval == 0:
                print(f"Epoch {trainer.epoch:03d} | Loss {loss:.6f}")
                ui.update(loss)
    except KeyboardInterrupt: pass
    plt.show()

if __name__ == "__main__":
    main()
