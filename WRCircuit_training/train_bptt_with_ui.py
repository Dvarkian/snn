"""
BPTT Training Script for Spatial.py Model with Real-time Visualization

This script implements backpropagation through time (BPTT) training for the Spatial.py
spiking neural network using rate matching as the training objective, with real-time
visualization of training progress and network activity. Training runs in a background thread
while visualization runs in the main thread (required by matplotlib).
"""

import threading
import queue
import warnings
import brainpy as bp
import brainpy.math as bm
from brainpy._src.context import share
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from src.models.Spatial import Spatial
import src.neurons as _neurons

# Suppress matplotlib GUI warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Monkey-patch FNSNeuron to FORCE float32 spikes via spk_dtype argument.
# This ensures that during initial setup, the spike Variable is created
# with float32 dtype, making it compatible with BPTT surrogate gradients.
from src.neurons import FNSNeuron
_orig_fns_init = FNSNeuron.__init__
def _patched_fns_init(self, *args, **kwargs):
    kwargs['spk_dtype'] = jnp.float32
    return _orig_fns_init(self, *args, **kwargs)
FNSNeuron.__init__ = _patched_fns_init

# Monkey-patch missing stop_gradient dependency
_neurons.stop_gradient = jax.lax.stop_gradient

def training_worker(data_queue, stop_event, rho=10000, dx=1.0, target_rate=50.0, n_epochs=float('inf'), learning_rate=1e-5, window_size=1000, sim_duration=100, clip_norm=1.0):
    """
    Background thread that runs the training loop and sends results to the main thread.
    """
    # Set training mode
    bm.set_environment(mode=bm.training_mode, dt=1.0)

    # Initialize model normally
    key = jax.random.PRNGKey(42)
    model = Spatial(rho=rho, dx=dx, key=key)

    # Explicitly wrap weights as trainable variables
    w_ee = bm.TrainVar(model.E2E.proj.comm.weight)
    w_ei = bm.TrainVar(model.E2I.proj.comm.weight)
    w_ie = bm.TrainVar(model.I2E.proj.comm.weight)
    w_ii = bm.TrainVar(model.I2I.proj.comm.weight)

    # Initialize optimizer with all recurrent weights
    optimizer = bp.optim.Adam(lr=learning_rate, train_vars={
        'w_ee': w_ee,
        'w_ei': w_ei,
        'w_ie': w_ie,
        'w_ii': w_ii
    })
    
    # Define loss function (traced by JAX for gradient computation)
    def loss_fun():
        model.E._time_counter.value = bm.asarray(0.0)
        model.I._time_counter.value = bm.asarray(0.0)

        def step(i):
            share.save('t', i * bm.get_dt(), 'dt', bm.get_dt())
            model.update()
            return bm.sum(model.E.spike.value), bm.sum(model.ext.spike.value)

        results = bm.for_loop(step, bm.arange(sim_duration))
        e_spikes = bm.sum(jnp.stack([r[0] for r in results]))
        ext_spikes = bm.sum(jnp.stack([r[1] for r in results]))

        duration_ms = sim_duration * bm.get_dt()
        actual_rate = e_spikes / duration_ms * 1000
        loss = bm.square(actual_rate - target_rate)

        return loss, actual_rate, ext_spikes
    
    # Training loop
    epoch = 0
    try:
        while epoch < n_epochs and not stop_event.is_set():
            def compute_loss():
                return loss_fun()[0]

            # Compute gradients
            grads = bm.grad(compute_loss, grad_vars=[w_ee, w_ei, w_ie, w_ii])()

            # Gradient clipping
            grad_norm = jnp.sqrt(jnp.sum([jnp.sum(g ** 2) for g in grads]))
            if grad_norm > clip_norm:
                grads = [g * (clip_norm / grad_norm) for g in grads]

            # Update weights
            optimizer.update({
                'w_ee': grads[0],
                'w_ei': grads[1],
                'w_ie': grads[2],
                'w_ii': grads[3]
            })

            # Compute metrics
            loss_val, actual_rate, ext_spikes = loss_fun()

            # Send data to main thread for UI updates
            data_queue.put({
                'epoch': epoch + 1,
                'loss': float(loss_val),
                'rate': float(actual_rate),
                'ext_spikes': float(ext_spikes),
                'grad_norm': float(grad_norm),
                'total_epochs': n_epochs if n_epochs != float('inf') else 'inf'
            })

            print(f"Epoch {epoch+1}: Loss = {loss_val:.4f}, Rate = {actual_rate:.2f} Hz")

            epoch += 1

        # Signal completion
        data_queue.put({'done': True})
    except Exception as e:
        # Send error to main thread
        data_queue.put({'error': str(e), 'error_type': type(e).__name__})
        print(f"Training error: {type(e).__name__}: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("BPTT Training for Spatial.py Model with Visualization")
    print("=" * 60)
    print(f"\nNetwork Configuration:")
    print(f"  - rho: 10000")
    print(f"  - dx: 1.0")
    print(f"  - Target rate: 50.0 Hz")
    print(f"\nTraining parameters:")
    print(f"  - Epochs: Infinite")
    print(f"  - Learning rate: 1e-5")
    print("\n" + "=" * 60)
    print("Starting Training with Background Thread and Main Thread Visualization")
    print("=" * 60)
    
    # Create a queue for communication and stop event for graceful shutdown
    data_queue = queue.Queue()
    stop_event = threading.Event()
    window_size = 1000

    # Start background thread (training)
    training_thread = threading.Thread(target=training_worker, args=(data_queue, stop_event))
    training_thread.daemon = True
    training_thread.start()
    
    # Set up the visualization (runs in main thread)
    fig = plt.figure(figsize=(14, 6))

    # Loss curve subplot
    ax_loss = fig.add_subplot(131)
    ax_loss.set_title('Training Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True, alpha=0.3)
    line_loss, = ax_loss.plot([], [], 'b-', linewidth=2)

    # Rate curve subplot
    ax_rate = fig.add_subplot(132)
    ax_rate.set_title('Firing Rate')
    ax_rate.set_xlabel('Epoch')
    ax_rate.set_ylabel('Rate (Hz)')
    ax_rate.grid(True, alpha=0.3)
    line_rate, = ax_rate.plot([], [], 'g-', linewidth=2)
    ax_rate.axhline(y=50.0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax_rate.legend()

    # External spikes subplot
    ax_ext = fig.add_subplot(133)
    ax_ext.set_title('External Input Spikes')
    ax_ext.set_xlabel('Epoch')
    ax_ext.set_ylabel('Spikes')
    ax_ext.grid(True, alpha=0.3)
    line_ext, = ax_ext.plot([], [], 'm-', linewidth=2)
    
    # Text annotations
    text_epoch = fig.text(0.02, 0.95, '', fontsize=12, fontweight='bold')
    text_loss = fig.text(0.02, 0.92, '', fontsize=10)
    text_rate = fig.text(0.02, 0.89, '', fontsize=10)
    text_grad = fig.text(0.02, 0.86, '', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Data storage
    losses = []
    rates = []
    ext_spikes_list = []
    epochs = []
    
    plt.ion()
    plt.show()
    
    # Main thread visualization loop
    try:
        while True:
            try:
                # Check for new data with timeout
                data = data_queue.get(timeout=0.1)

                if 'done' in data:
                    break

                if 'error' in data:
                    print(f"\nTraining thread error: {data['error_type']}: {data['error']}")
                    break

                # Update data storage with rolling window
                epochs.append(data['epoch'])
                losses.append(data['loss'])
                rates.append(data['rate'])
                ext_spikes_list.append(data['ext_spikes'])

                # Keep only last window_size points to prevent memory leak
                if len(epochs) > window_size:
                    epochs.pop(0)
                    losses.pop(0)
                    rates.pop(0)
                    ext_spikes_list.pop(0)

                # Update plots
                line_loss.set_data(epochs, losses)
                ax_loss.relim()
                ax_loss.autoscale_view()

                line_rate.set_data(epochs, rates)
                ax_rate.relim()
                ax_rate.autoscale_view()

                line_ext.set_data(epochs, ext_spikes_list)
                ax_ext.relim()
                ax_ext.autoscale_view()

                # Update text
                text_epoch.set_text(f"Epoch: {data['epoch']}/{data['total_epochs']}")
                text_loss.set_text(f"Loss: {data['loss']:.4f}")
                text_rate.set_text(f"Rate: {data['rate']:.2f} Hz (Target: 50.0 Hz)")
                text_grad.set_text(f"Grad Norm: {data['grad_norm']:.4f}")

                plt.draw()
                plt.pause(0.01)

            except queue.Empty:
                # No data available, just continue
                plt.pause(0.01)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping training gracefully...")
        stop_event.set()
    
    plt.ioff()
    print("\nTraining complete. Close the plot window to exit.")
    plt.show()
    
    # Wait for training thread to complete
    training_thread.join()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
