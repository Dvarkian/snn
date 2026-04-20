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

# Suppress matplotlib GUI warnings
warnings.filterwarnings('ignore', category=UserWarning)

def training_worker(data_queue, rho=10000, dx=1.0, target_rate=5.0, n_epochs=float('inf'), learning_rate=1e-5):
    """
    Background thread that runs the training loop and sends results to the main thread.
    """
    # Set training mode
    bm.set_environment(mode=bm.training_mode, dt=1.0)
    
    # Initialize model in training mode
    key = jax.random.PRNGKey(42)
    model = Spatial(rho=rho, dx=dx, key=key, training_mode=True)
    
    # Initialize optimizer with all recurrent weights
    optimizer = bp.optim.Adam(lr=learning_rate, train_vars={
        'w_ee': model.w_ee_trainable,
        'w_ei': model.w_ei_trainable,
        'w_ie': model.w_ie_trainable,
        'w_ii': model.w_ii_trainable
    })
    
    # Define loss function (traced by JAX for gradient computation)
    def loss_fun():
        model.E._time_counter.value = bm.asarray(0.0)
        model.I._time_counter.value = bm.asarray(0.0)
        
        def step(i):
            share.save('t', i * bm.get_dt(), 'dt', bm.get_dt())
            model()
            return bm.sum(model.E.spike), bm.sum(model.ext.spike)
        
        results = bm.for_loop(step, bm.arange(100))
        e_spikes = bm.sum(jnp.stack([r[0] for r in results]))
        ext_spikes = bm.sum(jnp.stack([r[1] for r in results]))
        
        duration_ms = 100 * bm.get_dt()
        actual_rate = e_spikes / duration_ms * 1000
        loss = bm.square(actual_rate - target_rate)
        
        return loss, actual_rate, ext_spikes
    
    # Training loop (runs indefinitely)
    epoch = 0
    while True:
        def compute_loss():
            return loss_fun()[0]
        
        # Compute gradients
        grads = bm.grad(compute_loss, grad_vars=[
            model.w_ee_trainable,
            model.w_ei_trainable,
            model.w_ie_trainable,
            model.w_ii_trainable
        ])()
        
        # Gradient clipping
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
        
        # Compute metrics
        loss_val, actual_rate, ext_spikes = loss_fun()
        
        # Send data to main thread for UI updates
        data_queue.put({
            'epoch': epoch + 1,
            'loss': float(loss_val),
            'rate': float(actual_rate),
            'ext_spikes': float(ext_spikes),
            'grad_norm': float(grad_norm),
            'total_epochs': float('inf')
        })
        
        print(f"Epoch {epoch+1}: Loss = {loss_val:.4f}, Rate = {actual_rate:.2f} Hz")
        
        epoch += 1
    
    # Signal completion
    data_queue.put({'done': True})

if __name__ == '__main__':
    print("=" * 60)
    print("BPTT Training for Spatial.py Model with Visualization")
    print("=" * 60)
    print(f"\nNetwork Configuration:")
    print(f"  - rho: 10000")
    print(f"  - dx: 1.0")
    print(f"  - Target rate: 5.0 Hz")
    print(f"\nTraining parameters:")
    print(f"  - Epochs: Infinite")
    print(f"  - Learning rate: 1e-5")
    print("\n" + "=" * 60)
    print("Starting Training with Background Thread and Main Thread Visualization")
    print("=" * 60)
    
    # Create a queue for communication
    data_queue = queue.Queue()
    
    # Start background thread (training)
    training_thread = threading.Thread(target=training_worker, args=(data_queue,))
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
    ax_rate.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='Target')
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
    while True:
        try:
            # Check for new data with timeout
            data = data_queue.get(timeout=0.1)
            
            if 'done' in data:
                break
            
            # Update data storage
            epochs.append(data['epoch'])
            losses.append(data['loss'])
            rates.append(data['rate'])
            ext_spikes_list.append(data['ext_spikes'])
            
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
            text_rate.set_text(f"Rate: {data['rate']:.2f} Hz (Target: 5.0 Hz)")
            text_grad.set_text(f"Grad Norm: {data['grad_norm']:.4f}")
            
            plt.draw()
            plt.pause(0.01)
            
        except:
            # No data available, just continue
            plt.pause(0.01)
    
    plt.ioff()
    print("\nTraining complete. Close the plot window to exit.")
    plt.show()
    
    # Wait for training thread to complete
    training_thread.join()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
