# Helper routines to independently run the simulation outside of notebook environments.

import brainpy as bp # Used to define neuron models.
import jax # Only used for the random number generator key, here.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from src.models.Spatial import Spatial # Spatial neural network model.
from src.plots import animate_spiking_activity # Animates the spatial neural network.



MODE = "interactive"  # "interactive", "save", or "both"
OUTPUT = "spiking_animation.gif" # Filename to save with.

FPS = 30.0 # [frames/s]
FRAME_STEP_MS = 1.0 # [simulated ms/frame]
PLAYBACK_SPEED = 1 # Playback scaling. 1.0 = Real time. 
WINDOW_SIZE_MS = 10.0 # [ms] Integration window for visualisation.
DURATION_MS = 2000.0 # [ms] Total simulation duration. 

SEED = 42
RHO = 10000 # [neurons / mm^2] Density of excitatory neurons.
DX = 1 # [mm] Side length of spatial domain. 





def prepare_spike_histograms(
    FNSnet,
    runner,
    window_size_ms,
    frame_step_ms,
):
    """Precompute spike-count histograms for each animation frame."""
    ts = np.asarray(runner.mon["ts"])
    E_spikes = np.asarray(runner.mon["E.spike"])

    t_start, t_stop = ts[0], ts[-1]

    # Determine the time points for each animation frame
    frame_times = np.arange(t_start, t_stop, frame_step_ms)
    if frame_times.size == 0:
        frame_times = np.array([t_start])  # Ensure at least one frame

    # Get neuron positions and spatial grid for histograms
    E_positions = np.asarray(FNSnet.E.positions)
    domain = np.asarray(FNSnet.E.embedding.domain, dtype=float)
    grid_size = np.asarray(FNSnet.E.size, dtype=int)
    x_edges = np.linspace(0, domain[0], grid_size[0] + 1)
    y_edges = np.linspace(0, domain[1], grid_size[1] + 1)

    # Pre-allocate histogram array
    histograms = np.zeros((len(frame_times), grid_size[0], grid_size[1]), dtype=float)

    # Calculate histogram for each frame
    for i, frame_t in enumerate(frame_times):
        # Find time window for the current frame
        win_start_t = frame_t - window_size_ms
        idx_start = np.searchsorted(ts, win_start_t, side="left")
        idx_end = np.searchsorted(ts, frame_t, side="right")

        # Sum spikes within the window for each neuron
        spike_counts = np.sum(E_spikes[idx_start:idx_end, :], axis=0)

        # Create a 2D histogram of spike counts, by neuron positions
        hist, _, _ = np.histogram2d(
            E_positions[:, 0],
            E_positions[:, 1],
            bins=[x_edges, y_edges],
            weights=spike_counts,
        )
        histograms[i] = hist

    return histograms, frame_times, domain


def show_interactive_spiking_activity(
    histograms,
    frame_times,
    domain,
    fps,
    playback_speed,
):
    """Show spike activity as an interactive Matplotlib heatmap with controls."""
    num_frames = histograms.shape[0]
    max_hist_value = float(np.max(histograms)) if histograms.size else 1.0
    if max_hist_value <= 0:
        max_hist_value = 1.0

    fig, ax = plt.subplots(dpi=100)  # Hardcode DPI for simplicity
    fig.subplots_adjust(bottom=0.26)

    im = ax.imshow(
        histograms[0].T,
        origin="lower",
        extent=[0, domain[0], 0, domain[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=max_hist_value,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spike Count")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    state = {
        "frame_idx": 0,
        "playing": True,
        "speed": float(playback_speed),
    }
    title = ax.set_title("")

    time_slider_ax = fig.add_axes([0.12, 0.14, 0.76, 0.035])
    if num_frames > 1:
        time_slider = Slider(
            ax=time_slider_ax,
            label="Time (ms)",
            valmin=float(frame_times[0]),
            valmax=float(frame_times[-1]),
            valinit=float(frame_times[0]),
            valstep=frame_times.tolist(),
        )
    else:
        time_slider = Slider(
            ax=time_slider_ax,
            label="Time (ms)",
            valmin=float(frame_times[0]),
            valmax=float(frame_times[0] + 1.0),
            valinit=float(frame_times[0]),
        )
        time_slider.set_active(False)

    speed_slider_ax = fig.add_axes([0.12, 0.08, 0.54, 0.035])
    speed_slider = Slider(
        ax=speed_slider_ax,
        label="Speed (x)",
        valmin=0.05,
        valmax=3.0,
        valinit=float(playback_speed),
        valstep=0.05,
    )

    play_button_ax = fig.add_axes([0.72, 0.06, 0.16, 0.07])
    play_button = Button(play_button_ax, "Pause")

    base_interval_ms = 1000.0 / fps
    timer = fig.canvas.new_timer(
        interval=max(1, int(round(base_interval_ms / state["speed"])))
    )

    def draw_frame(frame_idx, sync_slider=True):
        frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))
        state["frame_idx"] = frame_idx
        im.set_data(histograms[frame_idx].T)
        title.set_text(
            f"Time: {frame_times[frame_idx]:.1f} ms | Speed: {state['speed']:.2f}x"
        )
        if sync_slider and num_frames > 1:
            time_slider.eventson = False
            time_slider.set_val(float(frame_times[frame_idx]))
            time_slider.eventson = True
        fig.canvas.draw_idle()

    def update_timer_interval():
        timer.interval = max(1, int(round(base_interval_ms / state["speed"])))

    def on_timer():
        if not state["playing"] or num_frames <= 1:
            return
        next_idx = (state["frame_idx"] + 1) % num_frames
        draw_frame(next_idx, sync_slider=True)

    def on_time_change(time_value):
        nearest_idx = int(np.abs(frame_times - time_value).argmin())
        draw_frame(nearest_idx, sync_slider=False)

    def on_speed_change(speed_value):
        state["speed"] = max(0.05, float(speed_value))
        update_timer_interval()
        draw_frame(state["frame_idx"], sync_slider=False)

    def on_play_pause(_event):
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")

    def on_key(event):
        if event.key == " ":
            on_play_pause(None)
        elif event.key == "right":
            state["playing"] = False
            play_button.label.set_text("Play")
            draw_frame((state["frame_idx"] + 1) % num_frames, sync_slider=True)
        elif event.key == "left":
            state["playing"] = False
            play_button.label.set_text("Play")
            draw_frame((state["frame_idx"] - 1) % num_frames, sync_slider=True)

    def on_close(_event):
        timer.stop()

    time_slider.on_changed(on_time_change)
    speed_slider.on_changed(on_speed_change)
    play_button.on_clicked(on_play_pause)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    timer.add_callback(on_timer)
    timer.start()
    draw_frame(0, sync_slider=True)
    plt.show()







def main():
    """
    Run a spiking neural network simulation and either view it interactively,
    save it as an animation, or both.
    """

    key = jax.random.PRNGKey(SEED)

    print("Creating Spatial model...")
    model = Spatial(key=key, rho=RHO, dx=DX) # Initiates SNN. Builds network topology. 

    print(f"Running simulation for {DURATION_MS} ms...")
    runner = bp.DSRunner(model, monitors=["E.spike", "I.spike"]) # Run the simulation using brainpy. 
    runner.run(DURATION_MS)
    print("Simulation finished.")

    if MODE in ("interactive", "both"):
        print("Preparing interactive heatmap data...")
        histograms, frame_times, domain = prepare_spike_histograms(
            model,
            runner,
            window_size_ms=WINDOW_SIZE_MS,
            frame_step_ms=FRAME_STEP_MS,
        )
        print(
            "Opening interactive Matplotlib viewer (space toggles play/pause, "
            "left/right arrows step frames)."
        )
        show_interactive_spiking_activity(
            histograms,
            frame_times,
            domain,
            fps=FPS,
            playback_speed=PLAYBACK_SPEED,
        )

    if MODE in ("save", "both"):
        print("Generating animation for file output...")
        ani = animate_spiking_activity(
            model,
            runner,
            window_size_ms=WINDOW_SIZE_MS,
            fps=FPS,
        )
        print(f"Saving animation to {OUTPUT} with {FPS} FPS...")
        ani.save(OUTPUT, writer="imagemagick", fps=FPS)
        print("Animation saved.")


if __name__ == "__main__":
    main()
