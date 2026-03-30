import brainpy as bp
import jax
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
from scipy.signal import welch

from run_simulation import (
    DURATION_MS,
    DX,
    FPS,
    FRAME_STEP_MS,
    PLAYBACK_SPEED,
    RHO,
    SEED,
    WINDOW_SIZE_MS,
    prepare_spike_histograms,
)
from src.models.Spatial import Spatial


PATCH_CENTER_FRACTION = (0.5, 0.5)
PATCH_RADIUS_FRACTION = 0.12

SPECTRUM_MIN_DURATION_MS = 250.0
SPECTRUM_MAX_HZ = 120.0
SPECTRUM_SEGMENT_MS = 512.0
SPECTRUM_OVERLAP = 0.5

THETA_BAND_HZ = (4.0, 12.0)
GAMMA_BAND_HZ = (30.0, 100.0)


def resolve_patch_geometry(domain):
    domain = np.asarray(domain, dtype=float)
    center = domain * np.asarray(PATCH_CENTER_FRACTION, dtype=float)
    radius = float(np.min(domain) * PATCH_RADIUS_FRACTION)
    return center, radius


def build_patch_mask(model, patch_center_mm, patch_radius_mm):
    positions = np.asarray(model.E.positions, dtype=float)
    distances = np.linalg.norm(positions - patch_center_mm[None, :], axis=1)
    mask = distances <= patch_radius_mm

    if not np.any(mask):
        raise ValueError(
            "The circular patch does not contain any excitatory neurons. "
            "Increase PATCH_RADIUS_FRACTION or move PATCH_CENTER_FRACTION."
        )

    return mask


def prepare_patch_rate_series(runner, patch_mask, frame_times, frame_step_ms):
    ts = np.asarray(runner.mon["ts"], dtype=float)
    e_spikes = np.asarray(runner.mon["E.spike"], dtype=float)

    patch_step_counts = np.sum(e_spikes[:, patch_mask], axis=1, dtype=float)
    cumulative_counts = np.concatenate(([0.0], np.cumsum(patch_step_counts, dtype=float)))

    bin_edges = np.concatenate(
        (np.asarray(frame_times, dtype=float), [float(frame_times[-1] + frame_step_ms)])
    )
    idx_start = np.searchsorted(ts, bin_edges[:-1], side="left")
    idx_end = np.searchsorted(ts, bin_edges[1:], side="left")

    patch_counts_per_frame = cumulative_counts[idx_end] - cumulative_counts[idx_start]
    patch_neuron_count = int(np.sum(patch_mask))
    dt_s = frame_step_ms / 1000.0
    patch_rate_hz = patch_counts_per_frame / (patch_neuron_count * dt_s)

    return patch_rate_hz, patch_neuron_count


def extract_band_peak(freqs_hz, power, band_hz):
    band_mask = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1]) & np.isfinite(power)
    if not np.any(band_mask):
        return np.nan, np.nan

    band_indices = np.flatnonzero(band_mask)
    peak_index = band_indices[int(np.nanargmax(power[band_indices]))]
    return float(freqs_hz[peak_index]), float(power[peak_index])


def next_power_of_two(value):
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def prepare_patch_spectra(patch_rate_hz, frame_step_ms):
    sample_rate_hz = 1000.0 / frame_step_ms
    min_samples = max(8, int(np.ceil(SPECTRUM_MIN_DURATION_MS / frame_step_ms)))
    segment_samples = max(64, int(round(SPECTRUM_SEGMENT_MS / frame_step_ms)))
    nfft = max(256, next_power_of_two(len(patch_rate_hz)))
    freqs_hz = np.fft.rfftfreq(nfft, d=1.0 / sample_rate_hz)

    spectra = np.full((len(patch_rate_hz), len(freqs_hz)), np.nan, dtype=float)
    theta_peak_hz = np.full(len(patch_rate_hz), np.nan, dtype=float)
    theta_peak_power = np.full(len(patch_rate_hz), np.nan, dtype=float)
    gamma_peak_hz = np.full(len(patch_rate_hz), np.nan, dtype=float)
    gamma_peak_power = np.full(len(patch_rate_hz), np.nan, dtype=float)

    for frame_idx in range(len(patch_rate_hz)):
        n_samples = frame_idx + 1
        if n_samples < min_samples:
            continue

        segment = np.asarray(patch_rate_hz[:n_samples], dtype=float)
        if np.allclose(segment, segment[0]):
            spectra[frame_idx] = 0.0
            continue

        nperseg = min(segment_samples, n_samples)
        noverlap = min(int(round(nperseg * SPECTRUM_OVERLAP)), max(0, nperseg - 1))
        _, power = welch(
            segment,
            fs=sample_rate_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )

        spectra[frame_idx] = power
        theta_peak_hz[frame_idx], theta_peak_power[frame_idx] = extract_band_peak(
            freqs_hz, power, THETA_BAND_HZ
        )
        gamma_peak_hz[frame_idx], gamma_peak_power[frame_idx] = extract_band_peak(
            freqs_hz, power, GAMMA_BAND_HZ
        )

    return {
        "freqs_hz": freqs_hz,
        "spectra": spectra,
        "theta_peak_hz": theta_peak_hz,
        "theta_peak_power": theta_peak_power,
        "gamma_peak_hz": gamma_peak_hz,
        "gamma_peak_power": gamma_peak_power,
    }


def show_interactive_spiking_and_spectrum(
    histograms,
    frame_times,
    domain,
    patch_center_mm,
    patch_radius_mm,
    patch_rate_hz,
    patch_neuron_count,
    spectrum_data,
    fps,
    playback_speed,
):
    num_frames = histograms.shape[0]
    max_hist_value = float(np.max(histograms)) if histograms.size else 1.0
    if max_hist_value <= 0:
        max_hist_value = 1.0

    freqs_hz = spectrum_data["freqs_hz"]
    spectra = spectrum_data["spectra"]
    theta_peak_hz = spectrum_data["theta_peak_hz"]
    theta_peak_power = spectrum_data["theta_peak_power"]
    gamma_peak_hz = spectrum_data["gamma_peak_hz"]
    gamma_peak_power = spectrum_data["gamma_peak_power"]

    freq_mask = (freqs_hz > 0.0) & (freqs_hz <= SPECTRUM_MAX_HZ)
    display_freqs_hz = freqs_hz[freq_mask]

    finite_power = spectra[:, freq_mask]
    finite_power = finite_power[np.isfinite(finite_power) & (finite_power > 0.0)]
    if finite_power.size:
        power_floor = max(1e-6, float(np.max(finite_power)) * 1e-6)
        power_ceiling = float(np.max(finite_power)) * 1.2
    else:
        power_floor = 1e-6
        power_ceiling = 1.0

    fig, (heat_ax, spectrum_ax) = plt.subplots(
        ncols=2,
        figsize=(13.5, 6.5),
        dpi=100,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    fig.subplots_adjust(bottom=0.24, left=0.07, right=0.96, wspace=0.28)

    im = heat_ax.imshow(
        histograms[0].T,
        origin="lower",
        extent=[0, domain[0], 0, domain[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=max_hist_value,
    )
    heat_patch = Circle(
        patch_center_mm,
        patch_radius_mm,
        edgecolor="deepskyblue",
        facecolor="none",
        linewidth=2.0,
        linestyle="--",
    )
    heat_ax.add_patch(heat_patch)
    heat_ax.set_xlabel("X Position (mm)")
    heat_ax.set_ylabel("Y Position (mm)")
    heat_title = heat_ax.set_title("")
    cbar = fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spike Count")

    spectrum_ax.axvspan(*THETA_BAND_HZ, color="tab:blue", alpha=0.08)
    spectrum_ax.axvspan(*GAMMA_BAND_HZ, color="tab:orange", alpha=0.08)
    spectrum_ax.text(
        np.mean(THETA_BAND_HZ),
        0.98,
        "theta",
        transform=spectrum_ax.get_xaxis_transform(),
        color="tab:blue",
        ha="center",
        va="top",
    )
    spectrum_ax.text(
        np.mean(GAMMA_BAND_HZ),
        0.98,
        "gamma",
        transform=spectrum_ax.get_xaxis_transform(),
        color="tab:orange",
        ha="center",
        va="top",
    )
    spectrum_line, = spectrum_ax.plot([], [], color="black", linewidth=1.8)
    theta_marker, = spectrum_ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        color="tab:blue",
        markersize=7,
        label="Theta peak",
    )
    gamma_marker, = spectrum_ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        color="tab:orange",
        markersize=7,
        label="Gamma peak",
    )
    theta_line = spectrum_ax.axvline(np.nan, color="tab:blue", linestyle=":", linewidth=1.4)
    gamma_line = spectrum_ax.axvline(np.nan, color="tab:orange", linestyle=":", linewidth=1.4)
    spectrum_ax.set_xlim(0.0, SPECTRUM_MAX_HZ)
    spectrum_ax.set_yscale("log")
    spectrum_ax.set_ylim(power_floor, power_ceiling)
    spectrum_ax.set_xlabel("Frequency (Hz)")
    spectrum_ax.set_ylabel("Power Spectral Density")
    spectrum_ax.legend(loc="upper right")
    spectrum_title = spectrum_ax.set_title("")
    spectrum_text = spectrum_ax.text(
        0.02,
        0.95,
        "",
        transform=spectrum_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    state = {
        "frame_idx": 0,
        "playing": True,
        "speed": float(playback_speed),
    }

    time_slider_ax = fig.add_axes([0.11, 0.12, 0.78, 0.035])
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

    speed_slider_ax = fig.add_axes([0.11, 0.06, 0.56, 0.035])
    speed_slider = Slider(
        ax=speed_slider_ax,
        label="Speed (x)",
        valmin=0.05,
        valmax=3.0,
        valinit=float(playback_speed),
        valstep=0.05,
    )

    play_button_ax = fig.add_axes([0.73, 0.04, 0.16, 0.07])
    play_button = Button(play_button_ax, "Pause")

    base_interval_ms = 1000.0 / fps
    timer = fig.canvas.new_timer(
        interval=max(1, int(round(base_interval_ms / state["speed"])))
    )

    def set_peak_artist(marker, line, peak_hz, peak_power):
        if np.isfinite(peak_hz) and np.isfinite(peak_power):
            marker.set_data([peak_hz], [max(peak_power, power_floor)])
            line.set_xdata([peak_hz, peak_hz])
        else:
            marker.set_data([], [])
            line.set_xdata([np.nan, np.nan])

    def peak_label(name, peak_hz):
        if np.isfinite(peak_hz):
            return f"{name}: {peak_hz:.1f} Hz"
        return f"{name}: warming up"

    def draw_frame(frame_idx, sync_slider=True):
        frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))
        state["frame_idx"] = frame_idx

        im.set_data(histograms[frame_idx].T)
        current_time_ms = float(frame_times[frame_idx])
        heat_title.set_text(
            f"Spiking Activity | t = {current_time_ms:.1f} ms | speed = {state['speed']:.2f}x"
        )

        current_spectrum = spectra[frame_idx, freq_mask]
        current_rate_hz = float(patch_rate_hz[frame_idx])

        if np.any(np.isfinite(current_spectrum)):
            safe_spectrum = np.maximum(current_spectrum, power_floor)
            spectrum_line.set_data(display_freqs_hz, safe_spectrum)
            set_peak_artist(
                theta_marker,
                theta_line,
                theta_peak_hz[frame_idx],
                theta_peak_power[frame_idx],
            )
            set_peak_artist(
                gamma_marker,
                gamma_line,
                gamma_peak_hz[frame_idx],
                gamma_peak_power[frame_idx],
            )
            spectrum_title.set_text(f"Circular Window PSD | using samples up to {current_time_ms:.1f} ms")
            spectrum_text.set_text(
                f"{patch_neuron_count} excitatory neurons in patch\n"
                f"instantaneous patch rate: {current_rate_hz:.1f} Hz\n"
                f"{peak_label('theta peak', theta_peak_hz[frame_idx])}\n"
                f"{peak_label('gamma peak', gamma_peak_hz[frame_idx])}"
            )
        else:
            spectrum_line.set_data([], [])
            set_peak_artist(theta_marker, theta_line, np.nan, np.nan)
            set_peak_artist(gamma_marker, gamma_line, np.nan, np.nan)
            spectrum_title.set_text("Circular Window PSD")
            spectrum_text.set_text(
                f"{patch_neuron_count} excitatory neurons in patch\n"
                f"instantaneous patch rate: {current_rate_hz:.1f} Hz\n"
                f"accumulating at least {SPECTRUM_MIN_DURATION_MS:.0f} ms of data"
            )

        if sync_slider and num_frames > 1:
            time_slider.eventson = False
            time_slider.set_val(current_time_ms)
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
    key = jax.random.PRNGKey(SEED)

    print("Creating Spatial model...")
    model = Spatial(key=key, rho=RHO, dx=DX)

    print(f"Running simulation for {DURATION_MS} ms...")
    runner = bp.DSRunner(model, monitors=["E.spike", "I.spike"])
    runner.run(DURATION_MS)
    print("Simulation finished.")

    print("Preparing spatial spike histograms...")
    histograms, frame_times, domain = prepare_spike_histograms(
        model,
        runner,
        window_size_ms=WINDOW_SIZE_MS,
        frame_step_ms=FRAME_STEP_MS,
    )

    patch_center_mm, patch_radius_mm = resolve_patch_geometry(domain)
    patch_mask = build_patch_mask(model, patch_center_mm, patch_radius_mm)

    print("Preparing circular-window activity trace...")
    patch_rate_hz, patch_neuron_count = prepare_patch_rate_series(
        runner,
        patch_mask,
        frame_times,
        frame_step_ms=FRAME_STEP_MS,
    )

    print("Preparing live power spectra...")
    spectrum_data = prepare_patch_spectra(
        patch_rate_hz,
        frame_step_ms=FRAME_STEP_MS,
    )

    print("Opening combined activity + spectrum viewer...")
    show_interactive_spiking_and_spectrum(
        histograms=histograms,
        frame_times=frame_times,
        domain=domain,
        patch_center_mm=patch_center_mm,
        patch_radius_mm=patch_radius_mm,
        patch_rate_hz=patch_rate_hz,
        patch_neuron_count=patch_neuron_count,
        spectrum_data=spectrum_data,
        fps=FPS,
        playback_speed=PLAYBACK_SPEED,
    )


if __name__ == "__main__":
    main()
