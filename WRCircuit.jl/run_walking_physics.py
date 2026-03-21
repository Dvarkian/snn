"""
Standalone passive physics demo for the 2D walker.

This script uses a rigid-body Pymunk backend with proper collision handling.
The walker starts slightly above the ground with a small pose perturbation so
it simply falls and flops onto the ground under gravity.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np


def _ensure_writable_runtime_dirs():
    runtime_home = os.path.expanduser("~")
    try:
        probe_path = os.path.join(runtime_home, ".wrcircuit_write_probe")
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write("")
        os.remove(probe_path)
    except OSError:
        runtime_home = os.path.join("/tmp", "wrcircuit-home")
        os.makedirs(runtime_home, exist_ok=True)
        os.environ["HOME"] = runtime_home

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = os.path.join(runtime_home, ".config", "matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir


_ensure_writable_runtime_dirs()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from walking_physics_pymunk import PymunkPassiveWalker, make_initial_state


@dataclass
class PassiveWalkerConfig:
    n_legs: int = 2
    mass: float = 2.0
    thigh_mass: float = 0.25
    shank_mass: float = 0.18
    drag: float = 0.12
    angular_drag: float = 1.2
    joint_drag: float = 0.18
    gravity: float = 9.81
    ground_k: float = 1500.0
    ground_c: float = 35.0
    ground_tangent_damping: float = 80.0
    friction_mu: float = 0.9
    body_length: float = 0.45
    body_height: float = 0.18
    hip_x_offset: float = 0.16
    thigh_length: float = 0.22
    shank_length: float = 0.22
    leg_radius: float = 0.022
    foot_radius: float = 0.028
    body_corner_radius: float = 0.012
    height_target: float = 0.48
    hip_limit: float = 0.95
    knee_min: float = 0.05
    knee_max: float = 1.45
    hip_kp: float = 0.0
    hip_kd: float = 0.0
    knee_kp: float = 0.0
    knee_kd: float = 0.0
    hip_torque_limit: float = 0.0
    knee_torque_limit: float = 0.0

    dt_ms: float = 2.0
    duration_ms: float = 2500.0
    animation_fps: float = 30.0
    physics_substeps: int = 10
    solver_iterations: int = 50
    collision_slop: float = 1e-3
    contact_epsilon: float = 2e-3

    drop_height: float = 0.16
    initial_pitch: float = 0.22
    initial_omega: float = 0.0
    initial_body_x: float = 0.0
    initial_body_vx: float = 0.0
    initial_body_vy: float = 0.0
    hip_offsets: tuple[float, float] = (0.18, -0.10)
    knee_offsets: tuple[float, float] = (-0.10, 0.06)


# Edit these directly in code instead of passing CLI flags.
RUN_CONFIG = PassiveWalkerConfig()
SAVE_ANIMATION_PATH: Optional[str] = None
SHOW_ANIMATION = True


def _to_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def simulate_passive_walker(cfg: PassiveWalkerConfig):
    initial_state = make_initial_state(
        cfg,
        hip_offsets=np.asarray(cfg.hip_offsets, dtype=float),
        knee_offsets=np.asarray(cfg.knee_offsets, dtype=float),
    )
    physics = PymunkPassiveWalker(cfg, initial_state)
    sensed = physics.observe()

    num_steps = int(round(cfg.duration_ms / cfg.dt_ms))
    ts_ms = np.arange(num_steps + 1, dtype=float) * cfg.dt_ms
    pos = np.zeros((num_steps + 1, 2), dtype=float)
    vel = np.zeros((num_steps + 1, 2), dtype=float)
    angle = np.zeros((num_steps + 1,), dtype=float)
    omega = np.zeros((num_steps + 1,), dtype=float)
    hip_angle = np.zeros((num_steps + 1, cfg.n_legs), dtype=float)
    knee_angle = np.zeros((num_steps + 1, cfg.n_legs), dtype=float)
    foot_pos = np.zeros((num_steps + 1, cfg.n_legs, 2), dtype=float)
    foot_vel = np.zeros((num_steps + 1, cfg.n_legs, 2), dtype=float)
    ground_contact = np.zeros((num_steps + 1, cfg.n_legs), dtype=float)
    force = np.zeros((num_steps + 1, 2), dtype=float)

    def record(
        i: int,
        current_state,
        current_foot_pos,
        current_foot_vel,
        current_ground_contact,
        current_force,
    ):
        pos[i] = _to_np(current_state.pos)
        vel[i] = _to_np(current_state.vel)
        angle[i] = float(current_state.angle)
        omega[i] = float(current_state.omega)
        hip_angle[i] = _to_np(current_state.hip_angle)
        knee_angle[i] = _to_np(current_state.knee_angle)
        foot_pos[i] = _to_np(current_foot_pos)
        foot_vel[i] = _to_np(current_foot_vel)
        ground_contact[i] = _to_np(current_ground_contact)
        force[i] = _to_np(current_force)

    record(
        0,
        sensed,
        sensed.foot_pos,
        sensed.foot_vel,
        sensed.ground_contact,
        sensed.total_ground_force,
    )

    for i in range(num_steps):
        physics.step()
        state = physics.observe()
        record(
            i + 1,
            state,
            state.foot_pos,
            state.foot_vel,
            state.ground_contact,
            state.total_ground_force,
        )

    return {
        "ts_ms": ts_ms,
        "pos": pos,
        "vel": vel,
        "angle": angle,
        "omega": omega,
        "hip_angle": hip_angle,
        "knee_angle": knee_angle,
        "foot_pos": foot_pos,
        "foot_vel": foot_vel,
        "ground_contact": ground_contact,
        "force": force,
    }


def animate_passive_walker(
    rollout,
    cfg: PassiveWalkerConfig,
    save_path: Optional[str] = None,
    show: bool = True,
):
    ts_ms = rollout["ts_ms"]
    pos = rollout["pos"]
    angle = rollout["angle"]
    hip_angle = rollout["hip_angle"]
    knee_angle = rollout["knee_angle"]
    foot_pos = rollout["foot_pos"]
    ground_contact = rollout["ground_contact"]
    force = rollout["force"]

    hip_local = np.array(
        [
            [cfg.hip_x_offset, -0.5 * cfg.body_height],
            [-cfg.hip_x_offset, -0.5 * cfg.body_height],
        ],
        dtype=float,
    )
    body_outline_local = np.array(
        [
            [0.5 * cfg.body_length, 0.5 * cfg.body_height],
            [0.5 * cfg.body_length, -0.5 * cfg.body_height],
            [-0.5 * cfg.body_length, -0.5 * cfg.body_height],
            [-0.5 * cfg.body_length, 0.5 * cfg.body_height],
            [0.5 * cfg.body_length, 0.5 * cfg.body_height],
        ],
        dtype=float,
    )

    def rotation_matrix(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def world_points(local_points: np.ndarray, body_pos: np.ndarray, theta: float) -> np.ndarray:
        rot = rotation_matrix(theta)
        return local_points @ rot.T + body_pos[None, :]

    def segment_dir(joint_angle: float) -> np.ndarray:
        return np.array([np.sin(joint_angle), -np.cos(joint_angle)], dtype=float)

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], hspace=0.35, wspace=0.30)
    ax_robot = fig.add_subplot(gs[:, 0])
    ax_pose = fig.add_subplot(gs[0, 1])
    ax_contact = fig.add_subplot(gs[1, 1])

    ax_robot.set_aspect("equal", adjustable="box")
    ax_robot.set_xlabel("x")
    ax_robot.set_ylabel("y")
    ax_robot.axhline(0.0, color="0.6", lw=1.0)

    path_line, = ax_robot.plot([], [], color="0.75", lw=1.5)
    body_box, = ax_robot.plot([], [], "-", color="black", lw=2)
    body_com, = ax_robot.plot([], [], "o", color="black", markersize=4)
    thighs = [ax_robot.plot([], [], "-", lw=2)[0] for _ in range(cfg.n_legs)]
    shanks = [ax_robot.plot([], [], "-", lw=2)[0] for _ in range(cfg.n_legs)]
    knees = [ax_robot.plot([], [], "o", color="tab:orange", ms=3)[0] for _ in range(cfg.n_legs)]
    feet = [ax_robot.plot([], [], "o", color="tab:blue", ms=4)[0] for _ in range(cfg.n_legs)]
    force_line, = ax_robot.plot([], [], "-", color="tab:red", lw=2)

    min_x = float(
        min(np.min(pos[:, 0]) - cfg.body_length, np.min(foot_pos[:, :, 0]) - 0.1)
    )
    max_x = float(
        max(np.max(pos[:, 0]) + cfg.body_length, np.max(foot_pos[:, :, 0]) + 0.1)
    )
    min_y = float(
        min(
            np.min(pos[:, 1]) - cfg.body_height,
            np.min(foot_pos[:, :, 1]) - 0.05,
            0.0,
        )
    )
    max_y = float(
        max(
            np.max(pos[:, 1]) + cfg.body_height,
            np.max(foot_pos[:, :, 1]) + 0.15,
            cfg.height_target + cfg.drop_height + 0.05,
        )
    )
    ax_robot.set_xlim(min_x, max_x)
    ax_robot.set_ylim(min_y, max_y)
    force_reference = max(1.0, float(np.max(np.linalg.norm(force, axis=1))))
    force_scale = 0.35 * cfg.body_length / force_reference

    ax_pose.set_title("Body Pose")
    ax_pose.plot(ts_ms, pos[:, 1], color="tab:purple", label="height")
    ax_pose.plot(ts_ms, angle, color="tab:brown", label="pitch")
    pose_cursor = ax_pose.axvline(ts_ms[0], color="black", lw=1.0, ls="--", alpha=0.6)
    ax_pose.set_xlabel("time (ms)")
    ax_pose.legend(loc="upper right")

    ax_contact.set_title("Ground Contact")
    colors = ["tab:blue", "tab:orange"]
    contact_lines = []
    for i in range(cfg.n_legs):
        contact_lines.append(
            ax_contact.plot(ts_ms, ground_contact[:, i], color=colors[i], label=f"Leg {i + 1}")[0]
        )
    contact_cursor = ax_contact.axvline(
        ts_ms[0], color="black", lw=1.0, ls="--", alpha=0.6
    )
    ax_contact.set_xlabel("time (ms)")
    ax_contact.set_ylim(-0.05, 1.05)
    ax_contact.legend(loc="upper right")

    frame_stride = max(1, int(round((1000.0 / cfg.animation_fps) / cfg.dt_ms)))
    frame_indices = np.arange(0, len(ts_ms), frame_stride, dtype=int)
    if frame_indices[-1] != len(ts_ms) - 1:
        frame_indices = np.append(frame_indices, len(ts_ms) - 1)

    def init():
        path_line.set_data([], [])
        body_box.set_data([], [])
        body_com.set_data([], [])
        force_line.set_data([], [])
        for artist in thighs + shanks + knees + feet:
            artist.set_data([], [])
        return [path_line, body_box, body_com, force_line, pose_cursor, contact_cursor] + thighs + shanks + knees + feet + contact_lines

    def update(frame_number: int):
        i = int(frame_indices[frame_number])
        p = np.asarray(pos[i], dtype=float)
        theta = float(angle[i])
        path_line.set_data(pos[: i + 1, 0], pos[: i + 1, 1])
        body_outline = world_points(body_outline_local, p, theta)
        hip_pos = world_points(hip_local, p, theta)
        body_box.set_data(body_outline[:, 0], body_outline[:, 1])
        body_com.set_data([p[0]], [p[1]])

        for j in range(cfg.n_legs):
            hip = hip_pos[j]
            thigh_abs = theta + float(hip_angle[i, j])
            knee_abs = thigh_abs + float(knee_angle[i, j])
            knee_pos = hip + cfg.thigh_length * segment_dir(thigh_abs)
            foot = foot_pos[i, j]
            thighs[j].set_data([hip[0], knee_pos[0]], [hip[1], knee_pos[1]])
            shanks[j].set_data([knee_pos[0], foot[0]], [knee_pos[1], foot[1]])
            knees[j].set_data([knee_pos[0]], [knee_pos[1]])
            feet[j].set_data([foot[0]], [max(0.0, foot[1])])

        f = force[i]
        force_line.set_data(
            [p[0], p[0] + f[0] * force_scale],
            [p[1], p[1] + f[1] * force_scale],
        )
        pose_cursor.set_xdata([ts_ms[i], ts_ms[i]])
        contact_cursor.set_xdata([ts_ms[i], ts_ms[i]])
        ax_robot.set_title(
            f"Passive Walker Physics | t={ts_ms[i]:.0f} ms | "
            f"x={p[0]:.3f} y={p[1]:.3f} pitch={theta:.3f}"
        )
        return [path_line, body_box, body_com, force_line, pose_cursor, contact_cursor] + thighs + shanks + knees + feet + contact_lines

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=max(1, int(round(1000.0 / cfg.animation_fps))),
        blit=False,
    )

    if save_path:
        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, writer="pillow", fps=cfg.animation_fps)
        print("Saved animation.")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


def main():
    cfg = RUN_CONFIG
    rollout = simulate_passive_walker(cfg)
    print(
        "Passive physics rollout complete: "
        f"final x={rollout['pos'][-1, 0]:.3f}, "
        f"final y={rollout['pos'][-1, 1]:.3f}, "
        f"final pitch={rollout['angle'][-1]:.3f}"
    )
    if not SHOW_ANIMATION and SAVE_ANIMATION_PATH is None:
        return
    animate_passive_walker(
        rollout,
        cfg,
        save_path=SAVE_ANIMATION_PATH,
        show=SHOW_ANIMATION,
    )


if __name__ == "__main__":
    main()
