from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import brainpy as bp
import brainpy.math as bm
import jax

from src.models.Spatial import Spatial


LEFT_COLOR = "#1f77b4"
RIGHT_COLOR = "#ff7f0e"
SPEED_COLOR = "#9467bd"
SENSOR_COLOR = "cyan"


def wrap_angle(theta: float) -> float:
    # Normalize angle to [-pi, pi].
    return (theta + np.pi) % (2.0 * np.pi) - np.pi # Wrap angle to [-pi, pi].


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # Standard logistic nonlinearity.
    return 1.0 / (1.0 + np.exp(-x))


def to_numpy(x) -> np.ndarray:
    # Conversion for BrainPy/JAX arrays and Variables.
    if hasattr(x, "value"):
        x = x.value
    return np.asarray(x)


def set_input_var(input_var, value: np.ndarray) -> None:
    
    # Assign BrainPy InputVar value 
    # Used to feed input data into a running simulation. 

    target = input_var.input
    target_np = to_numpy(target)
    value_np = np.asarray(value, dtype=np.float32)

    if value_np.shape != target_np.shape:
        value_np = value_np.reshape(target_np.shape)

    v = bm.asarray(value_np)

    if hasattr(target, "value"):
        target.value = v
    else:
        target[:] = v


def assign_weight_array(comm, new_weight: np.ndarray) -> None:
    
    # Updates synaptic weights.
    
    new_w = bm.asarray(new_weight)
    
    if hasattr(comm.weight, "value"):
        comm.weight.value = new_w
    else:
        comm.weight = new_w




@dataclass
class TrainConfig:
    # SNN / simulation
    seed: int = 42
    dt_ms: float = 0.1
    rho: int = 10000
    dx: float = 1.0
    control_window_ms: float = 10.0

    # RL / environment
    episodes: int = 30
    steps_per_episode: int = 250
    world_size: float = 1.0
    n_positive: int = 6
    n_negative: int = 6
    robot_radius: float = 0.03
    collect_radius: float = 0.06
    sensor_range: float = 0.50
    sensor_fov: float = np.deg2rad(50.0)
    sensor_ring_radius_frac: float = 0.42
    sensor_patch_radius_frac: float = 0.05
    sensor_min_neurons: int = 40
    max_speed: float = 0.60
    wheel_base: float = 0.08
    action_noise_std: float = 0.10

    # Learning
    learning_rate: float = 3e-4
    weight_decay: float = 2e-3
    reward_scale: float = 1.0
    train_projections: Tuple[str, ...] = ("E2E", "E2I", "I2E", "I2I")
    edge_sample_fraction: float = 0.01
    weight_update_interval_windows: int = 1
    weight_sync_interval_windows: int = 5

    # Sensor/action interface
    sensor_gain: float = 65.0
    tonic_current_bias: float = 4.0
    action_turn_sensitivity: float = 8.0
    action_thrust_sensitivity: float = 10.0
    action_thrust_bias: float = 0.04
    action_turn_scale: float = 0.55

    # New parameters for reward shaping
    reward_shaping_pos: float = 0.08
    reward_shaping_neg: float = 0.05
    reward_time_cost: float = 0.003
    reward_boundary_penalty: float = 0.05

    # Rendering / logging
    render: bool = True
    render_every_episode: int = 1
    save_weights_path: str = "trained_spatial_robot_weights.npz"




class DifferentialSteeringWorld:
    def __init__(self, cfg: TrainConfig, rng: np.random.Generator):
        # Initialize world state, reward markers, and robot pose buffers.
        self.cfg = cfg
        self.rng = rng
        self.size = cfg.world_size
        self.pos_rewards = np.zeros((cfg.n_positive, 2), dtype=np.float32)
        self.neg_rewards = np.zeros((cfg.n_negative, 2), dtype=np.float32)
        self.robot = np.zeros(3, dtype=np.float32)  # x, y, theta
        self.trail: List[Tuple[float, float]] = []


    def _random_point(self, margin: float = 0.05) -> np.ndarray:
        # Sample a random point inside the arena away from boundaries.
        p = self.rng.uniform(margin, self.size - margin, size=(2,))
        return p.astype(np.float32)


    def reset(self) -> np.ndarray:
        # Reset robot pose, respawn reward markers, and clear the trail.
        self.robot = np.array(
            [
                self.rng.uniform(0.2, 0.8),
                self.rng.uniform(0.2, 0.8),
                self.rng.uniform(-np.pi, np.pi),
            ],
            dtype=np.float32,
        )
        for i in range(self.cfg.n_positive):
            self.pos_rewards[i] = self._random_point()
        for i in range(self.cfg.n_negative):
            self.neg_rewards[i] = self._random_point()
        self.trail = [(float(self.robot[0]), float(self.robot[1]))]
        return self.robot.copy()


    def observe(self, sensor_angles: np.ndarray) -> np.ndarray:
        
        # Signed directional sensors in [-1, 1]:
        # positive items push sensor values up; negative items push them down.

        x, y, theta = self.robot
        robot_xy = np.array([x, y], dtype=np.float32)
        sensors = np.zeros_like(sensor_angles, dtype=np.float32)

        def add_objects(points: np.ndarray, sign: float):
            # Accumulate signed sensor responses from nearby objects.
            for p in points:
                vec = p - robot_xy
                dist = float(np.linalg.norm(vec))
                if dist < 1e-6 or dist > self.cfg.sensor_range:
                    continue
                obj_angle = math.atan2(vec[1], vec[0])
                dist_gain = math.exp(-dist / max(1e-6, self.cfg.sensor_range))
                for i, rel_sensor in enumerate(sensor_angles):
                    sensor_dir = theta + rel_sensor
                    d_ang = wrap_angle(obj_angle - sensor_dir)
                    if abs(d_ang) <= self.cfg.sensor_fov * 0.5:
                        ang_gain = math.exp(
                            -0.5 * (d_ang / (self.cfg.sensor_fov * 0.35)) ** 2
                        )
                        sensors[i] += sign * float(dist_gain * ang_gain)

        add_objects(self.pos_rewards, +1.0)
        add_objects(self.neg_rewards, -1.0)


        # Wall proximity penalty in directional sensors.
        wall_dist = np.array([x, self.size - x, y, self.size - y])
        wall_angles = np.array([np.pi, 0.0, 0.5 * np.pi, -0.5 * np.pi])

        for d, wall_angle in zip(wall_dist, wall_angles):
            if d > self.cfg.sensor_range:
                continue
            dist_gain = math.exp(-d / max(1e-6, self.cfg.sensor_range))
            for i, rel_sensor in enumerate(sensor_angles):
                sensor_dir = theta + rel_sensor
                d_ang = wrap_angle(wall_angle - sensor_dir)
                if abs(d_ang) <= self.cfg.sensor_fov * 0.5:
                    ang_gain = math.exp(
                        -0.5 * (d_ang / (self.cfg.sensor_fov * 0.35)) ** 2
                    )
                    sensors[i] += -0.3 * float(dist_gain * ang_gain)
        
        return np.clip(sensors, -1.0, 1.0)


    def _nearest_distance(self, points: np.ndarray) -> float:
        # Return nearest Euclidean distance from robot to a point set.
        robot_xy = self.robot[:2]
        d = np.linalg.norm(points - robot_xy[None, :], axis=1)
        return float(np.min(d)) if len(d) else 0.0


    def step(self, wheel_left: float, wheel_right: float, dt_s: float) -> Tuple[np.ndarray, float, Dict]:
        # Differential-drive step with sparse and dense reward shaping.
        # +1 for collecting positive marker.
        # -1 for collecting negative marker.
        # Shape toward nearest positive and away from nearest negative.
        # Apply a boundary penalty on wall contact.
        
        wl = float(np.clip(wheel_left, 0.0, 1.0))
        wr = float(np.clip(wheel_right, 0.0, 1.0))
        v = 0.5 * (wl + wr) * self.cfg.max_speed
        omega = ((wr - wl) / max(1e-6, self.cfg.wheel_base)) * 0.9

        before_pos_d = self._nearest_distance(self.pos_rewards)
        before_neg_d = self._nearest_distance(self.neg_rewards)

        x, y, theta = self.robot
        theta = wrap_angle(theta + omega * dt_s)
        x = x + v * math.cos(theta) * dt_s
        y = y + v * math.sin(theta) * dt_s

        boundary_hit = False
        if x < 0.0:
            x = 0.0
            boundary_hit = True
        if x > self.size:
            x = self.size
            boundary_hit = True
        if y < 0.0:
            y = 0.0
            boundary_hit = True
        if y > self.size:
            y = self.size
            boundary_hit = True

        self.robot = np.array([x, y, theta], dtype=np.float32)
        self.trail.append((float(x), float(y)))

        reward = 0.0

        # Sparse collection rewards.
        for i in range(self.cfg.n_positive):
            if np.linalg.norm(self.pos_rewards[i] - self.robot[:2]) <= self.cfg.collect_radius:
                reward += 1.0
                self.pos_rewards[i] = self._random_point()
        for i in range(self.cfg.n_negative):
            if np.linalg.norm(self.neg_rewards[i] - self.robot[:2]) <= self.cfg.collect_radius:
                reward -= 1.0
                self.neg_rewards[i] = self._random_point()

        # Dense shaping.
        after_pos_d = self._nearest_distance(self.pos_rewards)
        after_neg_d = self._nearest_distance(self.neg_rewards)
        reward += self.cfg.reward_shaping_pos * (before_pos_d - after_pos_d)
        reward += self.cfg.reward_shaping_neg * (after_neg_d - before_neg_d)

        # Small time cost + boundary penalty.
        reward -= self.cfg.reward_time_cost
        if boundary_hit:
            reward -= self.cfg.reward_boundary_penalty

        info = {
            "v": v,
            "omega": omega,
            "before_pos_d": before_pos_d,
            "after_pos_d": after_pos_d,
            "before_neg_d": before_neg_d,
            "after_neg_d": after_neg_d,
            "boundary_hit": boundary_hit,
        }
        return self.robot.copy(), float(reward), info


class ProjectionUpdater:
    # Reward-modulated Hebbian updater over a fixed sparse edge set.
    # Topology stays fixed; only existing edge weights are changed.

    def __init__(
        self,
        projection,
        pre_size: int,
        post_size: int,
        lr: float,
        decay: float,
        rng: np.random.Generator,
        edge_sample_fraction: float,
        clip_scale: float = 4.0,
    ):
        # Cache sparse connectivity and initialize trainable weight buffers.
        self.proj = projection
        self.lr = lr
        self.decay = decay

        comm = self.proj.proj.comm
        self.indices = to_numpy(comm.indices).astype(np.int32)
        self.indptr = to_numpy(comm.indptr).astype(np.int32)
        self.pre_ids = np.repeat(np.arange(pre_size, dtype=np.int32), np.diff(self.indptr))
        self.post_size = post_size

        w0 = to_numpy(comm.weight).astype(np.float32)
        if w0.ndim != 1:
            w0 = w0.reshape(-1)
        self.w = w0.copy()
        self.w_min = 0.0
        self.w_max = max(1e-6, float(np.percentile(w0, 99.0) * clip_scale))

        n_edges = self.w.shape[0]
        if edge_sample_fraction >= 1.0:
            self.train_idx = None
        else:
            n_train = max(1, int(round(n_edges * edge_sample_fraction)))
            self.train_idx = rng.choice(n_edges, size=n_train, replace=False).astype(np.int32)

    def update(
        self,
        pre_rate: np.ndarray,
        post_rate: np.ndarray,
        reward: float,
        do_sync: bool,
    ) -> None:
        # Apply one reward-modulated Hebbian update and optionally sync to model.
        comm = self.proj.proj.comm

        # Hebbian term on existing edges only.
        if self.train_idx is None:
            edge_pre = pre_rate[self.pre_ids]
            edge_post = post_rate[self.indices]
            hebb = edge_pre * edge_post
            dw = self.lr * reward * (hebb - self.decay * self.w)
            self.w = np.clip(self.w + dw, self.w_min, self.w_max)
        else:
            idx = self.train_idx
            edge_pre = pre_rate[self.pre_ids[idx]]
            edge_post = post_rate[self.indices[idx]]
            hebb = edge_pre * edge_post
            dw = self.lr * reward * (hebb - self.decay * self.w[idx])
            self.w[idx] = np.clip(self.w[idx] + dw, self.w_min, self.w_max)

        if do_sync:
            assign_weight_array(comm, self.w)


class SpatialRobotTrainer:
    def __init__(self, cfg: TrainConfig):
        # Build model, world, control interface, trainers, and render state.
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        bm.set_dt(cfg.dt_ms)

        self.model = self._build_model()
        self.world = DifferentialSteeringWorld(cfg, self.rng)
        self.control_steps = max(1, int(round(cfg.control_window_ms / cfg.dt_ms)))
        self.control_dt_s = self.control_steps * cfg.dt_ms / 1000.0
        self.e_positions = to_numpy(self.model.E.positions)
        self.e_domain = to_numpy(self.model.E.embedding.domain).astype(np.float32)
        self.e_grid_size = np.asarray(to_numpy(self.model.E.size), dtype=np.int32)
        self.e_x_edges = np.linspace(0.0, float(self.e_domain[0]), int(self.e_grid_size[0]) + 1)
        self.e_y_edges = np.linspace(0.0, float(self.e_domain[1]), int(self.e_grid_size[1]) + 1)

        self.sensor_angles = np.linspace(-np.pi * 0.8, np.pi * 0.8, 8, dtype=np.float32)
        self.left_motor_idx, self.right_motor_idx, self.speed_motor_idx = self._build_motor_pools()
        self.sensor_groups = self._build_sensor_groups()
        self.global_step = 0

        self.proj_updaters = self._build_projection_updaters()
        self.reward_history: List[float] = []

        self.fig = None
        self.ax_world = None
        self.ax_spike = None
        self.ax_io = None
        self.ax_reward = None
        self.artist_trail = None
        self.artist_robot = None
        self.artist_heading = None
        self.artist_pos = None
        self.artist_neg = None
        self.artist_spike = None
        self.spike_cbar = None
        self.sensor_markers = []
        self.sensor_lines = []
        self.sensor_pool_centers: List[Tuple[float, float]] = []
        self.wheel_bars = []
        self.io_text = None
        self.artist_reward = None
        self.world_text = None

        if cfg.render:
            self._init_render()

    def _build_model(self) -> Spatial:
        # Construct and reset the spatial SNN model.
        key = jax.random.PRNGKey(self.cfg.seed)
        model = Spatial(key=key, rho=self.cfg.rho, dx=self.cfg.dx)
        bp.reset_state(model)
        return model

    def _build_sensor_groups(self) -> List[np.ndarray]:
        # Build spatial neuron groups used as directional sensor input pads.
        pos = to_numpy(self.model.E.positions).reshape(-1, 2)
        domain = to_numpy(self.model.E.embedding.domain)
        cx, cy = 0.5 * float(domain[0]), 0.5 * float(domain[1])
        ring_r = self.cfg.sensor_ring_radius_frac * float(domain[0])
        patch_r = self.cfg.sensor_patch_radius_frac * float(domain[0])

        # Prevent sensor drive from directly exciting output pads.
        motor_union = np.concatenate(
            [self.left_motor_idx, self.right_motor_idx, self.speed_motor_idx]
        ).astype(np.int32)
        allowed_mask = np.ones(pos.shape[0], dtype=bool)
        allowed_mask[motor_union] = False
        allowed_idx = np.where(allowed_mask)[0]
        allowed_pos = pos[allowed_idx]

        groups: List[np.ndarray] = []
        for ang in self.sensor_angles:
            px = cx + ring_r * math.cos(float(ang))
            py = cy + ring_r * math.sin(float(ang))
            d2 = (allowed_pos[:, 0] - px) ** 2 + (allowed_pos[:, 1] - py) ** 2
            local = allowed_idx[np.where(d2 <= patch_r * patch_r)[0]]
            if local.size < self.cfg.sensor_min_neurons:
                k = min(self.cfg.sensor_min_neurons, allowed_idx.size)
                nearest = np.argpartition(d2, k - 1)[:k]
                local = allowed_idx[nearest]
            groups.append(local.astype(np.int32))
        return groups

    def _build_motor_pools(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Build left/right/speed motor output pools on the excitatory map.
        pos = to_numpy(self.model.E.positions).reshape(-1, 2)
        domain = to_numpy(self.model.E.embedding.domain)

        def disc_pool(cx: float, cy: float, r_frac: float) -> np.ndarray:
            # Select neurons within a circular pool.
            r = float(domain[0]) * r_frac
            d2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
            idx = np.where(d2 <= r * r)[0]
            return idx.astype(np.int32)

        # Compact motor pads with clear separation:
        # left/right near bottom corners, speed near top-center.
        left = disc_pool(0.20 * float(domain[0]), 0.18 * float(domain[1]), 0.12)
        right = disc_pool(0.80 * float(domain[0]), 0.18 * float(domain[1]), 0.12)
        speed = disc_pool(0.50 * float(domain[0]), 0.84 * float(domain[1]), 0.10)

        if left.size == 0:
            warnings.warn("Left motor pool is empty, using fallback.")
            left = np.arange(0, max(1, pos.shape[0] // 8), dtype=np.int32)
        if right.size == 0:
            warnings.warn("Right motor pool is empty, using fallback.")
            right = np.arange(max(1, pos.shape[0] // 2), max(2, 5 * pos.shape[0] // 8), dtype=np.int32)
        if speed.size == 0:
            warnings.warn("Speed motor pool is empty, using fallback.")
            speed = np.arange(max(1, pos.shape[0] // 3), max(2, 2 * pos.shape[0] // 3), dtype=np.int32)
        return left.astype(np.int32), right.astype(np.int32), speed.astype(np.int32)

    def _build_projection_updaters(self) -> Dict[str, ProjectionUpdater]:
        # Create projection updaters for the configured trainable projections.
        e_num = int(np.prod(to_numpy(self.model.E.size)))
        i_num = int(np.prod(to_numpy(self.model.I.size)))
        spec = {
            "E2E": (self.model.E2E, e_num, e_num),
            "E2I": (self.model.E2I, e_num, i_num),
            "I2E": (self.model.I2E, i_num, e_num),
            "I2I": (self.model.I2I, i_num, i_num),
        }
        updaters: Dict[str, ProjectionUpdater] = {}
        for name in self.cfg.train_projections:
            if name not in spec:
                continue
            proj, pre_size, post_size = spec[name]
            updaters[name] = ProjectionUpdater(
                projection=proj,
                pre_size=pre_size,
                post_size=post_size,
                lr=self.cfg.learning_rate,
                decay=self.cfg.weight_decay,
                rng=self.rng,
                edge_sample_fraction=self.cfg.edge_sample_fraction,
            )
        return updaters

    def encode_sensors_to_current(self, sensor_values: np.ndarray) -> np.ndarray:
        # Map sensor readings to flattened excitatory input current.
        e_num = int(np.prod(to_numpy(self.model.E.size)))
        current = np.zeros((e_num,), dtype=np.float32)

        # Signed sensor drive onto directional groups.
        for s, idx in zip(sensor_values, self.sensor_groups):
            current[idx] += self.cfg.sensor_gain * float(s)

        # Small tonic bias keeps activity alive.
        current += self.cfg.tonic_current_bias

        # Keep flattened shape; set_input_var() will match the exact target shape.
        return current

    def _step_model(self, e_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Run one simulation step and return flattened E/I spikes.
        set_input_var(self.model.Ein, e_input)
        set_input_var(self.model.Iin, np.zeros_like(to_numpy(self.model.I.spike), dtype=np.float32))

        # One simulation step.
        self.model.step_run(self.global_step)
        self.global_step += 1

        e_spike = to_numpy(self.model.E.spike).reshape(-1).astype(np.float32)
        i_spike = to_numpy(self.model.I.spike).reshape(-1).astype(np.float32)
        return e_spike, i_spike

    def run_control_window(self, e_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Run several simulation steps for one control window.
        e_hist = []
        i_hist = []
        for _ in range(self.control_steps):
            e_spk, i_spk = self._step_model(e_input)
            e_hist.append(e_spk)
            i_hist.append(i_spk)
        return np.stack(e_hist, axis=0), np.stack(i_hist, axis=0)

    def decode_action(self, e_spike_hist: np.ndarray, training: bool) -> Tuple[float, float]:
        # Decode motor pool rates into wheel commands.
        rates = e_spike_hist.mean(axis=0)
        left_rate = float(np.mean(rates[self.left_motor_idx]))
        right_rate = float(np.mean(rates[self.right_motor_idx]))
        speed_rate = float(np.mean(rates[self.speed_motor_idx]))

        turn = np.tanh(self.cfg.action_turn_sensitivity * (right_rate - left_rate))
        thrust = float(sigmoid(self.cfg.action_thrust_sensitivity * (speed_rate - self.cfg.action_thrust_bias)))

        wl = np.clip(thrust - self.cfg.action_turn_scale * turn, 0.0, 1.0)
        wr = np.clip(thrust + self.cfg.action_turn_scale * turn, 0.0, 1.0)

        if training:
            wl += self.rng.normal(0.0, self.cfg.action_noise_std)
            wr += self.rng.normal(0.0, self.cfg.action_noise_std)
            wl = float(np.clip(wl, 0.0, 1.0))
            wr = float(np.clip(wr, 0.0, 1.0))
        return wl, wr

    def apply_reward_weight_updates(
        self,
        e_spike_hist: np.ndarray,
        i_spike_hist: np.ndarray,
        reward: float,
        do_sync: bool,
    ) -> None:
        # Update configured projections using reward-modulated population rates.
        e_rate = e_spike_hist.mean(axis=0).astype(np.float32)
        i_rate = i_spike_hist.mean(axis=0).astype(np.float32)
        r = float(reward * self.cfg.reward_scale)

        for name, updater in self.proj_updaters.items():
            if name == "E2E":
                updater.update(e_rate, e_rate, r, do_sync=do_sync)
            elif name == "E2I":
                updater.update(e_rate, i_rate, r, do_sync=do_sync)
            elif name == "I2E":
                updater.update(i_rate, e_rate, r, do_sync=do_sync)
            elif name == "I2I":
                updater.update(i_rate, i_rate, r, do_sync=do_sync)

    def _init_render(self) -> None:
        # Initialize figure layout and all rendering panels.
        plt.ion()
        self.fig = plt.figure(figsize=(18, 5))
        gs = self.fig.add_gridspec(1, 4, width_ratios=[1.2, 1.0, 1.0, 1.0])
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_spike = self.fig.add_subplot(gs[0, 1])
        self.ax_io = self.fig.add_subplot(gs[0, 2])
        self.ax_reward = self.fig.add_subplot(gs[0, 3])

        self._setup_world_plot()
        self._setup_spike_plot()
        self._setup_io_plot()
        self._setup_reward_plot()

        self.fig.tight_layout()

    def _setup_world_plot(self) -> None:
        # Configure world panel artists and axes.
        self.ax_world.set_xlim(0.0, self.cfg.world_size)
        self.ax_world.set_ylim(0.0, self.cfg.world_size)
        self.ax_world.set_aspect("equal", adjustable="box")
        self.ax_world.set_title("Robot Arena")

        self.artist_pos = self.ax_world.scatter([], [], c="green", s=90, marker="o", label="Positive")
        self.artist_neg = self.ax_world.scatter([], [], c="red", s=90, marker="x", label="Negative")
        self.artist_robot = self.ax_world.scatter([], [], c="blue", s=120, marker="o", label="Robot")
        (self.artist_trail,) = self.ax_world.plot([], [], color="royalblue", alpha=0.6, lw=1.6, label="Trail")
        (self.artist_heading,) = self.ax_world.plot([], [], "b-", lw=2)
        self.ax_world.legend(loc="upper right")
        self.world_text = self.ax_world.text(0.01, 1.02, "", transform=self.ax_world.transAxes)

    def _setup_spike_plot(self) -> None:
        # Configure live spike heatmap and pool overlays.
        e_shape = (int(self.e_grid_size[0]), int(self.e_grid_size[1]))
        self.artist_spike = self.ax_spike.imshow(
            np.zeros(e_shape, dtype=np.float32).T,
            origin="lower",
            extent=[0.0, float(self.e_domain[0]), 0.0, float(self.e_domain[1])],
            interpolation="nearest",
            cmap="hot",
            vmin=0.0,
            vmax=max(1.0, self.control_steps * 0.3),
            aspect="auto",
        )
        self.ax_spike.set_title("Live E Spike Heatmap")
        self.ax_spike.set_xlabel("X Position")
        self.ax_spike.set_ylabel("Y Position")

        # Show where sensor and motor pools sit on the neural map.
        def pool_centroid(flat_idx: np.ndarray) -> Tuple[float, float]:
            # Compute centroid of a neuron index set in position space.
            p = self.e_positions[flat_idx]
            return float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))

        self.sensor_pool_centers = []
        for g in self.sensor_groups:
            cx, cy = pool_centroid(g)
            self.sensor_pool_centers.append((cx, cy))
            self.ax_spike.scatter([cx], [cy], s=45, marker="o", facecolors="none", edgecolors=SENSOR_COLOR, lw=1.0)
        for pool, color in [
            (self.left_motor_idx, LEFT_COLOR),
            (self.right_motor_idx, RIGHT_COLOR),
            (self.speed_motor_idx, SPEED_COLOR),
        ]:
            cx, cy = pool_centroid(pool)
            self.ax_spike.scatter([cx], [cy], s=55, marker="s", facecolors="none", edgecolors=color, lw=1.2)
        cx_l, cy_l = pool_centroid(self.left_motor_idx)
        cx_r, cy_r = pool_centroid(self.right_motor_idx)
        cx_s, cy_s = pool_centroid(self.speed_motor_idx)
        tx = 0.02 * float(self.e_domain[0])
        ty = 0.02 * float(self.e_domain[1])
        self.ax_spike.text(cx_l + tx, cy_l + ty, "L", color=LEFT_COLOR, fontsize=9, weight="bold")
        self.ax_spike.text(cx_r + tx, cy_r + ty, "R", color=RIGHT_COLOR, fontsize=9, weight="bold")
        self.ax_spike.text(cx_s + tx, cy_s + ty, "S", color=SPEED_COLOR, fontsize=9, weight="bold")

        spike_legend = [
            Line2D([0], [0], marker="o", color="none", markeredgecolor=SENSOR_COLOR, markerfacecolor="none", label="Sensor pools"),
            Line2D([0], [0], marker="s", color="none", markeredgecolor=LEFT_COLOR, markerfacecolor="none", label="Left pad"),
            Line2D([0], [0], marker="s", color="none", markeredgecolor=RIGHT_COLOR, markerfacecolor="none", label="Right pad"),
            Line2D([0], [0], marker="s", color="none", markeredgecolor=SPEED_COLOR, markerfacecolor="none", label="Speed pad"),
        ]
        self.ax_spike.legend(handles=spike_legend, loc="upper right", fontsize=8, frameon=True)

        self.spike_cbar = self.fig.colorbar(self.artist_spike, ax=self.ax_spike, fraction=0.046, pad=0.04)

    def _setup_io_plot(self) -> None:
        # Explicit I/O tie-in panel.
        self.ax_io.set_title("Sensor -> Spatial SNN -> Wheels")
        self.ax_io.set_xlim(0.0, 1.0)
        self.ax_io.set_ylim(0.0, 1.0)
        self.ax_io.axis("off")

        net_box = plt.Rectangle((0.36, 0.25), 0.28, 0.50, ec="black", fc="#F5F5F5", lw=1.5)
        self.ax_io.add_patch(net_box)
        self.ax_io.text(0.50, 0.50, "Spatial\nSNN", ha="center", va="center", fontsize=12, weight="bold")

        sensor_y = np.linspace(0.15, 0.85, len(self.sensor_angles))
        for y in sensor_y:
            line, = self.ax_io.plot([0.16, 0.36], [y, 0.5], color="#999999", lw=0.8, alpha=0.8)
            marker = self.ax_io.scatter([0.12], [y], s=70, c=[0.5], cmap="coolwarm", vmin=0.0, vmax=1.0)
            self.sensor_lines.append(line)
            self.sensor_markers.append(marker)
        self.ax_io.text(0.04, 0.50, "Sensors", rotation=90, va="center", ha="center")

        wheel_y = [0.66, 0.34]
        for y in wheel_y:
            line, = self.ax_io.plot([0.64, 0.80], [0.5, y], color="#666666", lw=1.2)
            _ = line  # Keep visual linkage.
        left_bar = self.ax_io.barh([0.66], [0.0], left=0.82, height=0.10, color=LEFT_COLOR)[0]
        right_bar = self.ax_io.barh([0.34], [0.0], left=0.82, height=0.10, color=RIGHT_COLOR)[0]
        self.wheel_bars = [left_bar, right_bar]
        self.ax_io.text(0.94, 0.73, "Left", color=LEFT_COLOR, ha="center", va="center", fontsize=9, weight="bold")
        self.ax_io.text(0.94, 0.27, "Right", color=RIGHT_COLOR, ha="center", va="center", fontsize=9, weight="bold")
        self.io_text = self.ax_io.text(0.50, 0.06, "", ha="center", va="center", fontsize=9)

    def _setup_reward_plot(self) -> None:
        # Configure reward history plot.
        self.ax_reward.set_title("Episode Reward")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Reward")
        (self.artist_reward,) = self.ax_reward.plot([], [], color="black", lw=2)
        self.ax_reward.grid(True, alpha=0.3)


    def _update_render(
        self,
        episode: int,
        step: int,
        episode_reward: float,
        e_spike_hist: np.ndarray,
        sensor_vals: np.ndarray,
        wheel_l: float,
        wheel_r: float,
    ) -> None:
        # Refresh all render panels for the current step.
        if not self.cfg.render:
            return

        x, y, theta = self.world.robot
        heading_len = 0.08
        hx = x + heading_len * math.cos(theta)
        hy = y + heading_len * math.sin(theta)

        self.artist_pos.set_offsets(self.world.pos_rewards)
        self.artist_neg.set_offsets(self.world.neg_rewards)
        self.artist_robot.set_offsets(np.array([[x, y]], dtype=np.float32))
        trail = np.asarray(self.world.trail, dtype=np.float32)
        self.artist_trail.set_data(trail[:, 0], trail[:, 1])
        self.artist_heading.set_data([x, hx], [y, hy])

        spike_counts = e_spike_hist.sum(axis=0)
        hist, _, _ = np.histogram2d(
            self.e_positions[:, 0],
            self.e_positions[:, 1],
            bins=[self.e_x_edges, self.e_y_edges],
            weights=spike_counts,
        )
        self.artist_spike.set_data(hist.T)
        vmax = max(1.0, float(np.max(hist)))
        self.artist_spike.set_clim(0.0, vmax)

        # I/O panel update: sensor magnitudes + wheel commands.
        sensor_norm = np.clip((sensor_vals + 1.0) * 0.5, 0.0, 1.0)
        for marker, val in zip(self.sensor_markers, sensor_norm):
            marker.set_array(np.array([val], dtype=np.float32))
        self.wheel_bars[0].set_width(0.16 * float(np.clip(wheel_l, 0.0, 1.0)))
        self.wheel_bars[1].set_width(0.16 * float(np.clip(wheel_r, 0.0, 1.0)))
        self.io_text.set_text(
            f"mean sensor={float(np.mean(sensor_vals)):+.2f}   "
            f"wheel L/R=({wheel_l:.2f}, {wheel_r:.2f})"
        )

        self.world_text.set_text(
            f"Ep {episode + 1} | Step {step + 1} | Reward {episode_reward:.2f} | WL {wheel_l:.2f} WR {wheel_r:.2f}"
        )

        xs = np.arange(1, len(self.reward_history) + 1)
        self.artist_reward.set_data(xs, np.array(self.reward_history))
        self.ax_reward.set_xlim(1, max(2, len(self.reward_history)))
        if self.reward_history:
            ymin = min(self.reward_history) - 0.5
            ymax = max(self.reward_history) + 0.5
            if abs(ymax - ymin) < 1e-6:
                ymax = ymin + 1.0
            self.ax_reward.set_ylim(ymin, ymax)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def train(self) -> List[float]:
        # Run training episodes with online reward-modulated plasticity.
        print("Starting training...")
        print(
            f"Spatial model: rho={self.cfg.rho}, dx={self.cfg.dx}, dt={self.cfg.dt_ms} ms, "
            f"control window={self.cfg.control_window_ms} ms ({self.control_steps} steps)."
        )
        print(f"Trainable projections: {list(self.proj_updaters.keys())}")

        for ep in range(self.cfg.episodes):
            bp.reset_state(self.model)
            self.global_step = 0
            self.world.reset()
            episode_reward = 0.0

            do_render = self.cfg.render and ((ep % self.cfg.render_every_episode) == 0)

            for step in range(self.cfg.steps_per_episode):
                sensor_vals = self.world.observe(self.sensor_angles)
                e_input = self.encode_sensors_to_current(sensor_vals)

                e_spk_hist, i_spk_hist = self.run_control_window(e_input)
                wl, wr = self.decode_action(e_spk_hist, training=True)
                _, reward, _ = self.world.step(wl, wr, dt_s=self.control_dt_s)
                do_update = (step % max(1, self.cfg.weight_update_interval_windows)) == 0
                do_sync = (step % max(1, self.cfg.weight_sync_interval_windows)) == 0
                if do_update:
                    self.apply_reward_weight_updates(
                        e_spk_hist,
                        i_spk_hist,
                        reward,
                        do_sync=do_sync,
                    )
                episode_reward += reward

                if do_render:
                    self._update_render(
                        episode=ep,
                        step=step,
                        episode_reward=episode_reward,
                        e_spike_hist=e_spk_hist,
                        sensor_vals=sensor_vals,
                        wheel_l=wl,
                        wheel_r=wr,
                    )

            # Ensure all local weight buffers are pushed to the model at episode end.
            for updater in self.proj_updaters.values():
                assign_weight_array(updater.proj.proj.comm, updater.w)

            self.reward_history.append(float(episode_reward))
            rolling = float(np.mean(self.reward_history[-10:]))
            print(
                f"Episode {ep + 1:03d}/{self.cfg.episodes}: "
                f"reward={episode_reward: .3f} | rolling10={rolling: .3f}"
            )

        print("Training complete.")
        return self.reward_history

    def run_demo(self, steps: int = 400) -> None:
        # Run one post-training episode with rendering and no exploration noise.
        print("Running post-training demo...")
        bp.reset_state(self.model)
        self.global_step = 0
        self.world.reset()
        total_reward = 0.0
        for step in range(steps):
            sensor_vals = self.world.observe(self.sensor_angles)
            e_input = self.encode_sensors_to_current(sensor_vals)
            e_spk_hist, _ = self.run_control_window(e_input)
            wl, wr = self.decode_action(e_spk_hist, training=False)
            _, reward, _ = self.world.step(wl, wr, dt_s=self.control_dt_s)
            total_reward += reward
            if self.cfg.render:
                self._update_render(
                    episode=len(self.reward_history),
                    step=step,
                    episode_reward=total_reward,
                    e_spike_hist=e_spk_hist,
                    sensor_vals=sensor_vals,
                    wheel_l=wl,
                    wheel_r=wr,
                )
        print(f"Demo total reward: {total_reward:.3f}")

    def save_weights(self, path: str) -> None:
        # Save selected projection weight arrays to a NumPy archive.
        out = {}
        for name in ["E2E", "E2I", "I2E", "I2I", "ext2E", "ext2I"]:
            proj = getattr(self.model, name, None)
            if proj is None:
                continue
            out[f"{name}_weight"] = to_numpy(proj.proj.comm.weight).astype(np.float32)
        np.savez(path, **out)
        print(f"Saved weights to: {path}")


def main():
    # =========================
    # In-code configuration only
    # =========================
    cfg = TrainConfig(
        seed=42,
        dt_ms=0.1,
        rho=10000,
        dx=1.0,
        control_window_ms=10.0,
        episodes=30,
        steps_per_episode=250,
        learning_rate=3e-4,
        weight_decay=2e-3,
        sensor_gain=65.0,
        tonic_current_bias=4.0,
        action_turn_sensitivity=8.0,
        action_thrust_sensitivity=10.0,
        action_thrust_bias=0.04,
        action_turn_scale=0.55,
        reward_shaping_pos=0.08,
        reward_shaping_neg=0.05,
        reward_time_cost=0.003,
        reward_boundary_penalty=0.05,
        render=True,
        render_every_episode=1,
        save_weights_path="trained_spatial_robot_weights.npz",
        edge_sample_fraction=0.01,
        weight_update_interval_windows=1,
        weight_sync_interval_windows=5,
    )
    demo_steps = 400

    trainer = SpatialRobotTrainer(cfg)
    trainer.train()
    trainer.save_weights(cfg.save_weights_path)
    trainer.run_demo(steps=demo_steps)

    if cfg.render:
        print("Close the plot window to exit.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
