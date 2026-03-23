"""
Robust rigid-body physics backend for the standalone passive walker demo.

This module uses Pymunk/Chipmunk2D to model the articulated walker as a set of
rigid bodies with joints, limits, friction, and proper collision resolution
against a static ground plane. It is intentionally separate from the JAX-based
``walking_physics`` module so the trainable pipeline can keep its current
differentiable implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pymunk


GROUND_COLLISION_TYPE = 1
LINK_COLLISION_TYPE = 2
FOOT_COLLISION_TYPES = (10, 11)


@dataclass(frozen=True)
class PassiveWalkerState:
    pos: np.ndarray
    vel: np.ndarray
    angle: float
    omega: float
    hip_angle: np.ndarray
    knee_angle: np.ndarray
    hip_omega: np.ndarray
    knee_omega: np.ndarray


@dataclass(frozen=True)
class PassiveWalkerObservation:
    pos: np.ndarray
    vel: np.ndarray
    angle: float
    omega: float
    hip_angle: np.ndarray
    knee_angle: np.ndarray
    hip_omega: np.ndarray
    knee_omega: np.ndarray
    foot_pos: np.ndarray
    foot_vel: np.ndarray
    ground_contact: np.ndarray
    total_ground_force: np.ndarray


def initial_joint_configuration(cfg) -> tuple[float, float]:
    desired_reach = cfg.height_target - 0.5 * cfg.body_height
    desired_reach = float(
        np.clip(
            desired_reach,
            abs(cfg.thigh_length - cfg.shank_length) + 1e-4,
            cfg.thigh_length + cfg.shank_length - 1e-4,
        )
    )
    cos_knee = (
        desired_reach**2 - cfg.thigh_length**2 - cfg.shank_length**2
    ) / (2.0 * cfg.thigh_length * cfg.shank_length)
    cos_knee = float(np.clip(cos_knee, -1.0, 1.0))
    knee = float(np.arccos(cos_knee))
    hip = float(
        -np.arctan2(
            cfg.shank_length * np.sin(knee),
            cfg.thigh_length + cfg.shank_length * np.cos(knee),
        )
    )
    hip = float(np.clip(hip, -cfg.hip_limit, cfg.hip_limit))
    knee = float(np.clip(knee, cfg.knee_min, cfg.knee_max))
    return hip, knee


def make_initial_state(
    cfg,
    hip_offsets: np.ndarray,
    knee_offsets: np.ndarray,
) -> PassiveWalkerState:
    hip0, knee0 = initial_joint_configuration(cfg)
    hip_angle = np.clip(
        np.full((cfg.n_legs,), hip0, dtype=float) + hip_offsets,
        -cfg.hip_limit,
        cfg.hip_limit,
    )
    knee_angle = np.clip(
        np.full((cfg.n_legs,), knee0, dtype=float) + knee_offsets,
        cfg.knee_min,
        cfg.knee_max,
    )
    return PassiveWalkerState(
        pos=np.asarray([cfg.initial_body_x, cfg.height_target + cfg.drop_height], dtype=float),
        vel=np.asarray([cfg.initial_body_vx, cfg.initial_body_vy], dtype=float),
        angle=float(cfg.initial_pitch),
        omega=float(cfg.initial_omega),
        hip_angle=np.asarray(hip_angle, dtype=float),
        knee_angle=np.asarray(knee_angle, dtype=float),
        hip_omega=np.zeros((cfg.n_legs,), dtype=float),
        knee_omega=np.zeros((cfg.n_legs,), dtype=float),
    )


def _segment_dir(angle: float) -> np.ndarray:
    return np.asarray([np.sin(angle), -np.cos(angle)], dtype=float)


def _cross_z(omega: float, r: np.ndarray) -> np.ndarray:
    return np.asarray([-omega * r[1], omega * r[0]], dtype=float)


def _safe_rate(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8:
        return 0.0
    return max(0.0, float(numerator) / float(denominator))


class PymunkPassiveWalker:
    """Passive articulated walker simulated with a rigid-body contact solver."""

    def __init__(self, cfg, initial_state: PassiveWalkerState):
        if cfg.n_legs != 2:
            raise ValueError("PymunkPassiveWalker expects n_legs == 2.")

        self.cfg = cfg
        self.initial_state = initial_state
        self.dt = float(cfg.dt_ms) / 1000.0
        self.substeps = max(1, int(getattr(cfg, "physics_substeps", 8)))
        self.sub_dt = self.dt / self.substeps
        self.leg_radius = float(getattr(cfg, "leg_radius", 0.022))
        self.foot_radius = float(getattr(cfg, "foot_radius", 0.028))
        self.body_corner_radius = float(getattr(cfg, "body_corner_radius", 0.012))
        self.contact_epsilon = float(getattr(cfg, "contact_epsilon", 2e-3))

        self.space = pymunk.Space()
        self.space.gravity = (0.0, -float(cfg.gravity))
        self.space.iterations = int(getattr(cfg, "solver_iterations", 40))
        self.space.collision_slop = float(getattr(cfg, "collision_slop", 1e-3))

        self._walker_filter = pymunk.ShapeFilter(group=1)
        self._frame_impulse = pymunk.Vec2d(0.0, 0.0)
        self._foot_contact_count = [0 for _ in range(cfg.n_legs)]
        self.hip_targets = np.zeros((cfg.n_legs,), dtype=float)
        self.knee_targets = np.zeros((cfg.n_legs,), dtype=float)
        self.last_joint_torque = np.zeros((cfg.n_legs, 2), dtype=float)

        self._build_ground()
        self._build_walker()
        self._register_collision_callbacks()

    def set_targets(self, hip_target: np.ndarray, knee_target: np.ndarray):
        self.hip_targets = np.asarray(hip_target, dtype=float).reshape(self.cfg.n_legs)
        self.knee_targets = np.asarray(knee_target, dtype=float).reshape(self.cfg.n_legs)

    def _build_ground(self):
        extent = float(getattr(self.cfg, "ground_extent", 1000.0))
        ground = pymunk.Segment(self.space.static_body, (-extent, 0.0), (extent, 0.0), 0.0)
        ground.friction = float(self.cfg.friction_mu)
        ground.elasticity = 0.0
        ground.collision_type = GROUND_COLLISION_TYPE
        self.ground = ground
        self.space.add(ground)

    def _build_walker(self):
        cfg = self.cfg
        state = self.initial_state

        body_moment = pymunk.moment_for_box(cfg.mass, (cfg.body_length, cfg.body_height))
        body = pymunk.Body(cfg.mass, body_moment)
        body.position = tuple(state.pos)
        body.velocity = tuple(state.vel)
        body.angle = float(state.angle)
        body.angular_velocity = float(state.omega)
        body.velocity_func = self._make_velocity_func(
            linear_rate=_safe_rate(cfg.drag, cfg.mass),
            angular_rate=_safe_rate(cfg.angular_drag, body_moment),
        )

        body_shape = pymunk.Poly.create_box(
            body,
            size=(cfg.body_length, cfg.body_height),
            radius=self.body_corner_radius,
        )
        body_shape.filter = self._walker_filter
        body_shape.collision_type = LINK_COLLISION_TYPE
        body_shape.friction = float(cfg.friction_mu)
        body_shape.elasticity = 0.0

        self.body = body
        self.body_shape = body_shape
        self.space.add(body, body_shape)

        hip_local = np.asarray(
            [
                [cfg.hip_x_offset, -0.5 * cfg.body_height],
                [-cfg.hip_x_offset, -0.5 * cfg.body_height],
            ],
            dtype=float,
        )

        self.thigh_bodies: List[pymunk.Body] = []
        self.shank_bodies: List[pymunk.Body] = []
        self.foot_shapes: List[pymunk.Circle] = []
        self.hip_limit_joints = []
        self.knee_limit_joints = []

        bodies = []
        shapes = []
        constraints = []

        for leg_idx in range(cfg.n_legs):
            hip_pos = self._body_local_to_world(hip_local[leg_idx])

            thigh_angle = float(state.angle + state.hip_angle[leg_idx])
            knee_pos = hip_pos + cfg.thigh_length * _segment_dir(thigh_angle)
            shank_angle = float(thigh_angle + state.knee_angle[leg_idx])
            thigh_center = hip_pos + 0.5 * cfg.thigh_length * _segment_dir(thigh_angle)
            shank_center = knee_pos + 0.5 * cfg.shank_length * _segment_dir(shank_angle)

            thigh_moment = pymunk.moment_for_segment(
                cfg.thigh_mass,
                (0.0, 0.5 * cfg.thigh_length),
                (0.0, -0.5 * cfg.thigh_length),
                self.leg_radius,
            )
            thigh_body = pymunk.Body(cfg.thigh_mass, thigh_moment)
            thigh_body.position = tuple(thigh_center)
            thigh_body.angle = thigh_angle
            thigh_body.angular_velocity = float(state.omega + state.hip_omega[leg_idx])
            thigh_body.velocity = tuple(
                self._point_velocity_from_body(hip_pos)
                + _cross_z(thigh_body.angular_velocity, thigh_center - hip_pos)
            )

            thigh_shape = pymunk.Segment(
                thigh_body,
                (0.0, 0.5 * cfg.thigh_length),
                (0.0, -0.5 * cfg.thigh_length),
                self.leg_radius,
            )
            thigh_shape.filter = self._walker_filter
            thigh_shape.collision_type = LINK_COLLISION_TYPE
            thigh_shape.friction = float(cfg.friction_mu)
            thigh_shape.elasticity = 0.0

            shank_moment = pymunk.moment_for_segment(
                cfg.shank_mass,
                (0.0, 0.5 * cfg.shank_length),
                (0.0, -0.5 * cfg.shank_length),
                self.leg_radius,
            )
            shank_body = pymunk.Body(cfg.shank_mass, shank_moment)
            shank_body.position = tuple(shank_center)
            shank_body.angle = shank_angle
            shank_body.angular_velocity = float(
                state.omega + state.hip_omega[leg_idx] + state.knee_omega[leg_idx]
            )
            shank_body.velocity = tuple(
                thigh_body.velocity_at_world_point(tuple(knee_pos))
                + _cross_z(shank_body.angular_velocity, shank_center - knee_pos)
            )

            shank_shape = pymunk.Segment(
                shank_body,
                (0.0, 0.5 * cfg.shank_length),
                (0.0, -0.5 * cfg.shank_length),
                self.leg_radius,
            )
            shank_shape.filter = self._walker_filter
            shank_shape.collision_type = LINK_COLLISION_TYPE
            shank_shape.friction = float(cfg.friction_mu)
            shank_shape.elasticity = 0.0

            foot_shape = pymunk.Circle(
                shank_body,
                self.foot_radius,
                offset=(0.0, -0.5 * cfg.shank_length),
            )
            foot_shape.filter = self._walker_filter
            foot_shape.collision_type = FOOT_COLLISION_TYPES[leg_idx]
            foot_shape.friction = float(cfg.friction_mu)
            foot_shape.elasticity = 0.0

            hip_pivot = pymunk.PivotJoint(self.body, thigh_body, tuple(hip_pos))
            hip_limit = pymunk.RotaryLimitJoint(
                self.body, thigh_body, -float(cfg.hip_limit), float(cfg.hip_limit)
            )
            knee_pivot = pymunk.PivotJoint(thigh_body, shank_body, tuple(knee_pos))
            knee_limit = pymunk.RotaryLimitJoint(
                thigh_body,
                shank_body,
                float(cfg.knee_min),
                float(cfg.knee_max),
            )

            self.thigh_bodies.append(thigh_body)
            self.shank_bodies.append(shank_body)
            self.foot_shapes.append(foot_shape)
            self.hip_limit_joints.append(hip_limit)
            self.knee_limit_joints.append(knee_limit)

            bodies.extend([thigh_body, shank_body])
            shapes.extend([thigh_shape, shank_shape, foot_shape])
            constraints.extend([hip_pivot, hip_limit, knee_pivot, knee_limit])

        self.space.add(*bodies, *shapes, *constraints)

    def _register_collision_callbacks(self):
        self.space.on_collision(
            GROUND_COLLISION_TYPE,
            LINK_COLLISION_TYPE,
            post_solve=self._on_ground_post_solve,
        )
        for leg_idx, collision_type in enumerate(FOOT_COLLISION_TYPES):
            self.space.on_collision(
                GROUND_COLLISION_TYPE,
                collision_type,
                begin=self._make_foot_begin_cb(leg_idx),
                separate=self._make_foot_separate_cb(leg_idx),
                post_solve=self._on_ground_post_solve,
            )

    def _make_velocity_func(self, linear_rate: float, angular_rate: float):
        def _velocity_func(body: pymunk.Body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            if linear_rate > 0.0:
                body.velocity = body.velocity * max(0.0, 1.0 - linear_rate * dt)
            if angular_rate > 0.0:
                body.angular_velocity = body.angular_velocity * max(
                    0.0, 1.0 - angular_rate * dt
                )

        return _velocity_func

    def _make_foot_begin_cb(self, leg_idx: int):
        def _begin(_arbiter, _space, _data):
            self._foot_contact_count[leg_idx] += 1
            return True

        return _begin

    def _make_foot_separate_cb(self, leg_idx: int):
        def _separate(_arbiter, _space, _data):
            self._foot_contact_count[leg_idx] = max(0, self._foot_contact_count[leg_idx] - 1)

        return _separate

    def _on_ground_post_solve(self, arbiter: pymunk.Arbiter, _space, _data):
        self._frame_impulse += arbiter.total_impulse

    def _body_local_to_world(self, local_point: np.ndarray) -> np.ndarray:
        return np.asarray(self.body.local_to_world(tuple(local_point)), dtype=float)

    def _point_velocity_from_body(self, world_point: np.ndarray) -> np.ndarray:
        return np.asarray(self.body.velocity_at_world_point(tuple(world_point)), dtype=float)

    def _apply_joint_torques(self):
        cfg = self.cfg
        torques = np.zeros((cfg.n_legs, 2), dtype=float)
        for leg_idx in range(cfg.n_legs):
            thigh = self.thigh_bodies[leg_idx]
            shank = self.shank_bodies[leg_idx]

            hip_angle = float(thigh.angle - self.body.angle)
            knee_angle = float(shank.angle - thigh.angle)
            hip_omega = float(thigh.angular_velocity - self.body.angular_velocity)
            knee_omega = float(shank.angular_velocity - thigh.angular_velocity)

            hip_tau = (
                cfg.hip_kp * (self.hip_targets[leg_idx] - hip_angle)
                - cfg.hip_kd * hip_omega
            )
            knee_tau = (
                cfg.knee_kp * (self.knee_targets[leg_idx] - knee_angle)
                - cfg.knee_kd * knee_omega
            )
            hip_tau = float(
                np.clip(hip_tau, -cfg.hip_torque_limit, cfg.hip_torque_limit)
                - cfg.joint_drag * hip_omega
            )
            knee_tau = float(
                np.clip(knee_tau, -cfg.knee_torque_limit, cfg.knee_torque_limit)
                - cfg.joint_drag * knee_omega
            )

            self.body.torque -= hip_tau
            thigh.torque += hip_tau - knee_tau
            shank.torque += knee_tau
            torques[leg_idx, 0] = hip_tau
            torques[leg_idx, 1] = knee_tau
        return torques

    def step(self, hip_target: np.ndarray | None = None, knee_target: np.ndarray | None = None):
        if hip_target is not None and knee_target is not None:
            self.set_targets(hip_target, knee_target)
        self._frame_impulse = pymunk.Vec2d(0.0, 0.0)
        torque_accum = np.zeros((self.cfg.n_legs, 2), dtype=float)
        for _ in range(self.substeps):
            torque_accum += self._apply_joint_torques()
            self.space.step(self.sub_dt)
        self.last_joint_torque = torque_accum / float(self.substeps)

    def observe(self) -> PassiveWalkerObservation:
        cfg = self.cfg
        foot_pos = np.zeros((cfg.n_legs, 2), dtype=float)
        foot_vel = np.zeros((cfg.n_legs, 2), dtype=float)
        hip_angle = np.zeros((cfg.n_legs,), dtype=float)
        knee_angle = np.zeros((cfg.n_legs,), dtype=float)
        hip_omega = np.zeros((cfg.n_legs,), dtype=float)
        knee_omega = np.zeros((cfg.n_legs,), dtype=float)

        for leg_idx in range(cfg.n_legs):
            thigh = self.thigh_bodies[leg_idx]
            shank = self.shank_bodies[leg_idx]
            foot_world = np.asarray(
                shank.local_to_world((0.0, -0.5 * cfg.shank_length)),
                dtype=float,
            )
            foot_pos[leg_idx] = foot_world
            foot_vel[leg_idx] = np.asarray(
                shank.velocity_at_world_point(tuple(foot_world)),
                dtype=float,
            )
            hip_angle[leg_idx] = float(thigh.angle - self.body.angle)
            knee_angle[leg_idx] = float(shank.angle - thigh.angle)
            hip_omega[leg_idx] = float(thigh.angular_velocity - self.body.angular_velocity)
            knee_omega[leg_idx] = float(shank.angular_velocity - thigh.angular_velocity)

        ground_contact = np.asarray(
            [1.0 if count > 0 else 0.0 for count in self._foot_contact_count],
            dtype=float,
        )
        for leg_idx in range(cfg.n_legs):
            if foot_pos[leg_idx, 1] - self.foot_radius <= self.contact_epsilon:
                ground_contact[leg_idx] = 1.0

        total_ground_force = -np.asarray(self._frame_impulse, dtype=float) / max(self.dt, 1e-9)

        return PassiveWalkerObservation(
            pos=np.asarray(self.body.position, dtype=float),
            vel=np.asarray(self.body.velocity, dtype=float),
            angle=float(self.body.angle),
            omega=float(self.body.angular_velocity),
            hip_angle=hip_angle,
            knee_angle=knee_angle,
            hip_omega=hip_omega,
            knee_omega=knee_omega,
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            ground_contact=ground_contact,
            total_ground_force=total_ground_force,
        )


__all__ = [
    "PassiveWalkerObservation",
    "PassiveWalkerState",
    "PymunkPassiveWalker",
    "initial_joint_configuration",
    "make_initial_state",
]
