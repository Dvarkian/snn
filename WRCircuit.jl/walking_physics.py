"""
Walking robot physics simulation for the 2D articulated walker.

This module isolates the rigid-body dynamics and ground-contact model used by
``TrainableWalkingSystem`` so the controller and visualization code can stay
focused on the neural network and rollout tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import jax

import brainpy.math as bm


@dataclass(frozen=True)
class WalkerState:
    pos: Any
    vel: Any
    angle: Any
    omega: Any
    hip_angle: Any
    knee_angle: Any
    hip_omega: Any
    knee_omega: Any


@dataclass(frozen=True)
class WalkerSensors:
    foot_pos: Any
    foot_vel: Any
    ground_contact: Any


@dataclass(frozen=True)
class WalkerStepResult:
    state: WalkerState
    foot_pos: Any
    foot_vel: Any
    ground_contact: Any
    total_ground_force: Any
    joint_torque: Any


class WalkerPhysics:
    """Rigid-body dynamics for a planar two-link walker with compliant contacts."""

    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.n_legs != 2:
            raise ValueError(
                "WalkerPhysics currently implements a front-leg/back-leg walker "
                "and expects n_legs == 2."
            )

        self.n_legs = cfg.n_legs
        self.body_inertia = cfg.mass * (
            cfg.body_length**2 + cfg.body_height**2
        ) / 12.0
        self.thigh_inertia = cfg.thigh_mass * cfg.thigh_length**2 / 12.0
        self.shank_inertia = cfg.shank_mass * cfg.shank_length**2 / 12.0

        hip_y = -0.5 * cfg.body_height
        self.hip_local = bm.asarray(
            [
                [cfg.hip_x_offset, hip_y],
                [-cfg.hip_x_offset, hip_y],
            ]
        )
        self.bottom_corners_local = bm.asarray(
            [
                [0.5 * cfg.body_length, -0.5 * cfg.body_height],
                [-0.5 * cfg.body_length, -0.5 * cfg.body_height],
            ]
        )

    def _rotation_matrix(self, theta):
        c = bm.cos(theta)
        s = bm.sin(theta)
        return bm.asarray([[c, -s], [s, c]])

    def _world_points(self, local_points, pos, theta):
        rot = self._rotation_matrix(theta)
        return local_points @ rot.T + pos[None, :]

    def _segment_dir(self, angle):
        return bm.stack([bm.sin(angle), -bm.cos(angle)], axis=-1)

    def _pack_q(self, state: WalkerState):
        return bm.concatenate(
            [
                state.pos,
                bm.asarray([state.angle]),
                state.hip_angle,
                state.knee_angle,
            ],
            axis=0,
        )

    def _pack_qd(self, state: WalkerState):
        return bm.concatenate(
            [
                state.vel,
                bm.asarray([state.omega]),
                state.hip_omega,
                state.knee_omega,
            ],
            axis=0,
        )

    def _state_from_q_qd(self, q, qd) -> WalkerState:
        return WalkerState(
            pos=q[:2],
            vel=qd[:2],
            angle=q[2],
            omega=qd[2],
            hip_angle=q[3 : 3 + self.n_legs],
            knee_angle=q[3 + self.n_legs : 3 + 2 * self.n_legs],
            hip_omega=qd[3 : 3 + self.n_legs],
            knee_omega=qd[3 + self.n_legs : 3 + 2 * self.n_legs],
        )

    def _body_com(self, q):
        return q[:2]

    def _body_corners(self, q):
        pos = q[:2]
        angle = q[2]
        return self._world_points(self.bottom_corners_local, pos, angle)

    def _hip_positions(self, q):
        pos = q[:2]
        angle = q[2]
        return self._world_points(self.hip_local, pos, angle)

    def _thigh_abs_angles(self, q):
        angle = q[2]
        hip_angle = q[3 : 3 + self.n_legs]
        return angle + hip_angle

    def _shank_abs_angles(self, q):
        angle = q[2]
        hip_angle = q[3 : 3 + self.n_legs]
        knee_angle = q[3 + self.n_legs : 3 + 2 * self.n_legs]
        return angle + hip_angle + knee_angle

    def _knee_positions(self, q):
        hip_pos = self._hip_positions(q)
        thigh_dir = self._segment_dir(self._thigh_abs_angles(q))
        return hip_pos + self.cfg.thigh_length * thigh_dir

    def _foot_positions(self, q):
        knee_pos = self._knee_positions(q)
        shank_dir = self._segment_dir(self._shank_abs_angles(q))
        return knee_pos + self.cfg.shank_length * shank_dir

    def _thigh_com_positions(self, q):
        hip_pos = self._hip_positions(q)
        thigh_dir = self._segment_dir(self._thigh_abs_angles(q))
        return hip_pos + 0.5 * self.cfg.thigh_length * thigh_dir

    def _shank_com_positions(self, q):
        knee_pos = self._knee_positions(q)
        shank_dir = self._segment_dir(self._shank_abs_angles(q))
        return knee_pos + 0.5 * self.cfg.shank_length * shank_dir

    def _ground_forces(self, points, velocities):
        penetration = bm.relu(-points[:, 1])
        normal = self.cfg.ground_k * penetration - self.cfg.ground_c * bm.minimum(
            velocities[:, 1], 0.0
        )
        normal = bm.maximum(normal, 0.0)
        max_tangent = self.cfg.friction_mu * normal
        tangent = -bm.clip(
            self.cfg.ground_tangent_damping * velocities[:, 0],
            -max_tangent,
            max_tangent,
        )
        contact = bm.where(penetration > 1e-6, 1.0, 0.0)
        return bm.stack([tangent, normal], axis=1), contact

    def _point_velocities_from_jacobian(self, jacobian, qd):
        return bm.einsum("aij,j->ai", jacobian, qd)

    def _kinetic_energy(self, q, qd):
        body_vel = qd[:2]
        body_omega = qd[2]

        thigh_jac = jax.jacobian(self._thigh_com_positions)(q)
        shank_jac = jax.jacobian(self._shank_com_positions)(q)
        thigh_vel = self._point_velocities_from_jacobian(thigh_jac, qd)
        shank_vel = self._point_velocities_from_jacobian(shank_jac, qd)

        thigh_angle_jac = jax.jacobian(self._thigh_abs_angles)(q)
        shank_angle_jac = jax.jacobian(self._shank_abs_angles)(q)
        thigh_omega = thigh_angle_jac @ qd
        shank_omega = shank_angle_jac @ qd

        body_ke = 0.5 * self.cfg.mass * bm.sum(body_vel**2)
        body_ke += 0.5 * self.body_inertia * body_omega**2

        thigh_ke = 0.5 * self.cfg.thigh_mass * bm.sum(thigh_vel**2)
        thigh_ke += 0.5 * self.thigh_inertia * bm.sum(thigh_omega**2)

        shank_ke = 0.5 * self.cfg.shank_mass * bm.sum(shank_vel**2)
        shank_ke += 0.5 * self.shank_inertia * bm.sum(shank_omega**2)
        return body_ke + thigh_ke + shank_ke

    def _potential_energy(self, q):
        body_com = self._body_com(q)
        thigh_com = self._thigh_com_positions(q)
        shank_com = self._shank_com_positions(q)

        pe = self.cfg.mass * self.cfg.gravity * body_com[1]
        pe += self.cfg.thigh_mass * self.cfg.gravity * bm.sum(thigh_com[:, 1])
        pe += self.cfg.shank_mass * self.cfg.gravity * bm.sum(shank_com[:, 1])
        return pe

    def _mass_and_bias(self, q, qd):
        kinetic = lambda qq, vv: self._kinetic_energy(qq, vv)
        mass_matrix = jax.jacobian(
            lambda vv: jax.grad(lambda ww: kinetic(q, ww))(vv)
        )(qd)
        d_dq_dT_dqd = jax.jacobian(
            lambda qq: jax.grad(lambda vv: kinetic(qq, vv))(qd)
        )(q)
        dT_dq = jax.grad(lambda qq: kinetic(qq, qd))(q)
        dV_dq = jax.grad(self._potential_energy)(q)
        bias = d_dq_dT_dqd @ qd - dT_dq + dV_dq
        return mass_matrix, bias

    def _contact_generalized_force(self, q, qd):
        foot_pos = self._foot_positions(q)
        foot_jac = jax.jacobian(self._foot_positions)(q)
        foot_vel = self._point_velocities_from_jacobian(foot_jac, qd)
        foot_force, foot_contact = self._ground_forces(foot_pos, foot_vel)

        corner_pos = self._body_corners(q)
        corner_jac = jax.jacobian(self._body_corners)(q)
        corner_vel = self._point_velocities_from_jacobian(corner_jac, qd)
        corner_force, _ = self._ground_forces(corner_pos, corner_vel)

        generalized = bm.einsum("aij,ai->j", foot_jac, foot_force)
        generalized = generalized + bm.einsum("aij,ai->j", corner_jac, corner_force)
        total_ground_force = bm.sum(foot_force, axis=0) + bm.sum(corner_force, axis=0)
        return generalized, foot_pos, foot_vel, foot_contact, total_ground_force

    def initial_joint_configuration(self):
        desired_reach = self.cfg.height_target - 0.5 * self.cfg.body_height
        desired_reach = float(
            np.clip(
                desired_reach,
                abs(self.cfg.thigh_length - self.cfg.shank_length) + 1e-4,
                self.cfg.thigh_length + self.cfg.shank_length - 1e-4,
            )
        )
        cos_knee = (
            desired_reach**2
            - self.cfg.thigh_length**2
            - self.cfg.shank_length**2
        ) / (2.0 * self.cfg.thigh_length * self.cfg.shank_length)
        cos_knee = float(np.clip(cos_knee, -1.0, 1.0))
        knee = float(np.arccos(cos_knee))
        hip = float(
            -np.arctan2(
                self.cfg.shank_length * np.sin(knee),
                self.cfg.thigh_length + self.cfg.shank_length * np.cos(knee),
            )
        )
        hip = float(np.clip(hip, -self.cfg.hip_limit, self.cfg.hip_limit))
        knee = float(np.clip(knee, self.cfg.knee_min, self.cfg.knee_max))
        return hip, knee

    def initial_state(self) -> WalkerState:
        hip0, knee0 = self.initial_joint_configuration()
        return WalkerState(
            pos=bm.asarray([0.0, self.cfg.height_target]),
            vel=bm.zeros((2,)),
            angle=bm.asarray(0.0),
            omega=bm.asarray(0.0),
            hip_angle=bm.zeros((self.n_legs,)) + hip0,
            knee_angle=bm.zeros((self.n_legs,)) + knee0,
            hip_omega=bm.zeros((self.n_legs,)),
            knee_omega=bm.zeros((self.n_legs,)),
        )

    def sense(self, state: WalkerState) -> WalkerSensors:
        q = self._pack_q(state)
        qd = self._pack_qd(state)
        foot_pos = self._foot_positions(q)
        foot_jac = jax.jacobian(self._foot_positions)(q)
        foot_vel = self._point_velocities_from_jacobian(foot_jac, qd)
        _, ground_contact = self._ground_forces(foot_pos, foot_vel)
        return WalkerSensors(
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            ground_contact=ground_contact,
        )

    def step(self, state: WalkerState, hip_target, knee_target) -> WalkerStepResult:
        q = self._pack_q(state)
        qd = self._pack_qd(state)

        mass_matrix, bias = self._mass_and_bias(q, qd)
        contact_force, _foot_pos, _foot_vel, _ground_contact, total_ground_force = (
            self._contact_generalized_force(q, qd)
        )

        hip_torque = self.cfg.hip_kp * (hip_target - state.hip_angle)
        hip_torque = hip_torque - self.cfg.hip_kd * state.hip_omega
        hip_torque = bm.clip(
            hip_torque, -self.cfg.hip_torque_limit, self.cfg.hip_torque_limit
        )

        knee_torque = self.cfg.knee_kp * (knee_target - state.knee_angle)
        knee_torque = knee_torque - self.cfg.knee_kd * state.knee_omega
        knee_torque = bm.clip(
            knee_torque, -self.cfg.knee_torque_limit, self.cfg.knee_torque_limit
        )

        generalized_torque = bm.concatenate(
            [
                bm.asarray(
                    [
                        -self.cfg.drag * state.vel[0],
                        -self.cfg.drag * state.vel[1],
                        -self.cfg.angular_drag * state.omega,
                    ]
                ),
                hip_torque - self.cfg.joint_drag * state.hip_omega,
                knee_torque - self.cfg.joint_drag * state.knee_omega,
            ],
            axis=0,
        )

        reg = 1e-5 * bm.eye(mass_matrix.shape[0])
        qdd = bm.linalg.solve(
            mass_matrix + reg, generalized_torque + contact_force - bias
        )

        dt = bm.get_dt() / 1000.0
        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        q_new = q_new.at[3 : 3 + self.n_legs].set(
            bm.clip(q_new[3 : 3 + self.n_legs], -self.cfg.hip_limit, self.cfg.hip_limit)
        )
        q_new = q_new.at[3 + self.n_legs : 3 + 2 * self.n_legs].set(
            bm.clip(
                q_new[3 + self.n_legs : 3 + 2 * self.n_legs],
                self.cfg.knee_min,
                self.cfg.knee_max,
            )
        )

        qd_new = qd_new.at[3 : 3 + self.n_legs].set(
            bm.where(
                bm.logical_or(
                    bm.logical_and(
                        q_new[3 : 3 + self.n_legs] <= -self.cfg.hip_limit + 1e-6,
                        qd_new[3 : 3 + self.n_legs] < 0.0,
                    ),
                    bm.logical_and(
                        q_new[3 : 3 + self.n_legs] >= self.cfg.hip_limit - 1e-6,
                        qd_new[3 : 3 + self.n_legs] > 0.0,
                    ),
                ),
                0.0,
                qd_new[3 : 3 + self.n_legs],
            )
        )
        qd_new = qd_new.at[3 + self.n_legs : 3 + 2 * self.n_legs].set(
            bm.where(
                bm.logical_or(
                    bm.logical_and(
                        q_new[3 + self.n_legs : 3 + 2 * self.n_legs]
                        <= self.cfg.knee_min + 1e-6,
                        qd_new[3 + self.n_legs : 3 + 2 * self.n_legs] < 0.0,
                    ),
                    bm.logical_and(
                        q_new[3 + self.n_legs : 3 + 2 * self.n_legs]
                        >= self.cfg.knee_max - 1e-6,
                        qd_new[3 + self.n_legs : 3 + 2 * self.n_legs] > 0.0,
                    ),
                ),
                0.0,
                qd_new[3 + self.n_legs : 3 + 2 * self.n_legs],
            )
        )

        next_state = self._state_from_q_qd(q_new, qd_new)
        sensed = self.sense(next_state)
        return WalkerStepResult(
            state=next_state,
            foot_pos=sensed.foot_pos,
            foot_vel=sensed.foot_vel,
            ground_contact=sensed.ground_contact,
            total_ground_force=total_ground_force,
            joint_torque=bm.stack([hip_torque, knee_torque], axis=1),
        )


__all__ = [
    "WalkerPhysics",
    "WalkerSensors",
    "WalkerState",
    "WalkerStepResult",
]
