from dataclasses import dataclass
import os
import sys

import numpy as np

import brainpy.math as bm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from walking_physics import WalkerPhysics


@dataclass
class DummyConfig:
    n_legs: int = 2
    mass: float = 2.0
    thigh_mass: float = 0.25
    shank_mass: float = 0.18
    drag: float = 0.15
    angular_drag: float = 1.5
    joint_drag: float = 0.2
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
    height_target: float = 0.48
    hip_limit: float = 0.95
    knee_min: float = 0.05
    knee_max: float = 1.45
    hip_kp: float = 18.0
    hip_kd: float = 1.8
    knee_kp: float = 24.0
    knee_kd: float = 2.2
    hip_torque_limit: float = 12.0
    knee_torque_limit: float = 14.0


def main():
    bm.set_dt(1.0)
    cfg = DummyConfig()
    physics = WalkerPhysics(cfg)

    state = physics.initial_state()
    sensed = physics.sense(state)
    step = physics.step(state, state.hip_angle, state.knee_angle)

    assert np.asarray(state.pos).shape == (2,)
    assert np.asarray(sensed.foot_pos).shape == (cfg.n_legs, 2)
    assert np.asarray(sensed.foot_vel).shape == (cfg.n_legs, 2)
    assert np.asarray(sensed.ground_contact).shape == (cfg.n_legs,)
    assert np.asarray(step.state.pos).shape == (2,)
    assert np.asarray(step.joint_torque).shape == (cfg.n_legs, 2)

    assert np.all(np.isfinite(np.asarray(step.state.pos)))
    assert np.all(np.isfinite(np.asarray(step.state.vel)))
    assert np.all(np.isfinite(np.asarray(step.foot_pos)))
    assert np.all(np.isfinite(np.asarray(step.foot_vel)))
    assert np.all(np.isfinite(np.asarray(step.total_ground_force)))
    assert np.allclose(np.asarray(step.joint_torque), 0.0, atol=1e-6)
    assert np.all(np.asarray(step.state.knee_angle) >= cfg.knee_min - 1e-6)
    assert np.all(np.asarray(step.state.knee_angle) <= cfg.knee_max + 1e-6)
    assert np.all(np.abs(np.asarray(step.state.hip_angle)) <= cfg.hip_limit + 1e-6)

    return True


if __name__ == "__main__":
    main()
