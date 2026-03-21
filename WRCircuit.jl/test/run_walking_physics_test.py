import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_walking_physics import PassiveWalkerConfig, simulate_passive_walker


def _min_body_clearance(cfg: PassiveWalkerConfig, pos: np.ndarray, angle: np.ndarray) -> float:
    bottom_corners_local = np.array(
        [
            [0.5 * cfg.body_length, -0.5 * cfg.body_height],
            [-0.5 * cfg.body_length, -0.5 * cfg.body_height],
        ],
        dtype=float,
    )

    def rotation_matrix(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    min_clearance = np.inf
    for body_pos, body_angle in zip(pos, angle):
        corners = bottom_corners_local @ rotation_matrix(float(body_angle)).T
        corners = corners + body_pos[None, :]
        min_clearance = min(min_clearance, float(np.min(corners[:, 1]) - cfg.body_corner_radius))
    return float(min_clearance)


def main():
    cfg = PassiveWalkerConfig(duration_ms=1500.0, dt_ms=2.0)
    rollout = simulate_passive_walker(cfg)

    foot_y = np.asarray(rollout["foot_pos"][:, :, 1], dtype=float)
    foot_penetration = np.max(np.maximum(cfg.foot_radius - foot_y, 0.0))
    body_clearance = _min_body_clearance(
        cfg,
        np.asarray(rollout["pos"], dtype=float),
        np.asarray(rollout["angle"], dtype=float),
    )
    final_speed = float(np.linalg.norm(np.asarray(rollout["vel"][-1], dtype=float)))
    contact_counts = np.asarray(rollout["ground_contact"], dtype=float).sum(axis=0)

    assert foot_penetration < 5e-3
    assert body_clearance > -5e-3
    assert final_speed < 5e-2
    assert np.all(contact_counts > 0.0)

    return True


if __name__ == "__main__":
    main()
