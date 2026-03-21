import os
import sys

import numpy as np

import brainpy.math as bm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainable_system import (
    Config,
    TrainableWalkingSystem,
    build_feature_sequence,
    collect_rollout,
)


def main():
    cfg = Config(
        rho=2000,
        episode_ms=3.0,
        dt_ms=1.0,
        vis_every=0,
        eval_every=1,
        train_epochs=1,
    )
    bm.set_dt(cfg.dt_ms)

    system = TrainableWalkingSystem(cfg)
    num_steps = int(cfg.episode_ms / cfg.dt_ms)
    features = build_feature_sequence(
        num_steps,
        cfg.dt_ms,
        cfg.target_vx,
        cfg.target_vy,
        cfg.base_freq_hz,
    )
    runner = collect_rollout(system, features)

    pos = np.asarray(runner.mon["pos"], dtype=float)
    foot_pos = np.asarray(runner.mon["foot_pos"], dtype=float)
    contact = np.asarray(runner.mon["ground_contact"], dtype=float)

    assert pos.shape == (num_steps, 2)
    assert foot_pos.shape == (num_steps, cfg.n_legs, 2)
    assert contact.shape == (num_steps, cfg.n_legs)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(foot_pos))
    assert np.all(np.isfinite(contact))

    return True


if __name__ == "__main__":
    main()
