import os
import sys

import numpy as np

import brainpy.math as bm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainable_system import Config, build_feature_sequence
from trainable_spatial_bptt_system import TrainableSpatialBPTTWalkingSystem


def _assert_finite_metrics(metrics):
    for key, value in metrics.items():
        arr = np.asarray(value, dtype=float)
        assert np.all(np.isfinite(arr)), f"metric {key} contains non-finite values: {value}"


def main():
    cfg = Config(
        random_seed=3,
        rho=2000,
        episode_ms=12.0,
        dt_ms=4.0,
        train_epochs=1,
        vis_every=0,
        eval_every=1,
        learning_rate=1e-3,
        gradient_clip=1.0,
    )
    bm.set_dt(cfg.dt_ms)

    system = TrainableSpatialBPTTWalkingSystem(cfg)
    features = build_feature_sequence(
        system.num_steps,
        cfg.dt_ms,
        cfg.target_vx,
        cfg.target_vy,
    )

    initial_fixed = {
        key: np.array(system.params[key], copy=True)
        for key in ("w_in", "bias_in", "w_out", "bias_out")
    }
    initial_recurrent = {
        key: np.array(system.params[key], copy=True)
        for key in ("w_ee_raw", "w_ei_raw", "w_ie_raw", "w_ii_raw")
    }

    rollout, metrics = system.evaluate(features)
    assert "pos" in rollout and "action" in rollout
    _assert_finite_metrics(metrics)

    system.rollout_model.reset_state()
    spike_e = np.asarray(system.rollout_model.spatial_model.E.spike.value)
    spike_i = np.asarray(system.rollout_model.spatial_model.I.spike.value)
    assert spike_e.dtype.kind == "f"
    assert spike_i.dtype.kind == "f"
    assert np.allclose(spike_e, 0.0)
    assert np.allclose(spike_i, 0.0)

    train_metrics = system.train_step(features)
    _assert_finite_metrics(train_metrics)

    updated_fixed = {
        key: np.array(system.params[key], copy=True)
        for key in ("w_in", "bias_in", "w_out", "bias_out")
    }
    updated_recurrent = {
        key: np.array(system.params[key], copy=True)
        for key in ("w_ee_raw", "w_ei_raw", "w_ie_raw", "w_ii_raw")
    }

    for key in initial_fixed:
        np.testing.assert_allclose(updated_fixed[key], initial_fixed[key])

    changed = [
        not np.allclose(updated_recurrent[key], initial_recurrent[key])
        for key in initial_recurrent
    ]
    assert any(changed), "expected at least one recurrent weight tensor to change after train_step()"

    return True


def test_main():
    assert main()


if __name__ == "__main__":
    main()
