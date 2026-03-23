from __future__ import annotations

import multiprocessing as mp
from typing import Optional

import numpy as np

from trainable_system import (
    AdamState,
    Config,
    RolloutRunner,
    TerminalProgressBar,
    TrainingViewer,
    TrainableWalkingSystem as _BaseTrainableWalkingSystem,
    _ensure_writable_runtime_dirs,
    _log,
    _metrics_summary,
    _queue_put_latest,
    _require_runtime as _require_base_runtime,
    _tree_global_norm,
    _tree_to_numpy,
    build_feature_sequence,
    jax,
    jnp,
)


_ensure_writable_runtime_dirs()


_SPATIAL_IMPORT_ERROR = None

try:
    import brainpy as bp
    import brainpy.math as bm

    from src.models.Spatial import Spatial
except Exception as exc:  # pragma: no cover - exercised only when runtime is missing.
    bp = None
    bm = None
    Spatial = None
    _SPATIAL_IMPORT_ERROR = exc


def _require_runtime():
    _require_base_runtime()
    if bp is None or bm is None or Spatial is None:
        raise ImportError(
            "trainable_spatial_system.py requires the spatial BrainPy runtime. "
            "Install at least: brainpy, jax, jaxlib, numpy, matplotlib. "
            f"Missing runtime pieces: brainpy/spatial ({_SPATIAL_IMPORT_ERROR})"
        )


def _compatible_spatial_connectivity(cfg: Config) -> dict[str, int]:
    ne = max(1, int(round(np.sqrt(cfg.rho * cfg.controller_dx_mm**2))))
    ne_total = ne * ne
    ni = max(1, int(round(ne_total / cfg.gamma)))

    requested = {
        "K_ee": 260,
        "K_ei": 340,
        "K_ie": 225,
        "K_ii": 290,
    }
    limits = {
        "K_ee": max(1, (ne_total * ne_total) // ne),
        "K_ei": max(1, ne_total),
        "K_ie": max(1, (ni * ne_total) // ne),
        "K_ii": max(1, ni),
    }
    return {name: int(min(requested[name], limits[name])) for name in requested}


class TrainableSpatialWalkingSystem(_BaseTrainableWalkingSystem):
    def _init_params(self, key):
        k1, k2 = jax.random.split(key, 2)
        scale_in = 1.0 / np.sqrt(max(1, self.obs_size))
        scale_out = 1.0 / np.sqrt(max(1, self.n_total))
        return {
            "w_in": jax.random.normal(k1, (self.obs_size, self.n_total)) * scale_in,
            "bias_in": jnp.zeros((self.n_total,), dtype=jnp.float32),
            "w_out": jax.random.normal(k2, (self.n_total, self.action_size)) * scale_out,
            "bias_out": jnp.zeros((self.action_size,), dtype=jnp.float32),
        }

    def __init__(self, cfg: Config):
        _require_runtime()
        super().__init__(cfg)
        self.compute_backend = "pymunk+spatial"

        bm.set_dt(cfg.dt_ms)

        spatial_key = jax.random.PRNGKey(cfg.random_seed)
        self.spatial_connectivity = _compatible_spatial_connectivity(cfg)
        self.spatial_model = Spatial(
            key=spatial_key,
            rho=cfg.rho,
            dx=cfg.controller_dx_mm,
            gamma=cfg.gamma,
            **self.spatial_connectivity,
        )

        self.exc_side = int(np.asarray(self.spatial_model.E.size, dtype=int)[0])
        self.n_exc = int(np.prod(np.asarray(self.spatial_model.E.size, dtype=int)))
        self.n_inh = int(np.prod(np.asarray(self.spatial_model.I.size, dtype=int)))
        self.n_total = self.n_exc + self.n_inh

        self.E = self.spatial_model.E
        self.I = self.spatial_model.I
        self.exc_positions = np.asarray(self.E.positions, dtype=np.float32).reshape(self.n_exc, 2)
        self.inh_positions = np.asarray(self.I.positions, dtype=np.float32).reshape(self.n_inh, 2)
        self.all_positions = jnp.asarray(
            np.concatenate([self.exc_positions, self.inh_positions], axis=0),
            dtype=jnp.float32,
        )

        self.exc_mask = np.concatenate(
            [np.ones((self.n_exc,), dtype=np.float32), np.zeros((self.n_inh,), dtype=np.float32)],
            axis=0,
        )
        self.fixed_recurrent_weights = self._build_recurrent_weight_matrix()

        if self.params["w_in"].shape[1] != self.n_total:
            raise ValueError(
                "Spatial controller size mismatch. "
                f"Expected params width {self.n_total}, got {self.params['w_in'].shape[1]}."
            )

    @staticmethod
    def _shape_tuple(shape) -> tuple[int, ...]:
        values = np.atleast_1d(np.asarray(shape, dtype=int)).tolist()
        return tuple(int(v) for v in values)

    @staticmethod
    def _value_to_numpy(value) -> np.ndarray:
        if hasattr(value, "value"):
            value = value.value
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @staticmethod
    def _assign_input_var(input_var, value):
        target = getattr(input_var, "input", input_var)
        if hasattr(target, "value"):
            target.value = bm.asarray(value)
        else:
            target[...] = bm.asarray(value)

    @staticmethod
    def _csr_to_dense(indices, indptr, weights, num_pre: int, num_post: int) -> np.ndarray:
        dense = np.zeros((num_pre, num_post), dtype=np.float32)
        idx = np.asarray(indices, dtype=int)
        ptr = np.asarray(indptr, dtype=int)
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        for pre_idx in range(num_pre):
            start = ptr[pre_idx]
            stop = ptr[pre_idx + 1]
            dense[pre_idx, idx[start:stop]] = w[start:stop]
        return dense

    def _projection_matrix(self, proj, num_pre: int, num_post: int, sign: float = 1.0) -> np.ndarray:
        return sign * self._csr_to_dense(
            proj.proj.comm.indices,
            proj.proj.comm.indptr,
            proj.proj.comm.weight,
            num_pre=num_pre,
            num_post=num_post,
        )

    def _build_recurrent_weight_matrix(self) -> np.ndarray:
        weights = np.zeros((self.n_total, self.n_total), dtype=np.float32)
        weights[: self.n_exc, : self.n_exc] = self._projection_matrix(
            self.spatial_model.E2E,
            num_pre=self.n_exc,
            num_post=self.n_exc,
            sign=1.0,
        )
        weights[: self.n_exc, self.n_exc :] = self._projection_matrix(
            self.spatial_model.E2I,
            num_pre=self.n_exc,
            num_post=self.n_inh,
            sign=1.0,
        )
        weights[self.n_exc :, : self.n_exc] = self._projection_matrix(
            self.spatial_model.I2E,
            num_pre=self.n_inh,
            num_post=self.n_exc,
            sign=-1.0,
        )
        weights[self.n_exc :, self.n_exc :] = self._projection_matrix(
            self.spatial_model.I2I,
            num_pre=self.n_inh,
            num_post=self.n_inh,
            sign=-1.0,
        )
        return weights

    def _effective_recurrent_weights(self, params):
        del params
        return self.fixed_recurrent_weights

    def _spatial_snapshot(self, filt: np.ndarray, step: int) -> dict[str, np.ndarray]:
        spike = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.spike),
                self._value_to_numpy(self.spatial_model.I.spike),
            ],
            axis=0,
        )
        membrane = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.V),
                self._value_to_numpy(self.spatial_model.I.V),
            ],
            axis=0,
        )
        syn_current = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.input),
                self._value_to_numpy(self.spatial_model.I.input),
            ],
            axis=0,
        )
        adaptation = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.g_K),
                self._value_to_numpy(self.spatial_model.I.g_K),
            ],
            axis=0,
        )
        return {
            "step": int(step),
            "v": membrane.astype(np.float32),
            "syn": syn_current.astype(np.float32),
            "adapt": adaptation.astype(np.float32),
            "spike": spike.astype(np.float32),
            "filt": np.asarray(filt, dtype=np.float32),
        }

    def _controller_initial_state(self):
        bp.reset_state(self.spatial_model)
        filt = np.zeros((self.n_total,), dtype=np.float32)
        return self._spatial_snapshot(filt=filt, step=0)

    def _controller_step(self, params, ctrl_state, obs):
        obs = np.asarray(obs, dtype=np.float32)
        drive = self.cfg.input_gain * (obs @ params["w_in"]) + params["bias_in"]
        e_drive = drive[: self.n_exc].reshape(self._shape_tuple(self.spatial_model.E.size))
        i_drive = drive[self.n_exc :].reshape(self._shape_tuple(self.spatial_model.I.size))

        self._assign_input_var(self.spatial_model.Ein, e_drive.astype(np.float32))
        self._assign_input_var(self.spatial_model.Iin, i_drive.astype(np.float32))
        self.spatial_model.step_run(int(ctrl_state["step"]))

        spike = np.concatenate(
            [
                self._value_to_numpy(self.spatial_model.E.spike),
                self._value_to_numpy(self.spatial_model.I.spike),
            ],
            axis=0,
        ).astype(np.float32)
        filt = self.readout_decay * np.asarray(ctrl_state["filt"], dtype=np.float32) + spike
        action = np.tanh(self.cfg.readout_gain * (filt @ params["w_out"] + params["bias_out"]))

        next_state = self._spatial_snapshot(filt=filt, step=int(ctrl_state["step"]) + 1)
        return next_state, action.astype(np.float32)


TrainableWalkingSystem = TrainableSpatialWalkingSystem


def collect_rollout(
    system: TrainableWalkingSystem, features: Optional[np.ndarray] = None
) -> RolloutRunner:
    rollout, _ = system.evaluate(features)
    return RolloutRunner(mon=rollout)


def _training_worker(cfg: Config, features: np.ndarray, message_queue, stop_event):
    try:
        _log("spatial backend worker booting")
        system = TrainableWalkingSystem(cfg)
        refresh_every = cfg.vis_every if cfg.vis_every > 0 else cfg.eval_every
        train_forever = cfg.train_epochs <= 0
        total_epochs_label = "inf" if train_forever else str(cfg.train_epochs)
        epoch_progress = (
            None if train_forever else TerminalProgressBar(max(1, cfg.train_epochs), "training epochs")
        )
        _log(
            f"spatial backend ready: epochs={total_epochs_label} refresh_every={refresh_every} "
            f"es_population={cfg.es_population}"
        )

        _queue_put_latest(
            message_queue,
            {
                "type": "status",
                "phase": "evaluating",
                "detail": "running initial rollout",
            },
        )
        _log("starting initial spatial rollout for viewer bootstrap")
        rollout, metrics = system.evaluate_with_progress(
            features,
            progress_label="initial rollout",
        )
        _log(f"initial spatial rollout complete {_metrics_summary(metrics)}")
        _queue_put_latest(
            message_queue,
            {
                "type": "snapshot",
                "epoch": 0,
                "rollout": rollout,
                "metrics": metrics,
                "params": system.params,
                "phase": "idle",
                "detail": "ready to train",
            },
        )

        epoch = 0
        while not stop_event.is_set() and (train_forever or epoch < cfg.train_epochs):
            _queue_put_latest(
                message_queue,
                {
                    "type": "status",
                    "phase": "training",
                    "detail": "running optimizer step",
                },
            )
            _log(f"epoch {epoch + 1}/{total_epochs_label}: starting spatial optimizer step")
            latest_metrics = system.train_step(
                features,
                progress_prefix=f"epoch {epoch + 1}/{total_epochs_label}",
            )
            epoch += 1
            if epoch_progress is not None:
                epoch_progress.update(epoch)
            _log(
                f"epoch {epoch}/{total_epochs_label}: spatial optimizer step complete "
                f"{_metrics_summary(latest_metrics)}"
            )

            _queue_put_latest(
                message_queue,
                {
                    "type": "metrics",
                    "epoch": epoch,
                    "metrics": latest_metrics,
                    "phase": "training",
                    "detail": "optimizer step complete",
                },
            )

            should_refresh = (
                epoch == 1
                or epoch % max(1, refresh_every) == 0
                or (not train_forever and epoch >= cfg.train_epochs)
            )
            if should_refresh and not stop_event.is_set():
                _log(f"epoch {epoch}/{total_epochs_label}: refreshing spatial viewer rollout")
                _queue_put_latest(
                    message_queue,
                    {
                        "type": "status",
                        "phase": "evaluating",
                        "detail": "refreshing rollout for UI",
                    },
                )
                rollout, eval_metrics = system.evaluate_with_progress(
                    features,
                    progress_label="refresh rollout",
                )
                _log(
                    f"epoch {epoch}/{total_epochs_label}: refresh spatial rollout complete "
                    f"{_metrics_summary(eval_metrics)}"
                )
                _queue_put_latest(
                    message_queue,
                    {
                        "type": "snapshot",
                        "epoch": epoch,
                        "rollout": rollout,
                        "metrics": eval_metrics,
                        "params": system.params,
                        "phase": "idle",
                        "detail": "snapshot ready",
                    },
                )

        if epoch_progress is not None:
            epoch_progress.finish()
        _log(f"spatial training worker finished at epoch {epoch}/{total_epochs_label}")
        _queue_put_latest(
            message_queue,
            {
                "type": "done",
                "epoch": epoch,
                "phase": "done",
                "detail": "training stopped" if stop_event.is_set() or train_forever else "training complete",
            },
        )
    except Exception as exc:  # pragma: no cover - worker-side runtime path.
        _log(f"spatial backend worker failed: {exc}")
        _queue_put_latest(
            message_queue,
            {
                "type": "error",
                "phase": "error",
                "detail": str(exc),
            },
        )


def main():
    _require_runtime()
    print("[trainable_spatial_system] building system...", flush=True)
    cfg = Config()
    system = TrainableWalkingSystem(cfg)
    print(
        "[trainable_spatial_system] system ready "
        f"(steps={system.num_steps}, exc={system.n_exc}, inh={system.n_inh}, "
        f"backend={system.compute_backend})",
        flush=True,
    )
    features = build_feature_sequence(
        system.num_steps,
        cfg.dt_ms,
        cfg.target_vx,
        cfg.target_vy,
    )
    ctx = mp.get_context("spawn")
    message_queue = ctx.Queue(maxsize=4)
    stop_event = ctx.Event()
    worker_process = ctx.Process(
        target=_training_worker,
        args=(cfg, features, message_queue, stop_event),
        daemon=True,
    )
    print("[trainable_spatial_system] starting backend worker...", flush=True)
    worker_process.start()
    print("[trainable_spatial_system] opening training viewer...", flush=True)
    viewer = TrainingViewer(system, features, message_queue, stop_event, worker_process)
    try:
        viewer.show()
    finally:
        stop_event.set()
        worker_process.join(timeout=5.0)
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=2.0)


__all__ = [
    "AdamState",
    "Config",
    "RolloutRunner",
    "TrainableSpatialWalkingSystem",
    "TrainableWalkingSystem",
    "build_feature_sequence",
    "collect_rollout",
    "main",
]


if __name__ == "__main__":
    main()
