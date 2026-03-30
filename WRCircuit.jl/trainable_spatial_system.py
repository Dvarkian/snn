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


_RUN_SIMULATION_SPATIAL_SEED = 42
_RUN_SIMULATION_SPATIAL_RHO = 10000
_RUN_SIMULATION_SPATIAL_DX_MM = 1.0


class TrainableSpatialWalkingSystem(_BaseTrainableWalkingSystem):
    _TRAINABLE_RECURRENT_KEYS = ("w_ee_raw", "w_ei_raw", "w_ie_raw", "w_ii_raw")

    def _init_params(self, key):
        k1, k2 = jax.random.split(key, 2)
        scale_in = 1.0 / np.sqrt(max(1, self.obs_size))
        scale_out = 1.0 / np.sqrt(max(1, self.n_total))
        return {
            "w_in": jax.random.normal(k1, (self.obs_size, self.n_total)) * scale_in,
            "bias_in": jnp.zeros((self.n_total,), dtype=jnp.float32),
            "w_out": jax.random.normal(k2, (self.n_total, self.action_size)) * scale_out,
            "bias_out": jnp.zeros((self.action_size,), dtype=jnp.float32),
            "w_ee_raw": jnp.zeros((1,), dtype=jnp.float32),
            "w_ei_raw": jnp.zeros((1,), dtype=jnp.float32),
            "w_ie_raw": jnp.zeros((1,), dtype=jnp.float32),
            "w_ii_raw": jnp.zeros((1,), dtype=jnp.float32),
        }

    def __init__(self, cfg: Config):
        _require_runtime()
        super().__init__(cfg)
        self.compute_backend = "pymunk+spatial"

        bm.set_dt(cfg.dt_ms)

        spatial_key = jax.random.PRNGKey(_RUN_SIMULATION_SPATIAL_SEED)
        self.spatial_model = Spatial(
            key=spatial_key,
            rho=_RUN_SIMULATION_SPATIAL_RHO,
            dx=_RUN_SIMULATION_SPATIAL_DX_MM,
        )
        self.spatial_connectivity = {
            "K_ee": int(self.spatial_model.K_ee),
            "K_ei": int(self.spatial_model.K_ei),
            "K_ie": int(self.spatial_model.K_ie),
            "K_ii": int(self.spatial_model.K_ii),
        }
        self.cfg.rho = _RUN_SIMULATION_SPATIAL_RHO
        self.cfg.controller_dx_mm = _RUN_SIMULATION_SPATIAL_DX_MM
        self.cfg.gamma = float(self.spatial_model.gamma)

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
        self.recurrent_specs = self._build_recurrent_specs()
        self.params = self._init_spatial_params(cfg.random_seed)
        self.opt_state = self._adam_init(self._trainable_params(self.params))
        self._apply_recurrent_params(self.params)

        if self.params["w_in"].shape[1] != self.n_total:
            raise ValueError(
                "Spatial controller size mismatch. "
                f"Expected params width {self.n_total}, got {self.params['w_in'].shape[1]}."
            )

    @staticmethod
    def _value_to_numpy(value) -> np.ndarray:
        if hasattr(value, "value"):
            value = value.value
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @staticmethod
    def _softplus(value) -> np.ndarray:
        return np.logaddexp(np.asarray(value, dtype=np.float32), 0.0).astype(np.float32)

    @staticmethod
    def _inverse_softplus(value) -> np.ndarray:
        value = np.maximum(np.asarray(value, dtype=np.float32), 1e-6)
        large = value > 20.0
        raw = np.empty_like(value, dtype=np.float32)
        raw[large] = value[large]
        raw[~large] = np.log(np.expm1(value[~large])).astype(np.float32)
        return raw

    @staticmethod
    def _row_sparse_matrix(
        rng: np.random.Generator,
        num_rows: int,
        num_cols: int,
        density: float,
        scale: float,
    ) -> np.ndarray:
        count = min(num_cols, max(1, int(round(float(np.clip(density, 0.0, 1.0)) * num_cols))))
        weights = np.zeros((num_rows, num_cols), dtype=np.float32)
        for row_idx in range(num_rows):
            cols = rng.choice(num_cols, size=count, replace=False)
            weights[row_idx, cols] = scale * rng.standard_normal(count).astype(np.float32)
        return weights

    @staticmethod
    def _column_sparse_matrix(
        rng: np.random.Generator,
        num_rows: int,
        num_cols: int,
        density: float,
        scale: float,
    ) -> np.ndarray:
        count = min(num_rows, max(1, int(round(float(np.clip(density, 0.0, 1.0)) * num_rows))))
        weights = np.zeros((num_rows, num_cols), dtype=np.float32)
        for col_idx in range(num_cols):
            rows = rng.choice(num_rows, size=count, replace=False)
            weights[rows, col_idx] = scale * rng.standard_normal(count).astype(np.float32)
        return weights

    @staticmethod
    def _assign_input_var(input_var, value):
        target = getattr(input_var, "input", input_var)
        if hasattr(target, "value"):
            target_value = target.value
        else:
            target_value = target

        target_shape = np.asarray(target_value).shape
        value = np.asarray(value, dtype=np.float32).reshape(target_shape)
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

    def _recurrent_spec(self, proj, num_pre: int, num_post: int, sign: float) -> dict[str, object]:
        return {
            "proj": proj,
            "indices": np.asarray(proj.proj.comm.indices, dtype=int),
            "indptr": np.asarray(proj.proj.comm.indptr, dtype=int),
            "weights": np.asarray(proj.proj.comm.weight, dtype=np.float32).reshape(-1),
            "num_pre": int(num_pre),
            "num_post": int(num_post),
            "sign": float(sign),
        }

    def _build_recurrent_specs(self) -> dict[str, dict[str, object]]:
        return {
            "w_ee_raw": self._recurrent_spec(
                self.spatial_model.E2E,
                num_pre=self.n_exc,
                num_post=self.n_exc,
                sign=1.0,
            ),
            "w_ei_raw": self._recurrent_spec(
                self.spatial_model.E2I,
                num_pre=self.n_exc,
                num_post=self.n_inh,
                sign=1.0,
            ),
            "w_ie_raw": self._recurrent_spec(
                self.spatial_model.I2E,
                num_pre=self.n_inh,
                num_post=self.n_exc,
                sign=-1.0,
            ),
            "w_ii_raw": self._recurrent_spec(
                self.spatial_model.I2I,
                num_pre=self.n_inh,
                num_post=self.n_inh,
                sign=-1.0,
            ),
        }

    def _init_spatial_params(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed + 101)
        density = float(np.clip(self.cfg.conn_sparsity, 0.0, 1.0))
        scale_in = 1.0 / np.sqrt(max(1, self.obs_size))
        scale_out = 1.0 / np.sqrt(max(1, self.n_total))
        params = {
            "w_in": self._row_sparse_matrix(
                rng,
                num_rows=self.obs_size,
                num_cols=self.n_total,
                density=density,
                scale=scale_in,
            ),
            "bias_in": np.zeros((self.n_total,), dtype=np.float32),
            "w_out": self._column_sparse_matrix(
                rng,
                num_rows=self.n_total,
                num_cols=self.action_size,
                density=density,
                scale=scale_out,
            ),
            "bias_out": np.zeros((self.action_size,), dtype=np.float32),
        }
        for key, spec in self.recurrent_specs.items():
            params[key] = self._inverse_softplus(spec["weights"])
        return params

    def _trainable_params(self, params) -> dict[str, np.ndarray]:
        return {key: np.asarray(params[key], dtype=np.float32) for key in self._TRAINABLE_RECURRENT_KEYS}

    def _merge_trainable_params(self, base_params, trainable_params) -> dict[str, np.ndarray]:
        merged = {key: np.asarray(value, dtype=np.float32) for key, value in base_params.items()}
        for key, value in trainable_params.items():
            merged[key] = np.asarray(value, dtype=np.float32)
        return merged

    def _dense_projection_from_params(self, params, key: str) -> np.ndarray:
        spec = self.recurrent_specs[key]
        weights = self._softplus(params[key])
        return spec["sign"] * self._csr_to_dense(
            spec["indices"],
            spec["indptr"],
            weights,
            num_pre=spec["num_pre"],
            num_post=spec["num_post"],
        )

    def _apply_recurrent_params(self, params):
        for key, spec in self.recurrent_specs.items():
            spec["proj"].proj.comm.weight = bm.asarray(self._softplus(params[key]))

    def _effective_recurrent_weights(self, params):
        weights = np.zeros((self.n_total, self.n_total), dtype=np.float32)
        weights[: self.n_exc, : self.n_exc] = self._dense_projection_from_params(params, "w_ee_raw")
        weights[: self.n_exc, self.n_exc :] = self._dense_projection_from_params(params, "w_ei_raw")
        weights[self.n_exc :, : self.n_exc] = self._dense_projection_from_params(params, "w_ie_raw")
        weights[self.n_exc :, self.n_exc :] = self._dense_projection_from_params(params, "w_ii_raw")
        return weights

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

    def _simulate(self, params, features, progress_label: Optional[str] = None):
        self._apply_recurrent_params(params)
        return super()._simulate(params, features, progress_label=progress_label)

    def _controller_step(self, params, ctrl_state, obs):
        obs = np.asarray(obs, dtype=np.float32)
        drive = self.cfg.input_gain * (obs @ params["w_in"]) + params["bias_in"]
        e_drive = drive[: self.n_exc]
        i_drive = drive[self.n_exc :]

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

    def train_step(
        self,
        features: Optional[np.ndarray] = None,
        progress_prefix: Optional[str] = None,
    ) -> dict[str, float]:
        features = self._coerce_features(features)
        sigma = float(self.cfg.es_noise_std)
        population = max(1, int(self.cfg.es_population))
        trainable_params = self._trainable_params(self.params)
        grad_accum = jax.tree_util.tree_map(
            lambda p: np.zeros_like(p, dtype=np.float32),
            trainable_params,
        )
        step_prefix = progress_prefix or "optimizer step"

        _log(f"{step_prefix}: estimating recurrent-only gradient with {population} perturbation pairs")

        for sample_idx in range(population):
            sample_tag = f"{step_prefix} pair {sample_idx + 1}/{population}"
            noise = jax.tree_util.tree_map(
                lambda p: self.rng.standard_normal(p.shape).astype(np.float32),
                trainable_params,
            )
            trainable_plus = jax.tree_util.tree_map(
                lambda p, n: p + sigma * n,
                trainable_params,
                noise,
            )
            trainable_minus = jax.tree_util.tree_map(
                lambda p, n: p - sigma * n,
                trainable_params,
                noise,
            )
            params_plus = self._merge_trainable_params(self.params, trainable_plus)
            params_minus = self._merge_trainable_params(self.params, trainable_minus)
            rollout_plus = self._simulate(params_plus, features, progress_label=f"{sample_tag} (+)")
            reward_plus = self._metrics_from_rollout(rollout_plus, features)[1]["reward"]
            rollout_minus = self._simulate(params_minus, features, progress_label=f"{sample_tag} (-)")
            reward_minus = self._metrics_from_rollout(rollout_minus, features)[1]["reward"]
            coeff = self.cfg.es_reward_scale * (reward_plus - reward_minus) / (2.0 * sigma * population)
            grad_accum = jax.tree_util.tree_map(
                lambda g, n: g + coeff * n,
                grad_accum,
                noise,
            )
            _log(
                f"{sample_tag}: reward_plus={reward_plus:.4f} "
                f"reward_minus={reward_minus:.4f} coeff={coeff:.4f}"
            )

        grad_norm = float(_tree_global_norm(grad_accum))
        if grad_norm > self.cfg.gradient_clip > 0.0:
            scale = self.cfg.gradient_clip / (grad_norm + 1e-8)
            grad_accum = jax.tree_util.tree_map(lambda g: g * scale, grad_accum)
            _log(f"{step_prefix}: clipped recurrent gradient to norm {self.cfg.gradient_clip:.4f}")

        _log(f"{step_prefix}: applying Adam update to recurrent weights")
        updated_trainable = self._adam_update(trainable_params, grad_accum)
        self.params = self._merge_trainable_params(self.params, updated_trainable)
        rollout = self._simulate(self.params, features, progress_label=f"{step_prefix} eval")
        loss, metrics = self._metrics_from_rollout(rollout, features)
        metrics["loss"] = float(loss)
        metrics["grad_norm"] = grad_norm
        _log(f"{step_prefix}: complete {_metrics_summary(metrics)}")
        return metrics


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
