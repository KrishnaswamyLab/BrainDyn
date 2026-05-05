"""PyTorch Dataset and DataLoader utilities for bulk simulated neuron data.

Expects ``dataset.npz`` from ``simulate_neuron_dataset.py``: rate arrays shaped
``[subjects, channels, time]`` plus perturbation metadata. Forecasting uses
unperturbed rates with context-only z-scoring. Perturbation mode returns paired
windows ``x_original`` / ``x_perturbed`` (length ``x``) and ``y_original`` /
``y_perturbed`` (length ``y``), aligned around the perturbation onset on the bin
grid, normalised using per-channel mean/std from the full original trajectory;
perturbed windows use the same statistics so originals and perturbations are
comparable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, cast
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_NPZ_PATH = os.path.join(ROOT_DIR, "data", "simulated_neuron_dataset", "dataset.npz")
SPLITS = Literal["train", "val", "test", "within"]
TASK_MODES = Literal["forecasting", "perturbation"]
CROSS_SUBJECT_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


def _read_run_config(npz_path: str) -> dict[str, Any]:
    path = os.path.join(os.path.dirname(npz_path), "run_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _partition_subjects(
    n: int, train_frac: float, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= train_frac <= 1.0 and 0.0 <= val_frac <= 1.0):
        raise ValueError("train_frac and val_frac must be in [0, 1].")
    if train_frac + val_frac > 1.0 + 1e-9:
        raise ValueError("train_frac + val_frac must not exceed 1.")
    if n < 1:
        raise ValueError("n must be at least 1.")

    perm = np.random.default_rng(seed).permutation(n)
    n_train = max(0, min(int(round(train_frac * n)), n))
    n_val = max(0, min(int(round(val_frac * n)), n - n_train))
    i1 = n_train + n_val
    return perm[:n_train], perm[n_train:i1], perm[i1:]


def _zscore_with_context_stats(
    context: np.ndarray, horizon: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = context.mean(axis=0, keepdims=True)
    std = context.std(axis=0, keepdims=True).clip(1e-6)
    x = torch.from_numpy(((context - mean) / std).astype(np.float32).copy())
    y = torch.from_numpy(((horizon - mean) / std).astype(np.float32).copy())
    return x, y


def _zscore_blocks_with_series_stats(
    reference_tc: np.ndarray, *blocks_tc: np.ndarray
) -> tuple[torch.Tensor, ...]:
    """Per-channel z-score using mean/std over full time in ``reference_tc`` (T, C)."""
    mean = reference_tc.mean(axis=0, keepdims=True)
    std = reference_tc.std(axis=0, keepdims=True).clip(1e-6)
    return tuple(
        torch.from_numpy(((b - mean) / std).astype(np.float32).copy()) for b in blocks_tc
    )


def _pert_start_ms_to_center_bin_index(pert_start_ms: float, bin_edges_ms: np.ndarray) -> int:
    """Map perturbation onset (ms) to the nearest discrete bin center index."""
    edges = np.asarray(bin_edges_ms, dtype=np.float64).reshape(-1)
    if edges.size < 2:
        raise ValueError("bin_edges_ms must contain at least two edge values.")
    n_bins = int(edges.size) - 1
    idx = int(np.searchsorted(edges, pert_start_ms, side="right") - 1)
    idx = max(0, min(idx, n_bins - 1))
    width = float(edges[idx + 1] - edges[idx])
    if width > 1e-12:
        frac = (pert_start_ms - float(edges[idx])) / width
        c_continuous = idx + frac
    else:
        c_continuous = float(idx)
    c = int(round(c_continuous))
    return max(0, min(c, n_bins - 1))


class SNDataset(Dataset):
    """Index over ``dataset.npz`` for forecasting windows or perturbation-aligned windows."""

    def __init__(
        self,
        npz_path: str = DATA_NPZ_PATH,
        *,
        task_mode: TASK_MODES = "forecasting",
        split: SPLITS = "train",
        x: int = 90,
        y: int = 30,
        stride: int = 10,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_seed: int = 0,
        cache: bool = True,
    ) -> None:
        """
        Args:
            x: Number of context time bins (`x[t : t + x]`) in forecasting mode;
               same length for ``x_original`` / ``x_perturbed`` in perturbation mode.
            y: Horizon length in forecasting; same length for ``y_original`` /
               ``y_perturbed`` in perturbation mode. Perturbation windows are fixed
               (one sample per subject) and ignore ``stride``.
        """
        if split not in CROSS_SUBJECT_SPLITS and split != "within":
            raise ValueError(f"invalid split: {split!r}")
        if task_mode not in ("forecasting", "perturbation"):
            raise ValueError(f"invalid task_mode: {task_mode!r}")

        path = os.path.abspath(os.fspath(npz_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        self.npz_path = path
        self.task_mode = task_mode
        self.split: SPLITS = split
        self.x = x
        self.y = y
        self.stride = stride
        self._cache_subject_tc = cache
        self._tc_cache: dict[int, np.ndarray] = {}

        mmap: str | None = None if cache else "r"
        self._npz = np.load(path, mmap_mode=mmap, allow_pickle=True)
        self._rates_o = self._npz["smoothed_rates_hz_original"]
        self._rates_p = self._npz["smoothed_rates_hz_perturbed"]

        self.n_subjects, self.n_channels, self.n_bins = (int(self._rates_o.shape[i]) for i in range(3))

        self._pert_start = np.asarray(self._npz["perturbation_start_ms"], dtype=np.float64)
        self._pert_end = np.asarray(self._npz["perturbation_end_ms"], dtype=np.float64)
        self._pert_n = np.asarray(self._npz["perturbation_n_nodes"], dtype=np.int64)
        self._pert_nodes = np.asarray(self._npz["perturbation_nodes"], dtype=np.int64)
        self._graph_seeds = np.asarray(self._npz["graph_seeds"], dtype=np.int64)
        _adj_full = np.asarray(self._npz["adjacency"], dtype=np.int8)

        run_cfg = _read_run_config(path)
        self._perturbation_mode: str | None = run_cfg.get("perturbation_mode")

        train_i, val_i, test_i = _partition_subjects(
            self.n_subjects, train_frac, val_frac, split_seed
        )
        by_split = {
            "train": train_i.astype(np.int64, copy=False),
            "val": val_i.astype(np.int64, copy=False),
            "test": test_i.astype(np.int64, copy=False),
        }
        if split in CROSS_SUBJECT_SPLITS:
            self._subject_ids = by_split[split]
        else:
            self._subject_ids = np.arange(self.n_subjects, dtype=np.int64)

        # Same subject order as time series indexing: row k matches self._subject_ids[k].
        self._adjacency = np.asarray(_adj_full[self._subject_ids], dtype=np.int8)

        if "bin_edges_ms" in self._npz.files:
            self._bin_edges_ms = np.asarray(self._npz["bin_edges_ms"], dtype=np.float64).reshape(-1)
        else:
            proc = (
                cast(dict[str, Any], run_cfg.get("base_config", {}))
                .get("processing", {})
                if isinstance(run_cfg.get("base_config", {}), dict)
                else {}
            )
            bin_size = float(proc.get("bin_size_ms", 0.0))
            if bin_size <= 0 or self.n_bins < 1:
                raise KeyError(
                    "dataset.npz must contain 'bin_edges_ms' or run_config.json must "
                    "define base_config.processing.bin_size_ms"
                )
            self._bin_edges_ms = np.arange(self.n_bins + 1, dtype=np.float64) * bin_size

        self._index: list[tuple[int, ...]] = []
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        self._index.clear()
        if self.task_mode == "perturbation":
            self._index.extend((k,) for k in range(len(self._subject_ids)))
            return

        T = self.n_bins
        for k in range(len(self._subject_ids)):
            if self.split == "within":
                t0 = T - self.x - self.y
                if t0 >= 0:
                    self._index.append((k, t0))
            else:
                last_t = T - self.x - 2 * self.y
                if last_t < 0:
                    continue
                for t in range(0, last_t + 1, self.stride):
                    self._index.append((k, t))

    def _rates_orig_tc(self, subject: int) -> np.ndarray:
        if self._cache_subject_tc:
            hit = self._tc_cache.get(subject)
            if hit is not None:
                return hit
            tc = np.asarray(self._rates_o[subject], dtype=np.float32).T.copy()
            self._tc_cache[subject] = tc
            return tc
        return np.asarray(self._rates_o[subject], dtype=np.float32).T

    def _rates_pert_tc(self, subject: int) -> np.ndarray:
        return np.asarray(self._rates_p[subject], dtype=np.float32).T

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.task_mode == "forecasting":
            k, t = self._index[idx]
            s = int(self._subject_ids[k])
            ts = self._rates_orig_tc(s)
            ctx = ts[t : t + self.x]
            hrz = ts[t + self.x : t + self.x + self.y]
            x_t, y_t = _zscore_with_context_stats(ctx, hrz)
            adj = torch.from_numpy(self._adjacency[k].copy())
            meta = {
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "t_start": int(t),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
                "adjacency": adj,
            }
            return {"x": x_t, "y": y_t, "meta": meta}

        else:
            (k,) = self._index[idx]
            s = int(self._subject_ids[k])
            orig = self._rates_orig_tc(s)
            pert = self._rates_pert_tc(s)
            T = self.n_bins
            need = self.x + self.y
            if need > T:
                raise ValueError(
                    f"perturbation mode requires x+y<={T}, got x={self.x}, y={self.y}, T={T}"
                )
            c_bin = _pert_start_ms_to_center_bin_index(float(self._pert_start[s]), self._bin_edges_ms)
            tx = c_bin - self.x // 2
            tx = max(0, min(tx, T - need))
            sl_x = slice(tx, tx + self.x)
            sl_y = slice(tx + self.x, tx + need)
            ctx_orig = orig[sl_x]
            hrz_orig = orig[sl_y]
            ctx_pert = pert[sl_x]
            hrz_pert = pert[sl_y]
            (
                x_original,
                y_original,
                x_perturbed,
                y_perturbed,
            ) = _zscore_blocks_with_series_stats(orig, ctx_orig, hrz_orig, ctx_pert, hrz_pert)
            kn = int(self._pert_n[s])
            nodes = torch.from_numpy(self._pert_nodes[s].astype(np.int64).copy())
            mode = self._perturbation_mode or ""
            adj = torch.from_numpy(self._adjacency[k].copy())
            meta = {
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "t_start": int(tx),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
                "perturbation_start_ms": float(self._pert_start[s]),
                "perturbation_end_ms": float(self._pert_end[s]),
                "perturbation_mode": mode,
                "perturbed_n_nodes": torch.tensor(kn, dtype=torch.long),
                "perturbed_nodes": nodes,
                "adjacency": adj,
            }
            return {
                "x_original": x_original,
                "y_original": y_original,
                "x_perturbed": x_perturbed,
                "y_perturbed": y_perturbed,
                "meta": meta,
            }

    def summary(self) -> str:
        return (
            f"SNDataset({self.task_mode=}, {self.split=}, {self.x=}, {self.y=}, {self.stride=}) "
            f"n={len(self)} pool={len(self._subject_ids)} shape=({len(self)},{self.n_bins},{self.n_channels})"
        )

    def close(self) -> None:
        if getattr(self, "_npz", None) is not None:
            self._npz.close()


def make_dataloaders(
    npz_path: str = DATA_NPZ_PATH,
    *,
    task_mode: TASK_MODES = "forecasting",
    x: int = 90,
    y: int = 30,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 2,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    split_seed: int = 0,
    cache: bool = False,
    pin_memory: bool = True,
    verbose: bool = False,
) -> dict[str, DataLoader]:
    pin = pin_memory and torch.cuda.is_available()
    out: dict[str, DataLoader] = {}
    for sp in ("train", "val", "test", "within"):
        split: SPLITS = cast(SPLITS, sp)
        ds = SNDataset(
            npz_path,
            task_mode=task_mode,
            split=split,
            x=x,
            y=y,
            stride=stride,
            train_frac=train_frac,
            val_frac=val_frac,
            split_seed=split_seed,
            cache=cache,
        )
        out[sp] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(sp == "train"),
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )
        if verbose:
            print(ds.summary())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unit tests for SNDataset.")
    ap.add_argument("npz_path", nargs="?", default=str(DATA_NPZ_PATH))
    ap.add_argument("--x", type=int, default=90)
    ap.add_argument("--y", type=int, default=30)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=2)
    args = ap.parse_args()

    for task_mode in ("forecasting", "perturbation"):
        print(f"\nSanity checking task mode: {task_mode}...\n\n")

        loaders = make_dataloaders(
            args.npz_path,
            task_mode=task_mode,
            x=args.x,
            y=args.y,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            verbose=True,
        )

        for name, loader in loaders.items():
            n = len(loader.dataset)
            if n == 0:
                print(f"{name}: empty")
                continue
            t0 = time.perf_counter()
            for i, batch in enumerate(loader):
                if i == 0:
                    if task_mode == "forecasting":
                        print(
                            f'{name}: x={tuple(batch["x"].shape)} y={tuple(batch["y"].shape)} '
                            f'subject={batch["meta"]["subject_index"]}'
                            f'graph adjacency shape={batch["meta"]["adjacency"].shape}'
                        )
                    else:
                        print(
                            f'{name}: x_original={tuple(batch["x_original"].shape)} '
                            f'y_original={tuple(batch["y_original"].shape)} '
                            f'x_perturbed={tuple(batch["x_perturbed"].shape)} '
                            f'y_perturbed={tuple(batch["y_perturbed"].shape)} '
                            f'perturbation_start_ms={batch["meta"]["perturbation_start_ms"]}'
                            f'perturbation_end_ms={batch["meta"]["perturbation_end_ms"]}'
                            f'perturbation_mode={batch["meta"]["perturbation_mode"]}'
                            f'perturbed_n_nodes={batch["meta"]["perturbed_n_nodes"]}'
                            f'perturbed_nodes={batch["meta"]["perturbed_nodes"]}'
                            f'adjacency shape={batch["meta"]["adjacency"].shape}'
                        )
                if i + 1 >= args.n_batches:
                    break
            print(f'{name}: {args.n_batches} batches in {time.perf_counter() - t0:.2f}s')
