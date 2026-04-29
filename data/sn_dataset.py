"""PyTorch Dataset and DataLoader utilities for bulk simulated neuron data.

Expects ``dataset.npz`` from ``simulate_neuron_dataset.py``: rate arrays shaped
``[subjects, channels, time]`` plus perturbation metadata. Forecasting uses
unperturbed rates with context-only z-scoring; perturbation mode returns full
original and perturbed trajectories normalised with statistics from the original.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, cast

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


def _zscore_pair_from_reference(
    reference: np.ndarray, other: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True).clip(1e-6)
    a = torch.from_numpy(((reference - mean) / std).astype(np.float32).copy())
    b = torch.from_numpy(((other - mean) / std).astype(np.float32).copy())
    return a, b


class SNDataset(Dataset):
    """Index over ``dataset.npz`` for forecasting windows or full-sequence perturbation pairs."""

    def __init__(
        self,
        npz_path: str = DATA_NPZ_PATH,
        *,
        task_mode: TASK_MODES = "forecasting",
        split: SPLITS = "train",
        x: int = 90,
        y: int = 30,
        stride: int = 1,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_seed: int = 0,
        cache: bool = False,
    ) -> None:
        """
        Args:
            x: Number of context time bins given to the model as input
                (`x[t : t + x]`) in forecasting mode.
            y: Number of future time bins predicted by the model
                (`y[t + x : t + x + y]`) in forecasting mode.
                In perturbation mode, full trajectories are returned and
                `x`/`y` are ignored.
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
        self._pert_nodes_pad = np.asarray(self._npz["perturbation_nodes_padded"], dtype=np.int64)
        self._graph_seeds = np.asarray(self._npz["graph_seeds"], dtype=np.int64)

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

        self._index: list[tuple[int, ...]] = []
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        self._index.clear()
        if self.task_mode == "perturbation":
            self._index.extend((int(s),) for s in self._subject_ids.tolist())
            return

        T = self.n_bins
        for s in self._subject_ids.tolist():
            if self.split == "within":
                t0 = T - self.x - self.y
                if t0 >= 0:
                    self._index.append((s, t0))
            else:
                last_t = T - self.x - 2 * self.y
                if last_t < 0:
                    continue
                for t in range(0, last_t + 1, self.stride):
                    self._index.append((s, t))

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
            s, t = self._index[idx]
            ts = self._rates_orig_tc(s)
            ctx = ts[t : t + self.x]
            hrz = ts[t + self.x : t + self.x + self.y]
            x_t, y_t = _zscore_with_context_stats(ctx, hrz)
            meta = {
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "t_start": int(t),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
            }
            return {"x": x_t, "y": y_t, "meta": meta}

        else:
            (s,) = self._index[idx]
            orig = self._rates_orig_tc(s)
            pert = self._rates_pert_tc(s)
            o_t, p_t = _zscore_pair_from_reference(orig, pert)
            kn = int(self._pert_n[s])
            pad = torch.from_numpy(self._pert_nodes_pad[s].astype(np.int64).copy())
            mode = self._perturbation_mode or ""
            meta = {
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
                "perturbation_start_ms": float(self._pert_start[s]),
                "perturbation_end_ms": float(self._pert_end[s]),
                "perturbation_mode": mode,
            }
            return {
                "x_original": o_t,
                "x_perturbed": p_t,
                "pert_n_nodes": torch.tensor(kn, dtype=torch.long),
                "pert_nodes_pad": pad,
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
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Unit tests for SNDataset.")
    ap.add_argument("npz_path", nargs="?", default=str(DATA_NPZ_PATH))
    ap.add_argument("--x", type=int, default=90)
    ap.add_argument("--y", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
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
                            f"{name}: x={tuple(batch['x'].shape)} y={tuple(batch['y'].shape)} "
                            f"subject={batch['meta']['subject_index']}"
                        )
                    else:
                        print(
                            f"{name}: x_original={tuple(batch['x_original'].shape)} "
                            f"{name}: x_perturbed={tuple(batch['x_perturbed'].shape)} "
                            f"metadata={batch['meta']}"
                        )
                if i + 1 >= args.n_batches:
                    break
            print(f"{name}: {args.n_batches} batches in {time.perf_counter() - t0:.2f}s")
