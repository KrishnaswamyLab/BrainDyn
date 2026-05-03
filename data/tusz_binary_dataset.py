"""Simple PyTorch loader for ``tusz_binary.h5`` + ``manifest.csv``.

Outline
-------
1. Read ``manifest.csv``, keep rows whose ``split`` matches the requested partition.
2. ``__getitem__`` reads ``x[window_id]`` and ``y[window_id]`` from the single HDF5 file.
3. Optional per-window z-score over time (per channel) on the 40-sample clip.
4. ``make_binary_dataloader`` wraps ``DataLoader`` with a tiny collate that stacks ``x``, ``y``.

Use ``num_workers=0`` unless you add worker-safe HDF5 handling (each worker would need its own file handle).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Split = Literal["train", "val", "test"]
_MANIFEST_SPLIT = {"train": "train", "val": "dev", "test": "eval"}


def _read_manifest(path: Path) -> List[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _collate_binary(batch: List[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "window_id": torch.tensor([b["window_id"] for b in batch], dtype=torch.long),
        "meta": [b["meta"] for b in batch],
    }


class TUSZBinaryDataset(Dataset):
    """Scalar-label TUSZ windows: ``x`` (19, 40) float, ``y`` in {0, 1}."""

    def __init__(
        self,
        h5_path: str | Path,
        manifest_csv: str | Path,
        *,
        split: Split = "train",
        zscore: bool = True,
        eps: float = 1e-6,
        cache: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.zscore = zscore
        self.eps = eps
        self.cache = cache
        self._x_cache: Optional[np.ndarray] = None
        self._y_cache: Optional[np.ndarray] = None

        want = _MANIFEST_SPLIT[split]
        rows = [r for r in _read_manifest(Path(manifest_csv)) if r["split"] == want]
        if not rows:
            raise RuntimeError(f"No manifest rows for split={split!r} (manifest split={want!r}).")

        self._rows = rows
        self._window_ids = [int(r["window_id"]) for r in rows]

        with h5py.File(self.h5_path, "r") as hf:
            self.fs = int(hf.attrs.get("fs", 200))
            self.window_length = int(hf.attrs.get("window_length", 40))

    def __len__(self) -> int:
        return len(self._rows)

    def _get_x_cache(self) -> np.ndarray:
        if self._x_cache is None:
            with h5py.File(self.h5_path, "r") as hf:
                self._x_cache = hf["x"][:].astype(np.float32)
        return self._x_cache

    def _get_y_cache(self) -> np.ndarray:
        if self._y_cache is None:
            with h5py.File(self.h5_path, "r") as hf:
                self._y_cache = hf["y"][:]
        return self._y_cache

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        wid = self._window_ids[idx]

        if self.cache:
            x = self._get_x_cache()[wid].copy()
            y = int(self._get_y_cache()[wid])
        else:
            with h5py.File(self.h5_path, "r") as hf:
                x = np.asarray(hf["x"][wid], dtype=np.float32)
                y = int(hf["y"][wid])

        if self.zscore:
            m = x.mean(axis=1, keepdims=True)
            s = np.clip(x.std(axis=1, keepdims=True), self.eps, None)
            x = (x - m) / s

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.tensor(y, dtype=torch.long),
            "window_id": wid,
            "meta": {
                "h5_relpath": row["h5_relpath"],
                "t0_sample": int(row["t0_sample"]),
                "split": row["split"],
            },
        }


def make_binary_dataloader(
    h5_path: str | Path,
    manifest_csv: str | Path,
    *,
    split: Split = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    zscore: bool = True,
    eps: float = 1e-6,
    cache: bool = False,
    **loader_kw,
) -> DataLoader:
    ds = TUSZBinaryDataset(h5_path, manifest_csv, split=split, zscore=zscore, eps=eps, cache=cache)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_binary,
        **loader_kw,
    )
