"""tusz_dataset.py
=================
PyTorch Dataset and DataLoader factory for the consolidated TUSZ EEG window
dataset (``tusz_consolidated.h5`` + ``manifest.csv``).

Overview
--------
Each sample is a fixed 40-sample clip at 200 Hz across 19 channels.  Two
task modes are supported:

``"forecasting"``
    Unsupervised pre-training.  The 40-sample window is split into a context
    portion and a horizon portion.  The model receives ``x`` (context) and
    must predict ``y`` (horizon).  No seizure labels are returned.

    x  : Tensor[19, x_len]   – context given to the model  (channels × time)
    y  : Tensor[19, y_len]   – horizon the model must predict

    where ``x_len + y_len == 40``.

``"detection"``
    Supervised seizure detection / onset characterisation.  The full 40-sample
    window is returned together with binary and (optionally) type labels.

    x        : Tensor[19, 40]   – EEG window (channels × time)
    y_binary : Tensor[40]       – per-sample global seizure label (0 / 1)
    y_type   : Tensor[19, 40]   – per-channel seizure type ID; −1 where
                                   no typed annotation is present or the
                                   sample is background  (int16)

Normalisation
~~~~~~~~~~~~~
Per-window z-score applied to ``x`` (and ``y`` in forecasting mode): mean and
std computed over the context portion only, then applied to both.  A small
epsilon (1e-6) prevents division by zero on flat channels.

Splits
~~~~~~
The manifest ``split`` column follows Temple directories:
    "train"  →  Temple ``h5/train``   (578 patients)
    "val"    →  Temple ``h5/dev``     (53 patients, no overlap with train)
    "test"   →  Temple ``h5/eval``    (43 patients, no overlap with train/val)

Quick start
-----------
    from tusz_dataset import make_dataloaders

    # --- forecasting (pre-training) ---
    loaders = make_dataloaders(
        h5_path      = "tusz_consolidated.h5",
        manifest_csv = "manifest.csv",
        task_mode    = "forecasting",
        x_len        = 30,   # context samples
        batch_size   = 64,
        num_workers  = 2,
    )
    for batch in loaders["train"]:
        x    = batch["x"]        # (B, 19, 30)
        y    = batch["y"]        # (B, 19, 10)
        meta = batch["meta"]     # dict of lists

    # --- detection (fine-tuning) ---
    loaders = make_dataloaders(
        h5_path      = "tusz_consolidated.h5",
        manifest_csv = "manifest.csv",
        task_mode    = "detection",
        batch_size   = 64,
    )
    for batch in loaders["train"]:
        x        = batch["x"]         # (B, 19, 40)
        y_bin    = batch["y_binary"]   # (B, 40)   long
        y_type   = batch["y_type"]     # (B, 19, 40) long; −1 = ignore
        meta     = batch["meta"]

    # --- detection without type labels ---
    loaders = make_dataloaders(..., task_mode="detection", return_type=False)
    # batch keys: "x", "y_binary", "meta"  (no "y_type")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TaskMode = Literal["forecasting", "detection"]
Split = Literal["train", "val", "test"]
_SPLIT_MAP = {"train": "train", "val": "dev", "test": "eval"}  # manifest column values

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TUSZDataset(Dataset):
    """Fixed-window EEG dataset over ``tusz_consolidated.h5``.

    Parameters
    ----------
    h5_path : str | Path
        Path to ``tusz_consolidated.h5``.
    manifest_csv : str | Path
        Path to the companion ``manifest.csv``.
    split : "train" | "val" | "test"
        Which partition to load.
    task_mode : "forecasting" | "detection"
        Controls which batch keys are returned (see module docstring).
    x_len : int
        Context length in samples for ``forecasting`` mode.  The horizon
        is ``40 - x_len`` samples.  Must satisfy ``1 <= x_len <= 39``.
        Ignored in ``detection`` mode.
    condition : str | None
        Restrict to a single condition: ``"clean"``, ``"transition"``, or
        ``"seiz_only"``.  ``None`` loads all conditions.
    return_type : bool
        In ``detection`` mode, include ``y_type`` (per-channel type grid)
        in the batch.  Set to ``False`` to save memory when type labels are
        not needed.  Ignored in ``forecasting`` mode.
    cache : bool
        If ``True``, load the entire HDF5 datasets into RAM on first access.
        Useful when the dataset fits in memory and ``num_workers > 0``.
    """

    _IGNORE_TYPE: int = -1
    _IGNORE_TYPE_TRAIN: int = -100  # PyTorch CrossEntropyLoss ignore_index

    def __init__(
        self,
        h5_path: str | Path,
        manifest_csv: str | Path,
        split: Split = "train",
        task_mode: TaskMode = "forecasting",
        x_len: int = 30,
        condition: Optional[str] = None,
        return_type: bool = True,
        cache: bool = False,
    ) -> None:
        if split not in _SPLIT_MAP:
            raise ValueError(f"split must be one of {list(_SPLIT_MAP)}; got {split!r}")
        if task_mode not in ("forecasting", "detection"):
            raise ValueError(f"task_mode must be 'forecasting' or 'detection'; got {task_mode!r}")
        if task_mode == "forecasting" and not (1 <= x_len <= 39):
            raise ValueError(f"x_len must be in [1, 39] for forecasting; got {x_len}")

        self.h5_path = Path(h5_path)
        self.split = split
        self.task_mode = task_mode
        self.x_len = x_len
        self.y_len = 40 - x_len
        self.condition = condition
        self.return_type = return_type and (task_mode == "detection")
        self.cache = cache

        self._x_cache: Optional[np.ndarray] = None
        self._yb_cache: Optional[np.ndarray] = None
        self._yt_cache: Optional[np.ndarray] = None

        # Read HDF5 attributes for metadata
        with h5py.File(self.h5_path, "r") as hf:
            ch_bytes = hf.attrs.get("channel_order", None)
            self.channel_names: List[str] = (
                [b.decode("utf-8") if isinstance(b, bytes) else b for b in ch_bytes]
                if ch_bytes is not None
                else []
            )
            type_json = hf.attrs.get("type_id_json", "{}")
            self.type_id_to_label: Dict[int, str] = {
                int(k): v for k, v in json.loads(type_json).items()
            }
            self.fs: int = int(hf.attrs.get("fs", 200))

        # Build index from manifest
        manifest_split_val = _SPLIT_MAP[split]
        rows = self._read_manifest(manifest_csv)
        self._index: List[dict] = [
            r for r in rows
            if r["split"] == manifest_split_val
            and (condition is None or r["condition"] == condition)
        ]
        if not self._index:
            raise RuntimeError(
                f"No windows found for split={split!r}, condition={condition!r}."
            )

        # Extract integer row indices into the HDF5 arrays
        self._h5_rows: List[int] = [int(r["window_id"]) for r in self._index]

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        h5_row = self._h5_rows[idx]
        row = self._index[idx]

        x_raw = self._load_x(h5_row)  # (19, 40) float32

        if self.task_mode == "forecasting":
            return self._build_forecasting_batch(x_raw, row)
        else:
            return self._build_detection_batch(x_raw, h5_row, row)

    # ------------------------------------------------------------------
    # Batch builders
    # ------------------------------------------------------------------

    def _build_forecasting_batch(self, x_raw: np.ndarray, row: dict) -> dict:
        ctx_np = x_raw[:, : self.x_len]   # (19, x_len)
        hrz_np = x_raw[:, self.x_len :]   # (19, y_len)

        # mean = ctx_np.mean(axis=1, keepdims=True)
        # std  = ctx_np.std(axis=1, keepdims=True).clip(1e-2)

        # ctx = torch.from_numpy(((ctx_np - mean) / std).copy())
        # hrz = torch.from_numpy(((hrz_np - mean) / std).copy())

        ctx = torch.from_numpy(ctx_np.copy())
        hrz = torch.from_numpy(hrz_np.copy())

        return {"x": ctx, "y": hrz, "meta": self._make_meta(row)}

    def _build_detection_batch(
        self, x_raw: np.ndarray, h5_row: int, row: dict
    ) -> dict:
        # Normalise full window using per-channel stats over all 40 samples
        mean = x_raw.mean(axis=1, keepdims=True)
        std  = x_raw.std(axis=1, keepdims=True).clip(1e-2)
        x = torch.from_numpy(((x_raw - mean) / std).copy())

        yb = self._load_yb(h5_row)  # (40,) int8
        y_binary = torch.from_numpy(yb.astype(np.int64))

        batch: dict = {"x": x, "y_binary": y_binary, "meta": self._make_meta(row)}

        if self.return_type:
            yt = self._load_yt(h5_row)  # (19, 40) int16
            # Remap −1 → −100 for CrossEntropyLoss(ignore_index=-100)
            yt64 = yt.astype(np.int64)
            yt64[yt64 == self._IGNORE_TYPE] = self._IGNORE_TYPE_TRAIN
            batch["y_type"] = torch.from_numpy(yt64)

        return batch

    # ------------------------------------------------------------------
    # HDF5 I/O with optional caching
    # ------------------------------------------------------------------

    def _load_x(self, row: int) -> np.ndarray:
        if self.cache:
            return self._get_cache_x()[row]
        with h5py.File(self.h5_path, "r") as hf:
            return np.array(hf["x"][row], dtype=np.float32)

    def _load_yb(self, row: int) -> np.ndarray:
        if self.cache:
            return self._get_cache_yb()[row]
        with h5py.File(self.h5_path, "r") as hf:
            return np.array(hf["y_binary"][row], dtype=np.int8)

    def _load_yt(self, row: int) -> np.ndarray:
        if self.cache:
            return self._get_cache_yt()[row]
        with h5py.File(self.h5_path, "r") as hf:
            return np.array(hf["y_type_channel"][row], dtype=np.int16)

    def _get_cache_x(self) -> np.ndarray:
        if self._x_cache is None:
            with h5py.File(self.h5_path, "r") as hf:
                self._x_cache = hf["x"][:]
        return self._x_cache

    def _get_cache_yb(self) -> np.ndarray:
        if self._yb_cache is None:
            with h5py.File(self.h5_path, "r") as hf:
                self._yb_cache = hf["y_binary"][:]
        return self._yb_cache

    def _get_cache_yt(self) -> np.ndarray:
        if self._yt_cache is None:
            with h5py.File(self.h5_path, "r") as hf:
                self._yt_cache = hf["y_type_channel"][:]
        return self._yt_cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_manifest(path: str | Path) -> List[dict]:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    @staticmethod
    def _make_meta(row: dict) -> dict:
        return {
            "window_id":          int(row["window_id"]),
            "h5_relpath":         row["h5_relpath"],
            "t0_sample":          int(row["t0_sample"]),
            "condition":          row["condition"],
            "onset_k":            row["onset_k"],
            "episode_type":       row["episode_type"],
        }

    def summary(self) -> str:
        n_stems = len({r["h5_relpath"] for r in self._index})
        conds = sorted({r["condition"] for r in self._index})
        return (
            f"TUSZDataset(split={self.split!r}, task={self.task_mode!r}, "
            f"condition={self.condition!r}) | "
            f"{len(self)} windows from {n_stems} recordings | "
            f"conditions: {conds}"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_dataloaders(
    h5_path: str | Path,
    manifest_csv: str | Path,
    task_mode: TaskMode = "forecasting",
    x_len: int = 30,
    condition: Optional[str] = None,
    return_type: bool = True,
    batch_size: int = 64,
    num_workers: int = 2,
    cache: bool = False,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Return a dict of DataLoaders keyed by split (``"train"``, ``"val"``, ``"test"``).

    Parameters
    ----------
    h5_path : str | Path
        Path to ``tusz_consolidated.h5``.
    manifest_csv : str | Path
        Path to the companion ``manifest.csv``.
    task_mode : "forecasting" | "detection"
        Task mode (see module docstring).
    x_len : int
        Context length in samples for ``"forecasting"`` mode.  Ignored in
        ``"detection"`` mode.
    condition : str | None
        Restrict all loaders to one condition (``"clean"``, ``"transition"``,
        ``"seiz_only"``).  ``None`` loads all conditions.
    return_type : bool
        In ``"detection"`` mode, include ``y_type`` per-channel type grid.
        Set ``False`` to omit it and save memory.
    batch_size : int
        Samples per mini-batch.
    num_workers : int
        Parallel data-loading workers.  Start with 0–2 on HPC GPFS and
        increase if I/O is the bottleneck; ``cache=True`` amortises repeated
        file opens when ``num_workers > 0``.
    cache : bool
        Cache the full HDF5 arrays in RAM after first access.
    pin_memory : bool
        Pin memory for faster GPU transfer when a CUDA device is available.
    """
    use_pin = pin_memory and torch.cuda.is_available()
    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        ds = TUSZDataset(
            h5_path=h5_path,
            manifest_csv=manifest_csv,
            split=split,  # type: ignore[arg-type]
            task_mode=task_mode,
            x_len=x_len,
            condition=condition,
            return_type=return_type,
            cache=cache,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0),
        )
        print(ds.summary())

    return loaders


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Smoke-test TUSZDataset.")
    ap.add_argument("--h5",       required=True, help="Path to tusz_consolidated.h5")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument(
        "--task", default="forecasting", choices=["forecasting", "detection"],
        help="Task mode",
    )
    ap.add_argument("--x-len",      type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers",    type=int, default=0)
    ap.add_argument("--condition",  default=None)
    ap.add_argument("--no-type",    action="store_true")
    args = ap.parse_args()

    loaders = make_dataloaders(
        h5_path=args.h5,
        manifest_csv=args.manifest,
        task_mode=args.task,
        x_len=args.x_len,
        condition=args.condition,
        return_type=not args.no_type,
        batch_size=args.batch_size,
        num_workers=args.workers,
        cache=False,
        pin_memory=False,
    )

    for split, loader in loaders.items():
        t0 = time.perf_counter()
        batch = next(iter(loader))
        dt = time.perf_counter() - t0
        keys = {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                for k, v in batch.items() if k != "meta"}
        print(f"[{split}] first batch in {dt:.3f}s | {keys}")
