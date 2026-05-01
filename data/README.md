# TUSZ Consolidated EEG Window Dataset

A fixed-window EEG dataset derived from the **Temple University Hospital Seizure Corpus (TUSZ) v2.0.6**, designed for seizure detection and onset characterization using deep learning models. The dataset provides pre-extracted 40-sample clips at 200 Hz across 19 electrodes, with per-timepoint binary seizure labels and per-channel seizure-type annotations.

---

## Overview

### Motivation

Standard TUSZ dataloaders operate on variable-length clips and require annotation parsing at training time, making them unsuitable for pipelines that benefit from a single consolidated, self-contained dataset. This dataset materializes a balanced fixed-window collection that can be loaded with a minimal PyTorch `Dataset` wrapper without any dependency on raw EDF files at inference time.

### Design Principles

- **Three balanced conditions per split**: clean background, seizure onset (transition), and fully ictal.
- **Global binary labels**: seizure intervals derived from `csv_bi` / `tse_bi` term-level annotations, ensuring a consistent label timeline across all 19 channels.
- **Per-channel seizure-type labels**: multi-class event annotations from `*.csv` mapped to each electrode where available, with −1 where no typed interval is present.
- **No patient overlap across splits**: Temple's train / dev / eval directory structure is used as-is, which enforces patient-disjoint partitions (verified: zero shared patient IDs across splits).
- **Reproducible sampling**: fixed random seed; all parameters documented in HDF5 attributes and this README.

---

## Dataset Statistics


| Split   | Source     | Clean    | Transition | Seizure-only | Total    |
| ------- | ---------- | -------- | ---------- | ------------ | -------- |
| Train   | `h5/train` | 1428     | 1428       | 1428         | **4284** |
| Val     | `h5/dev`   | 357      | 357        | 357          | **1071** |
| Test    | `h5/eval`  | 357      | 357        | 357          | **1071** |
| **All** |            | **2142** | **2142**   | **2142**     | **6426** |


Split ratio is **4 : 1 : 1** (train : val : test), with exactly ⅓ of each split per condition. The total size is bounded by the number of distinct seizure events with valid transition geometry in the eval split (~357 events × 18 = 6426 ceiling).

---

## Files

```
eeg/
├── tusz_consolidated.h5   # Main dataset (EEG + labels)
├── manifest.csv           # Per-window metadata index
└── README.md              # This file
```

### `tusz_consolidated.h5`


| Dataset          | Shape         | Dtype     | Description                                                                               |
| ---------------- | ------------- | --------- | ----------------------------------------------------------------------------------------- |
| `x`              | `[N, 19, 40]` | `float32` | EEG signal: N windows × 19 channels × 40 samples                                          |
| `y_binary`       | `[N, 40]`     | `int8`    | Per-sample global seizure label (0 = background, 1 = seizure)                             |
| `y_type_channel` | `[N, 19, 40]` | `int16`   | Per-channel seizure type ID at each sample; −1 = no typed annotation or global background |


**HDF5 attributes**


| Attribute           | Value                                         |
| ------------------- | --------------------------------------------- |
| `fs`                | 200 (Hz)                                      |
| `window_length`     | 40 (samples)                                  |
| `n_channels`        | 19                                            |
| `channel_order`     | List of 19 electrode names in signal order    |
| `type_ignore_value` | −1                                            |
| `type_id_json`      | JSON mapping integer ID → seizure type string |
| `half_open_rule`    | Label convention (see below)                  |
| `builder_seed`      | 42                                            |


### `manifest.csv`

One row per window. Columns:


| Column                 | Description                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `window_id`            | Row index into HDF5 datasets                                                                                                                            |
| `split`                | `train` / `dev` / `eval`                                                                                                                                |
| `h5_relpath`           | Source recording path relative to `h5/` root                                                                                                            |
| `t0_sample`            | Absolute start sample index in the source recording                                                                                                     |
| `condition`            | `clean`, `transition`, or `seiz_only`                                                                                                                   |
| `onset_k`              | For transition windows: number of clean samples before onset (25–35); empty otherwise                                                                   |
| `episode_type`         | Dominant seizure type for this window's episode (overlap-weighted vote from `*.csv`); `none` for clean windows or when no typed annotation is available |
| `primary_seizure_type` | Same as `episode_type` (retained for compatibility)                                                                                                     |


---

## Label Conventions

### Binary labels (`y_binary`)

Sample index **k** is labeled seizure (1) if and only if the half-open time interval `**[k/fs, (k+1)/fs)`** intersects any merged `csv_bi` seizure segment `**[st, en)**`. Background samples are labeled 0.

The `csv_bi` / `tse_bi` annotations provide a single global seizure timeline per recording (one label for the full montage), consistent with term-level seizure detection.

### Per-channel type labels (`y_type_channel`)

Typed intervals are read from multi-class `*.csv` files, which annotate specific bipolar montage pairs (e.g. `FP1-F7`, `CZ-C4`). Each interval is assigned to every INCLUDED_CHANNELS electrode that appears in that bipolar pair name. At each sample, the interval with the greatest time overlap is used; ties are broken by lower type ID.

**Sentinel value −1** indicates either: (a) no multi-class typed interval covers that channel at that time, or (b) the global binary label is 0 (background). Type labels are only meaningful at positions where `y_binary == 1`.

**Seizure type ID mapping** (from `constants.py` `ALL_LABEL_DICT`):


| ID  | Code   | Seizure type                                |
| --- | ------ | ------------------------------------------- |
| 0   | `fnsz` | Focal, not further specified                |
| 1   | `gnsz` | Generalized, not further specified          |
| 2   | `spsz` | Simple partial (focal, awareness preserved) |
| 3   | `cpsz` | Complex partial (focal, impaired awareness) |
| 4   | `absz` | Absence                                     |
| 5   | `tnsz` | Tonic                                       |
| 6   | `tcsz` | Tonic–clonic                                |
| 7   | `mysz` | Myoclonic                                   |


For PyTorch cross-entropy training over type labels, map −1 → −100 in `Dataset.__getitem_`_ and use `CrossEntropyLoss(ignore_index=-100)`.

---

## Window Conditions

### Clean

- All 40 samples fall within annotated global background.
- The window does not intersect the **expanded exclusion zone**: each merged seizure interval expanded by one full window length (40 samples / 0.2 s) on both sides.
- The window starts and ends at least 40 samples from the recording boundary.

### Transition (onset)

- The first **k** samples are background and the remaining **40 − k** samples are seizure, where **k ∈ [25, 35]** (chosen randomly per seizure from valid placements). This gives at least 5 and at most 15 ictal samples per window.
- Onset is anchored to the global `csv_bi` / `tse_bi` seizure start time (half-open label rule).
- The ictal portion of the window must lie entirely within the **same** merged seizure interval (no gap or different episode spanning the tail).
- Because all valid onset positions for a single seizure produce overlapping windows, exactly **one transition window is drawn per seizure event** (k chosen randomly from valid values).

### Seizure-only

- All 40 samples are global seizure.
- The window is drawn from the interior of the seizure interval: at least 10 samples inside the annotated seizure start, and at least 10 + 40 = 50 samples before the annotated end (10-sample interior trim on each side).
- Seizure intervals shorter than 60 samples (0.3 s) are excluded.
- The Aufbau (round-robin) sampling strategy is used across seizure types: type IDs are cycled in order (0 → 7 → 0 → …), and in each round one window per type is preferred before adding a second. When a rare type is exhausted, any type is accepted to fill the quota.

### Non-overlap

Windows drawn from the same source recording are **pairwise non-overlapping**: no two windows share a sample index. This applies across all conditions within a recording.

---

## PyTorch Dataloader (`tusz_dataset.py`)

`tusz_dataset.py` provides a `TUSZDataset` class and a `make_dataloaders` factory that return a dict of `DataLoader` objects keyed by `"train"`, `"val"`, and `"test"`.

### Task modes

Two mutually exclusive modes are controlled by the `task_mode` flag.

#### `"forecasting"` — unsupervised pre-training

The 40-sample window is split into a **context** and a **horizon**. The model receives the context and must predict the horizon. No seizure labels are returned.


| Batch key | Shape                 | Description                                     |
| --------- | --------------------- | ----------------------------------------------- |
| `x`       | `[B, 19, x_len]`      | Context (channels × time), z-scored per channel |
| `y`       | `[B, 19, 40 − x_len]` | Horizon, normalised with context statistics     |
| `meta`    | dict of lists         | Window metadata (see below)                     |


`x_len` (default 30) controls the context length. The horizon is `40 − x_len` samples. Both `x` and `y` are z-scored using the mean and std computed from the context portion only.

#### `"detection"` — supervised seizure detection

The full 40-sample window is returned with binary and (optionally) type labels.


| Batch key  | Shape         | Dtype   | Description                                                                                                                          |
| ---------- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `x`        | `[B, 19, 40]` | float32 | Full window, z-scored per channel over all 40 samples                                                                                |
| `y_binary` | `[B, 40]`     | int64   | Per-sample global seizure label: 0 = background, 1 = seizure                                                                         |
| `y_type`   | `[B, 19, 40]` | int64   | Per-channel seizure type ID; −100 where no typed annotation is present or sample is background (see type table in Label Conventions) |
| `meta`     | dict of lists |         | Window metadata                                                                                                                      |


`y_type` uses **−100** as the ignore sentinel (compatible with `CrossEntropyLoss(ignore_index=-100)`). The raw HDF5 stores −1; the dataloader remaps automatically.

### Flags


| Flag          | Type        | Default         | Description                                                                                                                                                  |
| ------------- | ----------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `task_mode`   | str         | `"forecasting"` | `"forecasting"` or `"detection"`                                                                                                                             |
| `x_len`       | int         | 30              | Context length in samples for forecasting mode (horizon = `40 − x_len`)                                                                                      |
| `condition`   | str or None | `None`          | Restrict to one window condition: `"clean"`, `"transition"`, or `"seiz_only"`. `None` loads all three                                                        |
| `return_type` | bool        | `True`          | In detection mode, include `y_type` in the batch. Set `False` to omit it when per-channel type supervision is not needed                                     |
| `batch_size`  | int         | 64              | Samples per mini-batch                                                                                                                                       |
| `num_workers` | int         | 2               | Parallel data-loading workers. Start with 0–2 on HPC GPFS                                                                                                    |
| `cache`       | bool        | `False`         | Load entire HDF5 arrays into RAM on first access. Recommended when the dataset fits in memory and `num_workers > 0`, to avoid repeated file opens per worker |
| `pin_memory`  | bool        | `True`          | Pin memory for faster host→GPU transfer. Automatically disabled when no CUDA device is available                                                             |


### `meta` dict keys


| Key            | Type | Description                                                                                          |
| -------------- | ---- | ---------------------------------------------------------------------------------------------------- |
| `window_id`    | int  | Row index into the HDF5 datasets                                                                     |
| `h5_relpath`   | str  | Source recording path relative to `h5/` root                                                         |
| `t0_sample`    | int  | Absolute start sample in the source recording                                                        |
| `condition`    | str  | `"clean"`, `"transition"`, or `"seiz_only"`                                                          |
| `onset_k`      | str  | Number of leading clean samples for transition windows (25–35); empty string for clean and seiz_only |
| `episode_type` | str  | Dominant seizure type for the episode (`"fnsz"`, `"gnsz"`, …, or `"none"`)                           |


### Quick start

```python
from tusz_dataset import make_dataloaders

# Forecasting (pre-training)
loaders = make_dataloaders(
    h5_path      = "tusz_consolidated.h5",
    manifest_csv = "manifest.csv",
    task_mode    = "forecasting",
    x_len        = 30,
    batch_size   = 64,
    num_workers  = 2,
)
for batch in loaders["train"]:
    x    = batch["x"]     # (B, 19, 30)
    y    = batch["y"]     # (B, 19, 10)
    meta = batch["meta"]

# Detection (fine-tuning), binary labels only
loaders = make_dataloaders(
    h5_path      = "tusz_consolidated.h5",
    manifest_csv = "manifest.csv",
    task_mode    = "detection",
    return_type  = False,
    batch_size   = 64,
)
for batch in loaders["train"]:
    x        = batch["x"]        # (B, 19, 40)
    y_binary = batch["y_binary"] # (B, 40)

# Detection, seizure-onset windows only, with type labels
loaders = make_dataloaders(
    h5_path      = "tusz_consolidated.h5",
    manifest_csv = "manifest.csv",
    task_mode    = "detection",
    condition    = "transition",
    return_type  = True,
)
for batch in loaders["train"]:
    x      = batch["x"]        # (B, 19, 40)
    y_bin  = batch["y_binary"] # (B, 40)
    y_type = batch["y_type"]   # (B, 19, 40); −100 = ignore
```

### Smoke-test

```bash
python /gpfs/radev/grand_challenge/foundation_models/models/braindyn/eeg/tusz_dataset.py \
  --h5       /gpfs/radev/grand_challenge/foundation_models/models/braindyn/eeg/tusz_consolidated.h5 \
  --manifest /gpfs/radev/grand_challenge/foundation_models/models/braindyn/eeg/manifest.csv \
  --task forecasting \
  --x-len 30 \
  --batch-size 8
```

---

## Building the Dataset

**Dependencies**: `h5py`, `numpy`, `pyedflib`, `scipy` (see `benchmarks/ODEBRAIN/requirements.txt`). No GPU required.

```bash
python benchmarks/ODEBRAIN/data/tusz_consolidated_dataset/build_tusz_consolidated_dataset.py \
  --h5-root   /path/to/braindyn/h5 \
  --edf-root  /path/to/braindyn/edf \
  --manifest  /path/to/braindyn/h5/tusz_manifest.csv \
  --out-dir   /path/to/braindyn/eeg \
  --seed 42
```

`--max-windows 0` (default) automatically computes the maximum feasible total from the transition stratum. Pass a positive integer to target a fixed size (must not exceed the transition ceiling).

---

## Corpus Source

**Temple University Hospital Seizure Corpus (TUSZ) v2.0.6**

> Shah, V., von Weltin, E., Lopez, S., McHugh, J., Veloso, L., Golmohammadi, M., Obeid, I., & Picone, J. (2018). The Temple University Hospital Seizure Detection Corpus. *Frontiers in Neuroinformatics*, 12:83. [https://doi.org/10.3389/fninf.2018.00083](https://doi.org/10.3389/fninf.2018.00083)

Access: [https://isip.piconepress.com/projects/tuh_eeg/](https://isip.piconepress.com/projects/tuh_eeg/)

---

