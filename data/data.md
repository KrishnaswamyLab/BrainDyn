# Technical specification: `h5/` resampled TUSZ EEG

## 1. Scope

This directory stores **8139** HDF5 files. Each file is the resampled multichannel EEG from a single European Data Format (EDF) recording drawn from the **Temple University Hospital EEG Seizure Corpus (TUSZ)**, the seizure-annotated subset of the TUH EEG corpus. The binary layout and sampling parameters are produced by `benchmarks/ODEBRAIN/data/resample_signals.py` using channel definitions and target rate in `benchmarks/ODEBRAIN/constants.py`. The directory tree under `h5/` is **isomorphic** to the tree under repository root `edf/`: every `.h5` file replaces the corresponding `.edf` at the same relative path (extension change only).

Raw annotations and sidecar files shipped with TUSZ (for example `.csv` seizure annotations adjacent to `.edf` in `edf/`) are **not** copied into `h5/`; only converted waveforms appear here.

## 2. Filesystem layout

Paths follow this template:

```text
h5/<split>/<subject>/<session>/<config>/<subject>_s<session>_t<token>.h5
```

| Component | Form | Meaning |
|-----------|------|--------|
| `<split>` | `train`, `dev`, or `eval` | Official TUSZ partition: model training, development, or held-out evaluation. |
| `<subject>` | eight lowercase letters (e.g. `aaaaaoec`) | Anonymized patient identifier assigned in the distributed corpus. All paths below one `<subject>` directory refer to that patient only. |
| `<session>` | `s` + three digits + `_` + four-digit year (e.g. `s003_2012`) | A **session**: one indexed recording episode for that patient in that calendar year, as named in the corpus file layout. |
| `<config>` | pattern `NN_tcp_XX` with two digits, the literal substring `tcp`, and a two-letter suffix (e.g. `01_tcp_ar`, `02_tcp_le`) | Subdirectory used in TUSZ to separate recordings by **acquisition configuration** (montage and reference scheme). The numeric and letter codes are defined in TUH EEG documentation for the corpus version; they are preserved verbatim in this tree. |
| File stem | `<subject>_s<session>_t<token>` | Matches the source EDF basename. `s` and `t` segments are three-digit zero-padded indices from the corpus filenames (`_t` indexes distinct continuous recordings—**tokens**—within that session and configuration folder). |

### 2.1 Multiple `s…` directories under one subject

Several sibling directories named `s001_2003`, `s002_2003`, … under the same `<subject>` denote **separate sessions** in the corpus indexing: different listed recording dates or session indices for the same anonymized patient. The preprocessing step does not merge or split sessions; it only mirrors the source hierarchy.

### 2.2 Multiple `*_t*.h5` files under one `<session>/<config>` directory

Each file is one **continuous EDF** converted to HDF5. The `_t` index distinguishes multiple such recordings placed in the same session and configuration directory by the corpus layout (separate acquisition files, not subwindows of a single array produced by this repository).

## 3. HDF5 file contents

Each `.h5` file is a valid HDF5 archive with exactly **two** datasets at the root level:

| Dataset | Description |
|---------|-------------|
| `resampled_signal` | Two-dimensional array, shape `(19, T)`, floating-point. Row `c` is time series for one canonical scalp channel; column `t` is discrete time. `T` depends on recording length after resampling (typically on the order of \(10^5\)–\(10^7\) samples for clinical-length recordings). |
| `resample_freq` | Scalar storing the sampling frequency in hertz after conversion. Value is **200** Hz for every file in this tree. |

### 3.1 Channel order

Rows of `resampled_signal` are fixed across all files, in this order (same as `INCLUDED_CHANNELS` in `benchmarks/ODEBRAIN/constants.py`):

| Index | Label |
|------:|-------|
| 0 | EEG FP1 |
| 1 | EEG FP2 |
| 2 | EEG F3 |
| 3 | EEG F4 |
| 4 | EEG C3 |
| 5 | EEG C4 |
| 6 | EEG P3 |
| 7 | EEG P4 |
| 8 | EEG O1 |
| 9 | EEG O2 |
| 10 | EEG F7 |
| 11 | EEG F8 |
| 12 | EEG T3 |
| 13 | EEG T4 |
| 14 | EEG T5 |
| 15 | EEG T6 |
| 16 | EEG FZ |
| 17 | EEG CZ |
| 18 | EEG PZ |

Source EDF channel labels are matched after stripping any suffix following a hyphen on the label string. If a required label is missing in a given recording, the corresponding row is stored as **zeros** for all `T` samples.

### 3.2 Resampling rule

If the native EDF sample rate for the stacked signals differs from 200 Hz, `benchmarks/ODEBRAIN/data/data_utils.py::resampleData` applies `scipy.signal.resample` along time so that the full recording duration maps to `T = int(200 × duration_seconds)` samples (integer duration in seconds derived from sample count and native rate in the converter). If the native rate is already 200 Hz, the waveform is written without that resampling step.

## 4. Consumption in this repository

`benchmarks/ODEBRAIN/args.py` sets `--input_dir` default to `<braindyn>/h5` and `--raw_data_dir` default to `<braindyn>/edf` for TUSZ experiments. Training code expects the layout above and the internal dataset names `resampled_signal` and `resample_freq`.
