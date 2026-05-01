# `eeg_binary`

40-sample TUSZ windows (19 ch @ 200 Hz) with a **binary label**: clean (`0`) or seizure (`1`). No transition windows. Per Temple split, clean count matches seizure count (reservoir sample over background gaps).

**Build** (from repo root; needs `numpy`, `h5py`, `tqdm`):

```bash
python eeg_binary/build_eeg_binary_dataset.py \
  --h5-root h5 --edf-root edf --manifest h5/tusz_manifest.csv \
  --out-dir eeg_binary --seed 42
```

**Outputs:** `eeg_binary/tusz_binary.h5` (`x`, `y`) and `eeg_binary/manifest.csv`. Progress bars print during the run.

**Loader:** `tusz_binary_dataset.py` — `TUSZBinaryDataset` / `make_binary_dataloader`; batch keys `x` `(B,19,40)`, `y` `(B,)`, `window_id`, `meta`.

For full rules (buffers, stride), see the docstring in `build_eeg_binary_dataset.py`.
