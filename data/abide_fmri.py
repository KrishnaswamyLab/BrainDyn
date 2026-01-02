import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from nilearn.datasets import fetch_abide_pcp, fetch_atlas_basc_multiscale_2015
from nilearn.input_data import NiftiLabelsMasker

def fetch_abide_balanced(data_dir, n_per_group=40):
    """
    Returns nilearn ABIDE dict with:
      - func_preproc: list of Nifti images
      - phenotypic: pandas DataFrame (same order as func_preproc)
    """
    meta = fetch_abide_pcp(
        n_subjects=None,
        derivatives=[],
        data_dir=data_dir,
        verbose=0
    )["phenotypic"]

    asd = meta[meta["DX_GROUP"] == 1].head(n_per_group)
    hc  = meta[meta["DX_GROUP"] == 2].head(n_per_group)
    selected = pd.concat([asd, hc])
    subject_ids = selected["SUB_ID"].tolist()

    return fetch_abide_pcp(
        n_subjects=None,
        SUB_ID=subject_ids,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=True,
        derivatives=["func_preproc"],
        data_dir=data_dir,
        verbose=1
    )


def extract_time_series(func_img, labels_img, t_r=2.5):
    """
    Returns ROI time series with shape (T, N).
    """
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize=True,
        detrend=True,
        t_r=t_r
    )
    ts = masker.fit_transform(func_img)  # (T, N)
    return ts


def compute_fc_adjacency(ts_TN, threshold=0.3):
    """
    ts_TN: (T, N)
    Returns adjacency (N, N) computed from ROI correlations, thresholded.
    """
    # correlation expects (N, T), so transpose
    ts_NT = ts_TN.T  # (N, T)

    # Drop near-constant ROIs (rare but possible)
    stds = ts_NT.std(axis=1)
    valid = stds > 1e-6
    if not np.all(valid):
        ts_NT = ts_NT[valid]
        # We'll rebuild a full NxN adjacency by padding invalid nodes with zeros
        # (so shapes still match N regions)
        n_valid = ts_NT.shape[0]
    else:
        n_valid = ts_NT.shape[0]

    fc = np.corrcoef(ts_NT)  # (n_valid, n_valid)
    np.fill_diagonal(fc, 0.0)
    fc[np.abs(fc) < threshold] = 0.0

    if not np.all(valid):
        N = valid.shape[0]
        full = np.zeros((N, N), dtype=np.float32)
        idx = np.where(valid)[0]
        full[np.ix_(idx, idx)] = fc.astype(np.float32)
        return full
    else:
        return fc.astype(np.float32)


def make_full_state(ts_TN):
    """
    Convert ABIDE ROI time series ts(T,N) into full_state(T,N,3) expected by main.py.

    We set:
      feature0 = ts
      feature1 = 1  (so feature0 * feature1 == ts)
      feature2 = 0  (unused)
    """
    T, N = ts_TN.shape
    full_state = np.zeros((T, N, 3), dtype=np.float32)
    full_state[..., 0] = ts_TN.astype(np.float32)
    full_state[..., 1] = 1.0
    full_state[..., 2] = 0.0
    return full_state


def export_for_braindyn_main(
    abide_data,
    out_dir="data",
    subject_index=0,
    atlas_scale="scale197",
    t_r=2.5,
    threshold=0.3,
    truncate_to=None
):
    """
    Writes:
      out_dir/full_state.npy   (T, N, 3)
      out_dir/adjacency.npy    (N, N)

    subject_index selects which subject in abide_data['func_preproc'] to export.
    """
    os.makedirs(out_dir, exist_ok=True)

    atlas = fetch_atlas_basc_multiscale_2015(version="sym")
    labels_img = atlas[atlas_scale]  # e.g., 'scale197'
    phenos = abide_data["phenotypic"]

    func_list = abide_data["func_preproc"]
    if len(func_list) == 0:
        raise RuntimeError("No func_preproc images found in abide_data.")

    if subject_index < 0 or subject_index >= len(func_list):
        raise ValueError(f"subject_index={subject_index} out of range (0..{len(func_list)-1}).")

    func_img = func_list[subject_index]
    pheno = phenos.iloc[subject_index]
    dx_group = pheno.get("DX_GROUP", None)
    sex = pheno.get("SEX", None)
    sub_id = pheno.get("SUB_ID", None)

    print(f"[export] Subject index: {subject_index}, SUB_ID: {sub_id}, DX_GROUP: {dx_group}, SEX: {sex}")
    print(f"[export] Extracting ROI time series using {atlas_scale}...")

    ts = extract_time_series(func_img, labels_img, t_r=t_r)  # (T, N)

    if truncate_to is not None:
        ts = ts[:truncate_to]
        print(f"[export] Truncated to T={ts.shape[0]}")

    print(f"[export] Time series shape: {ts.shape} (T, N)")
    adj = compute_fc_adjacency(ts, threshold=threshold)  # (N, N)
    print(f"[export] Adjacency shape: {adj.shape} (N, N), nonzeros={np.count_nonzero(adj)}")

    full_state = make_full_state(ts)  # (T, N, 3)
    print(f"[export] full_state shape: {full_state.shape} (T, N, 3)")

    np.save(os.path.join(out_dir, "full_state.npy"), full_state)
    np.save(os.path.join(out_dir, "adjacency.npy"), adj)

    meta = {
        "subject_index": subject_index,
        "SUB_ID": sub_id,
        "DX_GROUP": int(dx_group) if dx_group is not None else None,
        "SEX": int(sex) if sex is not None else None,
        "atlas_scale": atlas_scale,
        "t_r": float(t_r),
        "threshold": float(threshold),
        "T": int(full_state.shape[0]),
        "N": int(full_state.shape[1]),
    }
    np.savez(os.path.join(out_dir, "abide_export_meta.npz"), **meta)

    print(f"[export] Saved:\n  {out_dir}/full_state_fmri.npy\n  {out_dir}/adjacency_fmri.npy\n  {out_dir}/abide_export_meta.npz")


if __name__ == "__main__":
    data_dir = "/gpfs/gibbs/pi/krishnaswamy_smita/sv496/nilearn_data/ABIDE_pcp/"

    abide_data = fetch_abide_balanced(data_dir=data_dir, n_per_group=40)

    export_for_braindyn_main(
        abide_data=abide_data,
        out_dir="data",
        subject_index=0,      # change to try different subjects
        atlas_scale="scale197",
        t_r=2.5,
        threshold=0.3,
        truncate_to=None      # e.g., 200 if you want faster debugging
    )
