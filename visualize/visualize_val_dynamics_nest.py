from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.sn_dataset import DATA_NPZ_PATH, make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import total_loss


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize fixed-horizon dynamics on the NEST CV validation split.",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/braindyn_nest_unperturbed_best_fold1.pt",
        help="Path to trained NEST forecasting checkpoint.",
    )
    ap.add_argument(
        "--npz_path",
        type=str,
        default=None,
        help="Optional override for dataset path. Defaults to checkpoint args or sn_dataset default.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override. Defaults to checkpoint training batch size.",
    )
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--fold", type=int, default=None, help="Override checkpoint fold index (1-based).")
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--n_rois", type=int, default=6, help="Number of evenly-spaced ROI traces to show per sample")
    ap.add_argument("--n_extreme_rois", type=int, default=3, help="Number of best- and worst-MSE ROIs to show per sample")
    ap.add_argument("--n_samples", type=int, default=None, help="Deprecated alias: if set, uses this for both --n_best and --n_worst")
    ap.add_argument("--n_best", type=int, default=3, help="Number of best-MSE samples to save")
    ap.add_argument("--n_worst", type=int, default=3, help="Number of worst-MSE samples to save")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/nest_val_dynamics")
    return ap.parse_args()


def get_cfg_value(ckpt_cfg: dict[str, Any], key: str, default: Any):
    return ckpt_cfg[key] if key in ckpt_cfg and ckpt_cfg[key] is not None else default


def sanitize_label(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def batch_to_model_tensors_nest(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)
    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def pearson_corr(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    p = y_pred.reshape(-1).astype(np.float64)
    t = y_true.reshape(-1).astype(np.float64)
    p = p - p.mean()
    t = t - t.mean()
    den = np.sqrt(np.dot(p, p) * np.dot(t, t))
    if den <= eps:
        return 0.0
    return float(np.dot(p, t) / den)


def wape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    num = np.sum(np.abs(y_pred.reshape(-1).astype(np.float64) - y_true.reshape(-1).astype(np.float64)))
    den = np.sum(np.abs(y_true.reshape(-1).astype(np.float64)))
    return float(num / max(den, eps))


def spearman_corr(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = y_pred.reshape(-1).astype(np.float64)
    t = y_true.reshape(-1).astype(np.float64)
    res = spearmanr(p, t)
    val = res.statistic if hasattr(res, "statistic") else res.correlation
    return float(val) if not np.isnan(float(val)) else 0.0


def dtw_distance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p2d = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
    t2d = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    T, n = p2d.shape
    total = 0.0
    for roi in range(n):
        p = p2d[:, roi].astype(np.float64)
        t = t2d[:, roi].astype(np.float64)
        D = np.full((T + 1, T + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, T + 1):
            for j in range(1, T + 1):
                cost = abs(p[i - 1] - t[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        total += D[T, T]
    return total / max(n, 1)


def make_subset_loader(dataset, indices: np.ndarray, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def _draw_roi_section(
    axes_section,
    roi_ids,
    roi_mse_per_roi,
    context_raw,
    future_raw,
    pred_raw,
    context_steps,
    future_steps,
    full_t,
    future_t,
    section_label,
    show_legend,
):
    for row_idx, (ax, roi) in enumerate(zip(axes_section, roi_ids)):
        ax.axvspan(0, context_steps - 1, color="#dbeafe", alpha=0.35, zorder=0)
        ax.axvspan(context_steps - 1, context_steps + future_steps - 1, color="#fef3c7", alpha=0.26, zorder=0)
        ax.plot(full_t[:context_steps], context_raw[:, roi], label="context", linewidth=2.4, color="#2563eb")
        ax.plot(future_t, future_raw[:, roi], label="ground truth", linewidth=2.4, color="#0f766e")
        ax.plot(future_t, pred_raw[:, roi], label="prediction", linewidth=2.1, linestyle="--", color="#dc2626")
        ax.axvline(context_steps - 1, color="#4b5563", linestyle=":", linewidth=1.2)
        ax.set_ylabel(f"ROI {roi}\\nmse={roi_mse_per_roi[roi]:.4f}", fontsize=9)
        ax.grid(alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)
        if row_idx == 0:
            ax.set_title(section_label, loc="left", fontsize=10, fontweight="bold", color="#374151", pad=4)
            if show_legend:
                ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cdd4de")


def plot_val_sample(
    out_dir: Path,
    group: str,
    sample_index: int,
    meta: dict,
    context_raw: np.ndarray,
    future_raw: np.ndarray,
    pred_raw: np.ndarray,
    n_rois: int,
    n_extreme_rois: int,
) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c7c9ce",
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#2f3542",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#d5d9e0",
            "grid.alpha": 0.55,
            "font.size": 10,
        }
    )

    n_nodes = future_raw.shape[1]
    roi_mse_per_roi = np.mean((pred_raw - future_raw) ** 2, axis=0)

    roi_count = min(n_rois, n_nodes)
    sampled_roi_ids = np.linspace(0, n_nodes - 1, roi_count, dtype=int)

    n_ext = min(n_extreme_rois, n_nodes)
    sorted_by_mse = np.argsort(roi_mse_per_roi)
    best_roi_ids = sorted_by_mse[:n_ext]
    worst_roi_ids = sorted_by_mse[-n_ext:][::-1]

    total_rows = roi_count + n_ext + n_ext
    context_steps = context_raw.shape[0]
    future_steps = future_raw.shape[0]
    full_t = np.arange(context_steps + future_steps)
    future_t = np.arange(context_steps, context_steps + future_steps)

    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 2.4 * total_rows), sharex=True)
    if total_rows == 1:
        axes = [axes]

    _draw_roi_section(
        axes[:roi_count],
        sampled_roi_ids,
        roi_mse_per_roi,
        context_raw,
        future_raw,
        pred_raw,
        context_steps,
        future_steps,
        full_t,
        future_t,
        section_label="Evenly-sampled ROIs",
        show_legend=True,
    )
    _draw_roi_section(
        axes[roi_count : roi_count + n_ext],
        best_roi_ids,
        roi_mse_per_roi,
        context_raw,
        future_raw,
        pred_raw,
        context_steps,
        future_steps,
        full_t,
        future_t,
        section_label="Best-MSE ROIs",
        show_legend=False,
    )
    _draw_roi_section(
        axes[roi_count + n_ext :],
        worst_roi_ids,
        roi_mse_per_roi,
        context_raw,
        future_raw,
        pred_raw,
        context_steps,
        future_steps,
        full_t,
        future_t,
        section_label="Worst-MSE ROIs",
        show_legend=False,
    )

    for sep_idx in [roi_count - 1, roi_count + n_ext - 1]:
        axes[sep_idx].spines["bottom"].set_linewidth(1.8)
        axes[sep_idx].spines["bottom"].set_color("#94a3b8")
        axes[sep_idx].spines["bottom"].set_alpha(1.0)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(
        (
            f"NEST Validation Window ({group}) | mse={meta['sample_mse']:.6f} | mae={meta['sample_mae']:.6f} | "
            f"wape={meta['sample_wape']:.4f} | dtw={meta['sample_dtw']:.4f} | "
            f"pcc={meta['sample_pcc']:.4f} | scc={meta['sample_scc']:.4f} | "
            f"subject={meta['subject_id']} t0={meta['t_start']}"
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    file_name = (
        f"{group}_{sample_index:03d}_mse-{meta['sample_mse']:.6f}_pcc-{meta['sample_pcc']:.4f}"
        f"_subject-{sanitize_label(meta['subject_id'])}_t0-{int(meta['t_start']):04d}.png"
    )
    out_path = out_dir / file_name
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def extract_raw_context_future(rates: np.ndarray, subject_id: int, t_start: int, x_len: int, y_len: int) -> tuple[np.ndarray, np.ndarray]:
    subject_tc = np.asarray(rates[subject_id], dtype=np.float32).T
    context_raw = subject_tc[t_start : t_start + x_len]
    future_raw = subject_tc[t_start + x_len : t_start + x_len + y_len]
    if context_raw.shape[0] != x_len or future_raw.shape[0] != y_len:
        raise RuntimeError(
            f"Invalid sample window for subject={subject_id}, t_start={t_start}: "
            f"context={context_raw.shape}, future={future_raw.shape}"
        )
    return context_raw, future_raw


def to_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_best = int(args.n_samples) if args.n_samples is not None else int(args.n_best)
    n_worst = int(args.n_samples) if args.n_samples is not None else int(args.n_worst)
    if n_best < 0 or n_worst < 0:
        raise ValueError("n_best and n_worst must be >= 0")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    npz_path = Path(args.npz_path or get_cfg_value(ckpt_args, "npz_path", str(DATA_NPZ_PATH)))
    if not npz_path.exists():
        raise FileNotFoundError(f"NEST dataset not found: {npz_path}")

    x_len = int(get_cfg_value(ckpt_args, "x", 30))
    y_len = int(get_cfg_value(ckpt_args, "y", 10))
    stride = int(get_cfg_value(ckpt_args, "stride", 1))
    train_frac = float(get_cfg_value(ckpt_args, "train_frac", 0.8))
    val_frac = float(get_cfg_value(ckpt_args, "val_frac", 0.1))
    split_seed = int(get_cfg_value(ckpt_args, "split_seed", 0))
    dt = float(get_cfg_value(ckpt_args, "dt", 1.0))
    lambda_mse = float(get_cfg_value(ckpt_args, "lambda_mse", 1.0))
    lambda_mae = float(get_cfg_value(ckpt_args, "lambda_mae", 0.0))

    batch_size = int(args.batch_size or int(get_cfg_value(ckpt_args, "batch_size", 32)))
    cv_folds = int(get_cfg_value(ckpt_args, "cv_folds", 5))
    split_rng_seed = int(get_cfg_value(ckpt_args, "seed", args.seed))

    loaders = make_dataloaders(
        npz_path=str(npz_path),
        task_mode="forecasting",
        x=x_len,
        y=y_len,
        stride=stride,
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_frac=train_frac,
        val_frac=val_frac,
        split_seed=split_seed,
        cache=args.cache,
        pin_memory=(not args.no_pin_memory),
        verbose=False,
    )
    train_dataset = loaders["train"].dataset
    val_dataset = loaders["val"].dataset
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    if len(combined_dataset) == 0:
        raise RuntimeError("Combined train+val dataset is empty.")
    if len(combined_dataset) < cv_folds:
        raise ValueError(f"Not enough combined train+val samples ({len(combined_dataset)}) for {cv_folds}-fold CV")

    fold_from_ckpt = int(ckpt.get("fold", 1))
    fold_idx_1based = int(args.fold) if args.fold is not None else fold_from_ckpt
    if fold_idx_1based < 1 or fold_idx_1based > cv_folds:
        raise ValueError(f"fold must be in [1, {cv_folds}], got {fold_idx_1based}")

    rng = np.random.default_rng(split_rng_seed)
    all_indices = np.arange(len(combined_dataset))
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, cv_folds)
    val_idx = fold_indices[fold_idx_1based - 1]

    val_loader = make_subset_loader(
        dataset=combined_dataset,
        indices=val_idx,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        shuffle=False,
    )

    edge_index = ckpt.get("edge_index", None)
    if edge_index is None:
        raise RuntimeError("Checkpoint does not include edge_index.")
    edge_index = edge_index.to(device)

    n_nodes = int(train_dataset.n_channels)
    model_cfg = BrainDynConfig(
        signal_dim=1,
        hidden_dim=int(get_cfg_value(ckpt_args, "hidden_dim", 64)),
        num_nodes=n_nodes,
        window_size=x_len,
        lstm_layers=int(get_cfg_value(ckpt_args, "lstm_layers", 1)),
        lstm_dropout=float(get_cfg_value(ckpt_args, "lstm_dropout", 0.0)),
        map_hidden_dim=int(get_cfg_value(ckpt_args, "map_hidden_dim", 16)),
        vf_hidden_dim=int(get_cfg_value(ckpt_args, "vf_hidden_dim", 128)),
        ode_method=str(get_cfg_value(ckpt_args, "ode_method", "rk4")),
        use_gat=bool(get_cfg_value(ckpt_args, "ablation_gat", False)),
        use_lstm_encoder=(not bool(get_cfg_value(ckpt_args, "ablation_no_lstm", False))),
        precompute_lap_h=bool(get_cfg_value(ckpt_args, "precompute_lap_h", False)),
    )
    model = BrainDyn(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_file = np.load(str(npz_path), mmap_mode=(None if args.cache else "r"), allow_pickle=True)
    rates = npz_file["smoothed_rates_hz_original"]

    total_running = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_wape = 0.0
    total_dtw = 0.0
    total_corr = 0.0
    total_scc = 0.0
    n_corr = 0
    n_batches = 0
    windows_processed = 0
    saved_paths: list[str] = []
    best_heap = []
    worst_heap = []
    unique_counter = 0

    pbar = tqdm(val_loader, desc="nest-val-visualize", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            x_history, y_true = batch_to_model_tensors_nest(batch, device)
            out = model(
                x_history=x_history,
                edge_index=edge_index,
                pred_steps=y_true.shape[0],
                dt=dt,
                autoregressive=False,
                return_aux=False,
            )
            y_pred = out["x_pred"]

            losses = total_loss(y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
            total_running += float(losses["total"].detach().cpu())
            total_mse += float(losses["mse"].detach().cpu())
            total_mae += float(losses["mae"].detach().cpu())
            n_batches += 1

            batch_size_curr = x_history.shape[0]
            windows_processed += batch_size_curr

            y_pred_bn = y_pred.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()
            y_true_bn = y_true.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()

            meta = batch["meta"]
            subj_ids = meta["subject_index"]
            t_starts = meta["t_start"]

            for i in range(batch_size_curr):
                sample_mse = float(np.mean((y_pred_bn[i] - y_true_bn[i]) ** 2))
                sample_mae = float(np.mean(np.abs(y_pred_bn[i] - y_true_bn[i])))
                sample_wape = wape(y_pred_bn[i], y_true_bn[i])
                sample_dtw = dtw_distance(y_pred_bn[i], y_true_bn[i])
                sample_pcc = pearson_corr(y_pred_bn[i], y_true_bn[i])
                sample_scc = spearman_corr(y_pred_bn[i], y_true_bn[i])

                total_corr += sample_pcc
                total_scc += sample_scc
                total_wape += sample_wape
                total_dtw += sample_dtw
                n_corr += 1

                record = {
                    "sample_mse": sample_mse,
                    "sample_mae": sample_mae,
                    "sample_wape": sample_wape,
                    "sample_dtw": sample_dtw,
                    "sample_pcc": sample_pcc,
                    "sample_scc": sample_scc,
                    "subject_id": to_int(subj_ids[i]),
                    "t_start": to_int(t_starts[i]),
                    "pred_norm": y_pred_bn[i].astype(np.float32),
                }

                if n_best > 0:
                    best_item = (-sample_mse, unique_counter, record)
                    if len(best_heap) < n_best:
                        heapq.heappush(best_heap, best_item)
                    elif best_item[0] > best_heap[0][0]:
                        heapq.heapreplace(best_heap, best_item)

                if n_worst > 0:
                    worst_item = (sample_mse, unique_counter, record)
                    if len(worst_heap) < n_worst:
                        heapq.heappush(worst_heap, worst_item)
                    elif worst_item[0] > worst_heap[0][0]:
                        heapq.heapreplace(worst_heap, worst_item)

                unique_counter += 1

            pbar.set_postfix(
                {
                    "total": f"{total_running / max(n_batches, 1):.4f}",
                    "mse": f"{total_mse / max(n_batches, 1):.4f}",
                    "mae": f"{total_mae / max(n_batches, 1):.4f}",
                    "wape": f"{total_wape / max(n_corr, 1):.4f}",
                    "dtw": f"{total_dtw / max(n_corr, 1):.4f}",
                    "pcc": f"{total_corr / max(n_corr, 1):.4f}",
                    "scc": f"{total_scc / max(n_corr, 1):.4f}",
                }
            )

    best_records = [item[2] for item in sorted(best_heap, key=lambda x: (-x[0], x[1]))]
    worst_records = [item[2] for item in sorted(worst_heap, key=lambda x: (-x[0], x[1]))]

    best_samples = []
    for idx, rec in enumerate(best_records):
        context_raw, future_raw = extract_raw_context_future(
            rates=rates,
            subject_id=rec["subject_id"],
            t_start=rec["t_start"],
            x_len=x_len,
            y_len=y_len,
        )
        mean = context_raw.mean(axis=0, keepdims=True)
        std = context_raw.std(axis=0, keepdims=True).clip(1e-6)
        pred_raw = rec["pred_norm"] * std + mean

        sample_meta = {
            "sample_mse": rec["sample_mse"],
            "sample_mae": rec["sample_mae"],
            "sample_wape": rec["sample_wape"],
            "sample_dtw": rec["sample_dtw"],
            "sample_pcc": rec["sample_pcc"],
            "sample_scc": rec["sample_scc"],
            "subject_id": rec["subject_id"],
            "t_start": rec["t_start"],
        }
        out_path = plot_val_sample(
            out_dir=out_dir,
            group="best",
            sample_index=idx,
            meta=sample_meta,
            context_raw=context_raw,
            future_raw=future_raw,
            pred_raw=pred_raw,
            n_rois=args.n_rois,
            n_extreme_rois=args.n_extreme_rois,
        )
        saved_paths.append(str(out_path))
        best_samples.append(
            {
                "sample_mse": rec["sample_mse"],
                "sample_mae": rec["sample_mae"],
                "sample_wape": rec["sample_wape"],
                "sample_dtw": rec["sample_dtw"],
                "sample_pcc": rec["sample_pcc"],
                "sample_scc": rec["sample_scc"],
                "subject_id": rec["subject_id"],
                "t_start": rec["t_start"],
                "plot_path": str(out_path),
            }
        )

    worst_samples = []
    for idx, rec in enumerate(worst_records):
        context_raw, future_raw = extract_raw_context_future(
            rates=rates,
            subject_id=rec["subject_id"],
            t_start=rec["t_start"],
            x_len=x_len,
            y_len=y_len,
        )
        mean = context_raw.mean(axis=0, keepdims=True)
        std = context_raw.std(axis=0, keepdims=True).clip(1e-6)
        pred_raw = rec["pred_norm"] * std + mean

        sample_meta = {
            "sample_mse": rec["sample_mse"],
            "sample_mae": rec["sample_mae"],
            "sample_wape": rec["sample_wape"],
            "sample_dtw": rec["sample_dtw"],
            "sample_pcc": rec["sample_pcc"],
            "sample_scc": rec["sample_scc"],
            "subject_id": rec["subject_id"],
            "t_start": rec["t_start"],
        }
        out_path = plot_val_sample(
            out_dir=out_dir,
            group="worst",
            sample_index=idx,
            meta=sample_meta,
            context_raw=context_raw,
            future_raw=future_raw,
            pred_raw=pred_raw,
            n_rois=args.n_rois,
            n_extreme_rois=args.n_extreme_rois,
        )
        saved_paths.append(str(out_path))
        worst_samples.append(
            {
                "sample_mse": rec["sample_mse"],
                "sample_mae": rec["sample_mae"],
                "sample_wape": rec["sample_wape"],
                "sample_dtw": rec["sample_dtw"],
                "sample_pcc": rec["sample_pcc"],
                "sample_scc": rec["sample_scc"],
                "subject_id": rec["subject_id"],
                "t_start": rec["t_start"],
                "plot_path": str(out_path),
            }
        )

    npz_file.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "npz_path": str(npz_path),
        "split": "val_cv",
        "fold": fold_idx_1based,
        "cv_folds": cv_folds,
        "x": x_len,
        "y": y_len,
        "stride": stride,
        "dt": dt,
        "lambda_mse": lambda_mse,
        "lambda_mae": lambda_mae,
        "n_best_requested": n_best,
        "n_worst_requested": n_worst,
        "batches_evaluated": n_batches,
        "windows_processed": windows_processed,
        "samples_visualized": len(saved_paths),
        "mean_total": (total_running / max(n_batches, 1)),
        "mean_mse": (total_mse / max(n_batches, 1)),
        "mean_mae": (total_mae / max(n_batches, 1)),
        "mean_wape": (total_wape / max(n_corr, 1)),
        "mean_dtw": (total_dtw / max(n_corr, 1)),
        "mean_pcc": (total_corr / max(n_corr, 1)),
        "mean_scc": (total_scc / max(n_corr, 1)),
        "best_samples": best_samples,
        "worst_samples": worst_samples,
        "out_dir": str(out_dir),
        "saved_plots": saved_paths,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved NEST validation dynamics plots and summary:")
    print(f"  out_dir: {out_dir}")
    print(f"  summary: {summary_path}")
    print(f"  mean val total: {summary['mean_total']:.6f}")
    print(f"  mean val MSE:   {summary['mean_mse']:.6f}")
    print(f"  mean val MAE:   {summary['mean_mae']:.6f}")
    print(f"  mean val WAPE:  {summary['mean_wape']:.6f}")
    print(f"  mean val DTW:   {summary['mean_dtw']:.6f}")
    print(f"  mean val PCC:   {summary['mean_pcc']:.6f}")
    print(f"  mean val SCC:   {summary['mean_scc']:.6f}")


if __name__ == "__main__":
    main()
