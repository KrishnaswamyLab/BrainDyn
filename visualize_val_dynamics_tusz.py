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
from tqdm import tqdm

from data.tusz_binary_dataset import make_binary_dataloader
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import total_loss


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize fixed-horizon dynamics on the TUSZ validation split (non-autoregressive)."
    )
    ap.add_argument("--checkpoint", type=str, default="checkpoints/braindyn_tusz_binary_forecast_best_fold4.pt")
    ap.add_argument("--h5_path", type=str, default=None, help="Defaults to training args from checkpoint")
    ap.add_argument("--manifest_csv", type=str, default=None, help="Defaults to training args from checkpoint")
    ap.add_argument("--zscore", action="store_true", help="Apply per-window z-score normalisation")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--max_batches", type=int, default=None, help="Optional cap on number of val batches to evaluate")
    ap.add_argument("--n_channels", type=int, default=6, help="Number of evenly-spaced channels to show per sample")
    ap.add_argument(
        "--n_extreme_channels",
        type=int,
        default=3,
        help="Number of best- and worst-MSE channels to show per sample",
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Deprecated alias: if set, uses this for both --n_best and --n_worst",
    )
    ap.add_argument("--n_best", type=int, default=3, help="Number of best-MSE samples to save")
    ap.add_argument("--n_worst", type=int, default=3, help="Number of worst-MSE samples to save")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/val_dynamics_tusz")
    return ap.parse_args()


def get_cfg_value(cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg and cfg[key] is not None else default


def sanitize_label(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def as_int(meta_value: Any, index: int) -> int:
    if isinstance(meta_value, torch.Tensor):
        return int(meta_value[index].item())
    if isinstance(meta_value, np.ndarray):
        return int(meta_value[index])
    return int(meta_value[index])


def as_str(meta_value: Any, index: int) -> str:
    if isinstance(meta_value, torch.Tensor):
        return str(meta_value[index].item())
    return str(meta_value[index])


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
    time_steps, n_channels = p2d.shape
    total = 0.0
    for ch in range(n_channels):
        p = p2d[:, ch].astype(np.float64)
        t = t2d[:, ch].astype(np.float64)
        d = np.full((time_steps + 1, time_steps + 1), np.inf)
        d[0, 0] = 0.0
        for i in range(1, time_steps + 1):
            for j in range(1, time_steps + 1):
                cost = abs(p[i - 1] - t[j - 1])
                d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
        total += d[time_steps, time_steps]
    return total / max(n_channels, 1)


def _channel_label(channel_names: list[str], ch_idx: int) -> str:
    if 0 <= ch_idx < len(channel_names):
        return f"Ch {channel_names[ch_idx]}"
    return f"Ch {ch_idx}"


def _draw_channel_section(
    axes_section,
    channel_ids: np.ndarray,
    ch_mse_per_channel: np.ndarray,
    context_raw: np.ndarray,
    future_raw: np.ndarray,
    pred_raw: np.ndarray,
    context_steps: int,
    future_steps: int,
    full_t: np.ndarray,
    future_t: np.ndarray,
    section_label: str,
    show_legend: bool,
    channel_names: list[str],
) -> None:
    for row_idx, (ax, ch) in enumerate(zip(axes_section, channel_ids)):
        ax.axvspan(0, context_steps - 1, color="#dbeafe", alpha=0.35, zorder=0)
        ax.axvspan(context_steps - 1, context_steps + future_steps - 1, color="#fef3c7", alpha=0.26, zorder=0)
        ax.plot(full_t[:context_steps], context_raw[:, ch], label="context", linewidth=2.4, color="#2563eb")
        ax.plot(future_t, future_raw[:, ch], label="ground truth", linewidth=2.4, color="#0f766e")
        ax.plot(future_t, pred_raw[:, ch], label="prediction", linewidth=2.1, linestyle="--", color="#dc2626")
        ax.axvline(context_steps - 1, color="#4b5563", linestyle=":", linewidth=1.2)
        ax.set_ylabel(f"{_channel_label(channel_names, int(ch))}\nmse={ch_mse_per_channel[ch]:.4f}", fontsize=9)
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
    meta: dict[str, Any],
    context_raw: np.ndarray,
    future_raw: np.ndarray,
    pred_raw: np.ndarray,
    n_channels: int,
    n_extreme_channels: int,
    channel_names: list[str],
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
    ch_mse_per_channel = np.mean((pred_raw - future_raw) ** 2, axis=0)

    ch_count = min(n_channels, n_nodes)
    sampled_ch_ids = np.linspace(0, n_nodes - 1, ch_count, dtype=int)

    n_ext = min(n_extreme_channels, n_nodes)
    sorted_by_mse = np.argsort(ch_mse_per_channel)
    best_ch_ids = sorted_by_mse[:n_ext]
    worst_ch_ids = sorted_by_mse[-n_ext:][::-1]

    total_rows = ch_count + n_ext + n_ext
    context_steps = context_raw.shape[0]
    future_steps = future_raw.shape[0]
    full_t = np.arange(context_steps + future_steps)
    future_t = np.arange(context_steps, context_steps + future_steps)

    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 2.4 * total_rows), sharex=True)
    if total_rows == 1:
        axes = [axes]

    _draw_channel_section(
        axes[:ch_count],
        sampled_ch_ids,
        ch_mse_per_channel,
        context_raw,
        future_raw,
        pred_raw,
        context_steps,
        future_steps,
        full_t,
        future_t,
        section_label="Evenly-sampled channels",
        show_legend=True,
        channel_names=channel_names,
    )
    _draw_channel_section(
        axes[ch_count : ch_count + n_ext],
        best_ch_ids,
        ch_mse_per_channel,
        context_raw,
        future_raw,
        pred_raw,
        context_steps,
        future_steps,
        full_t,
        future_t,
        section_label="Best-MSE channels",
        show_legend=False,
        channel_names=channel_names,
    )
    _draw_channel_section(
        axes[ch_count + n_ext :],
        worst_ch_ids,
        ch_mse_per_channel,
        context_raw,
        future_raw,
        pred_raw,
        context_steps,
        future_steps,
        full_t,
        future_t,
        section_label="Worst-MSE channels",
        show_legend=False,
        channel_names=channel_names,
    )

    for sep_idx in [ch_count - 1, ch_count + n_ext - 1]:
        if sep_idx >= 0:
            axes[sep_idx].spines["bottom"].set_linewidth(1.8)
            axes[sep_idx].spines["bottom"].set_color("#94a3b8")
            axes[sep_idx].spines["bottom"].set_alpha(1.0)

    axes[-1].set_xlabel("Time step")

    fig.suptitle(
        (
            f"TUSZ Validation Window ({group}) | mse={meta['sample_mse']:.6f} | mae={meta['sample_mae']:.6f} | "
            f"wape={meta['sample_wape']:.4f} | dtw={meta['sample_dtw']:.4f} | "
            f"pcc={meta['sample_pcc']:.4f} | scc={meta['sample_scc']:.4f} | "
            f"condition={meta['condition']} window={meta['window_id']} t0={meta['t0_sample']} file={meta['h5_relpath']}"
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    relpath = Path(meta["h5_relpath"]).name
    file_name = (
        f"{group}_{sample_index:03d}_mse-{meta['sample_mse']:.6f}_pcc-{meta['sample_pcc']:.4f}"
        f"_cond-{sanitize_label(meta['condition'])}"
        f"_file-{sanitize_label(relpath)}"
        f"_wid-{int(meta['window_id']):07d}_t0-{int(meta['t0_sample']):06d}.png"
    )
    out_path = out_dir / file_name
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_best = int(args.n_samples) if args.n_samples is not None else int(args.n_best)
    n_worst = int(args.n_samples) if args.n_samples is not None else int(args.n_worst)
    if n_best < 0 or n_worst < 0:
        raise ValueError("n_best and n_worst must be >= 0")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("args", ckpt.get("config", {}))

    h5_path = Path(args.h5_path or get_cfg_value(ckpt_cfg, "h5_path", "data/tusz_binary.h5"))
    manifest_csv = Path(args.manifest_csv or get_cfg_value(ckpt_cfg, "manifest_csv", "data/manifest_tusz_binary.csv"))
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    x_len = int(get_cfg_value(ckpt_cfg, "x_len", 30))
    y_len = 40 - x_len
    if y_len <= 0:
        raise ValueError(f"Invalid x_len={x_len}; expected 1..39")

    dt = float(get_cfg_value(ckpt_cfg, "dt", 1.0))
    lambda_mse = float(get_cfg_value(ckpt_cfg, "lambda_mse", 1.0))
    lambda_mae = float(get_cfg_value(ckpt_cfg, "lambda_mae", 0.0))
    ode_method = str(get_cfg_value(ckpt_cfg, "ode_method", "rk4"))

    hidden_dim = int(get_cfg_value(ckpt_cfg, "hidden_dim", 64))
    lstm_layers = int(get_cfg_value(ckpt_cfg, "lstm_layers", 1))
    lstm_dropout = float(get_cfg_value(ckpt_cfg, "lstm_dropout", 0.0))
    map_hidden_dim = int(get_cfg_value(ckpt_cfg, "map_hidden_dim", 16))
    vf_hidden_dim = int(get_cfg_value(ckpt_cfg, "vf_hidden_dim", 128))
    use_gat = bool(get_cfg_value(ckpt_cfg, "ablation_gat", False))
    use_lstm_encoder = not bool(get_cfg_value(ckpt_cfg, "ablation_no_lstm", False))
    precompute_lap_h = bool(get_cfg_value(ckpt_cfg, "precompute_lap_h", False))

    val_loader = make_binary_dataloader(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        zscore=args.zscore,
        cache=args.cache,
        pin_memory=(not args.no_pin_memory),
    )

    channel_names = []
    if hasattr(val_loader.dataset, "channel_names"):
        channel_names = list(val_loader.dataset.channel_names)

    if "edge_index" not in ckpt:
        raise KeyError("Checkpoint missing 'edge_index'. Use a forecasting checkpoint saved by train_tusz.py.")
    edge_index = ckpt["edge_index"].to(device)

    num_nodes = int(get_cfg_value(ckpt_cfg, "num_nodes", int(edge_index.max().item()) + 1))
    model_cfg = BrainDynConfig(
        signal_dim=1,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        window_size=x_len,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        map_hidden_dim=map_hidden_dim,
        vf_hidden_dim=vf_hidden_dim,
        ode_method=ode_method,
        use_gat=use_gat,
        use_lstm_encoder=use_lstm_encoder,
        precompute_lap_h=precompute_lap_h,
    )
    model = BrainDyn(model_cfg).to(device)

    model_state = ckpt.get("model_state", ckpt.get("model_state_dict", None))
    if model_state is None:
        raise KeyError("Checkpoint missing model weights: expected 'model_state' or 'model_state_dict'.")
    model.load_state_dict(model_state)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    best_heap: list[tuple[float, int, dict[str, Any]]] = []
    worst_heap: list[tuple[float, int, dict[str, Any]]] = []
    unique_counter = 0

    pbar = tqdm(val_loader, desc="val-visualize-tusz", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            x_full = batch["x"].to(device=device, dtype=torch.float32)  # (B, N, 40)
            x_ctx = x_full[:, :, :x_len]   # (B, N, x_len)
            y_future = x_full[:, :, x_len:]  # (B, N, y_len)

            x_history = x_ctx.unsqueeze(-1)
            y_true = y_future.permute(2, 0, 1).unsqueeze(-1)

            out = model(
                x_history=x_history,
                edge_index=edge_index,
                pred_steps=y_true.shape[0],
                dt=dt,
                autoregressive=False,
            )
            y_pred = out["x_pred"]

            losses = total_loss(y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
            total_running += float(losses["total"].detach().cpu())
            total_mse += float(losses["mse"].detach().cpu())
            total_mae += float(losses["mae"].detach().cpu())
            n_batches += 1

            batch_size = x_ctx.shape[0]
            windows_processed += batch_size
            y_pred_bn = y_pred.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()  # (B, y_len, N)
            y_true_bn = y_true.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()  # (B, y_len, N)
            x_ctx_bn = x_ctx.detach().cpu().numpy()  # (B, N, x_len)
            meta = batch["meta"]

            for i in range(batch_size):
                sample_mse = float(np.mean((y_pred_bn[i] - y_true_bn[i]) ** 2))
                sample_mae = float(np.mean(np.abs(y_pred_bn[i] - y_true_bn[i])))
                sample_wape = wape(y_pred_bn[i], y_true_bn[i])
                sample_dtw = dtw_distance(y_pred_bn[i], y_true_bn[i])
                sample_pcc = pearson_corr(y_pred_bn[i], y_true_bn[i])
                sample_scc = spearman_corr(y_pred_bn[i], y_true_bn[i])

                total_corr += sample_pcc
                total_wape += sample_wape
                total_dtw += sample_dtw
                total_scc += sample_scc
                n_corr += 1

                record = {
                    "sample_mse": sample_mse,
                    "sample_mae": sample_mae,
                    "sample_wape": sample_wape,
                    "sample_dtw": sample_dtw,
                    "sample_pcc": sample_pcc,
                    "sample_scc": sample_scc,
                    "condition": "n/a",
                    "h5_relpath": meta[i]["h5_relpath"],
                    "window_id": int(batch["window_id"][i].item()),
                    "t0_sample": int(meta[i]["t0_sample"]),
                    "context_raw": x_ctx_bn[i].transpose(1, 0).astype(np.float32),
                    "future_raw": y_true_bn[i].astype(np.float32),
                    "pred_raw": y_pred_bn[i].astype(np.float32),
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
        sample_meta = {
            "sample_mse": rec["sample_mse"],
            "sample_mae": rec["sample_mae"],
            "sample_wape": rec["sample_wape"],
            "sample_dtw": rec["sample_dtw"],
            "sample_pcc": rec["sample_pcc"],
            "sample_scc": rec["sample_scc"],
            "condition": rec["condition"],
            "h5_relpath": rec["h5_relpath"],
            "window_id": rec["window_id"],
            "t0_sample": rec["t0_sample"],
        }
        out_path = plot_val_sample(
            out_dir=out_dir,
            group="best",
            sample_index=idx,
            meta=sample_meta,
            context_raw=rec["context_raw"],
            future_raw=rec["future_raw"],
            pred_raw=rec["pred_raw"],
            n_channels=args.n_channels,
            n_extreme_channels=args.n_extreme_channels,
            channel_names=channel_names,
        )
        saved_paths.append(str(out_path))
        best_samples.append({**sample_meta, "plot_path": str(out_path)})

    worst_samples = []
    for idx, rec in enumerate(worst_records):
        sample_meta = {
            "sample_mse": rec["sample_mse"],
            "sample_mae": rec["sample_mae"],
            "sample_wape": rec["sample_wape"],
            "sample_dtw": rec["sample_dtw"],
            "sample_pcc": rec["sample_pcc"],
            "sample_scc": rec["sample_scc"],
            "condition": rec["condition"],
            "h5_relpath": rec["h5_relpath"],
            "window_id": rec["window_id"],
            "t0_sample": rec["t0_sample"],
        }
        out_path = plot_val_sample(
            out_dir=out_dir,
            group="worst",
            sample_index=idx,
            meta=sample_meta,
            context_raw=rec["context_raw"],
            future_raw=rec["future_raw"],
            pred_raw=rec["pred_raw"],
            n_channels=args.n_channels,
            n_extreme_channels=args.n_extreme_channels,
            channel_names=channel_names,
        )
        saved_paths.append(str(out_path))
        worst_samples.append({**sample_meta, "plot_path": str(out_path)})

    summary = {
        "checkpoint": str(ckpt_path),
        "h5_path": str(h5_path),
        "manifest_csv": str(manifest_csv),
        "split": "val",
        "x_len": x_len,
        "y_len": y_len,
        "dt": dt,
        "ode_method": ode_method,
        "ablation_gat": use_gat,
        "ablation_no_lstm": (not use_lstm_encoder),
        "precompute_lap_h": precompute_lap_h,
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
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved TUSZ validation dynamics plots and summary:")
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
