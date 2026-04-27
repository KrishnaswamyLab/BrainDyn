from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data.rbc_dataset import make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import total_loss


def parse_args():
    ap = argparse.ArgumentParser(
                description="Visualize autoregressive test-rollout dynamics on the TEST split."
    )
    ap.add_argument("--checkpoint", type=str, default="checkpoints/braindyn_rbc_pnc_best_fold1.pt")
    ap.add_argument("--manifest_csv", type=str, default=None, help="Defaults to training config from checkpoint")
    ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None (defaults to checkpoint config)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--max_runs", type=int, default=None, help="Optional cap on the number of test runs to roll out")
    ap.add_argument("--n_rois", type=int, default=6, help="Number of evenly-spaced ROI traces to show per sample")
    ap.add_argument("--n_extreme_rois", type=int, default=3, help="Number of best- and worst-MSE ROIs to show per sample")
    ap.add_argument("--n_samples", type=int, default=3, help="Number of sample plots to save")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/test_dynamics")
    return ap.parse_args()


def get_cfg_value(ckpt_cfg, key, default):
    return ckpt_cfg[key] if key in ckpt_cfg and ckpt_cfg[key] is not None else default


def batch_to_model_tensors(batch, device):
    """Convert RBC batch to BrainDyn tensor shapes."""
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)
    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "_")


def pearson_corr(y_pred, y_true, eps=1e-8):
    """Compute robust Pearson correlation on flattened arrays."""
    p = y_pred.reshape(-1).astype(np.float64)
    t = y_true.reshape(-1).astype(np.float64)
    p = p - p.mean()
    t = t - t.mean()
    den = np.sqrt(np.dot(p, p) * np.dot(t, t))
    if den <= eps:
        return 0.0
    return float(np.dot(p, t) / den)


def iter_test_runs(loader, max_runs=None):
    yielded = 0
    for batch in loader:
        meta = batch["meta"]
        for sample_idx, (path, t_start, run_len, cohort, subject_id, run) in enumerate(
            zip(
                meta["path"],
                meta["t_start"],
                meta["T"],
                meta["cohort"],
                meta["subject_id"],
                meta["run"],
            )
        ):
            yield {
                "path": path,
                "t_start": int(t_start),
                "run_len": int(run_len),
                "cohort": cohort,
                "subject_id": subject_id,
                "run": run,
                "sample_idx": sample_idx,
            }
            yielded += 1
            if max_runs is not None and yielded >= max_runs:
                return


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
    """Render one labelled section of ROI subplots."""
    for row_idx, (ax, roi) in enumerate(zip(axes_section, roi_ids)):
        ax.axvspan(0, context_steps - 1, color="#dbeafe", alpha=0.35, zorder=0)
        ax.axvspan(context_steps - 1, context_steps + future_steps - 1, color="#fef3c7", alpha=0.26, zorder=0)
        ax.plot(full_t[:context_steps], context_raw[:, roi], label="context", linewidth=2.4, color="#2563eb")
        ax.plot(future_t, future_raw[:, roi], label="ground truth", linewidth=2.4, color="#0f766e")
        ax.plot(future_t, pred_raw[:, roi], label="prediction", linewidth=2.1, linestyle="--", color="#dc2626")
        ax.axvline(context_steps - 1, color="#4b5563", linestyle=":", linewidth=1.2)
        ax.set_ylabel(f"ROI {roi}\nmse={roi_mse_per_roi[roi]:.4f}", fontsize=9)
        ax.grid(alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)
        if row_idx == 0:
            ax.set_title(section_label, loc="left", fontsize=10, fontweight="bold", color="#374151", pad=4)
            if show_legend:
                ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cdd4de")


def plot_rollout_sample(
    out_dir,
    sample_index,
    meta,
    context_raw,
    future_raw,
    pred_raw,
    n_rois,
    n_extreme_rois=3,
):
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

    # Per-ROI MSE over the prediction window
    roi_mse_per_roi = np.mean((pred_raw - future_raw) ** 2, axis=0)  # (n_nodes,)

    # Section 1: evenly-spaced ROIs (overview)
    roi_count = min(n_rois, n_nodes)
    sampled_roi_ids = np.linspace(0, n_nodes - 1, roi_count, dtype=int)

    # Section 2 & 3: best / worst ROIs by per-ROI MSE
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

    # Draw the three sections
    _draw_roi_section(
        axes[:roi_count], sampled_roi_ids, roi_mse_per_roi,
        context_raw, future_raw, pred_raw,
        context_steps, future_steps, full_t, future_t,
        section_label="Evenly-sampled ROIs", show_legend=True,
    )
    _draw_roi_section(
        axes[roi_count : roi_count + n_ext], best_roi_ids, roi_mse_per_roi,
        context_raw, future_raw, pred_raw,
        context_steps, future_steps, full_t, future_t,
        section_label="Best-MSE ROIs", show_legend=False,
    )
    _draw_roi_section(
        axes[roi_count + n_ext :], worst_roi_ids, roi_mse_per_roi,
        context_raw, future_raw, pred_raw,
        context_steps, future_steps, full_t, future_t,
        section_label="Worst-MSE ROIs", show_legend=False,
    )

    # Draw thin separator lines between sections
    for sep_idx in [roi_count - 1, roi_count + n_ext - 1]:
        axes[sep_idx].spines["bottom"].set_linewidth(1.8)
        axes[sep_idx].spines["bottom"].set_color("#94a3b8")
        axes[sep_idx].spines["bottom"].set_alpha(1.0)

    axes[-1].set_xlabel("Time step")

    fig.suptitle(
        (
            f"Test Rollout | mse={meta['sample_mse']:.6f} | corr={meta['sample_corr']:.4f} | "
            f"cohort={meta['cohort']} subject={meta['subject_id']} run={meta['run']}"
        ),
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    file_name = (
        f"sample_{sample_index:03d}_mse-{meta['sample_mse']:.6f}_corr-{meta['sample_corr']:.4f}"
        f"_cohort-{sanitize_label(meta['cohort'])}"
        f"_subject-{sanitize_label(meta['subject_id'])}_run-{sanitize_label(meta['run'])}.png"
    )
    out_path = out_dir / file_name
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    manifest_csv = Path(args.manifest_csv or get_cfg_value(ckpt_cfg, "manifest_csv", "data/manifest.csv"))
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    cohort = args.cohort if args.cohort is not None else get_cfg_value(ckpt_cfg, "cohort", None)
    x = int(get_cfg_value(ckpt_cfg, "x", 20))
    y = int(get_cfg_value(ckpt_cfg, "y", 10))
    stride = int(get_cfg_value(ckpt_cfg, "stride", 10))
    min_t = int(get_cfg_value(ckpt_cfg, "min_t", 0))
    dt = float(get_cfg_value(ckpt_cfg, "dt", 1.0))
    lambda_mse = float(get_cfg_value(ckpt_cfg, "lambda_mse", 1.0))
    lambda_mae = float(get_cfg_value(ckpt_cfg, "lambda_mae", 0.0))
    ode_method = str(get_cfg_value(ckpt_cfg, "ode_method", "rk4"))

    hidden_dim = int(get_cfg_value(ckpt_cfg, "hidden_dim", 64))
    lstm_layers = int(get_cfg_value(ckpt_cfg, "lstm_layers", 1))
    lstm_dropout = float(get_cfg_value(ckpt_cfg, "lstm_dropout", 0.0))
    map_hidden_dim = int(get_cfg_value(ckpt_cfg, "map_hidden_dim", 16))
    vf_hidden_dim = int(get_cfg_value(ckpt_cfg, "vf_hidden_dim", 128))
    forecast_mode = str(get_cfg_value(ckpt_cfg, "forecast_mode", "short"))
    ar_chunk_size = int(get_cfg_value(ckpt_cfg, "ar_chunk_size", 1))
    use_gat = bool(get_cfg_value(ckpt_cfg, "ablation_gat", False))
    use_lstm_encoder = not bool(get_cfg_value(ckpt_cfg, "ablation_no_lstm", False))

    loaders = make_dataloaders(
        manifest_csv=manifest_csv,
        x=x,
        y=y,
        stride=stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cohort=cohort,
        min_t=min_t,
        cache=args.cache,
        pin_memory=not args.no_pin_memory,
    )
    test_loader = loaders["test"]

    if len(test_loader.dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError("Test split is empty. Check cohort/x/y/stride/min_t settings.")

    if "edge_index" not in ckpt:
        raise KeyError("Checkpoint missing 'edge_index'. Use a checkpoint saved by main.py training.")
    edge_index = ckpt["edge_index"].to(device)
    num_nodes = int(get_cfg_value(ckpt_cfg, "num_nodes", int(edge_index.max().item()) + 1))

    config = BrainDynConfig(
        signal_dim=1,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        window_size=x,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        map_hidden_dim=map_hidden_dim,
        vf_hidden_dim=vf_hidden_dim,
        ode_method=ode_method,
        use_gat=use_gat,
        use_lstm_encoder=use_lstm_encoder,
    )
    model = BrainDyn(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_running = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_corr = 0.0
    n_corr = 0
    n_chunks = 0
    runs_processed = 0
    saved = 0
    saved_paths = []
    sample_summaries = []

    with torch.no_grad():
        if forecast_mode == "long":
            pbar = tqdm(iter_test_runs(test_loader, max_runs=args.max_runs), desc="test-rollout-visualize", leave=False)
            for run_meta in pbar:
                ts = np.loadtxt(run_meta["path"], delimiter=",", comments="#", dtype=np.float32)
                t0 = run_meta["t_start"]
                run_len = run_meta["run_len"]

                context_raw = ts[t0 : t0 + x]
                mean = context_raw.mean(axis=0, keepdims=True)
                std = context_raw.std(axis=0, keepdims=True).clip(1e-6)

                hist_norm = ((context_raw - mean) / std).astype(np.float32)
                hist = torch.from_numpy(hist_norm).to(device=device)
                hist = hist.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)

                current_t = t0 + x
                remaining = run_len - current_t
                pred_chunks = []
                gt_chunks = []
                gt_raw_chunks = []

                while remaining > 0:
                    step = min(y, remaining)
                    if ar_chunk_size > 0:
                        step = min(step, ar_chunk_size)

                    out = model(
                        x_history=hist,
                        edge_index=edge_index,
                        pred_steps=step,
                        dt=dt,
                        autoregressive=False,
                    )

                    y_pred = out["x_pred"]
                    gt_raw = ts[current_t : current_t + step]
                    gt_norm = ((gt_raw - mean) / std).astype(np.float32)
                    y_true = torch.from_numpy(gt_norm).to(device=edge_index.device)
                    y_true = y_true.unsqueeze(1).unsqueeze(-1)

                    losses = total_loss(y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
                    total_running += float(losses["total"].detach().cpu())
                    total_mse += float(losses["mse"].detach().cpu())
                    total_mae += float(losses["mae"].detach().cpu())
                    n_chunks += 1

                    pred_step = y_pred.squeeze(1).squeeze(-1).detach().cpu().numpy()
                    chunk_corr = pearson_corr(pred_step, gt_norm)
                    total_corr += chunk_corr
                    n_corr += 1

                    pred_chunks.append(pred_step)
                    gt_chunks.append(gt_norm)
                    gt_raw_chunks.append(gt_raw)

                    pred_hist = y_pred.permute(1, 2, 0, 3)
                    hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)

                    current_t += step
                    remaining = run_len - current_t

                    pbar.set_postfix(
                        {
                            "total": f"{total_running / max(n_chunks, 1):.4f}",
                            "mse": f"{total_mse / max(n_chunks, 1):.4f}",
                            "mae": f"{total_mae / max(n_chunks, 1):.4f}",
                            "corr": f"{total_corr / max(n_corr, 1):.4f}",
                        }
                    )

                runs_processed += 1

                sample_mse = float("nan")
                sample_corr = float("nan")
                if pred_chunks:
                    pred_norm = np.concatenate(pred_chunks, axis=0)
                    gt_norm_full = np.concatenate(gt_chunks, axis=0)
                    sample_mse = float(np.mean((pred_norm - gt_norm_full) ** 2))
                    sample_corr = pearson_corr(pred_norm, gt_norm_full)

                if saved < args.n_samples and pred_chunks:
                    future_raw = np.concatenate(gt_raw_chunks, axis=0)
                    pred_raw = pred_norm * std + mean
                    run_meta_vis = dict(run_meta)
                    run_meta_vis["sample_mse"] = sample_mse
                    run_meta_vis["sample_corr"] = sample_corr
                    out_path = plot_rollout_sample(
                        out_dir=out_dir,
                        sample_index=saved,
                        meta=run_meta_vis,
                        context_raw=context_raw,
                        future_raw=future_raw,
                        pred_raw=pred_raw,
                        n_rois=args.n_rois,
                        n_extreme_rois=args.n_extreme_rois,
                    )
                    saved_paths.append(str(out_path))
                    sample_summaries.append(
                        {
                            "sample_index": saved,
                            "sample_mse": sample_mse,
                            "sample_corr": sample_corr,
                            "cohort": run_meta["cohort"],
                            "subject_id": run_meta["subject_id"],
                            "run": run_meta["run"],
                            "t_start": run_meta["t_start"],
                            "plot_path": str(out_path),
                        }
                    )
                    saved += 1
        else:
            pbar = tqdm(test_loader, desc="test-window-visualize", leave=False)
            for batch in pbar:
                x_history, y_true = batch_to_model_tensors(batch, device)
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
                n_chunks += 1

                y_pred_bn = y_pred.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()
                y_true_bn = y_true.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()
                x_ctx_bn = batch["x"].detach().cpu().numpy()
                y_future_bn = batch["y"].detach().cpu().numpy()
                meta = batch["meta"]

                for i in range(y_pred_bn.shape[0]):
                    sample_corr = pearson_corr(y_pred_bn[i], y_true_bn[i])
                    total_corr += sample_corr
                    n_corr += 1
                    runs_processed += 1

                    if saved < args.n_samples:
                        sample_mse = float(np.mean((y_pred_bn[i] - y_true_bn[i]) ** 2))
                        run_meta_vis = {
                            "sample_mse": sample_mse,
                            "sample_corr": sample_corr,
                            "cohort": meta["cohort"][i],
                            "subject_id": meta["subject_id"][i],
                            "run": meta["run"][i],
                        }
                        out_path = plot_rollout_sample(
                            out_dir=out_dir,
                            sample_index=saved,
                            meta=run_meta_vis,
                            context_raw=x_ctx_bn[i],
                            future_raw=y_future_bn[i],
                            pred_raw=y_pred_bn[i],
                            n_rois=args.n_rois,
                            n_extreme_rois=args.n_extreme_rois,
                        )
                        saved_paths.append(str(out_path))
                        sample_summaries.append(
                            {
                                "sample_index": saved,
                                "sample_mse": sample_mse,
                                "sample_corr": sample_corr,
                                "cohort": meta["cohort"][i],
                                "subject_id": meta["subject_id"][i],
                                "run": meta["run"][i],
                                "t_start": int(meta["t_start"][i]),
                                "plot_path": str(out_path),
                            }
                        )
                        saved += 1

                pbar.set_postfix(
                    {
                        "total": f"{total_running / max(n_chunks, 1):.4f}",
                        "mse": f"{total_mse / max(n_chunks, 1):.4f}",
                        "mae": f"{total_mae / max(n_chunks, 1):.4f}",
                        "corr": f"{total_corr / max(n_corr, 1):.4f}",
                    }
                )

    summary = {
        "checkpoint": str(ckpt_path),
        "manifest_csv": str(manifest_csv),
        "split": "test",
        "cohort": cohort,
        "x": x,
        "y": y,
        "dt": dt,
        "ode_method": ode_method,
        "forecast_mode": forecast_mode,
        "ar_chunk_size": ar_chunk_size,
        "ablation_gat": use_gat,
        "ablation_no_lstm": (not use_lstm_encoder),
        "lambda_mse": lambda_mse,
        "lambda_mae": lambda_mae,
        "runs_processed": runs_processed,
        "chunks_evaluated": n_chunks,
        "samples_visualized": saved,
        "mean_total": (total_running / max(n_chunks, 1)),
        "mean_mse": (total_mse / max(n_chunks, 1)),
        "mean_mae": (total_mae / max(n_chunks, 1)),
        "mean_corr": (total_corr / max(n_corr, 1)),
        "sample_summaries": sample_summaries,
        "out_dir": str(out_dir),
        "saved_plots": saved_paths,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved test dynamics plots and summary:")
    print(f"  out_dir: {out_dir}")
    print(f"  summary: {summary_path}")
    print(f"  mean test total: {summary['mean_total']:.6f}")
    print(f"  mean test MSE: {summary['mean_mse']:.6f}")
    print(f"  mean test MAE: {summary['mean_mae']:.6f}")
    print(f"  mean test corr: {summary['mean_corr']:.6f}")


if __name__ == "__main__":
    main()
