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


def batch_to_model_tensors(batch, device):
    """Convert RBC batch to BrainDyn shapes.

    Input batch:
      x: (B, Lx, N)
      y: (B, Ly, N)

    Returns:
      x_history: (B, N, Lx, 1)
      y_true:    (Ly, B, N, 1)
    """
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)

    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize ground-truth vs predicted dynamics on TEST split only."
    )
    ap.add_argument("--checkpoint", type=str, default="checkpoints/braindyn_rbc_best.pt")
    ap.add_argument("--manifest_csv", type=str, default=None, help="Defaults to training config from checkpoint")
    ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None (defaults to checkpoint config)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--max_batches", type=int, default=10, help="How many test batches to visualize")
    ap.add_argument("--n_rois", type=int, default=6, help="Number of ROI traces to show per sample")
    ap.add_argument("--n_samples", type=int, default=3, help="Number of sample plots to save")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/test_dynamics")
    return ap.parse_args()


def get_cfg_value(ckpt_cfg, key, default):
    return ckpt_cfg[key] if key in ckpt_cfg and ckpt_cfg[key] is not None else default


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

    hidden_dim = int(get_cfg_value(ckpt_cfg, "hidden_dim", 64))
    lstm_layers = int(get_cfg_value(ckpt_cfg, "lstm_layers", 1))
    lstm_dropout = float(get_cfg_value(ckpt_cfg, "lstm_dropout", 0.0))
    map_hidden_dim = int(get_cfg_value(ckpt_cfg, "map_hidden_dim", 128))
    vf_hidden_dim = int(get_cfg_value(ckpt_cfg, "vf_hidden_dim", 128))

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

    config = BrainDynConfig(
        signal_dim=1,
        hidden_dim=hidden_dim,
        window_size=x,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        map_hidden_dim=map_hidden_dim,
        vf_hidden_dim=vf_hidden_dim,
    )
    model = BrainDyn(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "edge_index" not in ckpt:
        raise KeyError("Checkpoint missing 'edge_index'. Use a checkpoint saved by main.py training.")
    edge_index = ckpt["edge_index"].to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_mse = 0.0
    total_mae = 0.0
    n_items = 0
    saved = 0

    pbar = tqdm(test_loader, desc="test-visualize", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.max_batches:
                break

            x_history, y_true = batch_to_model_tensors(batch, device)
            pred_steps = y_true.shape[0]
            out = model(x_history=x_history, edge_index=edge_index, pred_steps=pred_steps, dt=dt)
            y_pred = out["x_pred"]  # (Ly, B, N, 1)

            err = y_pred - y_true
            mse_batch = float((err.pow(2)).mean().item())
            mae_batch = float(err.abs().mean().item())

            bsz = y_true.shape[1]
            total_mse += mse_batch * bsz
            total_mae += mae_batch * bsz
            n_items += bsz

            pbar.set_postfix({"mse": f"{total_mse / max(n_items, 1):.4f}", "mae": f"{total_mae / max(n_items, 1):.4f}"})

            y_true_np = y_true.squeeze(-1).permute(1, 0, 2).detach().cpu().numpy()  # (B, Ly, N)
            y_pred_np = y_pred.squeeze(-1).permute(1, 0, 2).detach().cpu().numpy()  # (B, Ly, N)

            for i in range(y_true_np.shape[0]):
                if saved >= args.n_samples:
                    break

                sample_true = y_true_np[i]  # (Ly, N)
                sample_pred = y_pred_np[i]  # (Ly, N)
                n_nodes = sample_true.shape[1]
                roi_count = min(args.n_rois, n_nodes)
                roi_ids = np.linspace(0, n_nodes - 1, roi_count, dtype=int)

                fig, axes = plt.subplots(roi_count, 1, figsize=(10, 2.2 * roi_count), sharex=True)
                if roi_count == 1:
                    axes = [axes]

                t_axis = np.arange(sample_true.shape[0])
                for ax, roi in zip(axes, roi_ids):
                    ax.plot(t_axis, sample_true[:, roi], label="ground_truth", linewidth=2.0, color="#1f77b4")
                    ax.plot(t_axis, sample_pred[:, roi], label="prediction", linewidth=1.8, linestyle="--", color="#d62728")
                    ax.set_ylabel(f"ROI {roi}")
                    ax.grid(alpha=0.25)

                axes[0].legend(loc="upper right")
                axes[-1].set_xlabel("Forecast time step")

                cohort_val = batch["meta"]["cohort"][i]
                sid_val = batch["meta"]["subject_id"][i]
                run_val = batch["meta"]["run"][i]
                fig.suptitle(
                    f"Test Dynamics | cohort={cohort_val} subject={sid_val} run={run_val}",
                    fontsize=12,
                )
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

                out_path = out_dir / f"sample_{saved:03d}_b{batch_idx:03d}_i{i:02d}.png"
                fig.savefig(out_path, dpi=160)
                plt.close(fig)
                saved += 1

            if saved >= args.n_samples:
                break

    summary = {
        "checkpoint": str(ckpt_path),
        "manifest_csv": str(manifest_csv),
        "split": "test",
        "cohort": cohort,
        "x": x,
        "y": y,
        "dt": dt,
        "samples_visualized": saved,
        "mean_mse": (total_mse / max(n_items, 1)),
        "mean_mae": (total_mae / max(n_items, 1)),
        "out_dir": str(out_dir),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved test dynamics plots and summary:")
    print(f"  out_dir: {out_dir}")
    print(f"  summary: {summary_path}")
    print(f"  mean test MSE: {summary['mean_mse']:.6f}")
    print(f"  mean test MAE: {summary['mean_mae']:.6f}")


if __name__ == "__main__":
    main()
