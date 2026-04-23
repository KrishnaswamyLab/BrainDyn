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


def get_cfg_value(ckpt_cfg, key, default):
    return ckpt_cfg[key] if key in ckpt_cfg and ckpt_cfg[key] is not None else default


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "grid.alpha": 0.2,
            "font.size": 11,
        }
    )


def temporal_attention_from_history(model, x_hist):
    """Reconstruct temporal attention gamma from current history.

    x_hist: (B, N, T, F)
    returns gamma: (B, N, T)
    """
    encoder = model.dynamics.temporal_encoder
    B, N, T, F = x_hist.shape
    if F != encoder.input_dim:
        raise ValueError(f"Expected input_dim={encoder.input_dim}, got {F}")

    x_flat = x_hist.reshape(B * N, T, F)
    z_seq, _ = encoder.lstm(x_flat)
    z_seq = z_seq.reshape(B, N, T, encoder.hidden_dim)

    current = z_seq[:, :, -1, :]
    scores = encoder.temporal_score(current, z_seq)
    gamma = torch.softmax(scores, dim=-1)
    return gamma


def plot_temporal_heatmap(temporal_mean, out_path):
    """Plot mean temporal attention over forecast step and context lag."""
    # Flip history axis so lag 0 is most recent context point.
    heat = temporal_mean[:, ::-1]
    pred_steps, t_hist = heat.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean temporal attention")

    xticks = np.arange(0, t_hist, max(1, t_hist // 8))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(v)) for v in xticks])
    ax.set_xlabel("Context lag (0 = most recent)")
    ax.set_ylabel("Forecast step")
    ax.set_yticks(np.arange(pred_steps))
    ax.set_title("Temporal Attention: Which Past Moments Matter Per Forecast Step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_temporal_profiles(temporal_mean, out_path):
    """Plot temporal attention profiles for early/middle/late forecast steps."""
    pred_steps, t_hist = temporal_mean.shape
    chosen = sorted({0, pred_steps // 2, pred_steps - 1})

    lags = np.arange(t_hist)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#1d4ed8", "#0f766e", "#b91c1c"]
    labels = [f"step {s}" for s in chosen]

    for s, c, label in zip(chosen, colors, labels):
        ax.plot(lags, temporal_mean[s, ::-1], linewidth=2.3, color=c, label=label)

    ax.set_xlabel("Context lag (0 = most recent)")
    ax.set_ylabel("Mean temporal attention")
    ax.set_title("Temporal Attention Profiles Across Forecast Horizon")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_spatial_node_heatmap(spatial_in_mean, top_nodes, out_path):
    """Plot per-step inbound spatial attention for top-attended nodes."""
    mean_over_steps = spatial_in_mean.mean(axis=0)
    top_n = min(top_nodes, spatial_in_mean.shape[1])
    top_ids = np.argsort(mean_over_steps)[-top_n:][::-1]

    heat = spatial_in_mean[:, top_ids]

    fig, ax = plt.subplots(figsize=(max(9, top_n * 0.35), 5))
    im = ax.imshow(heat, aspect="auto", cmap="OrRd")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Inbound attention")

    ax.set_title("Spatial Attention by Node and Forecast Step")
    ax.set_xlabel("Node index (top by mean inbound attention)")
    ax.set_ylabel("Forecast step")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_xticks(np.arange(top_n))
    ax.set_xticklabels([str(int(i)) for i in top_ids], rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return top_ids


def plot_top_edges_over_time(
    spatial_edge_mean,
    edge_index,
    top_edges,
    out_path,
):
    """Plot trajectories of the strongest directed edges over forecast steps."""
    mean_edge = spatial_edge_mean.mean(axis=0)
    top_e = min(top_edges, spatial_edge_mean.shape[1])
    top_idx = np.argsort(mean_edge)[-top_e:][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(spatial_edge_mean.shape[0])

    for e_idx in top_idx:
        src = int(edge_index[0, e_idx])
        dst = int(edge_index[1, e_idx])
        label = f"{src}->{dst}"
        ax.plot(t, spatial_edge_mean[:, e_idx], linewidth=2.0, alpha=0.9, label=label)

    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Mean edge attention")
    ax.set_title("Top Directed Spatial Attention Edges Over Forecast Horizon")
    if top_e <= 10:
        ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return top_idx


def circular_layout(n_nodes):
    """Return deterministic 2D circular layout coordinates for nodes."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y], axis=1)


def plot_spatial_attention_network(
    edge_weights,
    edge_index,
    n_nodes,
    out_path,
    title,
) -> None:
    """Plot all directed edges in red where darkness reflects attention weight."""
    pos = circular_layout(n_nodes)
    src = edge_index[0]
    dst = edge_index[1]

    w = np.asarray(edge_weights, dtype=np.float64)
    w_min = float(w.min())
    w_max = float(w.max())
    denom = max(w_max - w_min, 1e-12)
    w_norm = (w - w_min) / denom

    fig, ax = plt.subplots(figsize=(9.5, 9.5))

    # Draw lighter edges first; darker/high-attention edges on top.
    draw_order = np.argsort(w_norm)
    for e_idx in draw_order:
        s = int(src[e_idx])
        d = int(dst[e_idx])
        intensity = float(w_norm[e_idx])
        color = (0.95, 0.75 - 0.6 * intensity, 0.75 - 0.6 * intensity)
        linewidth = 0.25 + 1.8 * intensity
        alpha = 0.08 + 0.82 * intensity
        ax.plot(
            [pos[s, 0], pos[d, 0]],
            [pos[s, 1], pos[d, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=1,
        )

    # Node size reflects total inbound attention in this view.
    inbound = np.zeros((n_nodes,), dtype=np.float64)
    np.add.at(inbound, dst, w)
    inbound_norm = (inbound - inbound.min()) / max(inbound.max() - inbound.min(), 1e-12)
    node_sizes = 22.0 + 90.0 * inbound_norm

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=node_sizes,
        c="#1f2937",
        edgecolors="#f8fafc",
        linewidths=0.5,
        alpha=0.95,
        zorder=2,
    )

    # Colorbar legend for edge attention intensity.
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
    sm.set_clim(vmin=w_min, vmax=w_max)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Edge attention")

    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Interpret learned temporal and spatial attention on TEST split."
    )
    ap.add_argument("--checkpoint", type=str, default="checkpoints/braindyn_rbc_best.pt")
    ap.add_argument("--manifest_csv", type=str, default=None, help="Defaults to training config from checkpoint")
    ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None (defaults to checkpoint config)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--max_batches", type=int, default=20, help="How many test batches to analyze")
    ap.add_argument("--top_nodes", type=int, default=25, help="Top nodes to show in spatial node heatmap")
    ap.add_argument("--top_edges", type=int, default=12, help="Top directed edges to show over time")
    ap.add_argument(
        "--network_step",
        type=int,
        default=-1,
        help="Forecast step for network edge plot. Use -1 to average across all forecast steps.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/attention_interpretation")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_plot_style()

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
    edge_index_np = edge_index.detach().cpu().numpy()
    n_nodes = int(edge_index.max().item()) + 1
    n_edges = edge_index.shape[1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal_sum = np.zeros((y, x), dtype=np.float64)
    temporal_count = np.zeros((y,), dtype=np.float64)
    spatial_edge_sum = np.zeros((y, n_edges), dtype=np.float64)
    spatial_edge_count = np.zeros((y,), dtype=np.float64)
    spatial_in_sum = np.zeros((y, n_nodes), dtype=np.float64)
    n_windows = 0
    batches_used = 0

    pbar = tqdm(test_loader, desc="attention-interpret", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.max_batches:
                break

            x_history, y_true = batch_to_model_tensors(batch, device)
            pred_steps = y_true.shape[0]
            if pred_steps != y:
                raise ValueError(f"Expected pred_steps={y}, got {pred_steps}")

            hist = x_history
            bsz = hist.shape[0]
            n_windows += bsz
            batches_used += 1

            for step in range(pred_steps):
                gamma = temporal_attention_from_history(model, hist)  # (B, N, T)
                gamma_np = gamma.detach().cpu().numpy()
                temporal_sum[step] += gamma_np.sum(axis=(0, 1))
                temporal_count[step] += gamma_np.shape[0] * gamma_np.shape[1]

                x_next, aux = model.rk4_step(hist, edge_index=edge_index, dt=dt)
                alpha = aux["alpha"]  # (B, E)
                alpha_np = alpha.detach().cpu().numpy()
                alpha_sum_edges = alpha_np.sum(axis=0)

                spatial_edge_sum[step] += alpha_sum_edges
                spatial_edge_count[step] += bsz

                dst = edge_index_np[1]
                inbound = np.zeros((n_nodes,), dtype=np.float64)
                np.add.at(inbound, dst, alpha_sum_edges)
                spatial_in_sum[step] += inbound

                hist = torch.cat([hist[:, :, 1:, :], x_next.unsqueeze(2)], dim=2)

    if batches_used == 0:
        raise RuntimeError("No batches were processed. Increase --max_batches or check test split.")

    temporal_mean = temporal_sum / np.maximum(temporal_count[:, None], 1.0)
    spatial_edge_mean = spatial_edge_sum / np.maximum(spatial_edge_count[:, None], 1.0)
    spatial_in_mean = spatial_in_sum / np.maximum(spatial_edge_count[:, None], 1.0)

    temporal_focus_recent = float(temporal_mean[:, -1].mean())
    temporal_focus_oldest = float(temporal_mean[:, 0].mean())
    temporal_peak_step = int(np.argmax(temporal_mean.mean(axis=0)))
    spatial_node_importance = spatial_in_mean.mean(axis=0)
    top_nodes_global = np.argsort(spatial_node_importance)[-args.top_nodes:][::-1]

    temporal_heatmap_path = out_dir / "temporal_attention_heatmap.png"
    temporal_profiles_path = out_dir / "temporal_attention_profiles.png"
    spatial_node_heatmap_path = out_dir / "spatial_attention_node_heatmap.png"
    top_edges_path = out_dir / "spatial_attention_top_edges.png"
    spatial_network_path = out_dir / "spatial_attention_network_edges.png"

    plot_temporal_heatmap(temporal_mean, temporal_heatmap_path)
    plot_temporal_profiles(temporal_mean, temporal_profiles_path)
    top_nodes_step = plot_spatial_node_heatmap(spatial_in_mean, args.top_nodes, spatial_node_heatmap_path)
    top_edge_ids = plot_top_edges_over_time(spatial_edge_mean, edge_index_np, args.top_edges, top_edges_path)

    if args.network_step < -1 or args.network_step >= y:
        raise ValueError(f"network_step must be -1 or in [0, {y - 1}], got {args.network_step}")
    if args.network_step == -1:
        edge_view = spatial_edge_mean.mean(axis=0)
        network_title = "Spatial Attention Network (mean across forecast steps)"
    else:
        edge_view = spatial_edge_mean[args.network_step]
        network_title = f"Spatial Attention Network (forecast step {args.network_step})"
    plot_spatial_attention_network(
        edge_weights=edge_view,
        edge_index=edge_index_np,
        n_nodes=n_nodes,
        out_path=spatial_network_path,
        title=network_title,
    )

    top_edges_detail = []
    for e_idx in top_edge_ids.tolist():
        top_edges_detail.append(
            {
                "edge_index": int(e_idx),
                "src": int(edge_index_np[0, e_idx]),
                "dst": int(edge_index_np[1, e_idx]),
                "mean_attention": float(spatial_edge_mean[:, e_idx].mean()),
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
        "max_batches": args.max_batches,
        "batches_used": batches_used,
        "windows_used": n_windows,
        "temporal_focus_recent_mean": temporal_focus_recent,
        "temporal_focus_oldest_mean": temporal_focus_oldest,
        "temporal_peak_lag_from_oldest": temporal_peak_step,
        "top_nodes_global": [int(v) for v in top_nodes_global.tolist()],
        "top_nodes_in_heatmap": [int(v) for v in top_nodes_step.tolist()],
        "network_step": args.network_step,
        "top_edges": top_edges_detail,
        "outputs": {
            "temporal_attention_heatmap": str(temporal_heatmap_path),
            "temporal_attention_profiles": str(temporal_profiles_path),
            "spatial_attention_node_heatmap": str(spatial_node_heatmap_path),
            "spatial_attention_top_edges": str(top_edges_path),
            "spatial_attention_network_edges": str(spatial_network_path),
        },
    }

    summary_path = out_dir / "summary_attention.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved attention interpretation plots and summary:")
    print(f"  out_dir: {out_dir}")
    print(f"  summary: {summary_path}")
    print(f"  temporal attention heatmap: {temporal_heatmap_path}")
    print(f"  temporal attention profiles: {temporal_profiles_path}")
    print(f"  spatial node heatmap: {spatial_node_heatmap_path}")
    print(f"  spatial top edges: {top_edges_path}")
    print(f"  spatial network edges: {spatial_network_path}")


if __name__ == "__main__":
    main()