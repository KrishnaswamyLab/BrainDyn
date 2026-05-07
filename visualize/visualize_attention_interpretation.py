from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
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


def make_subset_loader(dataset, indices, batch_size, num_workers, pin_memory, shuffle):
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


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
    """Build a temporal importance profile from encoder internals.

    The current TemporalEncoder no longer exposes explicit temporal-attention
    scores. We estimate per-lag importance by similarity to the final hidden
    state. For no-LSTM ablation, the most recent step gets all weight.
    """
    B, N, T, F = x_hist.shape
    if not model.dynamics.use_lstm_encoder:
        gamma = torch.zeros((B, N, T), device=x_hist.device, dtype=x_hist.dtype)
        gamma[:, :, -1] = 1.0
        return gamma

    encoder = model.dynamics.temporal_encoder
    if F != encoder.input_dim:
        raise ValueError(f"Expected input_dim={encoder.input_dim}, got {F}")

    x_flat = x_hist.reshape(B * N, T, F)
    z_seq, _ = encoder.lstm(x_flat)
    z_seq = z_seq.reshape(B, N, T, encoder.hidden_dim)

    current = z_seq[:, :, -1, :].unsqueeze(2)  # (B, N, 1, H)
    # Higher similarity to final hidden state => larger temporal importance.
    sim = torch.nn.functional.cosine_similarity(z_seq, current, dim=-1)
    gamma = torch.softmax(sim, dim=-1)
    return gamma


def rollout_aux_seq(model, x_history, edge_index, dt, pred_steps, chunk_size):
    """Collect per-step aux dictionaries for long autoregressive rollouts."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    hist = x_history
    remaining = pred_steps
    aux_seq = []
    while remaining > 0:
        step = min(chunk_size, remaining)
        out = model(
            x_history=hist,
            edge_index=edge_index,
            pred_steps=step,
            dt=dt,
            autoregressive=False,
            return_aux=True,
        )
        aux_seq.extend(out["aux_seq"])
        pred_chunk = out["x_pred"]
        pred_hist = pred_chunk.permute(1, 2, 0, 3)
        hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)
        remaining -= step

    return aux_seq


def edge_scores_from_aux(aux, edge_index):
    """Extract an edge-level score tensor of shape (B, E) from aux outputs."""
    if "alpha" in aux:
        return aux["alpha"]
    if "delta" in aux:
        delta = aux["delta"]
        if delta.ndim == 3:
            return delta.norm(dim=-1)

    h_t = aux["h_t"]
    src = edge_index[0]
    dst = edge_index[1]
    return (h_t[:, src, :] - h_t[:, dst, :]).norm(dim=-1)


def plot_temporal_profile(temporal_mean, out_path):
    """Plot mean temporal attention weight per context lag.

    temporal_mean: (T,)  — mean attention weight at each context timestep,
                           index 0 = oldest, index -1 = most recent.
    """
    set_plot_style()
    T = len(temporal_mean)
    lags = np.arange(T)  # 0 = oldest, T-1 = most recent

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(lags, temporal_mean[::-1], alpha=0.18, color="#1d4ed8")
    ax.plot(lags, temporal_mean[::-1], linewidth=2.5, color="#1d4ed8", label="mean attention")
    ax.axvline(T - 1, color="#dc2626", linestyle=":", linewidth=1.4, label="most recent context")

    ax.set_xlabel("Context lag (0 = most recent)")
    ax.set_ylabel("Mean temporal attention weight")
    ax.set_title("Temporal Attention: Which Past Moments Drive the Forecast")
    ax.set_xlim(0, T - 1)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _node_attention_from_edges(edge_weights, edge_index, n_nodes):
    """Return per-node degree-normalised inbound attention (N,).

    Divides each node's total inbound weight by its in-degree so that
    hub nodes are not artificially inflated.
    """
    w = np.asarray(edge_weights, dtype=np.float64)
    dst = edge_index[1]
    inbound = np.zeros(n_nodes, dtype=np.float64)
    in_deg = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(inbound, dst, w)
    np.add.at(in_deg, dst, 1.0)
    in_deg = np.maximum(in_deg, 1.0)
    return inbound / in_deg  # mean attention weight per in-edge, per node


def _assign_network_colors(networks_arr):
    """Return a dict mapping network name -> hex color from tab20."""
    unique_nets = sorted(set(networks_arr))
    cmap = plt.cm.get_cmap("tab20", len(unique_nets))
    return {net: cmap(i) for i, net in enumerate(unique_nets)}


def _spring_layout(adjacency_list, n_nodes, iterations=300, seed=42):
    """Force-directed spring layout; returns (n_nodes, 2) array."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1.0, size=(n_nodes, 2)).astype(np.float64)
    if n_nodes <= 1:
        return pos
    k = np.sqrt(1.0 / n_nodes)
    t = 0.1
    for _ in range(iterations):
        disp = np.zeros_like(pos)
        delta = pos[:, None, :] - pos[None, :, :]   # (N, N, 2)
        dist = np.linalg.norm(delta, axis=-1) + 1e-9  # (N, N)
        np.fill_diagonal(dist, 1e9)
        rep = (k * k / dist[..., None]) * delta / dist[..., None]
        disp += rep.sum(axis=1)
        for i, nbrs in enumerate(adjacency_list):
            for j in nbrs:
                if j <= i:
                    continue
                d = pos[i] - pos[j]
                dn = np.linalg.norm(d) + 1e-9
                force = (dn * dn / k) * (d / dn)
                disp[i] -= force
                disp[j] += force
        norms = np.linalg.norm(disp, axis=1) + 1e-9
        step = np.minimum(norms, t)[:, None] * disp / norms[:, None]
        pos += step
        t *= 0.995
    pos -= pos.mean(axis=0)
    scale = np.linalg.norm(pos, axis=1).max() + 1e-9
    return pos / scale


def plot_attention_graph_spring(
    node_attention,
    edge_weights,
    edge_index,
    n_nodes,
    networks_arr,
    out_path,
    title,
):
    """Spring-layout graph with nodes sized by attention and colored by network."""
    set_plot_style()
    # Build adjacency list for layout
    adj = [set() for _ in range(n_nodes)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[s].add(d)
        adj[d].add(s)
    pos = _spring_layout(adj, n_nodes, iterations=300, seed=42)

    net_colors = _assign_network_colors(networks_arr)
    node_colors = [net_colors[net] for net in networks_arr]

    w = np.asarray(edge_weights, dtype=np.float64)
    w_norm = (w - w.min()) / max(w.max() - w.min(), 1e-12)

    fig, ax = plt.subplots(figsize=(11, 11))

    # Edges — draw low-attention first
    draw_order = np.argsort(w_norm)
    for e_idx in draw_order:
        s, d = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        intensity = float(w_norm[e_idx])
        ax.plot(
            [pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
            color=(0.9, 0.85 - 0.6 * intensity, 0.85 - 0.6 * intensity),
            linewidth=0.3 + 1.5 * intensity,
            alpha=0.07 + 0.75 * intensity,
            solid_capstyle="round", zorder=1,
        )

    # Nodes
    attn_norm = (node_attention - node_attention.min()) / max(
        node_attention.max() - node_attention.min(), 1e-12
    )
    node_sizes = 18.0 + 120.0 * attn_norm
    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=node_sizes, c=node_colors,
        edgecolors="#1f2937", linewidths=0.3,
        alpha=0.92, zorder=2,
    )

    # Network legend (max 17 entries)
    unique_nets = sorted(set(networks_arr))
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=net_colors[net], markersize=7, label=net)
        for net in unique_nets
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              ncol=2, frameon=True, framealpha=0.85)

    # Colorbar for edge attention
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
    sm.set_clim(vmin=float(w.min()), vmax=float(w.max()))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Edge attention weight")

    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_attention_brain_mni(
    node_attention,
    mni_coords,
    networks_arr,
    out_path,
    title,
):
    """Render node attention as a glass-brain attention map (nilearn).

    Builds a NIfTI stat map from the Schaefer 400 parcellation, then uses
    nilearn's plot_glass_brain with a hot colormap on a black background to
    produce four anatomical views similar to standard glass-brain figures.
    Falls back to a plain scatter plot if nilearn / atlas is unavailable.
    """
    n_nodes = len(node_attention)
    node_attention = np.asarray(node_attention, dtype=np.float64)
    attn_min = float(node_attention.min())
    attn_max = float(node_attention.max())
    attn_den = max(attn_max - attn_min, 1e-12)
    # Normalize to [0, 1] for consistent interpretability across runs.
    node_attention_norm = (node_attention - attn_min) / attn_den

    # ── Try nilearn path ──────────────────────────────────────────────────────
    _nilearn_ok = False
    if n_nodes in (100, 200, 300, 400, 500, 600, 800, 1000):
        try:
            import nibabel as nib
            from nilearn import datasets as nl_datasets
            from nilearn import image as nl_image
            from nilearn import plotting as nl_plotting

            atlas_info = nl_datasets.fetch_atlas_schaefer_2018(
                n_rois=n_nodes, yeo_networks=17
            )
            atlas_img = nib.load(atlas_info["maps"])
            atlas_data = np.asarray(atlas_img.dataobj, dtype=np.float32)

            # Parcel labels are 1-indexed (parcel i+1 corresponds to node i)
            stat_data = np.zeros_like(atlas_data)
            for i, val in enumerate(node_attention_norm):
                stat_data[atlas_data == (i + 1)] = float(val)

            stat_img = nib.Nifti1Image(stat_data, atlas_img.affine, atlas_img.header)

            # Smooth the parcel map to create contiguous blobs like standard
            # glass-brain visualizations.
            stat_img = nl_image.smooth_img(stat_img, fwhm=4)

            # Robust display range and threshold from voxel intensities after
            # smoothing so we avoid empty-mask plots when attention is flat.
            vox = np.asarray(stat_img.get_fdata(), dtype=np.float64)
            pos = vox[vox > 0]
            if pos.size == 0:
                vmin = 0.0
                vmax = float(np.max(vox)) if vox.size else 1.0
                threshold = None
            else:
                vmin = float(np.percentile(pos, 5.0))
                vmax = float(np.percentile(pos, 99.5))
                if vmax <= vmin + 1e-12:
                    vmax = float(pos.max()) + 1e-6
                    threshold = None
                else:
                    threshold = float(np.percentile(pos, 70.0))
                    if threshold >= vmax:
                        threshold = 0.5 * (vmin + vmax)

            display = nl_plotting.plot_glass_brain(
                stat_img,
                display_mode="lyrz",   # left-sag, coronal, right-sag, axial
                cmap="hot",
                black_bg=True,
                colorbar=True,
                threshold=threshold,
                vmin=vmin,
                vmax=vmax,
                plot_abs=False,
                title=title,
                annotate=False,
            )
            display.savefig(str(out_path), dpi=220)
            display.close()
            _nilearn_ok = True
        except Exception as e:
            print(f"  WARNING: nilearn brain plot failed ({e}); falling back to scatter.")

    # ── Fallback: plain MNI scatter ───────────────────────────────────────────
    if not _nilearn_ok:
        set_plot_style()
        attn_norm = node_attention_norm
        node_sizes = 12.0 + 100.0 * attn_norm
        x_mni, y_mni, z_mni = mni_coords[:, 0], mni_coords[:, 1], mni_coords[:, 2]
        views = [
            (x_mni, z_mni, "MNI X", "MNI Z", "Coronal"),
            (y_mni, z_mni, "MNI Y", "MNI Z", "Sagittal"),
            (x_mni, y_mni, "MNI X", "MNI Y", "Axial"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        cmap = plt.cm.hot
        for ax, (px, py, xlabel, ylabel, vt) in zip(axes, views):
            sc = ax.scatter(px, py, s=node_sizes, c=attn_norm, cmap=cmap,
                            vmin=0, vmax=1, edgecolors="none", alpha=0.9)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(vt, fontsize=11)
            ax.set_facecolor("black")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label="Normalized Attention (0-1)")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_top_edges_ranked(
    edge_weights,
    edge_index,
    top_edges,
    out_path,
):
    """Horizontal bar chart of top directed edges ranked by mean attention weight."""
    set_plot_style()
    w = np.asarray(edge_weights, dtype=np.float64)
    top_e = min(top_edges, len(w))
    top_idx = np.argsort(w)[-top_e:][::-1]

    labels = [f"{int(edge_index[0, e])}→{int(edge_index[1, e])}" for e in top_idx]
    values = w[top_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_e * 0.35)))
    bars = ax.barh(np.arange(top_e), values[::-1], color="#dc2626", alpha=0.78)
    ax.set_yticks(np.arange(top_e))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Mean attention weight")
    ax.set_title("Top Directed Spatial Attention Edges")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return top_idx


def plot_sheaf_directional_weights(src_weight, dst_weight, out_path):
    """Plot learned directional sheaf weights as a compact bar chart."""
    set_plot_style()

    src_scalar = float(np.asarray(src_weight).mean())
    dst_scalar = float(np.asarray(dst_weight).mean())

    labels = ["src_weight", "dst_weight"]
    values = [src_scalar, dst_scalar]
    colors = ["#2563eb", "#dc2626"]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Learned weight (sigmoid)")
    ax.set_title("Sheaf Directional Weights")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Interpret learned temporal and spatial attention on the train/val CV split."
    )
    ap.add_argument("--checkpoint", type=str, default="checkpoints/braindyn_rbc_pnc_best_fold1.pt")
    ap.add_argument("--manifest_csv", type=str, default=None, help="Defaults to training config from checkpoint")
    ap.add_argument("--cohort", type=str, default="PNC", help="PNC, HBN, or None (defaults to checkpoint config)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--fold", type=int, default=None, help="Override checkpoint fold index (1-based)")
    ap.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "val", "both"],
        help="Which CV partition to analyse: train (non-fold), val (fold), or both.",
    )
    ap.add_argument("--max_batches", type=int, default=20, help="How many batches to analyze")
    ap.add_argument("--top_edges", type=int, default=12, help="Top directed edges to show in bar chart")
    ap.add_argument(
        "--coords_npy",
        type=str,
        default="data/schaefer400_17net_mni_coords.npy",
        help="Path to (N,3) MNI centroid coordinates for the atlas parcels.",
    )
    ap.add_argument(
        "--networks_npy",
        type=str,
        default="data/schaefer400_17net_networks.npy",
        help="Path to (N,) network label strings for the atlas parcels.",
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

    cv_folds = int(get_cfg_value(ckpt_cfg, "cv_folds", 5))
    split_seed = int(get_cfg_value(ckpt_cfg, "seed", args.seed))
    fold_from_ckpt = int(ckpt.get("fold", 1))
    fold_idx_1based = int(args.fold) if args.fold is not None else fold_from_ckpt
    if fold_idx_1based < 1 or fold_idx_1based > cv_folds:
        raise ValueError(f"fold must be in [1, {cv_folds}], got {fold_idx_1based}")

    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

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
        pin_memory=use_pin,
    )
    combined_dataset = ConcatDataset([loaders["train"].dataset, loaders["val"].dataset])
    if len(combined_dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError("Combined train+val dataset is empty. Check cohort/x/y/stride/min_t settings.")
    if len(combined_dataset) < cv_folds:  # type: ignore[arg-type]
        raise ValueError(
            f"Not enough combined samples ({len(combined_dataset)}) for {cv_folds}-fold CV"
        )

    rng = np.random.default_rng(split_seed)
    all_indices = np.arange(len(combined_dataset))
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, cv_folds)
    val_idx = fold_indices[fold_idx_1based - 1]
    train_idx = np.concatenate([fold_indices[i] for i in range(cv_folds) if i != fold_idx_1based - 1])

    if args.split == "val":
        analysis_idx = val_idx
        split_label = f"val_fold{fold_idx_1based}"
    elif args.split == "train":
        analysis_idx = train_idx
        split_label = f"train_fold{fold_idx_1based}"
    else:
        analysis_idx = all_indices
        split_label = f"train_val_fold{fold_idx_1based}"

    analysis_loader = make_subset_loader(
        dataset=combined_dataset,
        indices=analysis_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        shuffle=False,
    )

    if "edge_index" not in ckpt:
        raise KeyError("Checkpoint missing 'edge_index'. Use a checkpoint saved by main.py training.")
    edge_index = ckpt["edge_index"].to(device)
    edge_index_np = edge_index.detach().cpu().numpy()
    num_nodes = int(get_cfg_value(ckpt_cfg, "num_nodes", int(edge_index.max().item()) + 1))
    n_edges = edge_index.shape[1]

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

    # Temporal attention: shape (T,) — aggregated mean attention weight per context lag.
    # With fixed-history architecture the temporal attention is constant across pred steps,
    # so we accumulate a single (T,) profile rather than a (y, T) heatmap.
    temporal_sum = np.zeros((x,), dtype=np.float64)
    temporal_count = 0

    # Spatial attention: shape (y, E) per step
    spatial_edge_sum = np.zeros((y, n_edges), dtype=np.float64)
    spatial_edge_count = np.zeros((y,), dtype=np.float64)
    spatial_in_sum = np.zeros((y, num_nodes), dtype=np.float64)

    # Learned directional sheaf weights (if available in aux).
    src_weight_sum = 0.0
    dst_weight_sum = 0.0
    sheaf_weight_count = 0

    n_windows = 0
    batches_used = 0

    pbar = tqdm(analysis_loader, desc=f"attention-interpret ({split_label})", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.max_batches:
                break

            x_history, y_true = batch_to_model_tensors(batch, device)
            pred_steps = y_true.shape[0]
            if pred_steps != y:
                raise ValueError(f"Expected pred_steps={y}, got {pred_steps}")

            # ── Temporal attention ────────────────────────────────────────────
            # gamma: (B, N, T) — compute once; history is fixed during forward.
            gamma = temporal_attention_from_history(model, x_history)
            gamma_np = gamma.detach().cpu().numpy()         # (B, N, T)
            temporal_sum += gamma_np.mean(axis=(0, 1))      # accumulate (T,)
            temporal_count += 1

            # ── Spatial attention per forecast step ───────────────────────────
            # return_aux=True re-evaluates dynamics at each pred step's state,
            # yielding alpha (B, E) from SheafLaplacian per step.
            if forecast_mode == "long":
                aux_seq = rollout_aux_seq(
                    model=model,
                    x_history=x_history,
                    edge_index=edge_index,
                    dt=dt,
                    pred_steps=pred_steps,
                    chunk_size=ar_chunk_size,
                )
            else:
                out = model(
                    x_history=x_history,
                    edge_index=edge_index,
                    pred_steps=pred_steps,
                    dt=dt,
                    return_aux=True,
                )
                aux_seq = out["aux_seq"]

            for step, aux in enumerate(aux_seq):
                edge_scores = edge_scores_from_aux(aux, edge_index)   # (B, E)
                edge_np = edge_scores.detach().cpu().numpy().mean(axis=0)
                spatial_edge_sum[step] += edge_np
                spatial_edge_count[step] += 1

                if ("src_weight" in aux) and ("dst_weight" in aux):
                    src_weight_sum += float(aux["src_weight"].detach().cpu().mean().item())
                    dst_weight_sum += float(aux["dst_weight"].detach().cpu().mean().item())
                    sheaf_weight_count += 1

                inbound = np.zeros((num_nodes,), dtype=np.float64)
                np.add.at(inbound, edge_index_np[1], edge_np)
                spatial_in_sum[step] += inbound

            n_windows += x_history.shape[0]
            batches_used += 1
            pbar.set_postfix({"batches": batches_used, "windows": n_windows})

    if batches_used == 0:
        raise RuntimeError("No batches were processed. Increase --max_batches or check the split.")

    temporal_mean = temporal_sum / max(temporal_count, 1)        # (T,)
    spatial_edge_mean = spatial_edge_sum / np.maximum(spatial_edge_count[:, None], 1.0)   # (y, E)
    spatial_in_mean = spatial_in_sum / np.maximum(spatial_edge_count[:, None], 1.0)       # (y, N)

    # Global node importance (averaged over forecast steps)
    spatial_node_importance = spatial_in_mean.mean(axis=0)
    top_nodes_global = np.argsort(spatial_node_importance)[-args.top_edges:][::-1]

    # ── Load atlas metadata ───────────────────────────────────────────────────
    coords_path = Path(args.coords_npy)
    networks_path = Path(args.networks_npy)
    has_atlas = coords_path.exists() and networks_path.exists()
    if not has_atlas:
        print(f"  WARNING: atlas files not found ({coords_path}, {networks_path}). "
              "Brain and spring-layout plots will be skipped.")
        mni_coords = None
        networks_arr = None
    else:
        mni_coords = np.load(coords_path)          # (N, 3)
        networks_arr = np.load(networks_path)      # (N,)
        if len(mni_coords) != num_nodes or len(networks_arr) != num_nodes:
            print(f"  WARNING: atlas size ({len(mni_coords)}) != model num_nodes ({num_nodes}). "
                  "Brain and spring-layout plots will be skipped.")
            mni_coords = None
            networks_arr = None

    # ── Aggregate spatial attention to a single (E,) and (N,) vector ─────────
    edge_weights_mean = spatial_edge_mean.mean(axis=0)   # (E,) averaged over y steps
    node_attention = _node_attention_from_edges(edge_weights_mean, edge_index_np, num_nodes)  # (N,)

    # ── Output paths ─────────────────────────────────────────────────────────
    temporal_profile_path = out_dir / "temporal_attention_profile.png"
    top_edges_path = out_dir / "spatial_attention_top_edges_ranked.png"
    spring_graph_path = out_dir / "spatial_attention_graph_spring.png"
    brain_mni_path = out_dir / "spatial_attention_brain_mni.png"
    sheaf_directional_weights_path = out_dir / "sheaf_directional_weights.png"

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_temporal_profile(temporal_mean, temporal_profile_path)

    top_edge_ids = plot_top_edges_ranked(edge_weights_mean, edge_index_np, args.top_edges, top_edges_path)

    if has_atlas:
        spring_title = f"Spatial Attention — Spring Layout ({split_label})"
        plot_attention_graph_spring(
            node_attention=node_attention,
            edge_weights=edge_weights_mean,
            edge_index=edge_index_np,
            n_nodes=num_nodes,
            networks_arr=networks_arr,
            out_path=spring_graph_path,
            title=spring_title,
        )
        brain_title = f"Spatial Attention on Brain ({split_label})"
        plot_attention_brain_mni(
            node_attention=node_attention,
            mni_coords=mni_coords,
            networks_arr=networks_arr,
            out_path=brain_mni_path,
            title=brain_title,
        )

    sheaf_directional_weights = None
    if sheaf_weight_count > 0:
        src_weight_mean = src_weight_sum / sheaf_weight_count
        dst_weight_mean = dst_weight_sum / sheaf_weight_count
        sheaf_directional_weights = {
            "src_weight": float(src_weight_mean),
            "dst_weight": float(dst_weight_mean),
            "src_minus_dst": float(src_weight_mean - dst_weight_mean),
            "src_to_dst_ratio": float(src_weight_mean / (dst_weight_mean + 1e-8)),
            "samples": int(sheaf_weight_count),
        }
        plot_sheaf_directional_weights(
            src_weight=src_weight_mean,
            dst_weight=dst_weight_mean,
            out_path=sheaf_directional_weights_path,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    top_edges_detail = [
        {
            "edge_index": int(e_idx),
            "src": int(edge_index_np[0, e_idx]),
            "dst": int(edge_index_np[1, e_idx]),
            "mean_attention": float(edge_weights_mean[e_idx]),
        }
        for e_idx in top_edge_ids.tolist()
    ]

    # Temporal attention statistics
    peak_lag = int(np.argmax(temporal_mean[::-1]))       # lag 0 = most recent
    temporal_focus_recent = float(temporal_mean[-1])
    temporal_focus_oldest = float(temporal_mean[0])

    # Top nodes by degree-normalised inbound attention
    top_nodes_global = np.argsort(node_attention)[-args.top_edges:][::-1]

    outputs = {
        "temporal_attention_profile": str(temporal_profile_path),
        "spatial_attention_top_edges_ranked": str(top_edges_path),
    }
    if has_atlas:
        outputs["spatial_attention_graph_spring"] = str(spring_graph_path)
        outputs["spatial_attention_brain_mni"] = str(brain_mni_path)
    if sheaf_directional_weights is not None:
        outputs["sheaf_directional_weights"] = str(sheaf_directional_weights_path)

    summary = {
        "checkpoint": str(ckpt_path),
        "manifest_csv": str(manifest_csv),
        "split": split_label,
        "fold": fold_idx_1based,
        "cv_folds": cv_folds,
        "cohort": cohort,
        "x": x,
        "y": y,
        "dt": dt,
        "ode_method": ode_method,
        "forecast_mode": forecast_mode,
        "ar_chunk_size": ar_chunk_size,
        "ablation_gat": use_gat,
        "ablation_no_lstm": (not use_lstm_encoder),
        "max_batches": args.max_batches,
        "batches_used": batches_used,
        "windows_used": n_windows,
        "temporal_peak_lag_from_recent": peak_lag,
        "temporal_focus_most_recent": temporal_focus_recent,
        "temporal_focus_oldest": temporal_focus_oldest,
        "top_nodes_by_attention": [int(v) for v in top_nodes_global.tolist()],
        "top_edges": top_edges_detail,
        "sheaf_directional_weights": sheaf_directional_weights,
        "outputs": outputs,
    }

    summary_path = out_dir / "summary_attention.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved attention interpretation plots and summary:")
    print(f"  out_dir:  {out_dir}")
    print(f"  summary:  {summary_path}")
    print(f"  split:    {split_label}  ({n_windows} windows, {batches_used} batches)")
    print(f"  temporal attention peak lag (from most recent): {peak_lag}")
    print(f"  temporal profile:    {temporal_profile_path}")
    print(f"  top edges ranked:    {top_edges_path}")
    if sheaf_directional_weights is not None:
        print(
            "  sheaf directional weights: "
            f"src={sheaf_directional_weights['src_weight']:.4f}, "
            f"dst={sheaf_directional_weights['dst_weight']:.4f}"
        )
        print(f"  sheaf weight plot:   {sheaf_directional_weights_path}")
    if has_atlas:
        print(f"  spring graph:        {spring_graph_path}")
        print(f"  brain MNI:           {brain_mni_path}")


if __name__ == "__main__":
    main()