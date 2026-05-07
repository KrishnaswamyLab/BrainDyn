from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from data.sn_dataset import DATA_NPZ_PATH, make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Visualize inferred vs ground-truth NEST connectivity from a trained BrainDyn checkpoint."
        )
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
        help="Optional override for dataset path. Defaults to checkpoint args or data/sn_dataset.py default.",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "within"])
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override. Defaults to checkpoint training batch size.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers for visualization pass.",
    )
    ap.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on number of dataloader batches to aggregate.",
    )
    ap.add_argument(
        "--max_subjects",
        type=int,
        default=6,
        help="How many subjects to render in the summary panel.",
    )
    ap.add_argument(
        "--edge_score_mode",
        type=str,
        default="delta_norm",
        choices=["inverse_delta", "delta_norm", "gate_product", "hidden_distance"],
        help=(
            "How to score candidate edges from model aux outputs. "
            "delta_norm uses ||delta|| directly: higher magnitude = stronger edge."
        ),
    )
    ap.add_argument(
        "--binarize_mode",
        type=str,
        default="topk",
        choices=["match_true_density", "threshold", "percentile", "topk"],
        help="How to convert inferred scores to a binary graph for GED.",
    )
    ap.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Used only when --binarize_mode=threshold.",
    )
    ap.add_argument(
        "--score_percentile",
        type=float,
        default=90.0,
        help="Used only when --binarize_mode=percentile. Keeps edges with scores >= this percentile.",
    )
    ap.add_argument(
        "--n_edges",
        type=int,
        default=400,
        help="Density-matched top-k edge count used for baselines and eval-style nGED.",
    )
    ap.add_argument(
        "--topk_edges",
        type=int,
        default=400,
        help="Used only when --binarize_mode=topk. Keeps the top-K inferred edges.",
    )
    ap.add_argument(
        "--baseline_graph_mode",
        type=str,
        default="granger",
        choices=["functional", "granger"],
        help="Graph construction for the input-window baseline.",
    )
    ap.add_argument(
        "--granger_lag",
        type=int,
        default=1,
        help="Lag used when --baseline_graph_mode=granger.",
    )
    ap.add_argument(
        "--baseline_threshold_mode",
        type=str,
        default="gt_superset_topk",
        choices=["manual", "gt_superset_auto", "gt_superset_topk"],
        help="How to convert baseline graph scores into binary edges.",
    )
    ap.add_argument(
        "--baseline_threshold",
        type=float,
        default=0.0,
        help="Used when --baseline_threshold_mode=manual.",
    )
    ap.add_argument(
        "--baseline_extra_edges",
        type=int,
        default=200,
        help="Extra non-GT edges when --baseline_threshold_mode=gt_superset_topk.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Only used for deterministic subject ordering when tied.",
    )
    ap.add_argument(
        "--nx_layout_base",
        type=str,
        default="true",
        choices=["true", "union"],
        help=(
            "How to compute node positions for NetworkX panels: "
            "'true' anchors layout to ground-truth graph only (stable across binarization modes), "
            "'union' uses union of true + inferred-binary edges (previous behavior)."
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/nest_connectivity_inference",
    )
    return ap.parse_args()


def get_cfg_value(ckpt_cfg: dict, key: str, default):
    return ckpt_cfg[key] if key in ckpt_cfg and ckpt_cfg[key] is not None else default


def batch_to_model_tensors_nest(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)
    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def adjacency_to_edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix (N, N) to edge_index (2, E).
    
    Args:
        adjacency: (N, N) binary or continuous adjacency matrix where adj[i,j] represents edge i->j
    
    Returns:
        edge_index: (2, E) where E is the number of edges, format is [source, destination]
    """
    adj = adjacency.bool() if not adjacency.dtype == torch.bool else adjacency
    adj.fill_diagonal_(False)
    src_idx, dst_idx = torch.where(adj)  # torch.where returns (row_indices, col_indices)
    if src_idx.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=adjacency.device)
    edge_index = torch.stack([src_idx, dst_idx], dim=0).long()
    return edge_index


def get_subject_edge_indices_from_batch(
    batch: dict,
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract per-subject edge indices from batch metadata.
    
    Args:
        batch: dictionary with "meta" containing "adjacency" (B, N, N)
        device: torch device
    
    Returns:
        List of edge_index tensors, one per subject in batch
    """
    if "meta" not in batch or "adjacency" not in batch["meta"]:
        raise ValueError("Batch must contain 'meta' with 'adjacency' field")
    
    adjacency_batch = batch["meta"]["adjacency"].to(device=device)  # (B, N, N)
    
    edge_indices = []
    for b in range(adjacency_batch.shape[0]):
        edge_idx = adjacency_to_edge_index(adjacency_batch[b])
        edge_indices.append(edge_idx)
    
    return edge_indices


def build_full_edge_index(n_nodes: int, device: torch.device) -> torch.Tensor:
    """Build a complete directed edge_index (all N×N edges except diagonal).
    
    Returns:
        edge_index: (2, N*(N-1)) with all possible edges
    """
    all_pairs = torch.combinations(torch.arange(n_nodes, device=device), r=2, with_replacement=False)
    src = []
    dst = []
    for i, j in all_pairs:
        src.append(i.item())
        src.append(j.item())
        dst.append(j.item())
        dst.append(i.item())
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index


def edge_scores_from_aux(
    aux: dict,
    edge_index: torch.Tensor,
    x_eval: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Return edge scores with shape (B, E) from an aux dictionary.

    For the sheaf model, ||delta|| magnitude is used as edge strength: higher delta = stronger edge.
    Use inverse_delta if you want to treat delta as a mismatch (lower = better).
    """
    if mode in {"inverse_delta", "delta_norm"} and "delta" in aux:
        delta = aux["delta"]
        if delta.ndim == 3:
            delta_norm = delta.norm(p=2, dim=-1)
            if mode == "delta_norm":
                return delta_norm
            return 1.0 / (1e-6 + delta_norm)

    if mode == "gate_product" and "src_weight" in aux and "dst_weight" in aux:
        src_w = aux["src_weight"]
        dst_w = aux["dst_weight"]
        if src_w.ndim == 3 and dst_w.ndim == 3:
            score = (src_w * dst_w).mean(dim=-1)
            if score.shape[0] == 1 and x_eval is not None:
                score = score.expand(x_eval.shape[0], -1)
            return score

    if mode == "hidden_distance":
        h_t = aux.get("h_t", None)
        if h_t is not None:
            src = edge_index[0]
            dst = edge_index[1]
            return (h_t[:, dst, :] - h_t[:, src, :]).norm(p=2, dim=-1)

    # Fallback order if the requested mode is unavailable in aux.
    if "delta" in aux and aux["delta"].ndim == 3:
        delta_norm = aux["delta"].norm(p=2, dim=-1)
        return 1.0 / (1e-6 + delta_norm)
    if "src_weight" in aux and "dst_weight" in aux:
        src_w = aux["src_weight"]
        dst_w = aux["dst_weight"]
        if src_w.ndim == 3 and dst_w.ndim == 3:
            score = (src_w * dst_w).mean(dim=-1)
            if score.shape[0] == 1 and x_eval is not None:
                score = score.expand(x_eval.shape[0], -1)
            return score
    h_t = aux.get("h_t", None)
    if h_t is not None:
        src = edge_index[0]
        dst = edge_index[1]
        return (h_t[:, dst, :] - h_t[:, src, :]).norm(p=2, dim=-1)

    raise RuntimeError("Unable to derive edge scores from aux outputs.")


def matrix_from_edge_scores(num_nodes: int, edge_index: np.ndarray, edge_scores: np.ndarray) -> np.ndarray:
    mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    mat[edge_index[0], edge_index[1]] = edge_scores
    np.fill_diagonal(mat, 0.0)
    return mat


def binarize_match_true_density(scores: np.ndarray, true_adj: np.ndarray) -> np.ndarray:
    """Binarize inferred scores by selecting top-K edges where K matches true edge count."""
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    true_k = int(np.count_nonzero(true_adj[mask]))
    out = np.zeros_like(true_adj, dtype=np.int8)

    if true_k <= 0:
        return out

    flat_scores = scores[mask]
    if true_k >= flat_scores.size:
        out[mask] = 1
        return out

    top_idx = np.argpartition(flat_scores, -true_k)[-true_k:]
    out_flat = out[mask]
    out_flat[top_idx] = 1
    out[mask] = out_flat
    np.fill_diagonal(out, 0)
    return out


def binarize_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Binarize inferred scores by selecting the top-K edges globally (excluding diagonal)."""
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    out = np.zeros((n, n), dtype=np.int8)

    if k <= 0:
        return out

    flat_scores = scores[mask]
    if k >= flat_scores.size:
        out[mask] = 1
        np.fill_diagonal(out, 0)
        return out

    top_idx = np.argpartition(flat_scores, -k)[-k:]
    out_flat = out[mask]
    out_flat[top_idx] = 1
    out[mask] = out_flat
    np.fill_diagonal(out, 0)
    return out


def binarize_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    out = (scores >= threshold).astype(np.int8)
    np.fill_diagonal(out, 0)
    return out


def binarize_percentile(scores: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    flat_scores = scores[mask]
    if flat_scores.size == 0:
        out = np.zeros_like(scores, dtype=np.int8)
        return out, float("inf")
    thr = float(np.percentile(flat_scores, percentile))
    out = (scores >= thr).astype(np.int8)
    np.fill_diagonal(out, 0)
    return out, thr


def frobenius_norm_adj(inferred: np.ndarray, true_bin: np.ndarray) -> float:
    """Frobenius norm ||inferred - true||_F (off-diagonal only).

    Works for both binary inferred matrices and continuous score matrices.
    For binary-vs-binary, this equals sqrt(GED).
    """
    n = inferred.shape[0]
    mask = ~np.eye(n, dtype=bool)
    diff = inferred[mask].astype(np.float64) - true_bin[mask].astype(np.float64)
    return float(np.sqrt(np.sum(diff ** 2)))


def edge_binary_metrics(inferred_bin: np.ndarray, true_bin: np.ndarray) -> dict:
    if inferred_bin.shape != true_bin.shape:
        raise ValueError(f"Shape mismatch: inferred={inferred_bin.shape}, true={true_bin.shape}")
    n = inferred_bin.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pred = inferred_bin[mask].astype(bool)
    gt = true_bin[mask].astype(bool)

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & (~gt)))
    fn = int(np.count_nonzero((~pred) & gt))
    tn = int(np.count_nonzero((~pred) & (~gt)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def graph_edit_distance_fixed_nodes(inferred_bin: np.ndarray, true_bin: np.ndarray) -> tuple[int, float]:
    """GED with fixed node correspondence and unit edge insert/delete cost.

    With aligned node identities, this equals the number of edge flips needed to
    transform one directed graph into the other.
    """
    if inferred_bin.shape != true_bin.shape:
        raise ValueError(f"Shape mismatch: inferred={inferred_bin.shape}, true={true_bin.shape}")
    n = inferred_bin.shape[0]
    mask = ~np.eye(n, dtype=bool)
    flips = int(np.count_nonzero(inferred_bin[mask] != true_bin[mask]))
    norm = flips / max(mask.sum(), 1)
    return flips, float(norm)


def _pearson_fc_from_context(x_norm: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation adjacency from a z-scored context window.

    x_norm : (T, N) float
    Returns (N, N) in [0, 1], diagonal zeroed.
    """
    T, N = x_norm.shape
    x = x_norm - x_norm.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=0, keepdims=True).clip(1e-6)
    x_n = x / norms
    corr = (x_n.T @ x_n) / max(T - 1, 1)
    np.fill_diagonal(corr, 0.0)
    return np.abs(corr).astype(np.float64)


def _granger_graph_from_context(x_ctx: np.ndarray, lag: int = 1) -> np.ndarray:
    """Pairwise Granger score matrix (src -> dst) from context window (T, N)."""
    T, N = x_ctx.shape
    if T <= lag + 1:
        return np.zeros((N, N), dtype=np.float64)

    x = x_ctx.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y_t = x[lag:, :]
    x_lag = x[:-lag, :]

    out = np.zeros((N, N), dtype=np.float64)
    n_obs = y_t.shape[0]
    ones = np.ones((n_obs, 1), dtype=np.float64)

    for dst in range(N):
        y = y_t[:, dst]
        xr = np.concatenate([ones, x_lag[:, [dst]]], axis=1)
        beta_r, *_ = np.linalg.lstsq(xr, y, rcond=None)
        rss_r = float(np.square(y - xr @ beta_r).sum()) + 1e-12

        for src in range(N):
            if src == dst:
                continue
            xf = np.concatenate([ones, x_lag[:, [dst]], x_lag[:, [src]]], axis=1)
            beta_f, *_ = np.linalg.lstsq(xf, y, rcond=None)
            rss_f = float(np.square(y - xf @ beta_f).sum()) + 1e-12
            out[src, dst] = max(0.0, np.log(rss_r / rss_f))

    np.fill_diagonal(out, 0.0)
    return out


def _baseline_graph_from_context(x_ctx: np.ndarray, mode: str, granger_lag: int) -> np.ndarray:
    if mode == "granger":
        return _granger_graph_from_context(x_ctx, lag=granger_lag)
    return _pearson_fc_from_context(x_ctx)


def _top_k_binarize_matrix(scores_mat: np.ndarray, k: int) -> np.ndarray:
    """Density-matched top-k binarization of an (N, N) score matrix."""
    n = scores_mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    out = np.zeros((n, n), dtype=np.int8)
    if k <= 0:
        return out
    flat = scores_mat[mask]
    k = min(k, int(flat.size))
    top_idx = np.argpartition(flat, -k)[-k:]
    out_flat = np.zeros(flat.size, dtype=np.int8)
    out_flat[top_idx] = 1
    out[mask] = out_flat
    return out


def _binarize_baseline_with_gt_mode(
    score_mat: np.ndarray,
    true_adj: np.ndarray,
    mode: str,
    manual_threshold: float,
    extra_edges: int,
) -> np.ndarray:
    """Match train_nest.py Granger threshold modes for baseline graph binarization."""
    n = score_mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    gt_mask = true_adj.astype(bool) & mask

    if mode == "manual":
        out = (score_mat > float(manual_threshold)).astype(np.int8)
        np.fill_diagonal(out, 0)
        return out

    if mode == "gt_superset_auto":
        gt_scores = score_mat[gt_mask]
        if gt_scores.size == 0:
            out = np.zeros((n, n), dtype=np.int8)
            return out
        gt_pos = gt_scores[gt_scores > 0]
        if gt_pos.size > 0:
            thr = float(np.min(gt_pos))
            out = ((score_mat >= thr) | gt_mask).astype(np.int8)
        else:
            out = gt_mask.astype(np.int8)
        np.fill_diagonal(out, 0)
        return out

    if mode == "gt_superset_topk":
        out = gt_mask.astype(np.int8)
        cand_mask = (~gt_mask) & mask
        cand_scores = score_mat[cand_mask]
        k = max(0, min(int(extra_edges), int(cand_scores.size)))
        if k > 0:
            top_idx = np.argpartition(cand_scores, -k)[-k:]
            out_flat = out[cand_mask]
            out_flat[top_idx] = 1
            out[cand_mask] = out_flat
        np.fill_diagonal(out, 0)
        return out

    raise ValueError(f"Unknown baseline_threshold_mode: {mode}")


def _random_baseline_ged(n_true_edges: int, n_nodes: int) -> tuple[float, float]:
    """Expected GED for a density-matched uniformly random directed graph.

    A random graph with k edges on n nodes (no self-loops) has expected
    GED = 2k(1 - k / (n*(n-1))) against a fixed k-edge ground truth.
    """
    max_edges = n_nodes * (n_nodes - 1)
    ged = 2.0 * n_true_edges * (1.0 - n_true_edges / max_edges)
    # Match eval_odebrain_sn_graph.py convention: nGED = GED / (2k)
    nged = 1.0 - n_true_edges / max_edges
    return float(ged), float(nged)


def _nged_eval_style(ged: float | np.ndarray, k: int) -> float | np.ndarray:
    """Match eval_odebrain_sn_graph.py: nGED = GED / (2k)."""
    return ged / (2.0 * max(k, 1))

def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "font.size": 10,
        }
    )


def plot_subject_panel(
    subject_id: int,
    true_adj: np.ndarray,
    inferred_scores: np.ndarray,
    inferred_bin: np.ndarray,
    ged: int,
    ged_norm: float,
    out_path: Path,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    im0 = axes[0].imshow(true_adj, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth Adjacency")
    axes[0].set_xlabel("Destination node")
    axes[0].set_ylabel("Source node")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax = float(max(np.max(inferred_scores), 1e-8))
    im1 = axes[1].imshow(inferred_scores, cmap="magma", interpolation="nearest", vmin=0.0, vmax=vmax)
    axes[1].set_title("Inferred Edge Score")
    axes[1].set_xlabel("Destination node")
    axes[1].set_ylabel("Source node")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(inferred_bin, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    axes[2].set_title("Inferred Binary Graph")
    axes[2].set_xlabel("Destination node")
    axes[2].set_ylabel("Source node")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        (
            f"Subject {subject_id} | Graph Edit Distance={ged} "
            f"| Normalized GED={ged_norm:.4f}"
        ),
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_nx_graph(adj: np.ndarray, edge_scores: np.ndarray | None = None) -> "nx.DiGraph":
    """Build a directed NetworkX graph from an adjacency matrix.

    If edge_scores is provided it is used as the 'weight' attribute on each edge.
    """
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(adj.astype(bool))
    for r, c in zip(rows, cols):
        w = float(edge_scores[r, c]) if edge_scores is not None else 1.0
        G.add_edge(int(r), int(c), weight=w)
    return G


def _nx_layout(G: "nx.DiGraph", seed: int = 42) -> dict:
    """Shared spring layout for consistent node positions across panels."""
    return nx.spring_layout(G, seed=seed, weight=None)


def plot_nx_panel(
    subject_id: int,
    true_adj: np.ndarray,
    inferred_scores: np.ndarray,
    inferred_bin: np.ndarray,
    ged: int,
    ged_norm: float,
    out_path: Path,
    layout_base: str,
) -> None:
    """Side-by-side NetworkX plots: ground truth vs inferred (score) vs inferred (binary)."""
    if not _NX_AVAILABLE:
        print("networkx not installed — skipping NetworkX panel. Install with: pip install networkx")
        return

    set_plot_style()

    G_true = _build_nx_graph(true_adj)
    # For readability, display only strongest inferred-score edges (match GT edge count).
    n_true_edges = int(np.count_nonzero(true_adj))
    inferred_score_bin = _top_k_binarize_matrix(inferred_scores, k=n_true_edges)
    G_inferred_score = _build_nx_graph(inferred_score_bin, edge_scores=inferred_scores)
    G_inferred_bin = _build_nx_graph(inferred_bin)

    # Keep layout deterministic and optionally independent of inferred binarization.
    if layout_base == "true":
        pos = _nx_layout(G_true, seed=42)
    else:
        G_union = nx.DiGraph()
        G_union.add_nodes_from(range(true_adj.shape[0]))
        G_union.add_edges_from(G_true.edges())
        G_union.add_edges_from(G_inferred_bin.edges())
        pos = _nx_layout(G_union, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # ---- panel 0: ground truth ----
    ax = axes[0]
    ax.set_facecolor("#f0f4f8")
    nx.draw_networkx_nodes(G_true, pos, ax=ax, node_size=120, node_color="#2563eb", alpha=0.85)
    nx.draw_networkx_edges(
        G_true, pos, ax=ax,
        edge_color="#1e3a5f", alpha=0.55,
        arrows=True, arrowsize=12,
        width=1.2,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(G_true, pos, ax=ax, font_size=6, font_color="white")
    ax.set_title("Ground Truth", fontsize=12)
    ax.axis("off")

    # ---- panel 1: inferred edge scores (continuous weights as edge width/color) ----
    ax = axes[1]
    ax.set_facecolor("#f0f4f8")
    score_edges = list(G_inferred_score.edges(data="weight", default=0.0))
    if score_edges:
        raw_weights = np.array([w for _, _, w in score_edges], dtype=np.float64)
        w_min, w_max = raw_weights.min(), raw_weights.max()
        w_norm = (raw_weights - w_min) / max(w_max - w_min, 1e-8)
        edge_widths = 0.4 + 3.0 * w_norm
        edge_colors = plt.cm.magma(0.2 + 0.7 * w_norm)
        nx.draw_networkx_nodes(G_inferred_score, pos, ax=ax, node_size=120, node_color="#dc2626", alpha=0.85)
        nx.draw_networkx_edges(
            G_inferred_score, pos, ax=ax,
            edgelist=[(u, v) for u, v, _ in score_edges],
            edge_color=edge_colors.tolist(),
            alpha=0.65,
            arrows=True, arrowsize=10,
            width=edge_widths.tolist(),
            connectionstyle="arc3,rad=0.08",
        )
    else:
        nx.draw_networkx_nodes(G_inferred_score, pos, ax=ax, node_size=120, node_color="#dc2626", alpha=0.85)
    nx.draw_networkx_labels(G_inferred_score, pos, ax=ax, font_size=6, font_color="white")
    ax.set_title("Inferred Edge Scores", fontsize=12)
    ax.axis("off")

    # ---- panel 2: inferred binary ----
    ax = axes[2]
    ax.set_facecolor("#f0f4f8")
    true_edge_set = set(G_true.edges())
    inferred_edge_set = set(G_inferred_bin.edges())
    tp_edges = list(true_edge_set & inferred_edge_set)
    fp_edges = list(inferred_edge_set - true_edge_set)
    fn_edges = list(true_edge_set - inferred_edge_set)
    nx.draw_networkx_nodes(G_inferred_bin, pos, ax=ax, node_size=120, node_color="#475569", alpha=0.85)
    if tp_edges:
        nx.draw_networkx_edges(G_inferred_bin, pos, ax=ax, edgelist=tp_edges,
                               edge_color="#16a34a", alpha=0.7, arrows=True, arrowsize=12,
                               width=1.6, connectionstyle="arc3,rad=0.08", label="TP")
    if fp_edges:
        nx.draw_networkx_edges(G_inferred_bin, pos, ax=ax, edgelist=fp_edges,
                               edge_color="#dc2626", alpha=0.7, arrows=True, arrowsize=12,
                               width=1.6, connectionstyle="arc3,rad=0.08", label="FP")
    if fn_edges:
        nx.draw_networkx_edges(G_inferred_bin, pos, ax=ax, edgelist=fn_edges,
                               edge_color="#d97706", alpha=0.55, arrows=True, arrowsize=10,
                               width=1.0, style="dashed", connectionstyle="arc3,rad=0.08", label="FN")
    nx.draw_networkx_labels(G_inferred_bin, pos, ax=ax, font_size=6, font_color="white")
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#16a34a", label="TP (correct)"),
        Patch(facecolor="#dc2626", label="FP (false edge)"),
        Patch(facecolor="#d97706", label="FN (missed edge)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.88)
    ax.set_title("Inferred Binary (TP/FP/FN)", fontsize=12)
    ax.axis("off")

    fig.suptitle(
        f"Subject {subject_id} — NetworkX | GED={ged} | nGED={ged_norm:.4f}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prethreshold_nx(
    subject_id: int,
    edge_index: np.ndarray,
    edge_scores: np.ndarray,
    true_adj: np.ndarray,
    out_path: Path,
) -> None:
    """NetworkX plot of ALL edges in the subject's graph before thresholding.

    Edge brightness and width encode delta magnitude (attention strength).
    Green edges are GT edges, red edges are non-GT edges.
    """
    if not _NX_AVAILABLE:
        return

    set_plot_style()
    n_nodes = true_adj.shape[0]
    n_edges = edge_index.shape[1]
    n_gt = int(np.count_nonzero(true_adj))

    scores_norm = (edge_scores - edge_scores.min()) / max(edge_scores.max() - edge_scores.min(), 1e-8)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    gt_edges, non_gt_edges = [], []
    gt_colors, non_gt_colors = [], []
    gt_widths, non_gt_widths = [], []

    for idx in range(n_edges):
        src, dst = int(edge_index[0, idx]), int(edge_index[1, idx])
        sn = float(scores_norm[idx])
        G.add_edge(src, dst, weight=sn)
        is_gt = bool(true_adj[src, dst])
        c = plt.cm.Greens(0.3 + 0.7 * sn) if is_gt else plt.cm.Reds(0.3 + 0.7 * sn)
        w = 0.3 + 2.5 * sn
        if is_gt:
            gt_edges.append((src, dst))
            gt_colors.append(c)
            gt_widths.append(w)
        else:
            non_gt_edges.append((src, dst))
            non_gt_colors.append(c)
            non_gt_widths.append(w)

    pos = nx.spring_layout(G, seed=42, weight=None)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.set_facecolor("#0f172a")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80, node_color="#94a3b8", alpha=0.9)
    if non_gt_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=non_gt_edges,
                               edge_color=[list(c) for c in non_gt_colors],
                               alpha=0.6, arrows=True, arrowsize=8, width=non_gt_widths,
                               connectionstyle="arc3,rad=0.08")
    if gt_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=gt_edges,
                               edge_color=[list(c) for c in gt_colors],
                               alpha=0.85, arrows=True, arrowsize=10, width=gt_widths,
                               connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5, font_color="white")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#16a34a", label=f"GT edge ({n_gt})"),
        Patch(facecolor="#dc2626", label=f"Non-GT edge ({n_edges - n_gt})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.8, facecolor="#1e293b", labelcolor="white")

    sc_min = float(edge_scores.min())
    sc_max = float(edge_scores.max())
    sc_mean = float(edge_scores.mean())
    sc_std = float(edge_scores.std())
    ax.set_title(
        f"Subject {subject_id} — Pre-threshold graph ({n_edges} edges total, {n_gt} GT)\n"
        f"delta min={sc_min:.4f}  max={sc_max:.4f}  mean={sc_mean:.4f}  std={sc_std:.4f}\n"
        "Brightness ∝ delta magnitude  |  Green=GT  Red=non-GT",
        fontsize=10, color="#0f172a",
    )
    ax.axis("off")
    fig.savefig(out_path, dpi=180, facecolor="#f8fafc")
    plt.close(fig)


def plot_overview(
    summary_rows: list[dict],
    out_path: Path,
    random_ged: float | None = None,
) -> None:
    set_plot_style()
    n = len(summary_rows)
    if n == 0:
        return

    subjects = [str(row["subject_id"]) for row in summary_rows]
    ged_vals = np.array([row["ged"] for row in summary_rows], dtype=np.float64)
    has_input_fc = any("ged_input_fc" in r and r["ged_input_fc"] is not None for r in summary_rows)
    y = np.arange(n)

    if has_input_fc:
        input_fc_vals = np.array([row.get("ged_input_fc") or 0.0 for row in summary_rows], dtype=np.float64)
        fig_h = max(3.2, 1.4 * n)
        fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
        bar_h = 0.35
        ax.barh(y - bar_h / 2, ged_vals, bar_h, color="#2563eb", alpha=0.85, label="Model inferred")
        ax.barh(y + bar_h / 2, input_fc_vals, bar_h, color="#f59e0b", alpha=0.85, label="Input FC baseline")
        if random_ged is not None:
            ax.axvline(
                random_ged, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Random baseline ({random_ged:.0f})",
            )
        ax.set_title("GED Comparison: Model vs Input FC vs Random Baseline")
        ax.legend(loc="lower right", fontsize=9)
    else:
        n_ged_vals = np.array([row["ged_norm"] for row in summary_rows], dtype=np.float64)
        fig_h = max(3.2, 1.15 * n)
        fig, ax = plt.subplots(figsize=(10.5, fig_h), constrained_layout=True)
        ax.barh(y, ged_vals, color="#2563eb", alpha=0.85, label="GED (edge flips)")
        if random_ged is not None:
            ax.axvline(
                random_ged, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Random baseline ({random_ged:.0f})",
            )
        ax.set_title("Inferred vs Ground Truth Connectivity: GED per Subject")
        for yi, gv, ng in zip(y, ged_vals, n_ged_vals):
            ax.text(gv + 0.02 * max(ged_vals.max(), 1.0), yi, f"nGED={ng:.4f}", va="center", fontsize=9)

    ax.set_yticks(y, subjects)
    ax.set_xlabel("GED (edge flips)")
    ax.set_ylabel("Subject index")
    ax.invert_yaxis()
    ax.grid(alpha=0.25, axis="x")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args: Dict = ckpt.get("args", {})

    npz_path = args.npz_path or get_cfg_value(ckpt_args, "npz_path", str(DATA_NPZ_PATH))
    batch_size = args.batch_size or int(get_cfg_value(ckpt_args, "batch_size", 32))
    num_workers = int(args.num_workers)
    x_len = int(get_cfg_value(ckpt_args, "x", 30))
    y_len = int(get_cfg_value(ckpt_args, "y", 10))
    stride = int(get_cfg_value(ckpt_args, "stride", 1))
    train_frac = float(get_cfg_value(ckpt_args, "train_frac", 0.8))
    val_frac = float(get_cfg_value(ckpt_args, "val_frac", 0.1))
    split_seed = int(get_cfg_value(ckpt_args, "split_seed", 0))
    cache = bool(get_cfg_value(ckpt_args, "cache", False))
    dt = float(get_cfg_value(ckpt_args, "dt", 1.0))

    loaders = make_dataloaders(
        npz_path=npz_path,
        task_mode="forecasting",
        x=x_len,
        y=y_len,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        train_frac=train_frac,
        val_frac=val_frac,
        split_seed=split_seed,
        cache=cache,
        pin_memory=torch.cuda.is_available(),
        verbose=False,
    )
    loader = loaders[args.split]

    n_nodes = int(loader.dataset.n_channels)
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

    edge_index = ckpt.get("edge_index", None)
    use_subject_graphs = ckpt.get("use_subject_graphs", False)
    
    if edge_index is None and not use_subject_graphs:
        raise RuntimeError("Checkpoint does not include edge_index and use_subject_graphs is False.")
    
    if use_subject_graphs:
        # Per-subject mode: each subject has their own edge_index from their GT adjacency.
        # edge_index is left as None here; it will be extracted per-subject from batch metadata.
        edge_index = None
        edge_index_np = None
        n_edges = None
    elif edge_index is not None:
        edge_index = edge_index.to(device)
        edge_index_np = edge_index.detach().cpu().numpy()
        n_edges = int(edge_index.shape[1])
    else:
        raise RuntimeError("Checkpoint does not include edge_index and use_subject_graphs is False.")

    n_edges_eval = int(args.n_edges)

    subj_score_sum: Dict[int, np.ndarray] = {}
    subj_score_count: Dict[int, np.ndarray] = {}
    subj_true_adj: Dict[int, np.ndarray] = {}
    subj_edge_index: Dict[int, np.ndarray] = {}  # per-subject edge_index (numpy) for score-to-matrix conversion
    subj_input_ged_sum: Dict[int, float] = {}
    subj_input_ged_count: Dict[int, int] = {}

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"infer-{args.split}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            x_history, y_true = batch_to_model_tensors_nest(batch, device)

            if use_subject_graphs:
                # Each subject has their own graph — extract per-subject edge_index from batch metadata
                subject_edge_indices = get_subject_edge_indices_from_batch(batch, device)
                aux_seq_list = []
                for b in range(x_history.shape[0]):
                    x_hist_b = x_history[b:b+1]
                    edge_idx_b = subject_edge_indices[b]
                    out_b = model(
                        x_history=x_hist_b,
                        edge_index=edge_idx_b,
                        pred_steps=y_true.shape[0],
                        dt=dt,
                        autoregressive=False,
                        return_aux=True,
                    )
                    if out_b["aux_seq"]:
                        aux_seq_list.append(out_b["aux_seq"])
            else:
                # Global shared graph for all subjects
                out = model(
                    x_history=x_history,
                    edge_index=edge_index,
                    pred_steps=y_true.shape[0],
                    dt=dt,
                    autoregressive=False,
                    return_aux=True,
                )
                aux_seq_list = [out["aux_seq"]] if out["aux_seq"] else []
                subject_edge_indices = [edge_index] * x_history.shape[0]
            
            if not aux_seq_list:
                continue

            meta = batch["meta"]
            subj_ids = [int(v) for v in meta["subject_index"]]
            adj_batch = meta["adjacency"].detach().cpu().numpy().astype(np.int8)
            x_ctx_np = batch["x"].cpu().numpy()  # (B, Lx, N)

            for i, subj_id in enumerate(subj_ids):
                if i < len(aux_seq_list):
                    aux_seq = aux_seq_list[i]
                else:
                    aux_seq = aux_seq_list[0] if aux_seq_list else []
                
                if not aux_seq:
                    continue
                
                edge_idx_i = subject_edge_indices[i]

                if subj_id not in subj_score_sum:
                    subj_score_sum[subj_id] = None
                    subj_score_count[subj_id] = 0
                    subj_true_adj[subj_id] = adj_batch[i]
                    subj_edge_index[subj_id] = edge_idx_i.detach().cpu().numpy()

                # Extract delta magnitude scores for this subject's edges
                step_scores = []
                for aux in aux_seq:
                    scores = edge_scores_from_aux(
                        aux=aux,
                        edge_index=edge_idx_i,
                        x_eval=aux.get("dxdt", None),
                        mode=args.edge_score_mode,
                    )
                    step_scores.append(scores)

                score_bt = torch.stack(step_scores, dim=0).mean(dim=0)
                score_np = score_bt.detach().cpu().numpy()

                n_edges_i = edge_idx_i.shape[1]
                if subj_score_sum[subj_id] is None:
                    subj_score_sum[subj_id] = np.zeros(n_edges_i, dtype=np.float64)
                    subj_score_count[subj_id] = np.zeros(n_edges_i, dtype=np.float64)

                subj_score_sum[subj_id] += score_np[0] if score_np.shape[0] == 1 else score_np[i]
                subj_score_count[subj_id] += 1.0

                # Input baseline: eval-style per-window graph -> top-k -> GED
                true_adj_i = adj_batch[i].astype(np.int8)
                np.fill_diagonal(true_adj_i, 0)
                input_graph_i = _baseline_graph_from_context(
                    x_ctx_np[i],
                    mode=args.baseline_graph_mode,
                    granger_lag=args.granger_lag,
                )
                if args.baseline_threshold_mode == "manual":
                    input_bin_i = _binarize_baseline_with_gt_mode(
                        score_mat=input_graph_i,
                        true_adj=true_adj_i,
                        mode="manual",
                        manual_threshold=float(args.baseline_threshold),
                        extra_edges=0,
                    )
                else:
                    input_bin_i = _binarize_baseline_with_gt_mode(
                        score_mat=input_graph_i,
                        true_adj=true_adj_i,
                        mode=args.baseline_threshold_mode,
                        manual_threshold=float(args.baseline_threshold),
                        extra_edges=int(args.baseline_extra_edges),
                    )
                ged_i_window, _ = graph_edit_distance_fixed_nodes(input_bin_i, true_adj_i)

                if subj_id not in subj_input_ged_sum:
                    subj_input_ged_sum[subj_id] = 0.0
                    subj_input_ged_count[subj_id] = 0
                subj_input_ged_sum[subj_id] += float(ged_i_window)
                subj_input_ged_count[subj_id] += 1

    if len(subj_score_sum) == 0:
        raise RuntimeError("No subject statistics were collected. Check data split / max_batches settings.")

    all_subject_ids = sorted(subj_score_sum.keys())
    plot_subject_ids = set(
        all_subject_ids[: max(0, args.max_subjects)] if args.max_subjects is not None else all_subject_ids
    )

    max_possible_edges = n_nodes * (n_nodes - 1)
    rand_ged, rand_nged = _random_baseline_ged(n_edges_eval, n_nodes)

    summary_rows = []
    model_ged_all: list[float] = []
    input_fc_ged_all: list[float] = []

    for subj_id in all_subject_ids:
        mean_scores = subj_score_sum[subj_id] / np.maximum(subj_score_count[subj_id], 1e-12)
        subj_ei_np = subj_edge_index[subj_id]  # (2, E) numpy

        # --- Diagnostic: pre-threshold edge count and delta distribution ---
        n_prethresh = subj_ei_np.shape[1]
        n_above = int((mean_scores > args.score_threshold).sum())
        true_adj_tmp = subj_true_adj[subj_id].astype(np.int8)
        np.fill_diagonal(true_adj_tmp, 0)
        n_gt = int(np.count_nonzero(true_adj_tmp))
        n_non_gt = n_prethresh - n_gt
        print(
            f"  Subject {subj_id}: {n_prethresh} edges pre-threshold "
            f"({n_gt} GT, {n_non_gt} non-GT) | "
            f"delta min={mean_scores.min():.4f} max={mean_scores.max():.4f} "
            f"mean={mean_scores.mean():.4f} std={mean_scores.std():.4f} | "
            f"above {args.score_threshold:.2f}: {n_above} | topk={int(args.topk_edges)}"
        )

        # Use the subject's own edge_index to map delta scores back to an N×N matrix
        inferred_scores = matrix_from_edge_scores(
            num_nodes=n_nodes,
            edge_index=subj_ei_np,
            edge_scores=mean_scores,
        )

        true_adj = subj_true_adj[subj_id].astype(np.int8)
        np.fill_diagonal(true_adj, 0)

        if args.binarize_mode == "match_true_density":
            inferred_bin = binarize_match_true_density(inferred_scores, true_adj)
            threshold_used = float("nan")
        elif args.binarize_mode == "topk":
            inferred_bin = binarize_topk(inferred_scores, k=int(args.topk_edges))
            threshold_used = float("nan")
        elif args.binarize_mode == "percentile":
            inferred_bin, threshold_used = binarize_percentile(
                inferred_scores,
                percentile=float(args.score_percentile),
            )
        else:
            inferred_bin = binarize_threshold(inferred_scores, threshold=float(args.score_threshold))
            threshold_used = float(args.score_threshold)

        ged, _ = graph_edit_distance_fixed_nodes(inferred_bin=inferred_bin, true_bin=true_adj)
        ged_norm = float(_nged_eval_style(float(ged), n_edges_eval))
        model_ged_all.append(float(ged))
        cls_metrics = edge_binary_metrics(inferred_bin=inferred_bin, true_bin=true_adj)

        # Frobenius norm metrics
        frob_bin = frobenius_norm_adj(inferred_bin, true_adj)
        # Soft version: normalize continuous scores to [0, 1] before comparing
        sc_min, sc_max = inferred_scores.min(), inferred_scores.max()
        scores_01 = (inferred_scores - sc_min) / max(sc_max - sc_min, 1e-8)
        frob_score = frobenius_norm_adj(scores_01, true_adj)

        # input FC baseline (subject mean of eval-style per-window GED)
        if subj_id in subj_input_ged_sum:
            ged_i = subj_input_ged_sum[subj_id] / max(subj_input_ged_count[subj_id], 1)
            nged_i = float(_nged_eval_style(float(ged_i), n_edges_eval))
        else:
            ged_i, nged_i = float("nan"), float("nan")
        input_fc_ged_all.append(float(ged_i))

        row: Dict = {
            "subject_id": subj_id,
            "ged": int(ged),
            "ged_norm": float(ged_norm),
            "threshold_used": None if np.isnan(threshold_used) else float(threshold_used),
            "ged_input_fc": int(ged_i) if not np.isnan(ged_i) else None,
            "nged_input_fc": float(nged_i) if not np.isnan(nged_i) else None,
            "n_true_edges": int(np.count_nonzero(true_adj)),
            "n_inferred_edges": int(np.count_nonzero(inferred_bin)),
            "frob_bin": float(frob_bin),
            "frob_score": float(frob_score),
            "tp": cls_metrics["tp"],
            "fp": cls_metrics["fp"],
            "fn": cls_metrics["fn"],
            "tn": cls_metrics["tn"],
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1": cls_metrics["f1"],
        }

        if subj_id in plot_subject_ids:
            panel_path = out_dir / f"subject_{subj_id:04d}_inferred_vs_gt.png"
            plot_subject_panel(
                subject_id=subj_id,
                true_adj=true_adj,
                inferred_scores=inferred_scores,
                inferred_bin=inferred_bin,
                ged=ged,
                ged_norm=ged_norm,
                out_path=panel_path,
            )
            prethresh_path = out_dir / f"subject_{subj_id:04d}_prethreshold_graph.png"
            plot_prethreshold_nx(
                subject_id=subj_id,
                edge_index=subj_ei_np,
                edge_scores=mean_scores,
                true_adj=true_adj,
                out_path=prethresh_path,
            )
            nx_panel_path = out_dir / f"subject_{subj_id:04d}_nx_graph.png"
            plot_nx_panel(
                subject_id=subj_id,
                true_adj=true_adj,
                inferred_scores=inferred_scores,
                inferred_bin=inferred_bin,
                ged=ged,
                ged_norm=ged_norm,
                out_path=nx_panel_path,
                layout_base=args.nx_layout_base,
            )
            row["panel"] = str(panel_path)

        summary_rows.append(row)

    summary_rows = sorted(summary_rows, key=lambda r: (r["ged_norm"], r["subject_id"]))
    plot_rows = [r for r in summary_rows if "panel" in r]

    overview_path = out_dir / "overview_ged.png"
    plot_overview(summary_rows=plot_rows, out_path=overview_path, random_ged=rand_ged)

    # aggregate stats across all subjects and print comparison table
    model_arr = np.array([g for g in model_ged_all if not np.isnan(g)])
    input_arr = np.array([g for g in input_fc_ged_all if not np.isnan(g)])
    frob_bin_arr = np.array([r["frob_bin"] for r in summary_rows], dtype=np.float64)
    frob_score_arr = np.array([r["frob_score"] for r in summary_rows], dtype=np.float64)
    precision_arr = np.array([r["precision"] for r in summary_rows], dtype=np.float64)
    recall_arr = np.array([r["recall"] for r in summary_rows], dtype=np.float64)
    f1_arr = np.array([r["f1"] for r in summary_rows], dtype=np.float64)
    tp_total = int(sum(r["tp"] for r in summary_rows))
    fp_total = int(sum(r["fp"] for r in summary_rows))
    fn_total = int(sum(r["fn"] for r in summary_rows))
    micro_precision = tp_total / max(tp_total + fp_total, 1)
    micro_recall = tp_total / max(tp_total + fn_total, 1)
    micro_f1 = 2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
    print(
        f"\n{'':=<64}\n"
        f"  GED Comparison \u2014 {args.split} split  (n={len(model_arr)} subjects)\n"
        f"{'':=<64}\n"
        f"  Model inferred graph : GED {model_arr.mean():.1f} \u00b1 {model_arr.std():.1f}  "
        f"nGED {np.asarray(_nged_eval_style(model_arr, n_edges_eval)).mean():.4f} "
        f"\u00b1 {np.asarray(_nged_eval_style(model_arr, n_edges_eval)).std():.4f}\n"
        f"  Input FC baseline    : GED {input_arr.mean():.1f} \u00b1 {input_arr.std():.1f}  "
        f"nGED {np.asarray(_nged_eval_style(input_arr, n_edges_eval)).mean():.4f} "
        f"\u00b1 {np.asarray(_nged_eval_style(input_arr, n_edges_eval)).std():.4f}\n"
        f"  Edge metrics (macro) : Precision {precision_arr.mean():.4f} \u00b1 {precision_arr.std():.4f}  "
        f"Recall {recall_arr.mean():.4f} \u00b1 {recall_arr.std():.4f}  "
        f"F1 {f1_arr.mean():.4f} \u00b1 {f1_arr.std():.4f}\n"
        f"  Edge metrics (micro) : Precision {micro_precision:.4f}  Recall {micro_recall:.4f}  F1 {micro_f1:.4f}\n"
        f"  Frobenius (binary)   : {frob_bin_arr.mean():.4f} \u00b1 {frob_bin_arr.std():.4f}  "
        f"(= sqrt(GED); sanity check: {np.sqrt(model_arr).mean():.4f})\n"
        f"  Frobenius (scores)   : {frob_score_arr.mean():.4f} \u00b1 {frob_score_arr.std():.4f}  "
        f"(continuous norm. scores vs binary GT)\n"
        f"  Random baseline      : GED {rand_ged:.1f}  nGED {rand_nged:.4f}\n"
        f"{'':=<64}"
    )

    metrics_path = out_dir / "summary_ged.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "n_subjects_total": len(model_arr),
        "n_subjects_visualized": len(plot_rows),
        "edge_score_mode": args.edge_score_mode,
        "binarize_mode": args.binarize_mode,
        "score_threshold": float(args.score_threshold),
        "score_percentile": float(args.score_percentile),
        "topk_edges": int(args.topk_edges),
        "n_edges_threshold": n_edges_eval,
        "baseline_graph_mode": args.baseline_graph_mode,
        "baseline_threshold_mode": args.baseline_threshold_mode,
        "baseline_threshold": float(args.baseline_threshold),
        "baseline_extra_edges": int(args.baseline_extra_edges),
        "model_ged": {
            "mean": float(model_arr.mean()),
            "std": float(model_arr.std()),
            "nged_mean": float(np.asarray(_nged_eval_style(model_arr, n_edges_eval)).mean()),
            "nged_std": float(np.asarray(_nged_eval_style(model_arr, n_edges_eval)).std()),
        },
        "frobenius_norm": {
            "frob_bin_mean": float(frob_bin_arr.mean()),
            "frob_bin_std": float(frob_bin_arr.std()),
            "frob_score_mean": float(frob_score_arr.mean()),
            "frob_score_std": float(frob_score_arr.std()),
        },
        "edge_metrics": {
            "macro_precision_mean": float(precision_arr.mean()),
            "macro_precision_std": float(precision_arr.std()),
            "macro_recall_mean": float(recall_arr.mean()),
            "macro_recall_std": float(recall_arr.std()),
            "macro_f1_mean": float(f1_arr.mean()),
            "macro_f1_std": float(f1_arr.std()),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
        },
        "input_fc_baseline_ged": {
            "mean": float(input_arr.mean()),
            "std": float(input_arr.std()),
            "nged_mean": float(np.asarray(_nged_eval_style(input_arr, n_edges_eval)).mean()),
            "nged_std": float(np.asarray(_nged_eval_style(input_arr, n_edges_eval)).std()),
        },
        "random_baseline_ged": {
            "expected_ged": rand_ged,
            "expected_nged": rand_nged,
        },
        "results": summary_rows,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved NEST connectivity inference visualizations:")
    print(f"  - subject panels: {out_dir}")
    print(f"  - GED overview:   {overview_path}")
    print(f"  - GED summary:    {metrics_path}")


if __name__ == "__main__":
    main()
