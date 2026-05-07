"""Evaluate ODEBRAIN graph prediction on NEST/SN test data.

For each test sample:
  1. Run a forward pass with the trained EvoBrainForecaster on the context window.
  2. Build a predicted FC matrix by applying the same Pearson correlation used during
     training (_per_batch_adj) to the K=y_len predicted node-feature timesteps.
  3. Binarise the predicted FC via density-matched top-k thresholding (k = n_edges,
     default 800, matching the exact edge count of every ground-truth NEST graph).
  4. Compute GED (= Hamming distance between two binary graphs on the same node set)
     against the per-sample structural adjacency stored in the dataset.

Also computes a baseline: apply the same pipeline to the INPUT context window FC
(what the model itself used as the graph during the forward pass).

Results are printed and optionally saved to --save_dir as JSON + .npy files.

Optional ``--viz_dir`` writes per-sample PNG panels (GT vs predicted FC vs binarized).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[misc, assignment]

# ── path setup ───────────────────────────────────────────────────────────────
_BASELINES = Path(__file__).resolve().parent
_BRAINDYN_ROOT = _BASELINES.parent
_REPO_ROOT = _BRAINDYN_ROOT.parents[1]
_ODEBRAIN_ROOT = _REPO_ROOT / "benchmarks" / "ODEBRAIN"

if str(_BRAINDYN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAINDYN_ROOT))

from data.sn_dataset import make_dataloaders  # noqa: E402


# ── ODEBRAIN model loader (same namespace-swap as training script) ────────────
def _load_evobrain_forecaster():
    odebrain_s = str(_ODEBRAIN_ROOT)
    if odebrain_s not in sys.path:
        sys.path.insert(0, odebrain_s)
    conflict_prefixes = ("data", "model", "constants")
    saved: dict = {}
    for key in list(sys.modules.keys()):
        if key.split(".")[0] in conflict_prefixes:
            saved[key] = sys.modules.pop(key)
    try:
        from model.evobrain_forecaster import EvoBrainForecaster  # noqa: PLC0415
    finally:
        sys.modules.update(saved)
    return EvoBrainForecaster


# ── helpers (mirrors odebrain_sn_od_train.py) ────────────────────────────────
def _per_batch_adj(x_rates: torch.Tensor) -> torch.Tensor:
    """Absolute Pearson correlation adjacency: (B, T, N) -> (B, N, N)."""
    B, T, N = x_rates.shape
    x = x_rates.float()
    mu = x.mean(dim=1, keepdim=True)
    x_c = x - mu
    std = x_c.norm(dim=1, keepdim=True).clamp_min(1e-6)
    x_n = x_c / std
    adj = torch.bmm(x_n.transpose(1, 2), x_n) / max(T - 1, 1)
    return adj.clamp(-1.0, 1.0).abs()


def _granger_single_window(x_rates: np.ndarray, lag: int = 1) -> np.ndarray:
    """Pairwise lag-1 Granger score matrix (src -> dst) for one sample.

    x_rates : (T, N)
    Returns (N, N) non-negative scores, diagonal zero.
    """
    T, N = x_rates.shape
    if T <= lag + 1:
        return np.zeros((N, N), dtype=np.float32)

    x = x_rates.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y_t = x[lag:, :]      # (T-lag, N)
    x_lag = x[:-lag, :]   # (T-lag, N)

    out = np.zeros((N, N), dtype=np.float32)
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
            out[src, dst] = float(max(0.0, np.log(rss_r / rss_f)))

    return out


def _per_batch_granger_adj(x_rates: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """Granger-causality adjacency: (B, T, N) -> (B, N, N), directed."""
    x_np = x_rates.detach().cpu().numpy()
    B, _, N = x_np.shape
    out = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        out[b] = _granger_single_window(x_np[b], lag=lag)
    return torch.from_numpy(out).to(device=x_rates.device)


def _build_batch_graph(x_rates: torch.Tensor, graph_mode: str, granger_lag: int) -> torch.Tensor:
    if graph_mode == "granger":
        return _per_batch_granger_adj(x_rates, lag=granger_lag)
    return _per_batch_adj(x_rates)


def _sn_batch_to_odebrain(
    batch: dict,
    device: torch.device,
    raw_s: int = 50,
    graph_mode: str = "functional",
    granger_lag: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, dtype=torch.float32)
    y = batch["y"].to(device, dtype=torch.float32)
    context_feat = x.unsqueeze(-1)
    raw_ctx = x.unsqueeze(-1).expand(-1, -1, -1, raw_s).contiguous()
    adj = _build_batch_graph(x, graph_mode=graph_mode, granger_lag=granger_lag)
    y_target = y.unsqueeze(-1)
    return context_feat, raw_ctx, adj, y_target


# ── graph utilities ──────────────────────────────────────────────────────────
def _top_k_binarize(adj: np.ndarray, k: int) -> np.ndarray:
    """Density-matched binarisation: keep top-k entries per sample (exclude diagonal).

    adj : (B, N, N) float  (absolute Pearson, values in [0, 1])
    Returns (B, N, N) int8 binary adjacency.
    """
    B, N, _ = adj.shape
    diag_mask = np.eye(N, dtype=bool)
    out = np.zeros((B, N, N), dtype=np.int8)
    k_eff = max(0, min(int(k), N * N - N))
    if k_eff == 0:
        return out
    for b in range(B):
        flat = adj[b].copy()
        flat[diag_mask] = -1.0  # exclude self-loops
        top_idx = np.argpartition(flat.ravel(), -k_eff)[-k_eff:]
        out[b].ravel()[top_idx] = 1
    return out


def _ged_per_sample(a_hat: np.ndarray, a_true: np.ndarray) -> np.ndarray:
    """GED = Hamming distance (edge insertions + deletions), shape (B,)."""
    return (a_hat != a_true).sum(axis=(1, 2)).astype(np.float64)


def _nged_per_sample(ged: np.ndarray, k: int) -> np.ndarray:
    """Normalised GED = GED / (2k), range [0, 1]. 0=perfect, 1=worst possible."""
    return ged / (2 * k)


def _save_connectivity_figure(
    out_path: Path,
    adj_gt: np.ndarray,
    adj_pred: np.ndarray,
    adj_pred_bin: np.ndarray,
    adj_input_bin: np.ndarray,
    ged_m: float,
    ged_i: float,
    subject_index: int,
) -> None:
    """2×2 heatmaps: structural GT, continuous pred FC, binarized pred, binarized input FC."""
    if plt is None:
        raise RuntimeError("matplotlib is required for --viz_dir (pip install matplotlib)")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    titles_data = [
        ("Structural GT", adj_gt, "Greys"),
        ("Pred FC (continuous)", adj_pred, "viridis"),
        (f"Pred FC top-k (GED={ged_m:.0f})", adj_pred_bin, "Greys"),
        (f"Input FC top-k (GED={ged_i:.0f})", adj_input_bin, "Greys"),
    ]
    for ax, (title, arr, cmap) in zip(axes.ravel(), titles_data):
        ax.imshow(arr, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("dst")
        ax.set_ylabel("src")
    fig.suptitle(f"subject_index={subject_index}")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── evaluation loop ──────────────────────────────────────────────────────────
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EvoBrainForecaster = _load_evobrain_forecaster()

    # Load checkpoint and reconstruct model from saved hyperparameters
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    n_nodes  = int(ckpt.get("n_nodes",  args.n_nodes))
    d_feat   = int(ckpt.get("d_feat",   1))
    d_embed  = int(ckpt.get("d_embed",  args.d_embed))
    d_target = int(ckpt.get("d_target", args.d_target))
    d_cnn    = int(ckpt.get("d_cnn",    args.d_cnn))
    gnn      = str(ckpt.get("gnn",      args.gnn))
    integrator = str(ckpt.get("integrator", args.integrator))
    y_len    = args.y

    model = EvoBrainForecaster(
        n_nodes=n_nodes, d_feat=d_feat, d_embed=d_embed,
        d_target=d_target, d_cnn=d_cnn, gnn=gnn, y_len=y_len,
        ode_method=integrator,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = make_dataloaders(
        npz_path=args.npz_path,
        x=args.x,
        y=y_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        cache=args.cache,
        pin_memory=False,
    )
    loader = loaders[args.split]
    print(f"Evaluating on '{args.split}' split  ({len(loader.dataset)} samples)")

    # Load full adjacency array for lookup by subject_index.
    # sn_dataset.py does not include adjacency in batch meta, so we load it here.
    _npz = np.load(args.npz_path, mmap_mode="r")
    _adj_all = np.asarray(_npz["adjacency"], dtype=np.int8)  # (n_subjects, N, N)

    ged_model: list[float] = []
    ged_input: list[float] = []
    viz_dir = Path(args.viz_dir).resolve() if getattr(args, "viz_dir", "") else None
    viz_max = int(getattr(args, "viz_max", 0) or 0)
    n_viz_done = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="graph eval"):
            context_feat, raw_ctx, adj_input, _ = _sn_batch_to_odebrain(
                batch,
                device,
                graph_mode=args.graph_mode,
                granger_lag=args.granger_lag,
            )

            # ── model predicted node features ────────────────────────────────
            pred = model(context_feat, raw_ctx, adj_input)  # (B, K, N, 1)
            pred_rates = pred.squeeze(-1)                   # (B, K, N)

            # ── predicted adjacency: Pearson of predicted K-step time series ─
            adj_pred = _build_batch_graph(
                pred_rates,
                graph_mode=args.graph_mode,
                granger_lag=args.granger_lag,
            )

            # ── ground truth structural adjacency ────────────────────────────
            subj_ids = batch["meta"]["subject_index"]  # (B,) tensor or list
            if hasattr(subj_ids, "tolist"):
                subj_ids = subj_ids.tolist()
            adj_gt = np.stack([_adj_all[i] for i in subj_ids], axis=0)  # (B, N, N)

            # ── binarise with density-matched top-k ──────────────────────────
            adj_pred_bin  = _top_k_binarize(adj_pred.cpu().numpy(),  k=args.n_edges)
            adj_input_bin = _top_k_binarize(adj_input.cpu().numpy(), k=args.n_edges)

            ged_batch_m = _ged_per_sample(adj_pred_bin, adj_gt)
            ged_batch_i = _ged_per_sample(adj_input_bin, adj_gt)
            ged_model.extend(ged_batch_m.tolist())
            ged_input.extend(ged_batch_i.tolist())

            if viz_dir is not None and plt is not None and n_viz_done < viz_max:
                viz_dir.mkdir(parents=True, exist_ok=True)
                B = adj_pred_bin.shape[0]
                for bi in range(B):
                    if n_viz_done >= viz_max:
                        break
                    stem_ckpt = Path(args.checkpoint).stem
                    out_png = viz_dir / f"{stem_ckpt}_subj{subj_ids[bi]}_viz{n_viz_done:02d}.png"
                    _save_connectivity_figure(
                        out_png,
                        adj_gt[bi],
                        adj_pred.cpu().numpy()[bi],
                        adj_pred_bin[bi],
                        adj_input_bin[bi],
                        float(ged_batch_m[bi]),
                        float(ged_batch_i[bi]),
                        int(subj_ids[bi]),
                    )
                    n_viz_done += 1

    ged_m = np.array(ged_model)
    ged_i = np.array(ged_input)
    nged_m = _nged_per_sample(ged_m, k=args.n_edges)
    nged_i = _nged_per_sample(ged_i, k=args.n_edges)

    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(ged_m),
        "n_edges_threshold": args.n_edges,
        "model_pred_fc_ged": {
            "mean": float(ged_m.mean()),
            "std":  float(ged_m.std()),
            "median": float(np.median(ged_m)),
            "nged_mean": float(nged_m.mean()),
            "nged_std":  float(nged_m.std()),
        },
        "input_fc_baseline_ged": {
            "mean": float(ged_i.mean()),
            "std":  float(ged_i.std()),
            "median": float(np.median(ged_i)),
            "nged_mean": float(nged_i.mean()),
            "nged_std":  float(nged_i.std()),
        },
        "random_baseline_ged": float(2 * args.n_edges * (1 - args.n_edges / 9900)),
        "random_baseline_nged": float(1 - args.n_edges / 9900),
    }

    print(
        f"\n{'':=<60}\n"
        f"  Graph Edit Distance — {args.split} split  "
        f"(n={results['n_samples']}, k={args.n_edges})\n"
        f"{'':=<60}\n"
        f"  Model predicted FC : GED {results['model_pred_fc_ged']['mean']:.1f} "
        f"± {results['model_pred_fc_ged']['std']:.1f}  "
        f"nGED {results['model_pred_fc_ged']['nged_mean']:.4f} "
        f"± {results['model_pred_fc_ged']['nged_std']:.4f}\n"
        f"  Input FC baseline  : GED {results['input_fc_baseline_ged']['mean']:.1f} "
        f"± {results['input_fc_baseline_ged']['std']:.1f}  "
        f"nGED {results['input_fc_baseline_ged']['nged_mean']:.4f} "
        f"± {results['input_fc_baseline_ged']['nged_std']:.4f}\n"
        f"  Random baseline    : GED {results['random_baseline_ged']:.1f}  "
        f"nGED {results['random_baseline_nged']:.4f}\n"
        f"{'':=<60}"
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        stem = Path(args.checkpoint).stem
        json_path = os.path.join(args.save_dir, f"graph_ged_{stem}.json")
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)
        np.save(os.path.join(args.save_dir, f"ged_model_{stem}.npy"), ged_m)
        np.save(os.path.join(args.save_dir, f"ged_input_{stem}.npy"), ged_i)
        print(f"Saved results to {json_path}")

    if viz_dir is not None:
        print(f"Saved {n_viz_done} connectivity figure(s) under {viz_dir}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc graph prediction evaluation for ODEBRAIN on NEST data."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint saved by odebrain_sn_od_train.py")
    p.add_argument("--npz_path", default=
                   "/gpfs/radev/home/cl2482/project/BrainDyn/data/simulated_neuron_dataset/dataset.npz")
    p.add_argument("--split", default="test", choices=("train", "val", "test", "within"))
    p.add_argument("--x",          type=int,   default=30)
    p.add_argument("--y",          type=int,   default=10)
    p.add_argument("--stride",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--num_workers",type=int,   default=2)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--split_seed", type=int,   default=0)
    p.add_argument("--n_edges",    type=int,   default=800,
                   help="Edges in each ground-truth graph (used for density-matched top-k binarisation).")
    p.add_argument(
        "--graph_mode",
        type=str,
        default="granger",
        choices=("functional", "granger"),
        help="Graph construction mode for context/prediction windows.",
    )
    p.add_argument(
        "--granger_lag",
        type=int,
        default=1,
        help="Lag used when --graph_mode=granger.",
    )
    # model architecture fallbacks (overridden by checkpoint metadata if present)
    p.add_argument("--n_nodes",   type=int, default=100)
    p.add_argument("--d_embed",   type=int, default=100)
    p.add_argument("--d_target",  type=int, default=64)
    p.add_argument("--d_cnn",     type=int, default=36)
    p.add_argument("--gnn",       type=str, default="gcn")
    p.add_argument("--integrator",type=str, default="rk4")
    p.add_argument("--save_dir",  type=str, default="",
                   help="Directory to write JSON results and .npy arrays.")
    p.add_argument("--cache", action="store_true",
                   help="Cache dataset arrays in RAM.")
    p.add_argument(
        "--viz_dir",
        type=str,
        default="",
        help="If set, save PNG panels (GT / pred FC / binarized) for the first --viz_max samples.",
    )
    p.add_argument("--viz_max", type=int, default=6,
                   help="Max samples to plot when --viz_dir is set.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
