from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.rbc_dataset import make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_node_features(loader, max_batches):
    """Collect node-wise time features from context windows.

    Returns:
        features: (N, S) where S is concatenated samples over batch/time.
    """
    chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        x_ctx = batch["x"].float()  # (B, Lx, N)
        node_series = x_ctx.permute(2, 0, 1).reshape(x_ctx.shape[-1], -1)  # (N, B*Lx)
        chunks.append(node_series)

    if not chunks:
        raise RuntimeError("Unable to build graph: train loader produced zero batches.")

    features = torch.cat(chunks, dim=1)  # (N, S)
    features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6)
    return features


def build_fc_graph(
    loader,
    max_batches,
    threshold: float,
):
    """Build an undirected functional connectivity graph from Pearson correlations.

    Node features are pooled from training context windows. An edge (i, j)
    exists if |corr(i, j)| >= threshold. The diagonal is always excluded.
    """
    features = collect_node_features(loader, max_batches=max_batches)  # (N, S)
    num_nodes = features.shape[0]

    if num_nodes < 2:
        raise ValueError(f"Need at least 2 nodes, got {num_nodes}")
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    # features are already z-scored by collect_node_features, so dot product / (S-1) = Pearson r
    f = features.float()
    corr = (f @ f.T) / (f.shape[1] - 1)  # (N, N), values in [-1, 1]
    corr = corr.clamp(-1.0, 1.0)

    adjacency = corr.abs() >= threshold
    adjacency.fill_diagonal_(False)

    dst_idx, src_idx = torch.where(adjacency)
    edge_index = torch.stack([src_idx, dst_idx], dim=0).long()
    if edge_index.shape[1] == 0:
        raise RuntimeError(
            f"FC graph has no edges at threshold={threshold:.3f}. Lower --fc_threshold."
        )
    return edge_index, corr, threshold


def batch_to_model_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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


class SubjectRunDataset(torch.utils.data.Dataset):
    """Dataset that yields one full timeseries per subject run.

    Each item is a dict with:
        ts   : (T, N) float32 tensor — raw z-scored timeseries
        meta : dict — subject metadata
    """

    def __init__(self, base_dataset, indices):
        """
        Args:
            base_dataset: ConcatDataset wrapping RBCDataset instances
            indices: subset of sample indices to use (global indices into ConcatDataset)
        """
        # Collect unique (path, meta) pairs — deduplicate by path
        seen_paths = {}
        for idx in indices:
            # For ConcatDataset, map global index to (dataset_idx, local_idx)
            if hasattr(base_dataset, 'datasets') and hasattr(base_dataset, 'cumulative_sizes'):
                dataset_idx = 0
                local_idx = idx
                for cumsum in base_dataset.cumulative_sizes:
                    if idx >= cumsum:
                        dataset_idx += 1
                    else:
                        local_idx = idx - (cumsum if dataset_idx > 0 else 0)
                        break
                # Ensure we're in the right dataset
                if dataset_idx > 0:
                    local_idx = idx - base_dataset.cumulative_sizes[dataset_idx - 1]
                path, meta, _ = base_dataset.datasets[dataset_idx]._samples[local_idx]
            else:
                path, meta, _ = base_dataset._samples[idx]
            if path not in seen_paths:
                seen_paths[path] = meta
        self._runs = [(path, meta) for path, meta in seen_paths.items()]

    def __len__(self):
        return len(self._runs)

    def __getitem__(self, idx):
        from data.rbc_dataset import _load_1d
        path, meta = self._runs[idx]
        ts = _load_1d(path)  # (T, N) float32
        ts_t = torch.from_numpy(ts)  # (T, N)
        # z-score across time per node
        mean = ts_t.mean(dim=0, keepdim=True)
        std = ts_t.std(dim=0, keepdim=True).clamp(min=1e-6)
        ts_t = (ts_t - mean) / std
        return {"ts": ts_t, "meta": meta}


def run_epoch_ar_train(
    model,
    subject_loader,
    edge_index,
    dt,
    x,
    chunk_size,
    tbptt_chunks,
    optimizer,
    lambda_mse,
    lambda_mae,
    grad_clip,
    desc,
    scaler=None,
    teacher_forcing_prob: float = 1.0,
):
    """Full run-level autoregressive training with truncated BPTT and scheduled sampling.

    For each subject run:
      - Use first x timepoints as initial context.
      - Roll forward across entire run in chunks of chunk_size.
      - With probability teacher_forcing_prob, feed ground truth back into
        context after each chunk (teacher forcing); otherwise feed model
        predictions (free running). Scheduled sampling linearly decays
        teacher_forcing_prob from 1.0 to 0.0 over training.
    """
    is_train = optimizer is not None
    use_amp = scaler is not None
    model.train(is_train)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    pcc_running = 0.0
    scc_running = 0.0
    n_chunks = 0
    n_updates = 0

    pbar = tqdm(subject_loader, desc=desc, leave=False)
    for batch in pbar:
        ts = batch["ts"]  # (B, T, N)
        T = ts.shape[1]
        N = ts.shape[2]

        if T < x + chunk_size:
            continue  # run too short to produce even one chunk

        # Initial context: (B, N, x, 1)
        hist = ts[:, :x, :].to(edge_index.device).permute(0, 2, 1).unsqueeze(-1)

        chunk_idx = 0
        t = x  # current position in the run

        while t + chunk_size <= T:
            # Zero gradients at the start of a TBPTT window
            if is_train and chunk_idx % tbptt_chunks == 0:
                optimizer.zero_grad(set_to_none=True)

            # Ground truth for this chunk: (chunk_size, B, N, 1)
            y_chunk = ts[:, t:t + chunk_size, :].to(edge_index.device)
            y_chunk = y_chunk.permute(1, 0, 2).unsqueeze(-1)

            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(
                        x_history=hist,
                        edge_index=edge_index,
                        pred_steps=chunk_size,
                        dt=dt,
                        autoregressive=False,
                    )
                    y_pred = out["x_pred"]  # (chunk_size, B, N, 1)
                    losses = total_loss(y_pred, y_chunk, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
                    loss = losses["total"] / tbptt_chunks  # normalize by tbptt_chunks for gradient averaging

                if is_train:
                    # Keep graph for intermediate chunks so gradients can flow
                    # through the full TBPTT window; free it at the window end.
                    will_step_now = ((chunk_idx + 1) % tbptt_chunks == 0) or (t + 2 * chunk_size > T)
                    retain = not will_step_now
                    if use_amp:
                        scaler.scale(loss).backward(retain_graph=retain)
                    else:
                        loss.backward(retain_graph=retain)

            total_running += float(losses["total"].detach().cpu())
            mse_running += float(losses["mse"].detach().cpu())
            mae_running += float(losses["mae"].detach().cpu())

            y_pred_np = y_pred.detach().cpu().numpy().ravel()
            y_chunk_np = y_chunk.detach().cpu().numpy().ravel()
            pcc_val, _ = pearsonr(y_pred_np, y_chunk_np)
            scc_val, _ = spearmanr(y_pred_np, y_chunk_np)
            pcc_running += float(pcc_val)
            scc_running += float(scc_val)
            n_chunks += 1

            # Decide whether to feed ground truth or predictions back into context
            # During accumulation, don't detach to allow gradients to flow
            if is_train and teacher_forcing_prob < 1.0 and random.random() > teacher_forcing_prob:
                # Free-running: feed model predictions back (gradients flow during accumulation)
                next_hist = y_pred.permute(1, 2, 0, 3)
            else:
                # Teacher forcing: feed ground truth back
                next_hist = y_chunk.detach().permute(1, 2, 0, 3)
            hist = torch.cat([hist[:, :, chunk_size:, :], next_hist], dim=2)

            chunk_idx += 1
            t += chunk_size

            # Update params at the end of a TBPTT window
            if is_train and (chunk_idx % tbptt_chunks == 0 or t + chunk_size > T):
                if grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                n_updates += 1
                # Detach context after update to break gradient chain
                hist = hist.detach()

        pbar.set_postfix(
            {
                "total": f"{total_running / max(n_chunks, 1):.4f}",
                "mse": f"{mse_running / max(n_chunks, 1):.4f}",
                "pcc": f"{pcc_running / max(n_chunks, 1):.4f}",
            }
        )

    if n_chunks == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan")}

    return {
        "total": total_running / n_chunks,
        "mse": mse_running / n_chunks,
        "mae": mae_running / n_chunks,
        "pcc": pcc_running / n_chunks,
        "scc": scc_running / n_chunks,
    }


def rollout_autoregressive(
    model,
    x_history,
    edge_index,
    dt,
    pred_steps,
    chunk_size,
):
    """Autoregressively roll out predictions by feeding chunks back into context."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    hist = x_history
    remaining = pred_steps
    chunks = []
    while remaining > 0:
        step = min(chunk_size, remaining)
        out = model(
            x_history=hist,
            edge_index=edge_index,
            pred_steps=step,
            dt=dt,
            autoregressive=False,
        )
        pred_chunk = out["x_pred"]
        chunks.append(pred_chunk)

        pred_hist = pred_chunk.permute(1, 2, 0, 3)
        hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)
        remaining -= step

    return torch.cat(chunks, dim=0)


def run_epoch(
    model,
    loader,
    edge_index,
    dt,
    optimizer,
    lambda_mse,
    lambda_mae,
    grad_clip,
    desc,
    forecast_mode,
    ar_chunk_size,
    scaler=None,
) :
    is_train = optimizer is not None
    use_amp = scaler is not None
    model.train(is_train)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    pcc_running = 0.0
    scc_running = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history, y_true = batch_to_model_tensors(batch, edge_index.device)
        pred_steps = y_true.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                if forecast_mode == "short":
                    out = model(
                        x_history=x_history,
                        edge_index=edge_index,
                        pred_steps=pred_steps,
                        dt=dt,
                        autoregressive=False,
                    )
                    y_pred = out["x_pred"]
                elif forecast_mode == "long":
                    y_pred = rollout_autoregressive(
                        model=model,
                        x_history=x_history,
                        edge_index=edge_index,
                        dt=dt,
                        pred_steps=pred_steps,
                        chunk_size=ar_chunk_size,
                    )
                else:
                    raise ValueError(f"Unknown forecast_mode='{forecast_mode}'. Use 'short' or 'long'.")

                losses = total_loss(y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
                loss = losses["total"]

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

        total_running += float(losses["total"].detach().cpu())
        mse_running += float(losses["mse"].detach().cpu())
        mae_running += float(losses["mae"].detach().cpu())

        y_pred_np = y_pred.detach().cpu().numpy().ravel()
        y_true_np = y_true.detach().cpu().numpy().ravel()
        pcc_val, _ = pearsonr(y_pred_np, y_true_np)
        scc_val, _ = spearmanr(y_pred_np, y_true_np)
        pcc_running += float(pcc_val)
        scc_running += float(scc_val)
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_batches:.4f}",
                "mse": f"{mse_running / n_batches:.4f}",
                "mae": f"{mae_running / n_batches:.4f}",
                "pcc": f"{pcc_running / n_batches:.4f}",
                "scc": f"{scc_running / n_batches:.4f}",
            }
        )

    if n_batches == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan")}

    return {
        "total": total_running / n_batches,
        "mse": mse_running / n_batches,
        "mae": mae_running / n_batches,
        "pcc": pcc_running / n_batches,
        "scc": scc_running / n_batches,
    }


def run_test_rollout_chunks(
    model,
    loader,
    edge_index,
    dt,
    chunk_steps,
    context_len,
    lambda_mse,
    lambda_mae,
    desc,
):
    """Evaluate test split by autoregressive chunk rollout over full runs."""
    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")

    # Use the earliest available context window per run to avoid duplicate run rollouts.
    run_starts: dict[str, tuple[int, int]] = {}
    for batch in loader:
        meta = batch["meta"]
        for path, t_start, run_len in zip(meta["path"], meta["t_start"], meta["T"]):
            t0 = int(t_start)
            T = int(run_len)
            if (path not in run_starts) or (t0 < run_starts[path][0]):
                run_starts[path] = (t0, T)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    pcc_running = 0.0
    scc_running = 0.0
    n_chunks = 0

    pbar = tqdm(run_starts.items(), desc=desc, leave=False)
    for path, (t0, T) in pbar:
        ts = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
        if t0 + context_len >= T:
            continue

        ctx_raw = ts[t0 : t0 + context_len]
        mean = ctx_raw.mean(axis=0, keepdims=True)
        std = ctx_raw.std(axis=0, keepdims=True).clip(1e-6)

        hist_norm = ((ctx_raw - mean) / std).astype(np.float32)
        hist = torch.from_numpy(hist_norm).to(device=edge_index.device)
        hist = hist.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, N, x, 1)

        current_t = t0 + context_len
        remaining = T - current_t
        while remaining > 0:
            step = min(chunk_steps, remaining)
            out = model(
                x_history=hist,
                edge_index=edge_index,
                pred_steps=step,
                dt=dt,
                autoregressive=False,
            )
            y_pred = out["x_pred"]  # (step, 1, N, 1)

            gt_raw = ts[current_t : current_t + step]
            gt_norm = ((gt_raw - mean) / std).astype(np.float32)
            y_true = torch.from_numpy(gt_norm).to(device=edge_index.device)
            y_true = y_true.unsqueeze(1).unsqueeze(-1)  # (step, 1, N, 1)

            losses = total_loss(y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
            total_running += float(losses["total"].detach().cpu())
            mse_running += float(losses["mse"].detach().cpu())
            mae_running += float(losses["mae"].detach().cpu())

            y_pred_np = y_pred.detach().cpu().numpy().ravel()
            y_true_np = y_true.detach().cpu().numpy().ravel()
            pcc_val, _ = pearsonr(y_pred_np, y_true_np)
            scc_val, _ = spearmanr(y_pred_np, y_true_np)
            pcc_running += float(pcc_val)
            scc_running += float(scc_val)
            n_chunks += 1

            pred_hist = y_pred.permute(1, 2, 0, 3)
            hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)

            current_t += step
            remaining = T - current_t

            pbar.set_postfix(
                {
                    "total": f"{total_running / n_chunks:.4f}",
                    "mse": f"{mse_running / n_chunks:.4f}",
                    "mae": f"{mae_running / n_chunks:.4f}",
                    "pcc": f"{pcc_running / n_chunks:.4f}",
                    "scc": f"{scc_running / n_chunks:.4f}",
                }
            )

    if n_chunks == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan")}

    return {
        "total": total_running / n_chunks,
        "mse": mse_running / n_chunks,
        "mae": mae_running / n_chunks,
        "pcc": pcc_running / n_chunks,
        "scc": scc_running / n_chunks,
    }


def make_subset_loader(dataset, indices, batch_size, num_workers, pin_memory, shuffle):
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Train BrainDyn on RBC.")

    ap.add_argument("--manifest_csv", type=str, default="data/manifest.csv")
    ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None for both")
    ap.add_argument("--x", type=int, default=30, help="context length")
    ap.add_argument("--y", type=int, default=10, help="forecast horizon length")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min_t", type=int, default=0)
    ap.add_argument("--cache", action="store_true")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cv_folds", type=int, default=5, help="number of train/val cross-validation folds")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau multiplicative decay factor")
    ap.add_argument("--lr_patience", type=int, default=2, help="Epochs with no val improvement before reducing LR")
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for ReduceLROnPlateau")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)
    ap.add_argument("--map_hidden_dim", type=int, default=16)
    ap.add_argument("--vf_hidden_dim", type=int, default=128)

    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument(
        "--ode_method",
        type=str,
        default="rk4",
        choices=["rk4", "dopri5", "euler", "midpoint"],
        help="torchdiffeq integration method",
    )

    ap.add_argument(
        "--fc_threshold",
        type=float,
        default=0.5,
        help="absolute Pearson correlation threshold for FC edges",
    )
    ap.add_argument("--fc_max_batches", type=int, default=8, help="train batches used to estimate functional connectivity")

    ap.add_argument("--save_path", type=str, default="checkpoints/braindyn_rbc_best.pt")
    ap.add_argument("--amp", action="store_true", help="Use automatic mixed precision (float16) to reduce GPU memory")
    ap.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory in DataLoader to reduce CPU RAM usage")
    ap.add_argument(
        "--forecast_mode",
        type=str,
        default="short",
        choices=["short", "long", "long_ar_train"],
        help=(
            "short: direct horizon prediction; "
            "long: window training + autoregressive inference; "
            "long_ar_train: full run-level autoregressive training + inference"
        ),
    )
    ap.add_argument(
        "--ar_chunk_size",
        type=int,
        default=1,
        help="chunk size for autoregressive rollout when --forecast_mode=long or long_ar_train",
    )
    ap.add_argument(
        "--tbptt_chunks",
        type=int,
        default=5,
        help="truncated BPTT: detach gradients every N chunks during long_ar_train",
    )
    ap.add_argument(
        "--ss_start",
        type=float,
        default=1.0,
        help="scheduled sampling: teacher forcing probability at epoch 1 (1.0 = full teacher forcing)",
    )
    ap.add_argument(
        "--ss_end",
        type=float,
        default=0.0,
        help="scheduled sampling: teacher forcing probability at final epoch (0.0 = full free-running)",
    )
    ap.add_argument(
        "--ablation_gat",
        action="store_true",
        help="Ablation 1: use GAT neighborhood aggregation instead of sheaf operator",
    )
    # Backward-compatible alias for earlier ablation flag naming.
    ap.add_argument(
        "--ablation_simple_graph",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--ablation_no_lstm",
        action="store_true",
        help="Ablation 2: disable LSTM temporal encoder and use last-step linear projection",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    args.ablation_gat = args.ablation_gat or args.ablation_simple_graph
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_csv = Path(args.manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    loaders = make_dataloaders(
        manifest_csv=manifest_csv,
        x=args.x,
        y=args.y,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cohort=args.cohort,
        min_t=args.min_t,
        cache=args.cache,
        pin_memory=not args.no_pin_memory,
    )
    train_dataset = loaders["train"].dataset
    val_dataset = loaders["val"].dataset
    test_loader = loaders["test"]
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    if len(combined_dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError("Combined train+val dataset is empty. Adjust x/y/stride/cohort/min_t settings.")
    if args.cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {args.cv_folds}")
    if len(combined_dataset) < args.cv_folds:  # type: ignore[arg-type]
        raise ValueError(
            f"Not enough combined train+val samples ({len(combined_dataset)}) for {args.cv_folds}-fold CV"
        )

    cohort_tag = (args.cohort or "all").lower()
    forecast_tag = args.forecast_mode
    ablation_parts = []
    if args.ablation_gat:
        ablation_parts.append("ablation_gat")
    if args.ablation_no_lstm:
        ablation_parts.append("ablation_no_lstm")
    ablation_tag = "main" if len(ablation_parts) == 0 else "_".join(ablation_parts)

    default_save_path = "checkpoints/braindyn_rbc_best.pt"
    resolved_save_path = (
        f"checkpoints/braindyn_rbc_{cohort_tag}_{forecast_tag}_{ablation_tag}_best.pt"
        if args.save_path == default_save_path
        else args.save_path
    )
    save_path = Path(resolved_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(combined_dataset))
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, args.cv_folds)

    fold_val_scores = []
    fold_val_metrics = []
    fold_test_scores = []

    for fold_idx in range(args.cv_folds):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(args.cv_folds) if i != fold_idx])

        train_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=train_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=True,
        )
        val_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=False,
        )

        edge_index_cpu, corr_matrix, threshold_used = build_fc_graph(
            loader=train_loader,
            max_batches=args.fc_max_batches,
            threshold=args.fc_threshold,
        )
        edge_index = edge_index_cpu.to(device)

        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} | FC graph built: "
            f"N={corr_matrix.shape[0]}, E={edge_index.shape[1]}, threshold={threshold_used:.3f}"
        )

        config = BrainDynConfig(
            signal_dim=1,
            hidden_dim=args.hidden_dim,
            num_nodes=corr_matrix.shape[0],
            window_size=args.x,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            map_hidden_dim=args.map_hidden_dim,
            vf_hidden_dim=args.vf_hidden_dim,
            ode_method=args.ode_method,
            use_gat=args.ablation_gat,
            use_lstm_encoder=(not args.ablation_no_lstm),
        )
        model = BrainDyn(config).to(device)
        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} | model ablations: "
            f"gat={args.ablation_gat}, no_lstm={args.ablation_no_lstm}"
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

        scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None
        if scaler is not None and fold_idx == 0:
            print("AMP enabled: using float16 for forward pass")

        # For long_ar_train, build subject run loaders (full timeseries, no windowing)
        if args.forecast_mode == "long_ar_train":
            train_run_dataset = SubjectRunDataset(combined_dataset, train_idx)
            val_run_dataset = SubjectRunDataset(combined_dataset, val_idx)
            train_run_loader = DataLoader(
                train_run_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
            )
            val_run_loader = DataLoader(
                val_run_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
            )

        fold_save_path = save_path.with_name(f"{save_path.stem}_fold{fold_idx + 1}{save_path.suffix}")
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
            if args.forecast_mode == "long_ar_train":
                # Linear decay of teacher forcing probability over epochs
                tf_prob = args.ss_start + (args.ss_end - args.ss_start) * (epoch - 1) / max(args.epochs - 1, 1)
                train_metrics = run_epoch_ar_train(
                    model=model,
                    subject_loader=train_run_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    x=args.x,
                    chunk_size=args.ar_chunk_size,
                    tbptt_chunks=args.tbptt_chunks,
                    optimizer=optimizer,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"fold {fold_idx + 1} train {epoch}/{args.epochs} [tf={tf_prob:.2f}]",
                    scaler=scaler,
                    teacher_forcing_prob=tf_prob,
                )
            else:
                train_metrics = run_epoch(
                    model=model,
                    loader=train_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    optimizer=optimizer,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"fold {fold_idx + 1} train {epoch}/{args.epochs}",
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                    scaler=scaler,
                )

            with torch.no_grad():
                if args.forecast_mode == "long_ar_train":
                    val_metrics = run_epoch_ar_train(
                        model=model,
                        subject_loader=val_run_loader,
                        edge_index=edge_index,
                        dt=args.dt,
                        x=args.x,
                        chunk_size=args.ar_chunk_size,
                        tbptt_chunks=args.tbptt_chunks,
                        optimizer=None,
                        lambda_mse=args.lambda_mse,
                        lambda_mae=args.lambda_mae,
                        grad_clip=args.grad_clip,
                        desc=f"fold {fold_idx + 1} val {epoch}/{args.epochs}",
                        teacher_forcing_prob=0.0,
                    )
                else:
                    val_metrics = run_epoch(
                        model=model,
                        loader=val_loader,
                        edge_index=edge_index,
                        dt=args.dt,
                        optimizer=None,
                        lambda_mse=args.lambda_mse,
                        lambda_mae=args.lambda_mae,
                        grad_clip=args.grad_clip,
                        desc=f"fold {fold_idx + 1} val {epoch}/{args.epochs}",
                        forecast_mode=args.forecast_mode,
                        ar_chunk_size=args.ar_chunk_size,
                    )

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["total"])
            new_lr = optimizer.param_groups[0]["lr"]
            lr_msg = f" | lr={new_lr:.2e}"
            if new_lr < prev_lr:
                lr_msg += " (reduced)"

            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} "
                f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f} | "
                f"val total={val_metrics['total']:.6f} mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f} "
                f"pcc={val_metrics['pcc']:.4f} scc={val_metrics['scc']:.4f}"
                f"{lr_msg}"
            )

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                        "best_val_total": best_val,
                        "edge_index": edge_index_cpu,
                        "fc_threshold": threshold_used,
                        "fold": fold_idx + 1,
                    },
                    fold_save_path,
                )
                print(f"Saved fold {fold_idx + 1} best checkpoint to {fold_save_path} (val total={best_val:.6f})")

        fold_val_scores.append(best_val)

        print(f"Evaluating fold {fold_idx + 1} best checkpoint on val and test splits...")
        ckpt = torch.load(fold_save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            best_val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                edge_index=edge_index,
                dt=args.dt,
                optimizer=None,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                desc=f"fold {fold_idx + 1} best-val",
                forecast_mode=args.forecast_mode,
                ar_chunk_size=args.ar_chunk_size,
            )
            if args.forecast_mode == "long":
                test_metrics = run_test_rollout_chunks(
                    model=model,
                    loader=test_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    chunk_steps=args.y,
                    context_len=args.x,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    desc=f"fold {fold_idx + 1} test-rollout",
                )
            else:
                test_metrics = run_epoch(
                    model=model,
                    loader=test_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    optimizer=None,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"fold {fold_idx + 1} test",
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                )
        fold_val_metrics.append(best_val_metrics)
        fold_test_scores.append(test_metrics)

        print(
            f"Fold {fold_idx + 1} Val  | total={best_val_metrics['total']:.6f} "
            f"mse={best_val_metrics['mse']:.6f} mae={best_val_metrics['mae']:.6f} "
            f"pcc={best_val_metrics['pcc']:.4f} scc={best_val_metrics['scc']:.4f}"
        )
        print(
            f"Fold {fold_idx + 1} Test | total={test_metrics['total']:.6f} "
            f"mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f} "
            f"pcc={test_metrics['pcc']:.4f} scc={test_metrics['scc']:.4f}"
        )

    def _ms(metrics_list, key):
        vals = [m[key] for m in metrics_list]
        return float(np.mean(vals)), float(np.std(vals))

    print("\nCV Val Summary (mean ± std across folds):")
    print(f"  MSE   : {_ms(fold_val_metrics, 'mse')[0]:.6f} ± {_ms(fold_val_metrics, 'mse')[1]:.6f}")
    print(f"  MAE   : {_ms(fold_val_metrics, 'mae')[0]:.6f} ± {_ms(fold_val_metrics, 'mae')[1]:.6f}")
    print(f"  PCC   : {_ms(fold_val_metrics, 'pcc')[0]:.4f}  ± {_ms(fold_val_metrics, 'pcc')[1]:.4f}")
    print(f"  SCC   : {_ms(fold_val_metrics, 'scc')[0]:.4f}  ± {_ms(fold_val_metrics, 'scc')[1]:.4f}")

    print("\nCV Test Summary (mean ± std across folds):")
    print(f"  MSE   : {_ms(fold_test_scores, 'mse')[0]:.6f} ± {_ms(fold_test_scores, 'mse')[1]:.6f}")
    print(f"  MAE   : {_ms(fold_test_scores, 'mae')[0]:.6f} ± {_ms(fold_test_scores, 'mae')[1]:.6f}")
    print(f"  PCC   : {_ms(fold_test_scores, 'pcc')[0]:.4f}  ± {_ms(fold_test_scores, 'pcc')[1]:.4f}")
    print(f"  SCC   : {_ms(fold_test_scores, 'scc')[0]:.4f}  ± {_ms(fold_test_scores, 'scc')[1]:.4f}")


if __name__ == "__main__":
    main()
