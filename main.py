from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.rbc_dataset import make_dataloaders
# from data.eeg_h5_dataset import (
#     NUM_CHANNELS,
#     FREQUENCY,
#     load_timeseries_seconds,
#     make_eeg_dataloaders,
# )
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import dtw_mean_normalized, total_loss


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


def minmax_normalize_pred_to_true(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8,
):
    """Normalize both y_pred and y_true to [0, 1] using y_true's min/max."""
    # dims = (0, 3)
    # true_min = y_true.amin(dim=dims, keepdim=True)
    # true_max = y_true.amax(dim=dims, keepdim=True)
    # denom = true_max - true_min + eps
    # return (y_pred - true_min) / denom, (y_true - true_min) / denom
    return y_pred, y_true

class SubjectRunDataset(torch.utils.data.Dataset):
    """Dataset that yields one full timeseries per subject run.

    Each item is a dict with:
        ts   : (T, N) float32 tensor — raw z-scored timeseries
        meta : dict — subject metadata
    """

    def __init__(self, base_dataset, indices, run_loader, use_cache: bool = False):
        """
        Args:
            base_dataset: ConcatDataset wrapping RBCDataset instances
            indices: subset of sample indices to use (global indices into ConcatDataset)
            use_cache: cache normalized run tensors by path in worker memory
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
        self._run_loader = run_loader
        self._use_cache = use_cache
        self._cache: dict[str, torch.Tensor] = {}

    def __len__(self):
        return len(self._runs)

    def __getitem__(self, idx):
        path, meta = self._runs[idx]
        if self._use_cache and path in self._cache:
            ts_t = self._cache[path]
        else:
            ts = self._run_loader(path)  # (T, N) float32
            ts_t = torch.from_numpy(ts)  # (T, N)
            # z-score across time per node
            mean = ts_t.mean(dim=0, keepdim=True)
            std = ts_t.std(dim=0, keepdim=True).clamp(min=1e-6)
            ts_t = (ts_t - mean) / std
            if self._use_cache:
                self._cache[path] = ts_t
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
    if not (0.0 <= teacher_forcing_prob <= 1.0):
        raise ValueError(f"teacher_forcing_prob must be in [0, 1], got {teacher_forcing_prob}")

    is_train = optimizer is not None
    use_amp = scaler is not None
    model.train(is_train)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    n_chunks = 0
    # Accumulate flattened predictions and targets for epoch-level PCC/SCC
    pred_accum: list[np.ndarray] = []
    true_accum: list[np.ndarray] = []

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
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        x_history=hist,
                        edge_index=edge_index,
                        pred_steps=chunk_size,
                        dt=dt,
                        autoregressive=False,
                    )
                    y_pred = out["x_pred"]  # (chunk_size, B, N, 1)
                    y_pred, y_chunk = minmax_normalize_pred_to_true(y_pred=y_pred, y_true=y_chunk)
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

            pred_accum.append(y_pred.detach().cpu().numpy().ravel())
            true_accum.append(y_chunk.detach().cpu().numpy().ravel())

            n_chunks += 1

            # Decide whether to feed ground truth or predictions back into context.
            # During eval, teacher_forcing_prob=0.0 means fully free-running rollout.
            if teacher_forcing_prob >= 1.0:
                use_teacher_forcing = True
            elif teacher_forcing_prob <= 0.0:
                use_teacher_forcing = False
            elif is_train:
                use_teacher_forcing = random.random() < teacher_forcing_prob
            else:
                # Keep eval deterministic for intermediate probabilities.
                use_teacher_forcing = False

            if use_teacher_forcing:
                # Teacher forcing: feed ground truth back.
                next_hist = y_chunk.detach().permute(1, 2, 0, 3)
            else:
                # Free-running: feed model predictions back.
                next_hist = y_pred.permute(1, 2, 0, 3)
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
                # Detach context after update to break gradient chain
                hist = hist.detach()

        pbar.set_postfix(
            {
                "total": f"{total_running / max(n_chunks, 1):.4f}",
                "mse": f"{mse_running / max(n_chunks, 1):.4f}",
            }
        )

    if n_chunks == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan"), "dtw": float("nan")}

    all_pred = np.concatenate(pred_accum)
    all_true = np.concatenate(true_accum)
    pcc_val, _ = pearsonr(all_pred, all_true)
    scc_val, _ = spearmanr(all_pred, all_true)
    dtw_val = dtw_mean_normalized(all_pred.reshape(-1, 1), all_true.reshape(-1, 1))

    return {
        "total": total_running / n_chunks,
        "mse": mse_running / n_chunks,
        "mae": mae_running / n_chunks,
        "pcc": float(pcc_val),
        "scc": float(scc_val),
        "dtw": float(dtw_val),
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
    dtw_running = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history, y_true = batch_to_model_tensors(batch, edge_index.device)
        pred_steps = y_true.shape[0]
        # y_true = y_true.permute(1, 2, 0, 3)
        # result = torch.cat((x_history, y_true), dim=2)
        # min_val, max_val = torch.aminmax(result, dim=2, keepdim = True)
        # denom = (max_val - min_val).clamp_min(1e-6)
        # x_history = (x_history - min_val) / denom
        # y_true = (y_true - min_val) / denom
        # y_true = y_true.permute(2, 0, 1, 3)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
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

                y_pred, y_true = minmax_normalize_pred_to_true(y_pred=y_pred, y_true=y_true)
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

        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        pcc_val, _ = pearsonr(y_pred_np.ravel(), y_true_np.ravel())
        scc_val, _ = spearmanr(y_pred_np.ravel(), y_true_np.ravel())
        pcc_running += float(pcc_val)
        scc_running += float(scc_val)
        dtw_running += dtw_mean_normalized(y_pred_np, y_true_np)
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_batches:.4f}",
                "mse": f"{mse_running / n_batches:.4f}",
                "mae": f"{mae_running / n_batches:.4f}",
                "pcc": f"{pcc_running / n_batches:.4f}",
                "scc": f"{scc_running / n_batches:.4f}",
                "dtw": f"{dtw_running / n_batches:.4f}",
            }
        )

    if n_batches == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan"), "dtw": float("nan")}

    return {
        "total": total_running / n_batches,
        "mse": mse_running / n_batches,
        "mae": mae_running / n_batches,
        "pcc": pcc_running / n_batches,
        "scc": scc_running / n_batches,
        "dtw": dtw_running / n_batches,
    }


def run_test_rollout_chunks(
    model,
    loader,
    edge_index,
    dt,
    chunk_steps,
    context_len,
    rollout_steps,
    lambda_mse,
    lambda_mae,
    desc,
    run_loader,
):
    """Evaluate test split by autoregressive chunk rollout over full runs."""
    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")
    if rollout_steps is not None and rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be positive when set, got {rollout_steps}")

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
    dtw_running = 0.0
    n_chunks = 0

    pbar = tqdm(run_starts.items(), desc=desc, leave=False)
    for path, (t0, T) in pbar:
        ts = run_loader(path)
        if t0 + context_len >= T:
            continue

        ctx_raw = ts[t0 : t0 + context_len]
        mean = ctx_raw.mean(axis=0, keepdims=True)
        std = ctx_raw.std(axis=0, keepdims=True).clip(1e-6)

        hist_norm = ((ctx_raw - mean) / std).astype(np.float32)
        hist = torch.from_numpy(hist_norm).to(device=edge_index.device)
        hist = hist.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, N, x, 1)

        current_t = t0 + context_len
        max_pred_steps = T - current_t
        if rollout_steps is not None:
            max_pred_steps = min(max_pred_steps, rollout_steps)
        remaining = max_pred_steps
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

            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            pcc_val, _ = pearsonr(y_pred_np.ravel(), y_true_np.ravel())
            scc_val, _ = spearmanr(y_pred_np.ravel(), y_true_np.ravel())
            pcc_running += float(pcc_val)
            scc_running += float(scc_val)
            dtw_running += dtw_mean_normalized(y_pred_np, y_true_np)
            n_chunks += 1

            pred_hist = y_pred.permute(1, 2, 0, 3)
            hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)

            current_t += step
            remaining -= step

            pbar.set_postfix(
                {
                    "total": f"{total_running / n_chunks:.4f}",
                    "mse": f"{mse_running / n_chunks:.4f}",
                    "mae": f"{mae_running / n_chunks:.4f}",
                    "pcc": f"{pcc_running / n_chunks:.4f}",
                    "scc": f"{scc_running / n_chunks:.4f}",
                    "dtw": f"{dtw_running / n_chunks:.4f}",
                }
            )

    if n_chunks == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan"), "dtw": float("nan")}

    return {
        "total": total_running / n_chunks,
        "mse": mse_running / n_chunks,
        "mae": mae_running / n_chunks,
        "pcc": pcc_running / n_chunks,
        "scc": scc_running / n_chunks,
        "dtw": dtw_running / n_chunks,
    }


def save_dynamics_plot(
    model,
    loader,
    edge_index,
    dt,
    forecast_mode,
    ar_chunk_size,
    out_path: Path,
    max_nodes: int = 4,
) -> bool:
    """Save a quick prediction-vs-truth dynamics plot from one loader batch."""
    try:
        batch = next(iter(loader))
    except StopIteration:
        return False

    x_history, y_true = batch_to_model_tensors(batch, edge_index.device)
    pred_steps = y_true.shape[0]
    model.eval()
    with torch.no_grad():
        if forecast_mode in {"short", "long_ar_train"}:
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
            raise ValueError(f"Unknown forecast_mode='{forecast_mode}'.")

    # Use first sample in batch and first few nodes to keep plots readable.
    y_true_np = y_true[:, 0, :, 0].detach().cpu().numpy()  # (Ly, N)
    y_pred_np = y_pred[:, 0, :, 0].detach().cpu().numpy()  # (Ly, N)

    node_count = min(max_nodes, y_true_np.shape[1])
    fig, axes = plt.subplots(node_count, 1, figsize=(10, 2.4 * node_count), sharex=True)
    if node_count == 1:
        axes = [axes]

    t = np.arange(y_true_np.shape[0])
    for node_idx in range(node_count):
        ax = axes[node_idx]
        ax.plot(t, y_true_np[:, node_idx], label="true", linewidth=2.0, color="#1f77b4")
        ax.plot(t, y_pred_np[:, node_idx], label="pred", linewidth=1.8, linestyle="--", color="#d62728")
        ax.set_ylabel(f"node {node_idx}")
        ax.grid(alpha=0.3)
        if node_idx == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("forecast step")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


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
    ap = argparse.ArgumentParser(description="Train BrainDyn on fMRI or EEG timeseries.")

    ap.add_argument(
        "--dataset",
        type=str,
        default="fmri",
        choices=["fmri", "eeg"],
        help="dataset backend: fmri (RBC manifest CSV) or eeg (HDF5 root)",
    )
    ap.add_argument("--manifest_csv", type=str, default="data/manifest.csv")
    # ap.add_argument("--eeg_h5_root", type=str, default="data/eeg_h5")
    # ap.add_argument(
    #     "--eeg_manifest_csv",
    #     type=str,
    #     default="data/tusz_manifest.csv",
    #     help="EEG manifest CSV (same role as --manifest_csv for fMRI)",
    # )
    # ap.add_argument("--eeg_frequency", type=int, default=FREQUENCY, help="EEG sampling rate used in HDF5 resampled_signal")
    # ap.add_argument("--eeg_num_channels", type=int, default=NUM_CHANNELS, help="EEG channel count (nodes); default 19")
    ap.add_argument("--cohort", type=str, default="PNC", help="PNC, HBN, or None for both")
    ap.add_argument("--x", type=int, default=40, help="context length")
    ap.add_argument("--y", type=int, default=10, help="forecast horizon length")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min_t", type=int, default=0)
    ap.add_argument("--cache", action="store_true")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--cv_folds", type=int, default=5, help="number of train/val cross-validation folds")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau multiplicative decay factor")
    ap.add_argument("--lr_patience", type=int, default=2, help="Epochs with no val improvement before reducing LR")
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for ReduceLROnPlateau")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=16)
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
        "--test_rollout_steps",
        type=int,
        default=None,
        help="when --forecast_mode=long, cap test autoregressive rollout to this many future timepoints (default: full remaining run)",
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
    ap.add_argument(
        "--precompute_lap_h",
        action="store_true",
        help="Pre-compute LSTM + graph Laplacian once per ODE chunk instead of at every RHS evaluation. "
             "Mathematically equivalent; faster when chunk_size > 1 and ode_method has multiple stages (e.g. rk4).",
    )
    ap.add_argument(
        "--run_batch_size",
        type=int,
        default=1,
        help="Batch size for the per-run AR training DataLoader. All runs must have identical T for batch_size>1 "
             "(true for fixed-protocol datasets like PNC). Larger values improve GPU utilization significantly.",
    )
    ap.add_argument(
        "--val_every",
        type=int,
        default=1,
        help="Run validation every N epochs (default 1 = every epoch). Skipped epochs still train and log train metrics.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    args.ablation_gat = args.ablation_gat or args.ablation_simple_graph
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "fmri":
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

        def run_loader(path: str) -> np.ndarray:
            return np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)

    elif args.dataset == "eeg":
        eeg_h5_root = Path(args.eeg_h5_root)
        if not eeg_h5_root.exists():
            raise FileNotFoundError(f"EEG HDF5 root not found: {eeg_h5_root}")
        eeg_manifest_csv = Path(args.eeg_manifest_csv)
        if not eeg_manifest_csv.exists():
            raise FileNotFoundError(f"EEG manifest not found: {eeg_manifest_csv}")

        loaders = make_eeg_dataloaders(
            h5_root=eeg_h5_root,
            x=args.x,
            y=args.y,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            manifest_csv=eeg_manifest_csv,
            min_t=args.min_t,
            cache=args.cache,
            pin_memory=not args.no_pin_memory,
            frequency=args.eeg_frequency,
            num_channels=args.eeg_num_channels,
        )

        def run_loader(path: str) -> np.ndarray:
            return load_timeseries_seconds(
                Path(path),
                frequency=args.eeg_frequency,
                num_channels=args.eeg_num_channels,
            )

    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

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

    if args.dataset == "fmri":
        cohort_tag = (args.cohort or "all").lower()
    else:
        cohort_tag = "all"
    forecast_tag = args.forecast_mode
    ablation_parts = []
    if args.ablation_gat:
        ablation_parts.append("ablation_gat")
    if args.ablation_no_lstm:
        ablation_parts.append("ablation_no_lstm")
    ablation_tag = "main" if len(ablation_parts) == 0 else "_".join(ablation_parts)

    default_save_path = "checkpoints/braindyn_rbc_best.pt"
    resolved_save_path = (
        f"checkpoints/braindyn_{args.dataset}_{cohort_tag}_{forecast_tag}_{ablation_tag}_dt02_trainviz_best.pt"
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
            precompute_lap_h=args.precompute_lap_h,
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

        scaler = torch.amp.GradScaler("cuda") if (args.amp and torch.cuda.is_available()) else None
        if scaler is not None and fold_idx == 0:
            print("AMP enabled: using float16 for forward pass")

        # For long_ar_train, build subject run loaders (full timeseries, no windowing)
        if args.forecast_mode == "long_ar_train":
            train_run_dataset = SubjectRunDataset(
                combined_dataset,
                train_idx,
                run_loader=run_loader,
                use_cache=args.cache,
            )
            val_run_dataset = SubjectRunDataset(
                combined_dataset,
                val_idx,
                run_loader=run_loader,
                use_cache=args.cache,
            )
            train_run_loader = DataLoader(
                train_run_dataset,
                batch_size=args.run_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
            )
            val_run_loader = DataLoader(
                val_run_dataset,
                batch_size=args.run_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
            )

        fold_save_path = save_path.with_name(f"{save_path.stem}_fold{fold_idx + 1}{save_path.suffix}")
        fold_plot_dir = Path("training") / f"{save_path.stem}_fold{fold_idx + 1}"
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            if args.forecast_mode == "long_ar_train":
                # Linear decay of teacher forcing probability over epochs
                # tf_prob = args.ss_start + (args.ss_end - args.ss_start) * (epoch - 1) / max(args.epochs - 1, 1)
                tf_prob = 0.4
                train_start = time.perf_counter()
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
                train_time_s = time.perf_counter() - train_start
            else:
                train_start = time.perf_counter()
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
                train_time_s = time.perf_counter() - train_start

            run_val_this_epoch = (epoch % args.val_every == 0) or (epoch == args.epochs)
            val_metrics = None
            val_time_s = 0.0
            if run_val_this_epoch:
                val_start = time.perf_counter()
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
                val_time_s = time.perf_counter() - val_start

            prev_lr = optimizer.param_groups[0]["lr"]
            if val_metrics is not None:
                scheduler.step(val_metrics["total"])
            new_lr = optimizer.param_groups[0]["lr"]
            lr_msg = f" | lr={new_lr:.2e}"
            if new_lr < prev_lr:
                lr_msg += " (reduced)"

            epoch_time_s = time.perf_counter() - epoch_start

            if val_metrics is not None:
                print(
                    f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                    f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} "
                    f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f} | "
                    f"val total={val_metrics['total']:.6f} mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f} "
                    f"pcc={val_metrics['pcc']:.4f} scc={val_metrics['scc']:.4f}"
                    f" | t_train={train_time_s:.1f}s t_val={val_time_s:.1f}s t_epoch={epoch_time_s:.1f}s"
                    f"{lr_msg}"
                )
            else:
                print(
                    f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                    f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} "
                    f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f}"
                    f" | t_train={train_time_s:.1f}s t_epoch={epoch_time_s:.1f}s (no val)"
                    f"{lr_msg}"
                )

            if val_metrics is not None and val_metrics["total"] < best_val:
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

            if epoch % 5 == 0:
                plot_path = fold_plot_dir / f"epoch_{epoch:03d}.png"
                saved = save_dynamics_plot(
                    model=model,
                    loader=val_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                    out_path=plot_path,
                )
                if saved:
                    print(f"Saved dynamics plot: {plot_path}")
                else:
                    print(f"Skipped dynamics plot at epoch {epoch}: validation loader had no batches")

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
                    rollout_steps=args.test_rollout_steps,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    desc=f"fold {fold_idx + 1} test-rollout",
                    run_loader=run_loader,
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
    print(f"  DTW   : {_ms(fold_val_metrics, 'dtw')[0]:.6f} ± {_ms(fold_val_metrics, 'dtw')[1]:.6f}")

    print("\nCV Test Summary (mean ± std across folds):")
    print(f"  MSE   : {_ms(fold_test_scores, 'mse')[0]:.6f} ± {_ms(fold_test_scores, 'mse')[1]:.6f}")
    print(f"  MAE   : {_ms(fold_test_scores, 'mae')[0]:.6f} ± {_ms(fold_test_scores, 'mae')[1]:.6f}")
    print(f"  PCC   : {_ms(fold_test_scores, 'pcc')[0]:.4f}  ± {_ms(fold_test_scores, 'pcc')[1]:.4f}")
    print(f"  SCC   : {_ms(fold_test_scores, 'scc')[0]:.4f}  ± {_ms(fold_test_scores, 'scc')[1]:.4f}")
    print(f"  DTW   : {_ms(fold_test_scores, 'dtw')[0]:.6f} ± {_ms(fold_test_scores, 'dtw')[1]:.6f}")


if __name__ == "__main__":
    main()
