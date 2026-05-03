from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.tusz_binary_dataset import make_binary_dataloader
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import dtw_mean_normalized, total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_binary_window(batch: dict, x_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert binary TUSZ full windows into context/horizon BrainDyn tensors.

    Input batch:
      x: (B, N, 40)

    Returns:
      x_history: (B, N, x_len, 1)
      y_true:    (Ly, B, N, 1), where Ly = 40 - x_len
    """
    x_full = batch["x"].to(device=device, dtype=torch.float32)
    if x_full.shape[-1] != 40:
        raise ValueError(f"Expected last dim 40 for TUSZ windows, got {x_full.shape[-1]}")
    if not (1 <= x_len <= 39):
        raise ValueError(f"x_len must be in [1, 39], got {x_len}")

    x_ctx = x_full[:, :, :x_len]
    y_future = x_full[:, :, x_len:]
    x_history = x_ctx.unsqueeze(-1)
    y_true = y_future.permute(2, 0, 1).unsqueeze(-1)
    return x_history, y_true


def collect_node_features_binary(loader: DataLoader, x_len: int, max_batches: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x_ctx = batch["x"].float()[:, :, :x_len]  # (B, N, x_len)
        node_series = x_ctx.permute(1, 0, 2).reshape(x_ctx.shape[1], -1)  # (N, B*x_len)
        chunks.append(node_series)

    if not chunks:
        raise RuntimeError("Unable to build FC graph: train loader produced zero batches.")

    features = torch.cat(chunks, dim=1)
    features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6)
    return features


def build_fc_graph_binary(loader: DataLoader, x_len: int, max_batches: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    features = collect_node_features_binary(loader=loader, x_len=x_len, max_batches=max_batches)
    f = features.float()
    corr = (f @ f.T) / (f.shape[1] - 1)
    corr = corr.clamp(-1.0, 1.0)

    adjacency = corr.abs() >= threshold
    adjacency.fill_diagonal_(False)

    dst_idx, src_idx = torch.where(adjacency)
    edge_index = torch.stack([src_idx, dst_idx], dim=0).long()
    if edge_index.shape[1] == 0:
        raise RuntimeError(
            f"FC graph has no edges at threshold={threshold:.3f}. Lower --fc_threshold."
        )

    return edge_index, corr


def run_forecasting_epoch(
    model: BrainDyn,
    loader: DataLoader,
    edge_index: torch.Tensor,
    x_len: int,
    dt: float,
    optimizer: torch.optim.Optimizer | None,
    lambda_mse: float,
    lambda_mae: float,
    grad_clip: float,
    use_amp: bool,
    desc: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
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
        x_history, y_true = split_binary_window(batch=batch, x_len=x_len, device=edge_index.device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
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
        pcc_val = float(pearsonr(y_pred_np.ravel(), y_true_np.ravel()).statistic)
        scc_val = float(spearmanr(y_pred_np.ravel(), y_true_np.ravel()).statistic)
        pcc_running += pcc_val
        scc_running += scc_val
        dtw_running += dtw_mean_normalized(y_pred_np, y_true_np)
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_batches:.4f}",
                "mse": f"{mse_running / n_batches:.4f}",
                "pcc": f"{pcc_running / n_batches:.4f}",
                "scc": f"{scc_running / n_batches:.4f}",
                "dtw": f"{dtw_running / n_batches:.4f}",
            }
        )

    if n_batches == 0:
        return {
            "total": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
            "dtw": float("nan"),
        }

    return {
        "total": total_running / n_batches,
        "mse": mse_running / n_batches,
        "mae": mae_running / n_batches,
        "pcc": pcc_running / n_batches,
        "scc": scc_running / n_batches,
        "dtw": dtw_running / n_batches,
    }


def binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC for binary labels without external dependencies.

    Returns NaN if only one class is present.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


class DynamicsBinaryHead(nn.Module):
    """Binary classification head over BrainDyn dynamic representations."""

    def __init__(self, hidden_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        rep = self.norm(rep)
        return self.mlp(rep).squeeze(-1)


def extract_dynamics_representation(
    model: BrainDyn,
    x_history: torch.Tensor,
    edge_index: torch.Tensor,
    rep_source: str,
) -> torch.Tensor:
    lap_h, aux = model.dynamics.compute_lap_h(x_history, edge_index)
    if rep_source == "lap_h":
        node_rep = lap_h
    elif rep_source == "h_t":
        node_rep = aux["h_t"]
    else:
        raise ValueError(f"Unsupported rep_source={rep_source!r}; choose lap_h or h_t")

    return node_rep.mean(dim=1)


def run_binary_classification_epoch(
    model: BrainDyn,
    classifier: DynamicsBinaryHead,
    loader: DataLoader,
    edge_index: torch.Tensor,
    x_len: int,
    rep_source: str,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    grad_clip: float,
    desc: str,
    pos_weight: float | None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train(is_train)
    classifier.train(is_train)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    tp = tn = fp = fn = 0
    score_accum: list[np.ndarray] = []
    label_accum: list[np.ndarray] = []

    if pos_weight is not None:
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=edge_index.device)
    else:
        pos_weight_t = None

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history = batch["x"].to(device=edge_index.device, dtype=torch.float32)[:, :, :x_len].unsqueeze(-1)
        labels = batch["y"].to(device=edge_index.device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                rep = extract_dynamics_representation(
                    model=model,
                    x_history=x_history,
                    edge_index=edge_index,
                    rep_source=rep_source,
                )
                logits = classifier(rep)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_t)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad],
                            max_norm=grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad],
                            max_norm=grad_clip,
                        )
                    optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        running_correct += int((preds == labels).sum().item())
        running_total += int(labels.numel())
        running_loss += float(loss.detach().cpu())

        tp += int(((preds == 1.0) & (labels == 1.0)).sum().item())
        tn += int(((preds == 0.0) & (labels == 0.0)).sum().item())
        fp += int(((preds == 1.0) & (labels == 0.0)).sum().item())
        fn += int(((preds == 0.0) & (labels == 1.0)).sum().item())

        score_accum.append(probs.detach().cpu().numpy().ravel())
        label_accum.append(labels.detach().cpu().numpy().ravel())

        pbar.set_postfix(
            {
                "loss": f"{running_loss / max(len(loader), 1):.4f}",
                "acc": f"{running_correct / max(running_total, 1):.4f}",
            }
        )

    if running_total == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "aucroc": float("nan"),
        }

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    y_score = np.concatenate(score_accum) if score_accum else np.array([], dtype=np.float64)
    y_true = np.concatenate(label_accum) if label_accum else np.array([], dtype=np.int64)
    aucroc = binary_auc_roc(y_true=y_true, y_score=y_score) if y_score.size > 0 else float("nan")

    return {
        "loss": running_loss / max(len(loader), 1),
        "acc": running_correct / running_total,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "aucroc": float(aucroc),
    }


def freeze_module(module: nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def estimate_pos_weight(loader: DataLoader) -> float:
    pos = 0
    total = 0
    for batch in loader:
        y = batch["y"].view(-1)
        pos += int(y.sum().item())
        total += int(y.numel())
    neg = total - pos
    if pos == 0:
        return 1.0
    return float(neg / pos)


def _targets_from_proportions(class_counts: dict[int, int], total_target: int) -> dict[int, int]:
    """Allocate per-class sample targets while preserving original class proportions."""
    total = sum(class_counts.values())
    if total == 0 or total_target <= 0:
        return {k: 0 for k in class_counts}

    raw = {k: (class_counts[k] / total) * total_target for k in class_counts}
    floor_targets = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total_target - sum(floor_targets.values())

    # Largest-remainder allocation keeps proportions close while summing exactly.
    frac_order = sorted(class_counts.keys(), key=lambda k: raw[k] - floor_targets[k], reverse=True)
    for k in frac_order[:remainder]:
        floor_targets[k] += 1
    return floor_targets


def _round_robin_recording_sample(
    indices: list[int],
    groups: list[str],
    rng: np.random.Generator,
    target_n: int,
) -> list[int]:
    """Sample indices by cycling recordings to maximize diversity."""
    by_group: dict[str, list[int]] = {}
    for i in indices:
        g = groups[i]
        by_group.setdefault(g, []).append(i)

    for g in by_group:
        rng.shuffle(by_group[g])

    group_keys = list(by_group.keys())
    rng.shuffle(group_keys)

    selected: list[int] = []
    while len(selected) < target_n:
        added_in_round = 0
        for g in group_keys:
            if by_group[g]:
                selected.append(by_group[g].pop())
                added_in_round += 1
                if len(selected) >= target_n:
                    break
        if added_in_round == 0:
            break
    return selected


def build_smart_subset_indices(
    dataset,
    h5_path: Path,
    target_n: int,
    seed: int,
) -> list[int]:
    """Create a smart subset preserving class ratio and recording diversity.

    Strategy:
      1) Preserve original seizure/non-seizure class proportions.
      2) Within each class, sample in recording round-robin order using h5_relpath.
      3) Fill any residual budget from remaining pool if one class exhausts early.
    """
    n_total = len(dataset)
    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")
    if target_n >= n_total:
        return list(range(n_total))

    rows = dataset._rows
    window_ids = np.asarray(dataset._window_ids, dtype=np.int64)

    with h5py.File(h5_path, "r") as hf:
        labels = np.asarray(hf["y"][window_ids], dtype=np.int64)

    groups = [r["h5_relpath"] for r in rows]
    class_to_indices: dict[int, list[int]] = {0: [], 1: []}
    for i, y in enumerate(labels.tolist()):
        class_to_indices[int(y)].append(i)

    class_counts = {c: len(v) for c, v in class_to_indices.items()}
    class_targets = _targets_from_proportions(class_counts, total_target=target_n)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_set: set[int] = set()

    for c in (0, 1):
        quota = min(class_targets.get(c, 0), len(class_to_indices[c]))
        picked = _round_robin_recording_sample(
            indices=class_to_indices[c],
            groups=groups,
            rng=rng,
            target_n=quota,
        )
        selected.extend(picked)
        selected_set.update(picked)

    remaining = target_n - len(selected)
    if remaining > 0:
        leftovers = [i for i in range(n_total) if i not in selected_set]
        picked = _round_robin_recording_sample(
            indices=leftovers,
            groups=groups,
            rng=rng,
            target_n=remaining,
        )
        selected.extend(picked)

    rng.shuffle(selected)
    return selected


def maybe_subset_loader(
    loader: DataLoader,
    h5_path: Path,
    target_n: int | None,
    seed: int,
    split_name: str,
) -> DataLoader:
    """Return a subset DataLoader when requested; otherwise return original."""
    if target_n is None:
        return loader

    dataset = loader.dataset
    n_total = len(dataset)
    if target_n >= n_total:
        print(f"[{split_name}] requested subset {target_n} >= total {n_total}; using full split.")
        return loader

    subset_idx = build_smart_subset_indices(
        dataset=dataset,
        h5_path=h5_path,
        target_n=target_n,
        seed=seed,
    )
    sub_ds = Subset(dataset, subset_idx)

    # Preserve DataLoader behavior while swapping in Subset.
    sub_loader = DataLoader(
        sub_ds,
        batch_size=loader.batch_size,
        shuffle=(split_name == "train"),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,
        drop_last=loader.drop_last,
        persistent_workers=(loader.num_workers > 0),
    )

    # Quick subset summary for sanity checks.
    with h5py.File(h5_path, "r") as hf:
        y_all = np.asarray(hf["y"][:], dtype=np.int64)
    y_sub = y_all[np.asarray(dataset._window_ids, dtype=np.int64)[np.asarray(subset_idx, dtype=np.int64)]]
    pos = int(y_sub.sum())
    neg = int(len(y_sub) - pos)
    print(
        f"[{split_name}] smart subset: {len(subset_idx)}/{n_total} windows "
        f"(neg={neg}, pos={pos}, pos_rate={pos / max(len(y_sub), 1):.3f})"
    )

    return sub_loader


def make_subset_loader(
    dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    collate_fn,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train BrainDyn on binary TUSZ windows with short-horizon forecasting and downstream seizure-vs-normal classification.",
    )

    ap.add_argument("--h5_path", type=str, default="data/tusz_binary.h5")
    ap.add_argument("--manifest_csv", type=str, default="data/manifest_tusz_binary.csv")

    ap.add_argument("--x_len", type=int, default=30, help="Context length for forecasting. Horizon is 40 - x_len.")
    ap.add_argument("--zscore", action="store_true", help="Apply per-window z-score from binary dataloader.")
    ap.add_argument("--eps", type=float, default=1e-2, help="z-score std floor when --zscore is enabled.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--subset_train_windows", type=int, default=10000, help="Smart-subset train split to this many windows.")
    ap.add_argument("--subset_eval_windows", type=int, default=2500, help="Smart-subset val/test splits to this many windows each.")
    ap.add_argument("--cv_folds", type=int, default=5, help="number of train/val cross-validation folds over train+val.")

    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)
    ap.add_argument("--map_hidden_dim", type=int, default=64)
    ap.add_argument("--vf_hidden_dim", type=int, default=128)
    ap.add_argument("--ode_method", type=str, default="rk4", choices=["rk4", "dopri5", "euler", "midpoint"])
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ablation_gat", action="store_true")
    ap.add_argument("--ablation_no_lstm", action="store_true")
    ap.add_argument("--precompute_lap_h", action="store_true")

    ap.add_argument("--fc_threshold", type=float, default=0.3)
    ap.add_argument("--fc_max_batches", type=int, default=30)

    ap.add_argument("--epochs_forecast", type=int, default=100)
    ap.add_argument("--epochs_cls", type=int, default=50)
    ap.add_argument("--lr_forecast", type=float, default=3e-4)
    ap.add_argument("--lr_cls_head", type=float, default=1e-3)
    ap.add_argument("--lr_cls_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--rep_source", type=str, default="lap_h", choices=["lap_h", "h_t"])
    ap.add_argument("--cls_dropout", type=float, default=0.3)
    ap.add_argument("--use_pos_weight", action="store_true", help="Use neg/pos weighting in BCE for class imbalance.")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_forecast_path", type=str, default="checkpoints/braindyn_tusz_binary_forecast_best.pt")
    ap.add_argument("--save_classifier_path", type=str, default="checkpoints/braindyn_tusz_binary_classifier_best.pt")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    h5_path = Path(args.h5_path)
    manifest_csv = Path(args.manifest_csv)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    train_loader = make_binary_dataloader(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        zscore=args.zscore,
        eps=args.eps,
    )
    val_loader = make_binary_dataloader(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        zscore=args.zscore,
        eps=args.eps,
    )
    test_loader = make_binary_dataloader(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        zscore=args.zscore,
        eps=args.eps,
    )

    train_loader = maybe_subset_loader(
        loader=train_loader,
        h5_path=h5_path,
        target_n=args.subset_train_windows,
        seed=args.seed,
        split_name="train",
    )
    val_loader = maybe_subset_loader(
        loader=val_loader,
        h5_path=h5_path,
        target_n=args.subset_eval_windows,
        seed=args.seed + 1,
        split_name="val",
    )
    test_loader = maybe_subset_loader(
        loader=test_loader,
        h5_path=h5_path,
        target_n=args.subset_eval_windows,
        seed=args.seed + 2,
        split_name="test",
    )

    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    if args.cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {args.cv_folds}")
    if len(combined_dataset) < args.cv_folds:
        raise ValueError(
            f"Not enough combined train+val samples ({len(combined_dataset)}) for {args.cv_folds}-fold CV"
        )

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(combined_dataset))
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, args.cv_folds)

    forecast_base_path = Path(args.save_forecast_path)
    classifier_base_path = Path(args.save_classifier_path)
    forecast_base_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_base_path.parent.mkdir(parents=True, exist_ok=True)

    fold_forecast_val_metrics: list[dict[str, float]] = []
    fold_forecast_test_metrics: list[dict[str, float]] = []
    fold_cls_val_metrics: list[dict[str, float]] = []
    fold_cls_test_metrics: list[dict[str, float]] = []

    print(f"\n=== {args.cv_folds}-Fold CV over train+val (binary TUSZ) ===")
    print(f"Combined CV pool: {len(combined_dataset)} windows | Held-out test: {len(test_loader.dataset)} windows")

    for fold_idx in range(args.cv_folds):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(args.cv_folds) if i != fold_idx])

        fold_train_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=train_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(not args.no_pin_memory),
            shuffle=True,
            collate_fn=train_loader.collate_fn,
        )
        fold_val_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(not args.no_pin_memory),
            shuffle=False,
            collate_fn=train_loader.collate_fn,
        )

        edge_index, corr = build_fc_graph_binary(
            loader=fold_train_loader,
            x_len=args.x_len,
            max_batches=args.fc_max_batches,
            threshold=args.fc_threshold,
        )
        edge_index = edge_index.to(device)
        n_edges = edge_index.shape[1]
        density = n_edges / (19 * 18)
        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} | FC graph: E={n_edges}, density={density:.3f}, "
            f"|corr| mean={corr.abs().mean().item():.3f}, max={corr.abs().max().item():.3f}"
        )

        model_cfg = BrainDynConfig(
            signal_dim=1,
            hidden_dim=args.hidden_dim,
            num_nodes=19,
            window_size=args.x_len,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            map_hidden_dim=args.map_hidden_dim,
            vf_hidden_dim=args.vf_hidden_dim,
            ode_method=args.ode_method,
            use_gat=args.ablation_gat,
            use_lstm_encoder=(not args.ablation_no_lstm),
            precompute_lap_h=args.precompute_lap_h,
        )
        model = BrainDyn(model_cfg).to(device)

        print(f"\nFold {fold_idx + 1}/{args.cv_folds} | Stage 1: Forecasting")
        optimizer_forecast = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr_forecast,
            weight_decay=args.weight_decay,
        )
        scheduler_forecast = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_forecast,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        fold_forecast_path = forecast_base_path.with_name(
            f"{forecast_base_path.stem}_fold{fold_idx + 1}{forecast_base_path.suffix}"
        )
        best_val_forecast = float("inf")
        for epoch in range(1, args.epochs_forecast + 1):
            train_metrics = run_forecasting_epoch(
                model=model,
                loader=fold_train_loader,
                edge_index=edge_index,
                x_len=args.x_len,
                dt=args.dt,
                optimizer=optimizer_forecast,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                use_amp=args.amp,
                desc=f"fold {fold_idx + 1} forecast train [{epoch}/{args.epochs_forecast}]",
            )
            val_metrics = run_forecasting_epoch(
                model=model,
                loader=fold_val_loader,
                edge_index=edge_index,
                x_len=args.x_len,
                dt=args.dt,
                optimizer=None,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                use_amp=args.amp,
                desc=f"fold {fold_idx + 1} forecast val   [{epoch}/{args.epochs_forecast}]",
            )
            scheduler_forecast.step(val_metrics["total"])

            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.4f} mse={train_metrics['mse']:.4f} mae={train_metrics['mae']:.4f} "
                f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f} | "
                f"val total={val_metrics['total']:.4f} mse={val_metrics['mse']:.4f} mae={val_metrics['mae']:.4f} "
                f"pcc={val_metrics['pcc']:.4f} scc={val_metrics['scc']:.4f}"
            )

            if val_metrics["total"] < best_val_forecast:
                best_val_forecast = val_metrics["total"]
                torch.save(
                    {
                        "stage": "forecasting",
                        "fold": fold_idx + 1,
                        "model_state": model.state_dict(),
                        "edge_index": edge_index.detach().cpu(),
                        "args": vars(args),
                        "val_metrics": val_metrics,
                    },
                    fold_forecast_path,
                )

        ckpt_forecast = torch.load(fold_forecast_path, map_location=device)
        model.load_state_dict(ckpt_forecast["model_state"])

        val_forecast = run_forecasting_epoch(
            model=model,
            loader=fold_val_loader,
            edge_index=edge_index,
            x_len=args.x_len,
            dt=args.dt,
            optimizer=None,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
            desc=f"fold {fold_idx + 1} forecast best-val",
        )
        test_forecast = run_forecasting_epoch(
            model=model,
            loader=test_loader,
            edge_index=edge_index,
            x_len=args.x_len,
            dt=args.dt,
            optimizer=None,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
            desc=f"fold {fold_idx + 1} forecast test",
        )
        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} Forecast | "
            f"val mse={val_forecast['mse']:.4f} mae={val_forecast['mae']:.4f} "
            f"pcc={val_forecast['pcc']:.4f} scc={val_forecast['scc']:.4f} | "
            f"test mse={test_forecast['mse']:.4f} mae={test_forecast['mae']:.4f} "
            f"pcc={test_forecast['pcc']:.4f} scc={test_forecast['scc']:.4f}"
        )

        fold_forecast_val_metrics.append(val_forecast)
        fold_forecast_test_metrics.append(test_forecast)

        print(f"\nFold {fold_idx + 1}/{args.cv_folds} | Stage 2: Binary classification")
        classifier = DynamicsBinaryHead(hidden_dim=args.hidden_dim, dropout=args.cls_dropout).to(device)
        freeze_module(model, args.freeze_backbone)

        if args.freeze_backbone:
            cls_params = list(classifier.parameters())
            print("Classification mode: frozen BrainDyn backbone; training classifier head only.")
        else:
            cls_params = [
                {"params": [p for p in classifier.parameters() if p.requires_grad], "lr": args.lr_cls_head},
                {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_cls_backbone},
            ]
            print("Classification mode: fine-tuning backbone + classifier head.")

        if args.freeze_backbone:
            optimizer_cls = torch.optim.AdamW(
                cls_params,
                lr=args.lr_cls_head,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer_cls = torch.optim.AdamW(
                cls_params,
                weight_decay=args.weight_decay,
            )

        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cls,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        pos_weight = estimate_pos_weight(fold_train_loader) if args.use_pos_weight else None
        if pos_weight is not None:
            print(f"Fold {fold_idx + 1}/{args.cv_folds} BCE pos_weight={pos_weight:.4f}")

        fold_classifier_path = classifier_base_path.with_name(
            f"{classifier_base_path.stem}_fold{fold_idx + 1}{classifier_base_path.suffix}"
        )
        best_val_f1 = -1.0
        for epoch in range(1, args.epochs_cls + 1):
            train_cls = run_binary_classification_epoch(
                model=model,
                classifier=classifier,
                loader=fold_train_loader,
                edge_index=edge_index,
                x_len=args.x_len,
                rep_source=args.rep_source,
                optimizer=optimizer_cls,
                use_amp=args.amp,
                grad_clip=args.grad_clip,
                desc=f"fold {fold_idx + 1} cls train [{epoch}/{args.epochs_cls}]",
                pos_weight=pos_weight,
            )
            val_cls = run_binary_classification_epoch(
                model=model,
                classifier=classifier,
                loader=fold_val_loader,
                edge_index=edge_index,
                x_len=args.x_len,
                rep_source=args.rep_source,
                optimizer=None,
                use_amp=args.amp,
                grad_clip=args.grad_clip,
                desc=f"fold {fold_idx + 1} cls val   [{epoch}/{args.epochs_cls}]",
                pos_weight=pos_weight,
            )
            scheduler_cls.step(val_cls["f1"])

            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                f"train loss={train_cls['loss']:.4f} acc={train_cls['acc']:.4f} precision={train_cls['precision']:.4f} "
                f"recall={train_cls['recall']:.4f} f1={train_cls['f1']:.4f} aucroc={train_cls['aucroc']:.4f} | "
                f"val loss={val_cls['loss']:.4f} acc={val_cls['acc']:.4f} precision={val_cls['precision']:.4f} "
                f"recall={val_cls['recall']:.4f} f1={val_cls['f1']:.4f} aucroc={val_cls['aucroc']:.4f}"
            )

            if val_cls["f1"] > best_val_f1:
                best_val_f1 = val_cls["f1"]
                torch.save(
                    {
                        "stage": "classification",
                        "fold": fold_idx + 1,
                        "model_state": model.state_dict(),
                        "classifier_state": classifier.state_dict(),
                        "edge_index": edge_index.detach().cpu(),
                        "args": vars(args),
                        "val_metrics": val_cls,
                        "label_mapping": {0: "normal", 1: "seizure"},
                    },
                    fold_classifier_path,
                )

        ckpt_cls = torch.load(fold_classifier_path, map_location=device)
        model.load_state_dict(ckpt_cls["model_state"])
        classifier.load_state_dict(ckpt_cls["classifier_state"])

        val_cls_best = run_binary_classification_epoch(
            model=model,
            classifier=classifier,
            loader=fold_val_loader,
            edge_index=edge_index,
            x_len=args.x_len,
            rep_source=args.rep_source,
            optimizer=None,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
            desc=f"fold {fold_idx + 1} cls best-val",
            pos_weight=pos_weight,
        )
        test_cls = run_binary_classification_epoch(
            model=model,
            classifier=classifier,
            loader=test_loader,
            edge_index=edge_index,
            x_len=args.x_len,
            rep_source=args.rep_source,
            optimizer=None,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
            desc=f"fold {fold_idx + 1} cls test",
            pos_weight=pos_weight,
        )
        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} Classification | "
            f"val acc={val_cls_best['acc']:.4f} f1={val_cls_best['f1']:.4f} aucroc={val_cls_best['aucroc']:.4f} | "
            f"test acc={test_cls['acc']:.4f} f1={test_cls['f1']:.4f} aucroc={test_cls['aucroc']:.4f}"
        )

        fold_cls_val_metrics.append(val_cls_best)
        fold_cls_test_metrics.append(test_cls)

    def _ms(metrics_list: list[dict[str, float]], key: str) -> tuple[float, float]:
        vals = np.asarray([m[key] for m in metrics_list], dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    print("\n=== CV Summary ===")
    print("\nForecasting Val (mean +- std across folds):")
    print(f"  MSE   : {_ms(fold_forecast_val_metrics, 'mse')[0]:.6f} +- {_ms(fold_forecast_val_metrics, 'mse')[1]:.6f}")
    print(f"  MAE   : {_ms(fold_forecast_val_metrics, 'mae')[0]:.6f} +- {_ms(fold_forecast_val_metrics, 'mae')[1]:.6f}")
    print(f"  PCC   : {_ms(fold_forecast_val_metrics, 'pcc')[0]:.4f} +- {_ms(fold_forecast_val_metrics, 'pcc')[1]:.4f}")
    print(f"  SCC   : {_ms(fold_forecast_val_metrics, 'scc')[0]:.4f} +- {_ms(fold_forecast_val_metrics, 'scc')[1]:.4f}")
    print(f"  DTW   : {_ms(fold_forecast_val_metrics, 'dtw')[0]:.6f} +- {_ms(fold_forecast_val_metrics, 'dtw')[1]:.6f}")

    print("\nForecasting Test (mean +- std across folds):")
    print(f"  MSE   : {_ms(fold_forecast_test_metrics, 'mse')[0]:.6f} +- {_ms(fold_forecast_test_metrics, 'mse')[1]:.6f}")
    print(f"  MAE   : {_ms(fold_forecast_test_metrics, 'mae')[0]:.6f} +- {_ms(fold_forecast_test_metrics, 'mae')[1]:.6f}")
    print(f"  PCC   : {_ms(fold_forecast_test_metrics, 'pcc')[0]:.4f} +- {_ms(fold_forecast_test_metrics, 'pcc')[1]:.4f}")
    print(f"  SCC   : {_ms(fold_forecast_test_metrics, 'scc')[0]:.4f} +- {_ms(fold_forecast_test_metrics, 'scc')[1]:.4f}")
    print(f"  DTW   : {_ms(fold_forecast_test_metrics, 'dtw')[0]:.6f} +- {_ms(fold_forecast_test_metrics, 'dtw')[1]:.6f}")

    print("\nClassification Val (mean +- std across folds):")
    print(f"  Acc   : {_ms(fold_cls_val_metrics, 'acc')[0]:.4f} +- {_ms(fold_cls_val_metrics, 'acc')[1]:.4f}")
    print(f"  F1    : {_ms(fold_cls_val_metrics, 'f1')[0]:.4f} +- {_ms(fold_cls_val_metrics, 'f1')[1]:.4f}")
    print(f"  AUCROC: {_ms(fold_cls_val_metrics, 'aucroc')[0]:.4f} +- {_ms(fold_cls_val_metrics, 'aucroc')[1]:.4f}")

    print("\nClassification Test (mean +- std across folds):")
    print(f"  Acc   : {_ms(fold_cls_test_metrics, 'acc')[0]:.4f} +- {_ms(fold_cls_test_metrics, 'acc')[1]:.4f}")
    print(f"  F1    : {_ms(fold_cls_test_metrics, 'f1')[0]:.4f} +- {_ms(fold_cls_test_metrics, 'f1')[1]:.4f}")
    print(f"  AUCROC: {_ms(fold_cls_test_metrics, 'aucroc')[0]:.4f} +- {_ms(fold_cls_test_metrics, 'aucroc')[1]:.4f}")


if __name__ == "__main__":
    main()
