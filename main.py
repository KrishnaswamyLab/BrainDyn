from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
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


def build_gaussian_knn_graph(
    loader,
    max_batches,
    k,
    sigma: float | None,
):
    """Build an undirected kNN graph from Gaussian-kernel node similarities.

    The graph is symmetrized after top-k selection so if i is linked to j
    or j is linked to i, the undirected edge {i, j} is kept.
    """
    features = collect_node_features(loader, max_batches=max_batches)  # (N, S)
    num_nodes = features.shape[0]

    if num_nodes < 2:
        raise ValueError(f"Need at least 2 nodes, got {num_nodes}")
    if k <= 0 or k >= num_nodes:
        raise ValueError(f"k must be in [1, N-1], got k={k}, N={num_nodes}")

    dist = torch.cdist(features, features, p=2)  # (N, N)
    dist2 = dist.pow(2)

    if sigma is None:
        mask = ~torch.eye(num_nodes, dtype=torch.bool)
        sigma_val = float(torch.sqrt(dist2[mask].median() + 1e-12).item())
    else:
        sigma_val = float(sigma)
    if sigma_val <= 0:
        raise ValueError(f"sigma must be positive, got {sigma_val}")

    kernel = torch.exp(-dist2 / (2.0 * sigma_val * sigma_val))  # (N, N)

    # Build kNN neighborhoods then symmetrize to enforce an undirected graph.
    neighborhood = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for dst in range(num_nodes):
        row = kernel[dst].clone()
        row[dst] = -1.0
        nn_idx = torch.topk(row, k=k, largest=True).indices
        neighborhood[dst, nn_idx] = True

    neighborhood = neighborhood | neighborhood.T
    neighborhood.fill_diagonal_(False)

    dst_idx, src_idx = torch.where(neighborhood)
    edge_index = torch.stack([src_idx, dst_idx], dim=0).long()
    return edge_index, kernel, sigma_val


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
    autoregressive,
    scaler=None,
) :
    is_train = optimizer is not None
    use_amp = scaler is not None
    model.train(is_train)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history, y_true = batch_to_model_tensors(batch, edge_index.device)
        pred_steps = y_true.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(
                    x_history=x_history,
                    edge_index=edge_index,
                    pred_steps=pred_steps,
                    dt=dt,
                    autoregressive=autoregressive,
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
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_batches:.4f}",
                "mse": f"{mse_running / n_batches:.4f}",
                "mae": f"{mae_running / n_batches:.4f}",
            }
        )

    if n_batches == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan")}

    return {
        "total": total_running / n_batches,
        "mse": mse_running / n_batches,
        "mae": mae_running / n_batches,
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


def run_test_autoregressive_chunks(
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
    model.eval()

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    n_chunks = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        meta = batch["meta"]
        paths = meta["path"]
        t_starts = meta["t_start"]
        run_lengths = meta["T"]

        for path, t_start, run_len in zip(paths, t_starts, run_lengths):
            ts = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
            t0 = int(t_start)
            T = int(run_len)

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

                with torch.no_grad():
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
                n_chunks += 1

                pred_hist = y_pred.permute(1, 2, 0, 3)  # (1, N, step, 1)
                hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)

                current_t += step
                remaining = T - current_t

                pbar.set_postfix(
                    {
                        "total": f"{total_running / n_chunks:.4f}",
                        "mse": f"{mse_running / n_chunks:.4f}",
                        "mae": f"{mae_running / n_chunks:.4f}",
                    }
                )

    if n_chunks == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan")}

    return {
        "total": total_running / n_chunks,
        "mse": mse_running / n_chunks,
        "mae": mae_running / n_chunks,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Train BrainDyn on RBC using a Gaussian-kernel graph.")

    ap.add_argument("--manifest_csv", type=str, default="data/manifest.csv")
    ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None for both")
    ap.add_argument("--x", type=int, default=30, help="context length")
    ap.add_argument("--y", type=int, default=10, help="forecast horizon length")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min_t", type=int, default=0)
    ap.add_argument("--cache", action="store_true")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cv_folds", type=int, default=5, help="number of train/val cross-validation folds")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau multiplicative decay factor")
    ap.add_argument("--lr_patience", type=int, default=2, help="Epochs with no val improvement before reducing LR")
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for ReduceLROnPlateau")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)
    ap.add_argument("--map_hidden_dim", type=int, default=128)
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
        "--graph_k",
        type=int,
        default=5,
        help="neighbors per node before undirected symmetrization",
    )
    ap.add_argument("--graph_sigma", type=float, default=None, help="Gaussian sigma; default uses median distance")
    ap.add_argument("--graph_max_batches", type=int, default=8, help="train batches to estimate graph kernel")

    ap.add_argument("--save_path", type=str, default="checkpoints/braindyn_rbc_best.pt")
    ap.add_argument("--amp", action="store_true", help="Use automatic mixed precision (float16) to reduce GPU memory")
    ap.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory in DataLoader to reduce CPU RAM usage")
    return ap.parse_args()


def main():
    args = parse_args()
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
    default_save_path = "checkpoints/braindyn_rbc_best.pt"
    resolved_save_path = (
        f"checkpoints/braindyn_rbc_{cohort_tag}_best.pt"
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

        edge_index_cpu, kernel, sigma_used = build_gaussian_knn_graph(
            loader=train_loader,
            max_batches=args.graph_max_batches,
            k=args.graph_k,
            sigma=args.graph_sigma,
        )
        edge_index = edge_index_cpu.to(device)

        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} | Undirected Gaussian-kNN graph built: "
            f"N={kernel.shape[0]}, E={edge_index.shape[1]}, k={args.graph_k}, sigma={sigma_used:.6f}"
        )

        config = BrainDynConfig(
            signal_dim=1,
            hidden_dim=args.hidden_dim,
            window_size=args.x,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            map_hidden_dim=args.map_hidden_dim,
            vf_hidden_dim=args.vf_hidden_dim,
            ode_method=args.ode_method,
        )
        model = BrainDyn(config).to(device)
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

        fold_save_path = save_path.with_name(f"{save_path.stem}_fold{fold_idx + 1}{save_path.suffix}")
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
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
                autoregressive=False,
                scaler=scaler,
            )

            with torch.no_grad():
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
                    autoregressive=False,
                )

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["total"])
            new_lr = optimizer.param_groups[0]["lr"]
            lr_msg = f" | lr={new_lr:.2e}"
            if new_lr < prev_lr:
                lr_msg += " (reduced)"

            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} | "
                f"val total={val_metrics['total']:.6f} mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f}"
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
                        "sigma": sigma_used,
                        "fold": fold_idx + 1,
                    },
                    fold_save_path,
                )
                print(f"Saved fold {fold_idx + 1} best checkpoint to {fold_save_path} (val total={best_val:.6f})")

        fold_val_scores.append(best_val)

        print(f"Evaluating fold {fold_idx + 1} best checkpoint on untouched test split...")
        ckpt = torch.load(fold_save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_metrics = run_test_autoregressive_chunks(
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
        fold_test_scores.append(test_metrics)

        print(
            f"Fold {fold_idx + 1} Test Rollout | total={test_metrics['total']:.6f} "
            f"mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f}"
        )

    mean_val = float(np.mean(fold_val_scores))
    mean_test_total = float(np.mean([m["total"] for m in fold_test_scores]))
    mean_test_mse = float(np.mean([m["mse"] for m in fold_test_scores]))
    mean_test_mae = float(np.mean([m["mae"] for m in fold_test_scores]))

    print(f"CV Summary | mean best-val total={mean_val:.6f}")
    print(
        f"CV Test Rollout Summary | total={mean_test_total:.6f} "
        f"mse={mean_test_mse:.6f} mae={mean_test_mae:.6f}"
    )


if __name__ == "__main__":
    main()
