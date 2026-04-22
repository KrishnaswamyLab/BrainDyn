from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
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
                out = model(x_history=x_history, edge_index=edge_index, pred_steps=pred_steps, dt=dt)
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


def parse_args():
    ap = argparse.ArgumentParser(description="Train BrainDyn on RBC using a Gaussian-kernel graph.")

    ap.add_argument("--manifest_csv", type=str, default="data/manifest.csv")
    ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None for both")
    ap.add_argument("--x", type=int, default=10, help="context length")
    ap.add_argument("--y", type=int, default=5, help="forecast horizon length")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min_t", type=int, default=0)
    ap.add_argument("--cache", action="store_true")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
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
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError("Train split is empty. Adjust x/y/stride/cohort/min_t settings.")

    edge_index_cpu, kernel, sigma_used = build_gaussian_knn_graph(
        loader=train_loader,
        max_batches=args.graph_max_batches,
        k=args.graph_k,
        sigma=args.graph_sigma,
    )
    edge_index = edge_index_cpu.to(device)

    print(
        f"Undirected Gaussian-kNN graph built: N={kernel.shape[0]}, "
        f"E={edge_index.shape[1]}, k={args.graph_k}, sigma={sigma_used:.6f}"
    )

    config = BrainDynConfig(
        signal_dim=1,
        hidden_dim=args.hidden_dim,
        window_size=args.x,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        map_hidden_dim=args.map_hidden_dim,
        vf_hidden_dim=args.vf_hidden_dim,
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

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None
    if scaler is not None:
        print("AMP enabled: using float16 for forward pass")

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
            desc=f"train {epoch}/{args.epochs}",
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
                desc=f"val {epoch}/{args.epochs}",
            )

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["total"])
        new_lr = optimizer.param_groups[0]["lr"]
        lr_msg = f" | lr={new_lr:.2e}"
        if new_lr < prev_lr:
            lr_msg += " (reduced)"

        print(
            f"Epoch {epoch:03d} | "
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
                },
                save_path,
            )
            print(f"Saved new best checkpoint to {save_path} (val total={best_val:.6f})")

    print("Evaluating best checkpoint on test split...")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            edge_index=edge_index,
            dt=args.dt,
            optimizer=None,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            desc="test",
        )

    print(
        f"Test | total={test_metrics['total']:.6f} "
        f"mse={test_metrics['mse']:.6f} "
        f"mae={test_metrics['mae']:.6f}"
    )


if __name__ == "__main__":
    main()
