from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.sn_dataset import DATA_NPZ_PATH, make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import dtw_mean_normalized, total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_model_tensors_nest(
    batch: dict,
    device: torch.device,
    perturb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NEST forecasting batch to BrainDyn shapes.

    Input batch:
      x: (B, Lx, N)
      y: (B, Ly, N)

    Returns:
      x_history: (B, N, Lx, 1)
      y_true:    (Ly, B, N, 1)
    """
    if perturb:
        x_ctx = batch["x_perturbed"].to(device=device, dtype=torch.float32)
        y_future = batch["y_perturbed"].to(device=device, dtype=torch.float32)
    else:
        x_ctx = batch["x_original"].to(device=device, dtype=torch.float32)
        y_future = batch["y_original"].to(device=device, dtype=torch.float32)

    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def collect_node_features_nest(loader: DataLoader, max_batches: int) -> torch.Tensor:
    """Collect node-wise features from NEST forecasting context windows.

    Returns:
        (N, S) tensor with concatenated samples over batch/time.
    """
    chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x_ctx = batch["x_original"].float()  # (B, Lx, N)
        node_series = x_ctx.permute(2, 0, 1).reshape(x_ctx.shape[2], -1)  # (N, B*Lx)
        chunks.append(node_series)

    if not chunks:
        raise RuntimeError("Unable to build FC graph: train loader produced zero batches.")

    features = torch.cat(chunks, dim=1)
    features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6)
    return features


def build_fc_graph_nest(loader: DataLoader, max_batches: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build undirected FC graph from absolute Pearson correlations."""
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    features = collect_node_features_nest(loader=loader, max_batches=max_batches)
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


def run_perturbation_epoch(
    model: BrainDyn,
    loader: DataLoader,
    edge_index: torch.Tensor,
    dt: float,
    optimizer: torch.optim.Optimizer | None,
    lambda_mse: float,
    lambda_mae: float,
    grad_clip: float,
    use_amp: bool,
    desc: str,
    perturb: bool,
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

    # Slurm / log files are not a TTY: tqdm redraw becomes one line per refresh.
    pbar = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    for batch in pbar:
        x_history, y_true = batch_to_model_tensors_nest(batch, edge_index.device, perturb=perturb)

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

        y_pred_np = y_pred.detach().cpu().numpy().ravel()
        y_true_np = y_true.detach().cpu().numpy().ravel()
        try:
            pcc_val = float(pearsonr(y_pred_np, y_true_np).statistic)
        except Exception:
            pcc_val = float("nan")
        try:
            scc_val = float(spearmanr(y_pred_np, y_true_np).statistic)
        except Exception:
            scc_val = float("nan")

        dtw_val = dtw_mean_normalized(
            y_pred.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
        )

        pcc_running += pcc_val
        scc_running += scc_val
        dtw_running += float(dtw_val)
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


def evaluate_perturbation_split(
    model: BrainDyn,
    loader: DataLoader,
    edge_index: torch.Tensor,
    dt: float,
    lambda_mse: float,
    lambda_mae: float,
    grad_clip: float,
    use_amp: bool,
    split_name: str,
    perturb: bool,
) -> Dict[str, float]:
    metrics = run_perturbation_epoch(
        model=model,
        loader=loader,
        edge_index=edge_index,
        dt=dt,
        optimizer=None,
        lambda_mse=lambda_mse,
        lambda_mae=lambda_mae,
        grad_clip=grad_clip,
        use_amp=use_amp,
        desc=f"forecast {split_name}",
        perturb=perturb,
    )
    print(
        f"{split_name.capitalize()} forecast (perturb={perturb}) | total={metrics['total']:.4f} "
        f"mse={metrics['mse']:.4f} mae={metrics['mae']:.4f} "
        f"pcc={metrics['pcc']:.4f} scc={metrics['scc']:.4f} dtw={metrics['dtw']:.4f}"
    )
    return metrics


def make_subset_loader(
    dataset: ConcatDataset | Subset | torch.utils.data.Dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def metric_mean_std(metrics: list[Dict[str, float]], key: str) -> tuple[float, float]:
    values = np.asarray([m[key] for m in metrics], dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def print_cv_summary(title: str, metrics: list[Dict[str, float]]) -> None:
    print(f"\n{title} (mean +/- std across folds):")
    print(f"  total : {metric_mean_std(metrics, 'total')[0]:.4f} +/- {metric_mean_std(metrics, 'total')[1]:.4f}")
    print(f"  mse   : {metric_mean_std(metrics, 'mse')[0]:.4f} +/- {metric_mean_std(metrics, 'mse')[1]:.4f}")
    print(f"  mae   : {metric_mean_std(metrics, 'mae')[0]:.4f} +/- {metric_mean_std(metrics, 'mae')[1]:.4f}")
    print(f"  pcc   : {metric_mean_std(metrics, 'pcc')[0]:.4f} +/- {metric_mean_std(metrics, 'pcc')[1]:.4f}")
    print(f"  scc   : {metric_mean_std(metrics, 'scc')[0]:.4f} +/- {metric_mean_std(metrics, 'scc')[1]:.4f}")
    print(f"  dtw   : {metric_mean_std(metrics, 'dtw')[0]:.4f} +/- {metric_mean_std(metrics, 'dtw')[1]:.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train BrainDyn on unperturbed simulated NEST trajectories for forecasting.",
    )

    ap.add_argument("--npz_path", type=str, default=str(DATA_NPZ_PATH))

    ap.add_argument("--x", type=int, default=30, help="Context length for unperturbed forecasting windows.")
    ap.add_argument("--y", type=int, default=10, help="Forecast horizon length.")
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--cv_folds", type=int, default=3, help="Number of cross-validation folds over the combined forecasting train+val pool.")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true", help="Cache arrays in RAM (instead of memory mapping).")
    ap.add_argument("--no_pin_memory", action="store_true")

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)
    ap.add_argument("--map_hidden_dim", type=int, default=16)
    ap.add_argument("--vf_hidden_dim", type=int, default=128)
    ap.add_argument("--ode_method", type=str, default="rk4", choices=["rk4", "dopri5", "euler", "midpoint"])
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ablation_gat", action="store_true")
    ap.add_argument("--ablation_no_lstm", action="store_true")
    ap.add_argument("--precompute_lap_h", action="store_true")

    ap.add_argument("--fc_threshold", type=float, default=0.3)
    ap.add_argument("--fc_max_batches", type=int, default=30)

    ap.add_argument("--epochs", "--epochs_forecast", dest="epochs", type=int, default=60)
    ap.add_argument("--lr", "--lr_forecast", dest="lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save_path",
        "--save_forecast_path",
        dest="save_path",
        type=str,
        default="checkpoints/braindyn_nest_perturbed_best.pt",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset.npz not found: {npz_path}")

    perturbation_loaders = make_dataloaders(
        npz_path=str(npz_path),
        task_mode="perturbation",
        x=args.x,
        y=args.y,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        cache=args.cache,
        pin_memory=(not args.no_pin_memory),
        verbose=True,
    )

    train_loader = perturbation_loaders["train"]
    val_loader = perturbation_loaders["val"]
    test_loader = perturbation_loaders["test"]
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

    n_nodes = int(train_loader.dataset.n_channels)

    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    if args.cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {args.cv_folds}")
    if len(combined_dataset) < args.cv_folds:
        raise ValueError(
            f"Not enough combined train+val samples ({len(combined_dataset)}) for {args.cv_folds}-fold CV"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(combined_dataset))
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, args.cv_folds)

    print(f"\n=== Unperturbed Forecasting (NEST, {args.cv_folds}-fold CV) ===")

    fold_val_metrics: list[Dict[str, float]] = []
    fold_test_metrics: list[Dict[str, float]] = []
    fold_infer_metrics: list[Dict[str, float]] = []

    for fold_idx in range(args.cv_folds):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(args.cv_folds) if i != fold_idx])

        fold_train_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=train_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=True,
        )
        fold_val_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=False,
        )

        edge_index_cpu, corr = build_fc_graph_nest(
            loader=fold_train_loader,
            max_batches=args.fc_max_batches,
            threshold=args.fc_threshold,
        )
        edge_index = edge_index_cpu.to(device)

        n_edges = edge_index.shape[1]
        density = n_edges / max(n_nodes * (n_nodes - 1), 1)
        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} | FC graph: {n_edges} directed edges "
            f"| density={density:.3f} | threshold={args.fc_threshold:.3f}"
        )
        print(f"Fold {fold_idx + 1}/{args.cv_folds} | FC abs(corr) mean={corr.abs().mean().item():.3f}, max={corr.abs().max().item():.3f}")

        model_cfg = BrainDynConfig(
            signal_dim=1,
            hidden_dim=args.hidden_dim,
            num_nodes=n_nodes,
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
        model = BrainDyn(model_cfg).to(device)

        optimizer_perturbation = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler_perturbation = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_perturbation,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        best_val_perturbation = float("inf")
        fold_save_path = save_path.with_name(f"{save_path.stem}_fold{fold_idx + 1}{save_path.suffix}")
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_perturbation_epoch(
                model=model,
                loader=fold_train_loader,
                edge_index=edge_index,
                dt=args.dt,
                optimizer=optimizer_perturbation,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
                desc=f"fold {fold_idx + 1} train [{epoch}/{args.epochs}]",
                perturb=False,
            )
            val_metrics = run_perturbation_epoch(
                model=model,
                loader=fold_val_loader,
                edge_index=edge_index,
                dt=args.dt,
                optimizer=None,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
                desc=f"fold {fold_idx + 1} val [{epoch}/{args.epochs}]",
                perturb=False,
            )
            scheduler_perturbation.step(val_metrics["total"])

            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.4f} mse={train_metrics['mse']:.4f} mae={train_metrics['mae']:.4f} "
                f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f} dtw={train_metrics['dtw']:.4f} | "
                f"val total={val_metrics['total']:.4f} mse={val_metrics['mse']:.4f} mae={val_metrics['mae']:.4f} "
                f"pcc={val_metrics['pcc']:.4f} scc={val_metrics['scc']:.4f} dtw={val_metrics['dtw']:.4f}"
            )

            if val_metrics["total"] < best_val_perturbation:
                best_val_perturbation = val_metrics["total"]
                torch.save(
                    {
                        "stage": "perturbation_forecasting",
                        "model_state": model.state_dict(),
                        "edge_index": edge_index.detach().cpu(),
                        "args": vars(args),
                        "val_metrics": val_metrics,
                        "fold": fold_idx + 1,
                    },
                    fold_save_path,
                )
                print(f"Saved fold {fold_idx + 1} best forecasting checkpoint -> {fold_save_path}")

        state = torch.load(fold_save_path, map_location=device)
        model.load_state_dict(state["model_state"])

        best_val_metrics = evaluate_perturbation_split(
            model=model,
            loader=fold_val_loader,
            edge_index=edge_index,
            dt=args.dt,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            split_name=f"fold {fold_idx + 1} val",
            perturb=False,
        )
        test_metrics = evaluate_perturbation_split(
            model=model,
            loader=test_loader,
            edge_index=edge_index,
            dt=args.dt,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            split_name=f"fold {fold_idx + 1} test",
            perturb=False,
        )
        infer_metrics = evaluate_perturbation_split(
            model=model,
            loader=test_loader,
            edge_index=edge_index,
            dt=args.dt,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            split_name=f"fold {fold_idx + 1} infer (perturbation)",
            perturb=True,
        )
        fold_val_metrics.append(best_val_metrics)
        fold_test_metrics.append(test_metrics)
        fold_infer_metrics.append(infer_metrics)

    print_cv_summary("CV Val Summary", fold_val_metrics)
    print_cv_summary("CV Test Summary", fold_test_metrics)
    print_cv_summary("CV Infer Summary", fold_infer_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
