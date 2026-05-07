from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tusz_dataset import make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import total_loss


CONDITION_TO_ID = {
    "clean": 0,
    "transition": 1,
    "seiz_only": 2,
}
ID_TO_CONDITION = {v: k for k, v in CONDITION_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_model_tensors_tusz(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert TUSZ forecasting batch to BrainDyn shapes.

    Input batch:
      x: (B, N, Lx)
      y: (B, N, Ly)

    Returns:
      x_history: (B, N, Lx, 1)
      y_true:    (Ly, B, N, 1)
    """
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)

    x_history = x_ctx.unsqueeze(-1)
    y_true = y_future.permute(2, 0, 1).unsqueeze(-1)
    return x_history, y_true


def collect_node_features_tusz(loader: DataLoader, max_batches: int) -> torch.Tensor:
    """Collect node-wise features from TUSZ forecasting context windows.

    Returns:
        (N, S) tensor with concatenated samples over batch/time.
    """
    chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x_ctx = batch["x"].float()  # (B, N, Lx)
        node_series = x_ctx.permute(1, 0, 2).reshape(x_ctx.shape[1], -1)  # (N, B*Lx)
        chunks.append(node_series)

    if not chunks:
        raise RuntimeError("Unable to build FC graph: train loader produced zero batches.")

    features = torch.cat(chunks, dim=1)
    features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6)
    return features


def build_fc_graph_tusz(loader: DataLoader, max_batches: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build undirected FC graph from absolute Pearson correlations."""
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    features = collect_node_features_tusz(loader=loader, max_batches=max_batches)
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
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history, y_true = batch_to_model_tensors_tusz(batch, edge_index.device)

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
        pcc_val = float(pearsonr(y_pred_np, y_true_np).statistic)
        pcc_running += pcc_val
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_batches:.4f}",
                "mse": f"{mse_running / n_batches:.4f}",
                "pcc": f"{pcc_running / n_batches:.4f}",
            }
        )

    if n_batches == 0:
        return {"total": float("nan"), "mse": float("nan"), "mae": float("nan"), "pcc": float("nan")}

    return {
        "total": total_running / n_batches,
        "mse": mse_running / n_batches,
        "mae": mae_running / n_batches,
        "pcc": pcc_running / n_batches,
    }


def condition_labels_from_meta(meta: dict, device: torch.device) -> torch.Tensor:
    labels = [CONDITION_TO_ID[c] for c in meta["condition"]]
    return torch.tensor(labels, dtype=torch.long, device=device)


class DynamicsConditionHead(nn.Module):
    """Classification head over BrainDyn dynamic representations."""

    def __init__(self, hidden_dim: int, dropout: float = 0.3, num_classes: int = 3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        rep = self.norm(rep)
        return self.mlp(rep)


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

    # Graph-level representation by node average pooling.
    return node_rep.mean(dim=1)


def run_classification_epoch(
    model: BrainDyn,
    classifier: DynamicsConditionHead,
    loader: DataLoader,
    edge_index: torch.Tensor,
    x_len: int,
    rep_source: str,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    grad_clip: float,
    desc: str,
) -> Dict[str, float]:
    del x_len  # kept as explicit arg for readability in call-sites

    is_train = optimizer is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train(is_train)
    classifier.train(is_train)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    confusion = torch.zeros(3, 3, dtype=torch.long)

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history = batch["x"].to(device=edge_index.device, dtype=torch.float32).unsqueeze(-1)
        labels = condition_labels_from_meta(batch["meta"], device=edge_index.device)

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
                loss = F.cross_entropy(logits, labels)

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

        preds = torch.argmax(logits, dim=1)
        running_correct += int((preds == labels).sum().item())
        running_total += int(labels.numel())
        running_loss += float(loss.detach().cpu())

        for t, p in zip(labels.detach().cpu(), preds.detach().cpu()):
            confusion[int(t), int(p)] += 1

        denom = max(running_total, 1)
        pbar.set_postfix({
            "loss": f"{running_loss / max(len(confusion), 1):.4f}",
            "acc": f"{running_correct / denom:.4f}",
        })

    if running_total == 0:
        return {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan")}

    macro_f1 = macro_f1_from_confusion(confusion)
    avg_loss = running_loss / max(len(loader), 1)
    acc = running_correct / running_total
    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    f1s: List[float] = []
    for c in range(confusion.shape[0]):
        tp = float(confusion[c, c].item())
        fp = float(confusion[:, c].sum().item() - tp)
        fn = float(confusion[c, :].sum().item() - tp)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def freeze_module(module: nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train BrainDyn on consolidated TUSZ with short-horizon forecasting and downstream 3-class dynamics classification.",
    )

    ap.add_argument("--h5_path", type=str, default="data/tusz_consolidated.h5")
    ap.add_argument("--manifest_csv", type=str, default="data/manifest_tusz.csv")

    ap.add_argument("--x_len", type=int, default=30, help="Context length for forecasting. Horizon is 40 - x_len.")
    ap.add_argument("--condition", type=str, default=None, choices=[None, "clean", "transition", "seiz_only"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true", help="Cache full HDF5 arrays in RAM.")
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

    ap.add_argument("--epochs_forecast", type=int, default=60)
    ap.add_argument("--epochs_cls", type=int, default=30)
    ap.add_argument("--lr_forecast", type=float, default=3e-4)
    ap.add_argument("--lr_cls_head", type=float, default=1e-3)
    ap.add_argument("--lr_cls_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze BrainDyn during downstream classification.")
    ap.add_argument("--rep_source", type=str, default="lap_h", choices=["lap_h", "h_t"])
    ap.add_argument("--cls_dropout", type=float, default=0.3)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_forecast_path", type=str, default="checkpoints/braindyn_tusz_forecast_best.pt")
    ap.add_argument("--save_classifier_path", type=str, default="checkpoints/braindyn_tusz_classifier_best.pt")
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

    loaders = make_dataloaders(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        task_mode="forecasting",
        x_len=args.x_len,
        condition=args.condition,
        return_type=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache=args.cache,
        pin_memory=(not args.no_pin_memory),
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    edge_index, corr = build_fc_graph_tusz(
        loader=train_loader,
        max_batches=args.fc_max_batches,
        threshold=args.fc_threshold,
    )
    edge_index = edge_index.to(device)
    n_edges = edge_index.shape[1]
    density = n_edges / (19 * 18)
    print(f"FC graph: {n_edges} directed edges | density={density:.3f} | threshold={args.fc_threshold:.3f}")
    print(f"FC abs(corr) mean={corr.abs().mean().item():.3f}, max={corr.abs().max().item():.3f}")

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

    print("\n=== Stage 1: Forecasting pre-training ===")
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

    best_val_forecast = float("inf")
    for epoch in range(1, args.epochs_forecast + 1):
        train_metrics = run_forecasting_epoch(
            model=model,
            loader=train_loader,
            edge_index=edge_index,
            dt=args.dt,
            optimizer=optimizer_forecast,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
            desc=f"forecast train [{epoch}/{args.epochs_forecast}]",
        )
        val_metrics = run_forecasting_epoch(
            model=model,
            loader=val_loader,
            edge_index=edge_index,
            dt=args.dt,
            optimizer=None,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
            desc=f"forecast val   [{epoch}/{args.epochs_forecast}]",
        )
        scheduler_forecast.step(val_metrics["total"])

        print(
            f"Epoch {epoch:03d} | "
            f"train total={train_metrics['total']:.4f} mse={train_metrics['mse']:.4f} pcc={train_metrics['pcc']:.4f} | "
            f"val total={val_metrics['total']:.4f} mse={val_metrics['mse']:.4f} pcc={val_metrics['pcc']:.4f}"
        )

        if val_metrics["total"] < best_val_forecast:
            best_val_forecast = val_metrics["total"]
            save_path = Path(args.save_forecast_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stage": "forecasting",
                    "model_state": model.state_dict(),
                    "edge_index": edge_index.detach().cpu(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            print(f"Saved best forecasting checkpoint -> {save_path}")

    test_forecast = run_forecasting_epoch(
        model=model,
        loader=test_loader,
        edge_index=edge_index,
        dt=args.dt,
        optimizer=None,
        lambda_mse=args.lambda_mse,
        lambda_mae=args.lambda_mae,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
        desc="forecast test",
    )
    print(
        f"Forecast test | total={test_forecast['total']:.4f} "
        f"mse={test_forecast['mse']:.4f} mae={test_forecast['mae']:.4f} pcc={test_forecast['pcc']:.4f}"
    )

    print("\n=== Stage 2: Dynamics condition classification (clean / transition / seiz_only) ===")
    classifier = DynamicsConditionHead(
        hidden_dim=args.hidden_dim,
        dropout=args.cls_dropout,
        num_classes=3,
    ).to(device)

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

    best_val_macro_f1 = -1.0
    for epoch in range(1, args.epochs_cls + 1):
        train_cls = run_classification_epoch(
            model=model,
            classifier=classifier,
            loader=train_loader,
            edge_index=edge_index,
            x_len=args.x_len,
            rep_source=args.rep_source,
            optimizer=optimizer_cls,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
            desc=f"cls train [{epoch}/{args.epochs_cls}]",
        )
        val_cls = run_classification_epoch(
            model=model,
            classifier=classifier,
            loader=val_loader,
            edge_index=edge_index,
            x_len=args.x_len,
            rep_source=args.rep_source,
            optimizer=None,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
            desc=f"cls val   [{epoch}/{args.epochs_cls}]",
        )
        scheduler_cls.step(val_cls["macro_f1"])

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_cls['loss']:.4f} acc={train_cls['acc']:.4f} macro_f1={train_cls['macro_f1']:.4f} | "
            f"val loss={val_cls['loss']:.4f} acc={val_cls['acc']:.4f} macro_f1={val_cls['macro_f1']:.4f}"
        )

        if val_cls["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_cls["macro_f1"]
            save_path = Path(args.save_classifier_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stage": "classification",
                    "model_state": model.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "edge_index": edge_index.detach().cpu(),
                    "args": vars(args),
                    "val_metrics": val_cls,
                    "label_mapping": CONDITION_TO_ID,
                },
                save_path,
            )
            print(f"Saved best classifier checkpoint -> {save_path}")

    test_cls = run_classification_epoch(
        model=model,
        classifier=classifier,
        loader=test_loader,
        edge_index=edge_index,
        x_len=args.x_len,
        rep_source=args.rep_source,
        optimizer=None,
        use_amp=args.amp,
        grad_clip=args.grad_clip,
        desc="cls test",
    )
    print(
        f"Classification test | loss={test_cls['loss']:.4f} "
        f"acc={test_cls['acc']:.4f} macro_f1={test_cls['macro_f1']:.4f}"
    )

    print("\nDone.")
    print("Classes:")
    for idx in sorted(ID_TO_CONDITION):
        print(f"  {idx}: {ID_TO_CONDITION[idx]}")


if __name__ == "__main__":
    main()
