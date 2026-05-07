from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sn_dataset import DATA_NPZ_PATH, make_dataloaders
from train_nest import (
    build_granger_graph_nest,
    collect_gt_reference_adjacency_nest,
    make_subset_loader,
    set_seed,
)


@dataclass
class FoldCalibration:
    fold: int
    threshold: float
    recall: float
    precision: float
    f1: float
    edge_count: int
    density: float
    gt_total: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Calibrate a manual Granger threshold for NEST using training folds only.",
    )
    ap.add_argument("--npz_path", type=str, default=str(DATA_NPZ_PATH))
    ap.add_argument("--x", type=int, default=30)
    ap.add_argument("--y", type=int, default=10)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--fc_max_batches", type=int, default=30)
    ap.add_argument("--granger_lag", type=int, default=1)
    ap.add_argument(
        "--granger_gt_reference",
        type=str,
        default="first_subject",
        choices=["first_subject", "union", "intersection"],
    )
    ap.add_argument(
        "--target_recall",
        type=float,
        default=0.8,
        help="Recommend the largest threshold whose mean train-fold GT recall stays above this target.",
    )
    ap.add_argument(
        "--max_candidate_thresholds",
        type=int,
        default=512,
        help="If pooled unique thresholds are too many, subsample to this many values.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_json",
        type=str,
        default="outputs/nest_granger_threshold_calibration.json",
    )
    return ap.parse_args()


def evaluate_threshold(score: torch.Tensor, gt_mask: torch.Tensor, threshold: float) -> dict[str, float]:
    diag_mask = torch.eye(score.shape[0], dtype=torch.bool)
    pred = score > float(threshold)
    pred &= ~diag_mask

    tp = int((pred & gt_mask).sum().item())
    gt_total = int(gt_mask.sum().item())
    pred_total = int(pred.sum().item())
    fp = max(pred_total - tp, 0)
    fn = max(gt_total - tp, 0)

    recall = tp / gt_total if gt_total > 0 else float("nan")
    precision = tp / pred_total if pred_total > 0 else 0.0
    f1 = 0.0 if tp == 0 else 2.0 * tp / max(2 * tp + fp + fn, 1)
    density = pred_total / max(score.shape[0] * (score.shape[0] - 1), 1)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "density": float(density),
    }


def downsample_candidates(candidates: np.ndarray, max_count: int) -> np.ndarray:
    if candidates.size <= max_count:
        return candidates
    keep_idx = np.linspace(0, candidates.size - 1, num=max_count, dtype=int)
    return candidates[keep_idx]


def select_recommended_threshold(metrics: list[dict[str, float]], target_recall: float) -> dict[str, float]:
    eligible = [row for row in metrics if row["mean_recall"] >= target_recall]
    if eligible:
        return max(eligible, key=lambda row: row["threshold"])
    return max(metrics, key=lambda row: row["mean_f1"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset.npz not found: {npz_path}")

    forecast_loaders = make_dataloaders(
        npz_path=str(npz_path),
        task_mode="forecasting",
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
        verbose=False,
    )

    train_loader = forecast_loaders["train"]
    val_loader = forecast_loaders["val"]
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()
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

    fold_scores: list[torch.Tensor] = []
    fold_gt_masks: list[torch.Tensor] = []
    pooled_positive_gt_scores: list[np.ndarray] = []

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

        gt_ref_adj = collect_gt_reference_adjacency_nest(
            loader=fold_train_loader,
            max_batches=args.fc_max_batches,
            reference=args.granger_gt_reference,
        )
        _, score, _ = build_granger_graph_nest(
            loader=fold_train_loader,
            max_batches=args.fc_max_batches,
            threshold=0.0,
            lag=args.granger_lag,
            threshold_mode="manual",
            extra_edges=0,
            gt_ref_adj=None,
        )

        gt_mask = gt_ref_adj.bool().clone()
        gt_mask.fill_diagonal_(False)

        fold_scores.append(score)
        fold_gt_masks.append(gt_mask)

        gt_scores = score[gt_mask]
        gt_pos = gt_scores[gt_scores > 0].detach().cpu().numpy()
        if gt_pos.size > 0:
            pooled_positive_gt_scores.append(gt_pos)

    candidate_thresholds = np.array([0.0], dtype=np.float64)
    if pooled_positive_gt_scores:
        pooled = np.unique(np.concatenate(pooled_positive_gt_scores))
        pooled = downsample_candidates(pooled, args.max_candidate_thresholds)
        candidate_thresholds = np.unique(np.concatenate([candidate_thresholds, pooled]))

    summary_metrics: list[dict[str, float]] = []
    best_per_fold: list[FoldCalibration] = []

    for threshold in candidate_thresholds:
        per_fold = [
            evaluate_threshold(score=score, gt_mask=gt_mask, threshold=float(threshold))
            for score, gt_mask in zip(fold_scores, fold_gt_masks)
        ]
        summary_metrics.append(
            {
                "threshold": float(threshold),
                "mean_recall": float(np.mean([row["recall"] for row in per_fold])),
                "mean_precision": float(np.mean([row["precision"] for row in per_fold])),
                "mean_f1": float(np.mean([row["f1"] for row in per_fold])),
                "mean_density": float(np.mean([row["density"] for row in per_fold])),
                "mean_edge_count": float(np.mean([row["pred_total"] for row in per_fold])),
            }
        )

    recommended = select_recommended_threshold(summary_metrics, args.target_recall)

    for fold_idx, (score, gt_mask) in enumerate(zip(fold_scores, fold_gt_masks), start=1):
        stats = evaluate_threshold(score=score, gt_mask=gt_mask, threshold=recommended["threshold"])
        best_per_fold.append(
            FoldCalibration(
                fold=fold_idx,
                threshold=float(recommended["threshold"]),
                recall=float(stats["recall"]),
                precision=float(stats["precision"]),
                f1=float(stats["f1"]),
                edge_count=int(stats["pred_total"]),
                density=float(stats["density"]),
                gt_total=int(stats["gt_total"]),
            )
        )

    best_f1 = max(summary_metrics, key=lambda row: row["mean_f1"])
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "args": vars(args),
        "recommendation_rule": {
            "target_recall": float(args.target_recall),
            "description": "largest threshold whose mean train-fold GT recall meets target; fallback is highest mean F1",
        },
        "recommended_threshold": recommended,
        "best_mean_f1_threshold": best_f1,
        "per_fold_at_recommended_threshold": [vars(row) for row in best_per_fold],
        "sweep": summary_metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Recommended manual threshold: {recommended['threshold']:.8f}")
    print(
        "Mean train-fold metrics at recommendation: "
        f"recall={recommended['mean_recall']:.4f}, "
        f"precision={recommended['mean_precision']:.4f}, "
        f"f1={recommended['mean_f1']:.4f}, "
        f"density={recommended['mean_density']:.4f}, "
        f"edges={recommended['mean_edge_count']:.1f}"
    )
    print(
        "Best mean-F1 threshold: "
        f"{best_f1['threshold']:.8f} "
        f"(recall={best_f1['mean_recall']:.4f}, precision={best_f1['mean_precision']:.4f}, "
        f"f1={best_f1['mean_f1']:.4f}, density={best_f1['mean_density']:.4f})"
    )
    print(f"Saved full sweep to {out_path}")


if __name__ == "__main__":
    main()