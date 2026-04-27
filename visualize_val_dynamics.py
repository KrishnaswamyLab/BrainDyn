from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.rbc_dataset import make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig
from model.losses import total_loss


def parse_args():
	ap = argparse.ArgumentParser(
		description="Visualize fixed-horizon dynamics on the CV validation split (non-autoregressive)."
	)
	ap.add_argument("--checkpoint", type=str, default="checkpoints/braindyn_rbc_pnc_best_fold1.pt")
	ap.add_argument("--manifest_csv", type=str, default=None, help="Defaults to training config from checkpoint")
	ap.add_argument("--cohort", type=str, default=None, help="PNC, HBN, or None (defaults to checkpoint config)")
	ap.add_argument("--batch_size", type=int, default=8)
	ap.add_argument("--num_workers", type=int, default=2)
	ap.add_argument("--cache", action="store_true")
	ap.add_argument("--no_pin_memory", action="store_true")
	ap.add_argument("--fold", type=int, default=None, help="Override checkpoint fold index (1-based)")
	ap.add_argument("--max_batches", type=int, default=None, help="Optional cap on number of val batches to evaluate")
	ap.add_argument("--n_rois", type=int, default=6, help="Number of evenly-spaced ROI traces to show per sample")
	ap.add_argument("--n_extreme_rois", type=int, default=3, help="Number of best- and worst-MSE ROIs to show per sample")
	ap.add_argument("--n_samples", type=int, default=None, help="Deprecated alias: if set, uses this for both --n_best and --n_worst")
	ap.add_argument("--n_best", type=int, default=3, help="Number of best-MSE samples to save")
	ap.add_argument("--n_worst", type=int, default=3, help="Number of worst-MSE samples to save")
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--out_dir", type=str, default="outputs/val_dynamics")
	return ap.parse_args()


def get_cfg_value(ckpt_cfg, key, default):
	return ckpt_cfg[key] if key in ckpt_cfg and ckpt_cfg[key] is not None else default


def sanitize_label(value):
	return str(value).replace("/", "-").replace(" ", "_")


def pearson_corr(y_pred, y_true, eps=1e-8):
	"""Compute robust Pearson correlation on flattened arrays."""
	p = y_pred.reshape(-1).astype(np.float64)
	t = y_true.reshape(-1).astype(np.float64)
	p = p - p.mean()
	t = t - t.mean()
	den = np.sqrt(np.dot(p, p) * np.dot(t, t))
	if den <= eps:
		return 0.0
	return float(np.dot(p, t) / den)


def make_subset_loader(dataset, indices, batch_size, num_workers, pin_memory, shuffle):
	return DataLoader(
		Subset(dataset, list(indices)),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=(num_workers > 0),
	)


def rollout_autoregressive(model, x_history, edge_index, dt, pred_steps, chunk_size):
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


def _draw_roi_section(
	axes_section,
	roi_ids,
	roi_mse_per_roi,
	context_raw,
	future_raw,
	pred_raw,
	context_steps,
	future_steps,
	full_t,
	future_t,
	section_label,
	show_legend,
):
	"""Render one labelled section of ROI subplots."""
	for row_idx, (ax, roi) in enumerate(zip(axes_section, roi_ids)):
		ax.axvspan(0, context_steps - 1, color="#dbeafe", alpha=0.35, zorder=0)
		ax.axvspan(context_steps - 1, context_steps + future_steps - 1, color="#fef3c7", alpha=0.26, zorder=0)
		ax.plot(full_t[:context_steps], context_raw[:, roi], label="context", linewidth=2.4, color="#2563eb")
		ax.plot(future_t, future_raw[:, roi], label="ground truth", linewidth=2.4, color="#0f766e")
		ax.plot(future_t, pred_raw[:, roi], label="prediction", linewidth=2.1, linestyle="--", color="#dc2626")
		ax.axvline(context_steps - 1, color="#4b5563", linestyle=":", linewidth=1.2)
		ax.set_ylabel(f"ROI {roi}\nmse={roi_mse_per_roi[roi]:.4f}", fontsize=9)
		ax.grid(alpha=0.45)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_alpha(0.4)
		ax.spines["bottom"].set_alpha(0.4)
		if row_idx == 0:
			ax.set_title(section_label, loc="left", fontsize=10, fontweight="bold", color="#374151", pad=4)
			if show_legend:
				ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cdd4de")


def plot_val_sample(
	out_dir,
	group,
	sample_index,
	meta,
	context_raw,
	future_raw,
	pred_raw,
	n_rois,
	n_extreme_rois=3,
):
	plt.style.use("seaborn-v0_8-whitegrid")
	plt.rcParams.update(
		{
			"figure.facecolor": "#f7f8fa",
			"axes.facecolor": "#ffffff",
			"axes.edgecolor": "#c7c9ce",
			"axes.titleweight": "semibold",
			"axes.labelcolor": "#2f3542",
			"xtick.color": "#475569",
			"ytick.color": "#475569",
			"grid.color": "#d5d9e0",
			"grid.alpha": 0.55,
			"font.size": 10,
		}
	)

	n_nodes = future_raw.shape[1]

	# Per-ROI MSE over the prediction window
	roi_mse_per_roi = np.mean((pred_raw - future_raw) ** 2, axis=0)  # (n_nodes,)

	# Section 1: evenly-spaced ROIs (overview)
	roi_count = min(n_rois, n_nodes)
	sampled_roi_ids = np.linspace(0, n_nodes - 1, roi_count, dtype=int)

	# Section 2 & 3: best / worst ROIs by per-ROI MSE
	n_ext = min(n_extreme_rois, n_nodes)
	sorted_by_mse = np.argsort(roi_mse_per_roi)
	best_roi_ids = sorted_by_mse[:n_ext]
	worst_roi_ids = sorted_by_mse[-n_ext:][::-1]

	total_rows = roi_count + n_ext + n_ext
	context_steps = context_raw.shape[0]
	future_steps = future_raw.shape[0]
	full_t = np.arange(context_steps + future_steps)
	future_t = np.arange(context_steps, context_steps + future_steps)

	fig, axes = plt.subplots(total_rows, 1, figsize=(12, 2.4 * total_rows), sharex=True)
	if total_rows == 1:
		axes = [axes]

	# Draw the three sections
	_draw_roi_section(
		axes[:roi_count], sampled_roi_ids, roi_mse_per_roi,
		context_raw, future_raw, pred_raw,
		context_steps, future_steps, full_t, future_t,
		section_label="Evenly-sampled ROIs", show_legend=True,
	)
	_draw_roi_section(
		axes[roi_count : roi_count + n_ext], best_roi_ids, roi_mse_per_roi,
		context_raw, future_raw, pred_raw,
		context_steps, future_steps, full_t, future_t,
		section_label="Best-MSE ROIs", show_legend=False,
	)
	_draw_roi_section(
		axes[roi_count + n_ext :], worst_roi_ids, roi_mse_per_roi,
		context_raw, future_raw, pred_raw,
		context_steps, future_steps, full_t, future_t,
		section_label="Worst-MSE ROIs", show_legend=False,
	)

	# Draw thin separator lines between sections
	for sep_idx in [roi_count - 1, roi_count + n_ext - 1]:
		axes[sep_idx].spines["bottom"].set_linewidth(1.8)
		axes[sep_idx].spines["bottom"].set_color("#94a3b8")
		axes[sep_idx].spines["bottom"].set_alpha(1.0)

	axes[-1].set_xlabel("Time step")

	fig.suptitle(
		(
			f"Validation Window ({group}) | mse={meta['sample_mse']:.6f} | corr={meta['sample_corr']:.4f} | "
			f"cohort={meta['cohort']} subject={meta['subject_id']} run={meta['run']} t0={meta['t_start']}"
		),
		fontsize=12.5,
	)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

	file_name = (
		f"{group}_{sample_index:03d}_mse-{meta['sample_mse']:.6f}_corr-{meta['sample_corr']:.4f}_cohort-{sanitize_label(meta['cohort'])}"
		f"_subject-{sanitize_label(meta['subject_id'])}_run-{sanitize_label(meta['run'])}"
		f"_t0-{int(meta['t_start']):04d}.png"
	)
	out_path = out_dir / file_name
	fig.savefig(out_path, dpi=160)
	plt.close(fig)
	return out_path


def main():
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	n_best = int(args.n_samples) if args.n_samples is not None else int(args.n_best)
	n_worst = int(args.n_samples) if args.n_samples is not None else int(args.n_worst)
	if n_best < 0 or n_worst < 0:
		raise ValueError("n_best and n_worst must be >= 0")

	ckpt_path = Path(args.checkpoint)
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	ckpt = torch.load(ckpt_path, map_location=device)
	ckpt_cfg = ckpt.get("config", {})

	manifest_csv = Path(args.manifest_csv or get_cfg_value(ckpt_cfg, "manifest_csv", "data/manifest.csv"))
	if not manifest_csv.exists():
		raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

	cohort = args.cohort if args.cohort is not None else get_cfg_value(ckpt_cfg, "cohort", None)
	x = int(get_cfg_value(ckpt_cfg, "x", 20))
	y = int(get_cfg_value(ckpt_cfg, "y", 10))
	stride = int(get_cfg_value(ckpt_cfg, "stride", 10))
	min_t = int(get_cfg_value(ckpt_cfg, "min_t", 0))
	dt = float(get_cfg_value(ckpt_cfg, "dt", 1.0))
	lambda_mse = float(get_cfg_value(ckpt_cfg, "lambda_mse", 1.0))
	lambda_mae = float(get_cfg_value(ckpt_cfg, "lambda_mae", 0.0))
	ode_method = str(get_cfg_value(ckpt_cfg, "ode_method", "rk4"))

	hidden_dim = int(get_cfg_value(ckpt_cfg, "hidden_dim", 64))
	lstm_layers = int(get_cfg_value(ckpt_cfg, "lstm_layers", 1))
	lstm_dropout = float(get_cfg_value(ckpt_cfg, "lstm_dropout", 0.0))
	map_hidden_dim = int(get_cfg_value(ckpt_cfg, "map_hidden_dim", 16))
	vf_hidden_dim = int(get_cfg_value(ckpt_cfg, "vf_hidden_dim", 128))
	forecast_mode = str(get_cfg_value(ckpt_cfg, "forecast_mode", "short"))
	ar_chunk_size = int(get_cfg_value(ckpt_cfg, "ar_chunk_size", 1))
	use_gat = bool(get_cfg_value(ckpt_cfg, "ablation_gat", False))
	use_lstm_encoder = not bool(get_cfg_value(ckpt_cfg, "ablation_no_lstm", False))

	cv_folds = int(get_cfg_value(ckpt_cfg, "cv_folds", 5))
	split_seed = int(get_cfg_value(ckpt_cfg, "seed", args.seed))
	fold_from_ckpt = int(ckpt.get("fold", 1))
	fold_idx_1based = int(args.fold) if args.fold is not None else fold_from_ckpt
	if fold_idx_1based < 1 or fold_idx_1based > cv_folds:
		raise ValueError(f"fold must be in [1, {cv_folds}], got {fold_idx_1based}")

	loaders = make_dataloaders(
		manifest_csv=manifest_csv,
		x=x,
		y=y,
		stride=stride,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		cohort=cohort,
		min_t=min_t,
		cache=args.cache,
		pin_memory=not args.no_pin_memory,
	)
	train_dataset = loaders["train"].dataset
	val_dataset = loaders["val"].dataset
	use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

	combined_dataset = ConcatDataset([train_dataset, val_dataset])
	if len(combined_dataset) == 0:  # type: ignore[arg-type]
		raise RuntimeError("Combined train+val dataset is empty. Check cohort/x/y/stride/min_t settings.")
	if len(combined_dataset) < cv_folds:  # type: ignore[arg-type]
		raise ValueError(f"Not enough combined train+val samples ({len(combined_dataset)}) for {cv_folds}-fold CV")

	rng = np.random.default_rng(split_seed)
	all_indices = np.arange(len(combined_dataset))
	rng.shuffle(all_indices)
	fold_indices = np.array_split(all_indices, cv_folds)
	val_idx = fold_indices[fold_idx_1based - 1]

	val_loader = make_subset_loader(
		dataset=combined_dataset,
		indices=val_idx,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=use_pin,
		shuffle=False,
	)

	if "edge_index" not in ckpt:
		raise KeyError("Checkpoint missing 'edge_index'. Use a checkpoint saved by main.py training.")
	edge_index = ckpt["edge_index"].to(device)
	num_nodes = int(get_cfg_value(ckpt_cfg, "num_nodes", int(edge_index.max().item()) + 1))

	config = BrainDynConfig(
		signal_dim=1,
		hidden_dim=hidden_dim,
		num_nodes=num_nodes,
		window_size=x,
		lstm_layers=lstm_layers,
		lstm_dropout=lstm_dropout,
		map_hidden_dim=map_hidden_dim,
		vf_hidden_dim=vf_hidden_dim,
		ode_method=ode_method,
		use_gat=use_gat,
		use_lstm_encoder=use_lstm_encoder,
	)
	model = BrainDyn(config).to(device)
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	total_running = 0.0
	total_mse = 0.0
	total_mae = 0.0
	total_corr = 0.0
	n_corr = 0
	n_batches = 0
	windows_processed = 0
	saved_paths = []
	best_heap = []  # max-heap via negative mse, stores best (smallest mse)
	worst_heap = []  # min-heap stores worst (largest mse)
	unique_counter = 0

	pbar = tqdm(val_loader, desc="val-visualize", leave=False)
	with torch.no_grad():
		for batch_idx, batch in enumerate(pbar):
			if args.max_batches is not None and batch_idx >= args.max_batches:
				break

			x_ctx = batch["x"].to(device=device, dtype=torch.float32)
			y_future = batch["y"].to(device=device, dtype=torch.float32)

			x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
			y_true = y_future.permute(1, 0, 2).unsqueeze(-1)

			if forecast_mode == "long":
				y_pred = rollout_autoregressive(
					model=model,
					x_history=x_history,
					edge_index=edge_index,
					dt=dt,
					pred_steps=y_true.shape[0],
					chunk_size=ar_chunk_size,
				)
			else:
				out = model(
					x_history=x_history,
					edge_index=edge_index,
					pred_steps=y_true.shape[0],
					dt=dt,
					autoregressive=False,
				)
				y_pred = out["x_pred"]

			losses = total_loss(y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae)
			total_running += float(losses["total"].detach().cpu())
			total_mse += float(losses["mse"].detach().cpu())
			total_mae += float(losses["mae"].detach().cpu())
			n_batches += 1

			batch_size = x_ctx.shape[0]
			windows_processed += batch_size
			y_pred_bn = y_pred.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()
			y_true_bn = y_true.permute(1, 0, 2, 3).squeeze(-1).detach().cpu().numpy()
			meta = batch["meta"]

			for i in range(batch_size):
				sample_mse = float(np.mean((y_pred_bn[i] - y_true_bn[i]) ** 2))
				sample_corr = pearson_corr(y_pred_bn[i], y_true_bn[i])
				total_corr += sample_corr
				n_corr += 1
				record = {
					"sample_mse": sample_mse,
					"sample_corr": sample_corr,
					"path": meta["path"][i],
					"t_start": int(meta["t_start"][i]),
					"cohort": meta["cohort"][i],
					"subject_id": meta["subject_id"][i],
					"run": meta["run"][i],
					"pred_norm": y_pred_bn[i].astype(np.float32),
				}

				if n_best > 0:
					best_item = (-sample_mse, unique_counter, record)
					if len(best_heap) < n_best:
						heapq.heappush(best_heap, best_item)
					elif best_item[0] > best_heap[0][0]:
						heapq.heapreplace(best_heap, best_item)

				if n_worst > 0:
					worst_item = (sample_mse, unique_counter, record)
					if len(worst_heap) < n_worst:
						heapq.heappush(worst_heap, worst_item)
					elif worst_item[0] > worst_heap[0][0]:
						heapq.heapreplace(worst_heap, worst_item)

				unique_counter += 1

			pbar.set_postfix(
				{
					"total": f"{total_running / max(n_batches, 1):.4f}",
					"mse": f"{total_mse / max(n_batches, 1):.4f}",
					"mae": f"{total_mae / max(n_batches, 1):.4f}",
					"corr": f"{total_corr / max(n_corr, 1):.4f}",
				}
			)

	best_records = [item[2] for item in sorted(best_heap, key=lambda x: (-x[0], x[1]))]
	worst_records = [item[2] for item in sorted(worst_heap, key=lambda x: (-x[0], x[1]))]

	best_samples = []
	for idx, rec in enumerate(best_records):
		ts = np.loadtxt(rec["path"], delimiter=",", comments="#", dtype=np.float32)
		t_start = rec["t_start"]
		context_raw = ts[t_start : t_start + x]
		future_raw = ts[t_start + x : t_start + x + y]
		mean = context_raw.mean(axis=0, keepdims=True)
		std = context_raw.std(axis=0, keepdims=True).clip(1e-6)
		pred_raw = rec["pred_norm"] * std + mean

		sample_meta = {
			"sample_mse": rec["sample_mse"],
			"sample_corr": rec["sample_corr"],
			"cohort": rec["cohort"],
			"subject_id": rec["subject_id"],
			"run": rec["run"],
			"t_start": t_start,
		}
		out_path = plot_val_sample(
			out_dir=out_dir,
			group="best",
			sample_index=idx,
			meta=sample_meta,
			context_raw=context_raw,
			future_raw=future_raw,
			pred_raw=pred_raw,
			n_rois=args.n_rois,
			n_extreme_rois=args.n_extreme_rois,
		)
		saved_paths.append(str(out_path))
		best_samples.append({
			"sample_mse": rec["sample_mse"],
			"sample_corr": rec["sample_corr"],
			"cohort": rec["cohort"],
			"subject_id": rec["subject_id"],
			"run": rec["run"],
			"t_start": t_start,
			"plot_path": str(out_path),
		})

	worst_samples = []
	for idx, rec in enumerate(worst_records):
		ts = np.loadtxt(rec["path"], delimiter=",", comments="#", dtype=np.float32)
		t_start = rec["t_start"]
		context_raw = ts[t_start : t_start + x]
		future_raw = ts[t_start + x : t_start + x + y]
		mean = context_raw.mean(axis=0, keepdims=True)
		std = context_raw.std(axis=0, keepdims=True).clip(1e-6)
		pred_raw = rec["pred_norm"] * std + mean

		sample_meta = {
			"sample_mse": rec["sample_mse"],
			"sample_corr": rec["sample_corr"],
			"cohort": rec["cohort"],
			"subject_id": rec["subject_id"],
			"run": rec["run"],
			"t_start": t_start,
		}
		out_path = plot_val_sample(
			out_dir=out_dir,
			group="worst",
			sample_index=idx,
			meta=sample_meta,
			context_raw=context_raw,
			future_raw=future_raw,
			pred_raw=pred_raw,
			n_rois=args.n_rois,
			n_extreme_rois=args.n_extreme_rois,
		)
		saved_paths.append(str(out_path))
		worst_samples.append({
			"sample_mse": rec["sample_mse"],
			"sample_corr": rec["sample_corr"],
			"cohort": rec["cohort"],
			"subject_id": rec["subject_id"],
			"run": rec["run"],
			"t_start": t_start,
			"plot_path": str(out_path),
		})

	summary = {
		"checkpoint": str(ckpt_path),
		"manifest_csv": str(manifest_csv),
		"split": "val_cv",
		"cohort": cohort,
		"fold": fold_idx_1based,
		"cv_folds": cv_folds,
		"x": x,
		"y": y,
		"dt": dt,
		"ode_method": ode_method,
		"forecast_mode": forecast_mode,
		"ar_chunk_size": ar_chunk_size,
		"ablation_gat": use_gat,
		"ablation_no_lstm": (not use_lstm_encoder),
		"lambda_mse": lambda_mse,
		"lambda_mae": lambda_mae,
		"n_best_requested": n_best,
		"n_worst_requested": n_worst,
		"batches_evaluated": n_batches,
		"windows_processed": windows_processed,
		"samples_visualized": len(saved_paths),
		"mean_total": (total_running / max(n_batches, 1)),
		"mean_mse": (total_mse / max(n_batches, 1)),
		"mean_mae": (total_mae / max(n_batches, 1)),
		"mean_corr": (total_corr / max(n_corr, 1)),
		"best_samples": best_samples,
		"worst_samples": worst_samples,
		"out_dir": str(out_dir),
		"saved_plots": saved_paths,
	}

	summary_path = out_dir / "summary.json"
	with open(summary_path, "w", encoding="utf-8") as fh:
		json.dump(summary, fh, indent=2)

	print("Saved validation dynamics plots and summary:")
	print(f"  out_dir: {out_dir}")
	print(f"  summary: {summary_path}")
	print(f"  mean val total: {summary['mean_total']:.6f}")
	print(f"  mean val MSE: {summary['mean_mse']:.6f}")
	print(f"  mean val MAE: {summary['mean_mae']:.6f}")
	print(f"  mean val corr: {summary['mean_corr']:.6f}")


if __name__ == "__main__":
	main()
