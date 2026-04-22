from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_forecast_loss(x_pred, x_true):
    """
    x_pred: (T_pred, B, N, F)
    x_true: (T_pred, B, N, F)
    """
    return F.mse_loss(x_pred, x_true)


def mae_forecast_loss(x_pred, x_true):
    return F.l1_loss(x_pred, x_true)


def total_loss(
    x_pred,
    x_true,
    lambda_mse=1.0,
    lambda_mae=0.0,
):
    mse = mse_forecast_loss(x_pred, x_true)
    mae = mae_forecast_loss(x_pred, x_true)
    total = lambda_mse * mse + lambda_mae * mae

    return {
        "total": total,
        "mse": mse,
        "mae": mae,
    }