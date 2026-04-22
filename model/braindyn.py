from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .dynamics import BrainDynDynamics


@dataclass
class BrainDynConfig:
    signal_dim: int
    hidden_dim: int
    window_size: int
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    map_hidden_dim: int = 128
    vf_hidden_dim: int = 128


class BrainDyn(nn.Module):
    """
    Autoregressive BrainDyn with RK4 stepping.

    At each forecast step:
      1. encode current history
      2. compute spatial attention + restriction maps + sheaf Laplacian
      3. compute dx/dt
      4. use RK4 to get x_next
      5. append x_next to history and repeat
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.dynamics = BrainDynDynamics(
            signal_dim=config.signal_dim,
            hidden_dim=config.hidden_dim,
            window_size=config.window_size,
            lstm_layers=config.lstm_layers,
            lstm_dropout=config.lstm_dropout,
            map_hidden_dim=config.map_hidden_dim,
            vf_hidden_dim=config.vf_hidden_dim,
        )

    def rk4_step(
        self,
        x_hist: torch.Tensor,
        edge_index: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        One autoregressive RK4 step.

        x_hist: (B, N, T, F)
        returns:
            x_next: (B, N, F)
            aux: dict from k1 computation
        """
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, dtype=x_hist.dtype, device=x_hist.device)

        x_t = x_hist[:, :, -1, :]  # (B, N, F)

        # k1
        k1, aux1 = self.dynamics(x_hist, edge_index, x_eval=x_t)

        # k2
        x_mid1 = x_t + 0.5 * dt * k1
        k2, _ = self.dynamics(x_hist, edge_index, x_eval=x_mid1)

        # k3
        x_mid2 = x_t + 0.5 * dt * k2
        k3, _ = self.dynamics(x_hist, edge_index, x_eval=x_mid2)

        # k4
        x_end = x_t + dt * k3
        k4, _ = self.dynamics(x_hist, edge_index, x_eval=x_end)

        x_next = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next, aux1

    def forward(
        self,
        x_history: torch.Tensor,
        edge_index: torch.Tensor,
        pred_steps: int,
        dt: float = 1.0,
        return_aux: bool = False,
    ) -> dict[str, Any]:
        """
        x_history: (B, N, T, F)
        edge_index: (2, E)
        pred_steps: number of future steps
        dt: step size

        returns:
            {
                "x_pred": (pred_steps, B, N, F)
                "aux_seq": optional list of dicts
            }
        """
        hist = x_history
        preds = []
        aux_seq = []

        for _ in range(pred_steps):
            x_next, aux = self.rk4_step(hist, edge_index, dt=dt)
            preds.append(x_next)

            if return_aux:
                aux_seq.append(aux)

            # roll history window
            hist = torch.cat([hist[:, :, 1:, :], x_next.unsqueeze(2)], dim=2)

        x_pred = torch.stack(preds, dim=0)

        out = {"x_pred": x_pred}
        if return_aux:
            out["aux_seq"] = aux_seq
        return out