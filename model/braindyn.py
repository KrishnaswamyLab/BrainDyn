from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .dynamics import BrainDynDynamics


@dataclass
class BrainDynConfig:
    signal_dim: int
    hidden_dim: int
    num_nodes: int
    window_size: int
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    map_hidden_dim: int = 16
    vf_hidden_dim: int = 128
    ode_method: str = "rk4"
    use_gat: bool = False
    use_lstm_encoder: bool = True


class BrainDyn(nn.Module):
    """
    BrainDyn with ODE solver stepping.

    Per forward call, integrates over the full requested horizon from
    a fixed context window. Longer rollouts are handled outside this
    module by feeding predicted chunks back into context.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.ode_method = getattr(config, "ode_method", "rk4")

        valid_methods = {"rk4", "dopri5", "euler", "midpoint"}
        if self.ode_method not in valid_methods:
            raise ValueError(
                f"Unsupported ode_method='{self.ode_method}'. "
                f"Choose one of: {sorted(valid_methods)}"
            )

        self.dynamics = BrainDynDynamics(
            signal_dim=config.signal_dim,
            hidden_dim=config.hidden_dim,
            num_nodes=config.num_nodes,
            window_size=config.window_size,
            lstm_layers=config.lstm_layers,
            lstm_dropout=config.lstm_dropout,
            map_hidden_dim=config.map_hidden_dim,
            vf_hidden_dim=config.vf_hidden_dim,
            use_gat=config.use_gat,
            use_lstm_encoder=config.use_lstm_encoder,
        )

    def forward(
        self,
        x_history: torch.Tensor,
        edge_index: torch.Tensor,
        pred_steps: int,
        dt: float = 1.0,
        autoregressive: bool = False,
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
        if pred_steps <= 0:
            raise ValueError(f"pred_steps must be positive, got {pred_steps}")

        # Keep the argument for backward compatibility; rollout strategy
        # is controlled by the caller by feeding predicted chunks back.
        _ = autoregressive

        dt_t = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=x_history.dtype, device=x_history.device)
        x0 = x_history[:, :, -1, :]

        def rhs(_t: torch.Tensor, x_eval: torch.Tensor) -> torch.Tensor:
            dxdt, _ = self.dynamics(x_history, edge_index, x_eval=x_eval)
            return dxdt

        t_eval = torch.arange(pred_steps + 1, device=x_history.device, dtype=x_history.dtype) * dt_t.to(dtype=x_history.dtype)
        x_traj = odeint(rhs, x0, t_eval, method=self.ode_method)
        x_pred = x_traj[1:]

        out = {"x_pred": x_pred}
        if return_aux:
            aux_seq = []
            for step_state in x_pred:
                _, aux = self.dynamics(x_history, edge_index, x_eval=step_state)
                aux_seq.append(aux)
            out["aux_seq"] = aux_seq
        return out