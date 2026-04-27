from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .sheaf import GATAggregator, SheafLaplacian


class BrainDynDynamics(nn.Module):
    """
    Computes dx/dt given a history window and current signal.

    h_t = temporal encoder(x_hist)
    lap_h = sheaf laplacian(h_t)
    dx/dt = f_theta(x_t, lap_h)
    """

    def __init__(
        self,
        signal_dim,
        hidden_dim,
        num_nodes,
        window_size,
        lstm_layers=1,
        lstm_dropout=0.0,
        map_hidden_dim=128,
        vf_hidden_dim=128,
        use_gat=False,
        use_lstm_encoder=True,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.use_gat = use_gat
        self.use_lstm_encoder = use_lstm_encoder

        if self.use_lstm_encoder:
            self.temporal_encoder = TemporalEncoder(
                input_dim=signal_dim,
                hidden_dim=hidden_dim,
                num_layers=lstm_layers,
                dropout=lstm_dropout,
            )
        else:
            self.no_lstm_proj = nn.Linear(signal_dim, hidden_dim)

        if self.use_gat:
            self.graph_laplacian = GATAggregator(hidden_dim=hidden_dim, num_nodes=num_nodes)
        else:
            self.graph_laplacian = SheafLaplacian(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                map_hidden_dim=map_hidden_dim,
            )

        self.vector_field = nn.Sequential(
            nn.Linear(signal_dim + hidden_dim, vf_hidden_dim),
            nn.Tanh(),
            nn.Linear(vf_hidden_dim, vf_hidden_dim),
            nn.Tanh(),
            nn.Linear(vf_hidden_dim, signal_dim),
        )

    def forward(
        self,
        x_hist,
        edge_index,
        x_eval=None,
    ):
        """
        x_hist: (B, N, T, F)
        edge_index: (2, E)
        x_eval: optional signal at which to evaluate the derivative, shape (B, N, F).
                If None, uses the last signal in the history window.

        returns:
            dxdt: (B, N, F)
            aux: dict
        """
        if x_hist.ndim != 4:
            raise ValueError(f"x_hist must have shape (B, N, T, F), got {tuple(x_hist.shape)}")
        if x_hist.shape[2] != self.window_size:
            raise ValueError(f"Expected history length {self.window_size}, got {x_hist.shape[2]}")

        x_t = x_hist[:, :, -1, :] if x_eval is None else x_eval
        if self.use_lstm_encoder:
            h_t = self.temporal_encoder(x_hist)  # (B, N, H)
        else:
            h_t = torch.tanh(self.no_lstm_proj(x_hist[:, :, -1, :]))

        lap_h, graph_aux = self.graph_laplacian(h_t, edge_index)  # (B, N, H)

        vf_input = torch.cat([x_t, lap_h], dim=-1)
        dxdt = self.vector_field(vf_input)

        aux = {
            "h_t": h_t,
            "lap_h": lap_h,
            "dxdt": dxdt,
            "encoder_mode": "lstm" if self.use_lstm_encoder else "last-step-linear",
            "graph_mode": "gat" if self.use_gat else "sheaf",
            **graph_aux,
        }
        return dxdt, aux