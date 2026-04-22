from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .sheaf import SheafLaplacian


class BrainDynODEFunc(nn.Module):
    """
    Option 2:
        h_v(t) = temporal encoder from recent signal history
        dx_v/dt = f_theta( x_v(t), (L_F h(t))_v )

    The ODE evolves x directly.
    """

    def __init__(
        self,
        signal_dim,
        hidden_dim,
        window_size,
        lstm_layers=1,
        lstm_dropout=0.0,
        map_hidden_dim=128,
        vf_hidden_dim=128,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        self.temporal_encoder = TemporalEncoder(
            input_dim=signal_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.sheaf_laplacian = SheafLaplacian(
            hidden_dim=hidden_dim,
            map_hidden_dim=map_hidden_dim,
        )

        self.vector_field = nn.Sequential(
            nn.Linear(signal_dim + hidden_dim, vf_hidden_dim),
            nn.Tanh(),
            nn.Linear(vf_hidden_dim, vf_hidden_dim),
            nn.Tanh(),
            nn.Linear(vf_hidden_dim, signal_dim),
        )

        self.edge_index: torch.Tensor | None = None
        self.x_history: torch.Tensor | None = None

    def set_graph(self, edge_index):
        self.edge_index = edge_index

    def set_history(self, x_history):
        """
        x_history: (B, N, T, F_signal)
        """
        self.x_history = x_history

    def forward(self, t, x_t):
        """
        x_t: (B, N, F_signal)
        returns:
            dx/dt: (B, N, F_signal)
        """
        if self.edge_index is None:
            raise RuntimeError("edge_index not set. Call set_graph first.")
        if self.x_history is None:
            raise RuntimeError("x_history not set. Call set_history first.")

        # Encode recent temporal history into latent node states
        h_t = self.temporal_encoder(self.x_history)              # (B, N, H)

        # Sheaf Laplacian in latent space
        lap_h, _ = self.sheaf_laplacian(h_t, self.edge_index)    # (B, N, H)

        # Evolve signal directly
        vf_input = torch.cat([x_t, lap_h], dim=-1)               # (B, N, F_signal + H)
        dxdt = self.vector_field(vf_input)                       # (B, N, F_signal)
        return dxdt