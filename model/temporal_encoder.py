from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    LSTM + temporal attention.

    Input:
        x_hist: (B, N, T, F)

    Output:
        h: (B, N, H)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.theta_t = nn.Linear(2 * hidden_dim, hidden_dim)
        self.a_t = nn.Parameter(torch.randn(hidden_dim))

    def temporal_score(self, current, history):
        """
        current: (B, N, H)
        history: (B, N, T, H)
        returns: (B, N, T)
        """
        T = history.shape[2]
        current_exp = current.unsqueeze(2).expand(-1, -1, T, -1)
        cat = torch.cat([current_exp, history], dim=-1)
        z = torch.tanh(self.theta_t(cat))
        scores = torch.einsum("bnth,h->bnt", z, self.a_t)
        return scores

    def forward(self, x_hist):
        """
        x_hist: (B, N, T, F)
        returns: (B, N, H)
        """
        if x_hist.ndim != 4:
            raise ValueError(f"x_hist must have shape (B, N, T, F), got {tuple(x_hist.shape)}")

        B, N, T, F = x_hist.shape
        if F != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {F}")

        x_flat = x_hist.reshape(B * N, T, F)
        z_seq, _ = self.lstm(x_flat)
        z_seq = z_seq.reshape(B, N, T, self.hidden_dim)

        current = z_seq[:, :, -1, :]
        scores = self.temporal_score(current, z_seq)
        gamma = torch.softmax(scores, dim=-1)

        h = torch.einsum("bnt,bnth->bnh", gamma, z_seq)
        return h