from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    score_s(a,b) = a_s^T tanh(Theta_s [a || b])
    alpha_{vu} = softmax over neighbors of v
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.theta_s = nn.Linear(2 * hidden_dim, hidden_dim)
        self.a_s = nn.Parameter(torch.randn(hidden_dim))

    def score(self, h_dst, h_src):
        """
        h_dst: (B, E, H)
        h_src: (B, E, H)
        returns: (B, E)
        """
        z = torch.tanh(self.theta_s(torch.cat([h_dst, h_src], dim=-1)))
        scores = torch.einsum("beh,h->be", z, self.a_s)
        return scores

    def forward(self, h, edge_index):

        """
        h: (B, N, H)
        edge_index: (2, E) with src=edge_index[0], dst=edge_index[1]
        returns: alpha: (B, E)
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (B, N, H), got {tuple(h.shape)}")

        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[:, src, :]
        h_dst = h[:, dst, :]

        scores = self.score(h_dst, h_src)

        B, E = scores.shape
        N = h.shape[1]
        scores_f = scores.float()  # (B, E)
        dst_exp = dst.unsqueeze(0).expand(B, -1)  # (B, E)

        # Segment softmax: subtract per-dst max for numerical stability
        max_s = torch.full((B, N), float("-inf"), dtype=torch.float32, device=h.device)
        max_s.scatter_reduce_(1, dst_exp, scores_f, reduce="amax", include_self=True)
        exp_s = (scores_f - max_s.gather(1, dst_exp)).exp()
        sum_exp = torch.zeros(B, N, dtype=torch.float32, device=h.device)
        sum_exp.scatter_add_(1, dst_exp, exp_s)
        alpha = (exp_s / sum_exp.gather(1, dst_exp)).to(scores.dtype)

        return alpha