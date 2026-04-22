from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SpatialAttention


class RestrictionMapGenerator(nn.Module):
    """
    Generates Phi_{v->e}(t) in R^{H x H}.
    """

    def __init__(self, hidden_dim, map_hidden_dim=128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.map_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, map_hidden_dim),
            nn.ReLU(),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.ReLU(),
            nn.Linear(map_hidden_dim, hidden_dim * hidden_dim),
        )

    def forward(self, h_src, h_dst):
        """
        h_src: (B, E, H)
        h_dst: (B, E, H)
        returns: (B, E, H, H)
        """
        B, E, H = h_src.shape
        out = self.map_mlp(torch.cat([h_src, h_dst], dim=-1))
        return out.view(B, E, H, H)


class SheafLaplacian(nn.Module):
    """
    (L_F h)_v = sum_{u in N_v} rho_{v->e}^T (tilde{h}_{v->e} - tilde{h}_{u->e})

    with rho_{v->e}(t) = alpha_{vu}(t) * Phi_{v->e}(t)
    """

    def __init__(self, hidden_dim, map_hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.restriction_map = RestrictionMapGenerator(hidden_dim, map_hidden_dim)

    def forward(self, h, edge_index):
        """
        h: (B, N, H)
        edge_index: (2, E)

        returns:
            lap: (B, N, H)
            aux: dict
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (B, N, H), got {tuple(h.shape)}")

        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[:, src, :]
        h_dst = h[:, dst, :]

        alpha = self.spatial_attention(h, edge_index)   # (B, E)

        Phi_src = self.restriction_map(h_src, h_dst)    # (B, E, H, H)
        Phi_dst = self.restriction_map(h_dst, h_src)    # (B, E, H, H)

        rho_src = alpha.unsqueeze(-1).unsqueeze(-1) * Phi_src
        rho_dst = alpha.unsqueeze(-1).unsqueeze(-1) * Phi_dst

        tilde_src = torch.einsum("beij,bej->bei", rho_src, h_src)
        tilde_dst = torch.einsum("beij,bej->bei", rho_dst, h_dst)

        delta = tilde_dst - tilde_src
        pulled_back = torch.einsum("beji,bej->bei", rho_dst, delta)

        B, N, H = h.shape
        lap = torch.zeros_like(h)
        # vectorised scatter over batch dimension: (B, E, H) -> (B, N, H)
        lap.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H),
            pulled_back,
        )

        aux = {
            "alpha": alpha,
            "rho_src": rho_src,
            "rho_dst": rho_dst,
            "tilde_src": tilde_src,
            "tilde_dst": tilde_dst,
            "delta": delta,
        }
        return lap, aux