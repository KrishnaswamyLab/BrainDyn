from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    GCNConv = None


class SheafLaplacian(nn.Module):
    """
    (L_F h)_v = sum_{u in N_v} rho_{v->e}^T (tilde{h}_{v->e} - tilde{h}_{u->e})
    """

    def __init__(self, hidden_dim, num_nodes, map_hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        # Directional gates are learned per node and per sheaf channel.
        self.src_weight_table = nn.Parameter(torch.ones(num_nodes, map_hidden_dim))
        self.dst_weight_table = nn.Parameter(torch.ones(num_nodes, map_hidden_dim))
        self.restriction_maps = nn.Parameter(torch.empty(num_nodes, hidden_dim, map_hidden_dim))
        nn.init.xavier_uniform_(self.restriction_maps)

    @staticmethod
    def _minmax_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, h, edge_index):
        """
        h: (B, N, D)
        edge_index: (2, E)

        returns:
            lap: (B, N, D)
            aux: dict
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (B, N, D), got {tuple(h.shape)}")
        if h.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {h.shape[1]}")

        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[:, src, :]
        h_dst = h[:, dst, :]

        Phi_src = self.restriction_maps[src]
        Phi_dst = self.restriction_maps[dst]

        rho_src = Phi_src
        rho_dst = Phi_dst

        tilde_src = torch.einsum("bed,edh->beh", h_src, rho_src)
        tilde_dst = torch.einsum("bed,edh->beh", h_dst, rho_dst)

        tilde_src_norm = self._minmax_normalize(tilde_src)
        tilde_dst_norm = self._minmax_normalize(tilde_dst)

        src_weight = torch.sigmoid(self.src_weight_table[src]).unsqueeze(0)
        dst_weight = torch.sigmoid(self.dst_weight_table[dst]).unsqueeze(0)

        delta = dst_weight * tilde_dst_norm - src_weight * tilde_src_norm
        msg_src = torch.einsum("beh,edh->bed", -delta, rho_src)
        msg_dst = torch.einsum("beh,edh->bed", delta, rho_dst)
        # pulled_back = torch.einsum("beji,bej->bei", rho_dst, delta)

        B, N, D = h.shape
        lap = torch.zeros_like(h)
        lap.index_add_(1, src, msg_src)
        lap.index_add_(1, dst, msg_dst)
        # # vectorised scatter over batch dimension: (B, E, D) -> (B, N, D)
        # lap.scatter_add_(
        #     1,
        #     dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D),
        #     pulled_back,
        # )

        aux = {
            "src_weight": src_weight,
            "dst_weight": dst_weight,
            "rho_src": rho_src,
            "rho_dst": rho_dst,
            "tilde_src": tilde_src,
            "tilde_dst": tilde_dst,
            "tilde_src_norm": tilde_src_norm,
            "tilde_dst_norm": tilde_dst_norm,
            "delta": delta,
        }
        return lap, aux


class GATAggregator(nn.Module):
    """GCN neighborhood aggregation implemented with PyG GCNConv."""

    def __init__(self, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        if GCNConv is None:
            raise ImportError(
                "GCN ablation requires torch-geometric. Install it with: pip install torch-geometric"
            )
        self.gat = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            add_self_loops=False,
            bias=True,
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        """
        h: (B, N, D)
        edge_index: (2, E)
        returns:
            agg: (B, N, D)
            aux: dict
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (B, N, D), got {tuple(h.shape)}")
        if h.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {h.shape[1]}")

        batch_outputs = []
        for b in range(h.shape[0]):
            # PyG GATConv expects node features of shape (N, D) per graph.
            out_b = self.gat(h[b], edge_index)
            batch_outputs.append(out_b)
        agg = torch.stack(batch_outputs, dim=0)

        aux = {
            "graph_mode": "gcn",
        }
        return agg, aux