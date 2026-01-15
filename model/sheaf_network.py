import torch
import torch.nn as nn


class EdgeRestrictionMLP(nn.Module):
    """
    Small MLP that produces a restriction matrix R_{v->e} for each edge
    from a learned embedding of the edge (or just an ID embedding).

    For now we assume:
      - all node features have dimension F_node
      - edge "fiber" dimension = F_edge = F_node (for simplicity)
    """

    def __init__(self, F_node, hidden_dim):
        super().__init__()
        # We parametrize R as a dense matrix of shape (F_edge, F_node)
        # produced by an MLP from a learnable edge embedding.
        self.edge_emb = nn.Parameter(torch.randn(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, F_node * F_node)
        )
        self.F_node = F_node

    def forward(self):
        """
        Returns a restriction matrix R of shape (F_node, F_node)
        """
        h = self.mlp(self.edge_emb)                 # (F_node * F_node,)
        R = h.view(self.F_node, self.F_node)        # (F_edge, F_node)
        return R

class SheafIncidenceFast_Message(nn.Module):
    def __init__(self, num_edges, F_node, hidden_dim=32):
        super().__init__()
        self.F_node = F_node
        self.num_edges = num_edges

        self.edge_emb = nn.Embedding(num_edges, hidden_dim)
        self.mlp_src = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, F_node * F_node)
        )
        self.mlp_tgt = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, F_node * F_node)
        )

    def _restriction_matrices(self):
        E, F = self.num_edges, self.F_node
        R_src = self.mlp_src(self.edge_emb.weight).view(E, F, F)  # R_i^(ij)
        R_tgt = self.mlp_tgt(self.edge_emb.weight).view(E, F, F)  # R_j^(ij)
        return R_src, R_tgt

    def compat(self, X, edge_index):
        """
        Compute edge-level compatibility (disagreement) compat_ij
        for each edge (i,j):

            compat_ij = R_i^(ij) x_i - R_j^(ij) x_j

        X: (N, F)
        edge_index: (2, E)
        Returns:
            compat: (E, F)
        """
        if X.dim() != 2:
            raise RuntimeError(
                f"SheafIncidenceFast.compat expects X dim==2, got {X.dim()}"
            )

        src, tgt = edge_index  # (E,), (E,)
        R_src, R_tgt = self._restriction_matrices()  # (E, F, F)

        X_src, X_tgt = X[src], X[tgt]  # (E, F)

        # h_src[e] = R_i^(ij) x_i, h_tgt[e] = R_j^(ij) x_j
        h_src = torch.einsum("ef, efg->eg", X_src, R_src.transpose(1, 2))
        h_tgt = torch.einsum("ef, efg->eg", X_tgt, R_tgt.transpose(1, 2))

        compat = h_src - h_tgt  # (E, F)
        return compat


class SheafMessagePassingLayer(nn.Module):
    """
    One sheaf neural network (SNN) layer:

        compat_ij = R_i^(ij) x_i - R_j^(ij) x_j
        m_i       = sum_{j:(i,j) in E} compat_ij
        x'_i      = x_i + sigma(W m_i + b)
    """

    def __init__(self, num_edges, F_node, hidden_dim=32):
        super().__init__()
        self.sheaf = SheafIncidenceFast_Message(num_edges, F_node, hidden_dim)
        self.lin = nn.Linear(F_node, F_node)  # W in the paper
        self.act = nn.Tanh()                  # or ReLU, etc.

    def forward(self, X, edge_index):
        """
        X: (N, F)
        edge_index: (2, E)
        """
        src, tgt = edge_index
        compat = self.sheaf.compat(X, edge_index)  # (E, F)

        # Aggregate compat_ij back to nodes:
        # m_i = sum_{j: (i,j) in E} compat_ij
        m = torch.zeros_like(X)
        m.index_add_(0, src, compat)   # add compat_ij to node i

        # NOTE: In the original formula, m_j would use compat_ji
        # = -compat_ij, so if you want symmetric treatment you can do:
        m.index_add_(0, tgt, -compat)

        delta = self.act(self.lin(m))  # sigma(W m_i + b)
        X_out = X + delta              # residual update

        return X_out
