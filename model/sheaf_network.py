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


class SheafIncidence(nn.Module):
    """
    Encodes the sheaf incidence operator B_F implicitly via per-edge
    restriction maps R_src[e], R_tgt[e].

    We never materialize the full B_F; instead we provide:
      - edge_disagreement(X, edge_index)
      - laplacian_apply(X, edge_index)
    """

    def __init__(self, num_edges, F_node, hidden_dim=32):
        """
        num_edges: number of edges E
        F_node: feature dimension at each node
        """
        super().__init__()
        self.F_node = F_node

        # One MLP per edge for src and tgt restriction maps.
        # (You can later replace this with a more neuro-inspired parametrization.)
        self.R_src_mlps = nn.ModuleList([
            EdgeRestrictionMLP(F_node, hidden_dim) for _ in range(num_edges)
        ])
        self.R_tgt_mlps = nn.ModuleList([
            EdgeRestrictionMLP(F_node, hidden_dim) for _ in range(num_edges)
        ])

    def edge_disagreement(self, X, edge_index):
        """
        Compute edge disagreements d = B_F X, without explicit B_F.

        X: (N, F)
        edge_index: (2, E)
        Returns:
            d: (E, F)
        """
        src, tgt = edge_index
        E = src.shape[0]
        F = self.F_node

        X_src = X[src]   # (E, F)
        X_tgt = X[tgt]   # (E, F)

        d_list = []
        for e in range(E):
            R_src = self.R_src_mlps[e]()  # (F, F)
            R_tgt = self.R_tgt_mlps[e]()  # (F, F)
            h_src = X_src[e].unsqueeze(0) @ R_src.T  # (1, F)
            h_tgt = X_tgt[e].unsqueeze(0) @ R_tgt.T  # (1, F)
            d_e = h_src - h_tgt                      # (1, F)
            d_list.append(d_e)

        d = torch.cat(d_list, dim=0)  # (E, F)
        return d

    def laplacian_apply(self, X, edge_index):
        """
        Apply the sheaf Laplacian L = B_F^T B_F to X, i.e., compute L X.

        X: (N, F)
        edge_index: (2, E)
        Returns:
            LX: (N, F)
        """
        src, tgt = edge_index
        E = src.shape[0]
        N, F = X.shape
        device = X.device

        # 1) d = B_F X (edge disagreements)
        d = self.edge_disagreement(X, edge_index)  # (E, F)

        # 2) m = B_F^T d (aggregate back to nodes), implementing
        #    the adjoint action of the restriction maps
        m = torch.zeros(N, F, device=device)

        for e in range(E):
            i = src[e]
            j = tgt[e]
            R_src = self.R_src_mlps[e]()  # (F, F)
            R_tgt = self.R_tgt_mlps[e]()  # (F, F)
            d_e = d[e].unsqueeze(0)       # (1, F)

            # Contribution to node i: -R_src^T d_e
            m[i:i+1] += -d_e @ R_src

            # Contribution to node j: +R_tgt^T d_e
            m[j:j+1] += +d_e @ R_tgt

        # This is L X = B_F^T B_F X
        return m


class SheafGeometry(nn.Module):
    """
    Thin wrapper that exposes Laplacian powers L^k X through the
    SheafIncidence object.
    """

    def __init__(self, num_edges, F_node, hidden_dim=32):
        super().__init__()
        self.incidence = SheafIncidence(
            num_edges=num_edges,
            F_node=F_node,
            hidden_dim=hidden_dim
        )

    def apply_laplacian_power(self, X, edge_index, k):
        """
        Compute L^k X by repeated application of L.

        X: (N, F)
        edge_index: (2, E)
        k: non-negative integer (0,1,2,3,...)
        Returns:
            Y = L^k X
        """
        if k == 0:
            return X
        Y = X
        for _ in range(k):
            Y = self.incidence.laplacian_apply(Y, edge_index)
        return Y