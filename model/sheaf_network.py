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


####THIS IS WHAT WE ARE USING!! DO NOT LOOK ABOVE####

class SheafIncidenceFast(nn.Module):
    def __init__(self, num_edges, F_node, hidden_dim=32):
        super().__init__()
        self.F_node = F_node
        self.num_edges = num_edges

        # Shared MLPs with per-edge embeddings
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
        R_src = self.mlp_src(self.edge_emb.weight).view(E, F, F)
        R_tgt = self.mlp_tgt(self.edge_emb.weight).view(E, F, F)
        return R_src, R_tgt

    def edge_disagreement(self, X, edge_index):
        src, tgt = edge_index
        R_src, R_tgt = self._restriction_matrices()  # (E, F, F)
        # Expect X: (N, F)
        if X.dim() != 2:
            raise RuntimeError(f'SheafIncidenceFast.edge_disagreement expects X dim==2, got {X.dim()}')

        # X_src: (E, F)
        X_src, X_tgt = X[src], X[tgt]
        # h_src: (E, F) where for each edge e: X_src[e] @ R_src[e].T
        h_src = torch.einsum('ef, efg->eg', X_src, R_src.transpose(1, 2))
        h_tgt = torch.einsum('ef, efg->eg', X_tgt, R_tgt.transpose(1, 2))
        return h_src - h_tgt

    def laplacian_apply(self, X, edge_index):
        src, tgt = edge_index
        R_src, R_tgt = self._restriction_matrices()
        d = self.edge_disagreement(X, edge_index)

        # aggregate back to nodes (expect X: (N, F))
        if X.dim() != 2:
            raise RuntimeError(f'SheafIncidenceFast.laplacian_apply expects X dim==2, got {X.dim()}')

        # d: (E, F)
        m = torch.zeros_like(X)
        contrib_src = -torch.einsum('ef, efg->eg', d, R_src)
        contrib_tgt =  torch.einsum('ef, efg->eg', d, R_tgt)
        m.index_add_(0, src, contrib_src)
        m.index_add_(0, tgt, contrib_tgt)
        return m

import torch
import torch.nn as nn

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
