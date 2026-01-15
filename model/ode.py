import torch
import torch.nn as nn
from torchdiffeq import odeint
from model.sheaf_network import SheafIncidenceFast, SheafIncidenceFast_Message
import torch
import torch.nn as nn
from torchdiffeq import odeint
from model.sheaf_network import SheafIncidenceFast


######with takens embeddings######  
class SheafMLPODEFunc(nn.Module):

    def __init__(
        self,
        num_edges: int,
        F_node: int,
        hidden_dim_geom: int = 32,
        hidden_dim_mlp: int = 64,
        window_size: int = 20,
        takens_m: int = 3,
        takens_tau: int = 1,
        symmetric_messages: bool = True,
    ):
        super().__init__()

        self.F_signal = F_node
        self.window_size = window_size

        self.takens_m = takens_m
        self.takens_tau = takens_tau
        self.symmetric_messages = symmetric_messages

        self.F_takens = takens_m * self.F_signal
        self.F_stalk = (takens_m + 2) * self.F_signal

        self.geometry = SheafIncidenceFast_Message(
            num_edges=num_edges,
            F_node=self.F_stalk,
            hidden_dim=hidden_dim_geom
        )

        self.temporal_encoder = nn.LSTM(
            input_size=self.F_signal,
            hidden_size=self.F_signal,
            batch_first=True
        )

        # Drift MLP input dims:
        #   X_signal: F
        #   m_stalk:  (m+2)F
        #   context:  F
        #   takens:   mF
        # total = F + (m+2)F + F + mF = (2m + 4)F
        in_dim = (2 * takens_m + 4) * self.F_signal

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim_mlp),
            nn.Tanh(),
            nn.Linear(hidden_dim_mlp, self.F_signal)
        )

        self.edge_index = None
        self.X_history = None  # (W, N, F) or (B, W, N, F)

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def set_history(self, X_history):
        """
        X_history can be:
          - (W, N, F)  single example
          - (B, W, N, F) batch

        NOTE: odeint will call forward multiple times; we typically keep X_history fixed
        across the integration for each rollout window.
        """
        self.X_history = X_history.detach() if X_history is not None else None

    def _temporal_context(self, X_signal):
        if self.X_history is None:
            return torch.zeros_like(X_signal)

        X_hist = self.X_history

        if X_hist.dim() == 3:
            X_hist_proc = X_hist.permute(1, 0, 2)
            _, (h_n, _) = self.temporal_encoder(X_hist_proc)  # (1, N, F)
            return h_n.squeeze(0)                             # (N, F)

        elif X_hist.dim() == 4:
            X_hist_proc = X_hist.permute(0, 2, 1, 3)
            B, Nn, W, Ff = X_hist_proc.shape
            X_hist_reshape = X_hist_proc.reshape(B * Nn, W, Ff)
            _, (h_n, _) = self.temporal_encoder(X_hist_reshape)  # (1, B*N, F)
            return h_n.squeeze(0).view(B, Nn, Ff)                # (B, N, F)

        else:
            raise RuntimeError(f"Unsupported X_history dim: {X_hist.dim()}")

    def _takens_embedding(self):
        """
        Returns:
          - (N, mF) if X_history is (W, N, F)
          - (B, N, mF) if X_history is (B, W, N, F)

        Uses the last frame in the history as "current" time.
        """
        if self.X_history is None:
            raise RuntimeError("Takens embedding requires X_history to be set.")

        m, tau = self.takens_m, self.takens_tau
        X_hist = self.X_history

        W = X_hist.shape[0] if X_hist.dim() == 3 else X_hist.shape[1]
        needed = 1 + (m - 1) * tau
        if W < needed:
            raise RuntimeError(
                f"Not enough history for Takens embedding: need W >= {needed}, got W={W}. "
                f"Either increase window_size, or decrease takens_m/takens_tau."
            )

        idxs = [W - 1 - k * tau for k in reversed(range(m))]

        if X_hist.dim() == 3:
            slices = [X_hist[t] for t in idxs]
            return torch.cat(slices, dim=-1)

        elif X_hist.dim() == 4:
            slices = [X_hist[:, t] for t in idxs]
            return torch.cat(slices, dim=-1)

        else:
            raise RuntimeError(f"Unsupported X_history dim: {X_hist.dim()}")

    def _sheaf_messages(self, X_stalk):
        if self.edge_index is None:
            raise RuntimeError("edge_index not set")

        src, tgt = self.edge_index
        compat = self.geometry.compat(X_stalk, self.edge_index)  # (E, F_stalk)

        m = torch.zeros_like(X_stalk)
        m.index_add_(0, src, compat)

        if self.symmetric_messages:
            m.index_add_(0, tgt, -compat)

        return m

    def forward(self, t, X_signal):
        if self.edge_index is None:
            raise RuntimeError("edge_index not set")
        if self.X_history is None:
            raise RuntimeError("X_history not set (needed for LSTM + Takens).")

        context = self._temporal_context(X_signal)

        takens = self._takens_embedding()

        X_stalk = torch.cat([X_signal, context, takens], dim=-1)

        m_stalk = self._sheaf_messages(X_stalk)

        H = torch.cat([X_signal, m_stalk, context, takens], dim=-1)
        dX_signal = self.mlp(H)

        return dX_signal


def integrate_sheaf_ode(odefunc, X0, tspan, **odeint_kwargs):
    return odeint(odefunc, X0, tspan, **odeint_kwargs)
