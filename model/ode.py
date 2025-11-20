import torch
import torch.nn as nn
from torchdiffeq import odeint
from model.sheaf_network import SheafIncidenceFast


class SheafMLPODEFunc(nn.Module):
    """
    Temporal Sheaf ODE:
        dX/dt = MLP([X, L*X, temporal_context(X_history)])

    - X_history is a tensor of shape (window_size, N, F)
    - temporal_context aggregates past window information per node
    """

    def __init__(self, num_edges, F_node,
                 hidden_dim_geom=32,
                 hidden_dim_mlp=64,
                 window_size=20):
        super().__init__()

        self.geometry = SheafIncidenceFast(num_edges, F_node, hidden_dim=hidden_dim_geom)
        self.F_node = F_node
        self.window_size = window_size

        # Temporal encoder that compresses (window_size, F) → (F)
        # Use LSTM for better long-term context retention
        self.temporal_encoder = nn.LSTM(
            input_size=F_node,
            hidden_size=F_node,
            batch_first=True
        )

        # Main drift MLP (takes X, L*X, and context)
        self.mlp = nn.Sequential(
            nn.Linear(3 * F_node, hidden_dim_mlp),
            nn.Tanh(),
            nn.Linear(hidden_dim_mlp, F_node)
        )

        self.edge_index = None
        self.X_history = None  # (window_size, N, F) to be set externally

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def set_history(self, X_history):
        """
        Set the full time window before integration.
        X_history: (window_size, N, F)
        """
        self.X_history = X_history.detach() if X_history is not None else None

    def forward(self, t, X):
        if self.edge_index is None:
            raise RuntimeError("edge_index not set")

        # Laplacian
        LX = self.geometry.laplacian_apply(X, self.edge_index)

        # Temporal context per node
        if self.X_history is not None:
            # Pass the entire window (T_in, N, F) through LSTM per node
            # Treat each node as an independent batch element
            X_hist = self.X_history.permute(1, 0, 2)  # (N, window_size, F)
            _, (h_n, c_n) = self.temporal_encoder(X_hist)  # h_n: (1, N, F)
            context = h_n.squeeze(0)                       # (N, F)
        else:
            context = torch.zeros_like(X)

        # Concatenate [X, L*X, context]
        H = torch.cat([X, LX, context], dim=-1)
        dX = self.mlp(H)
        return dX


def integrate_sheaf_ode(odefunc, X0, tspan, **odeint_kwargs):
    return odeint(odefunc, X0, tspan, **odeint_kwargs)
