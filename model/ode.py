import torch
import torch.nn as nn
from torchdiffeq import odeint

from model.sheaf_network import SheafGeometry


class SheafMLPODEFunc(nn.Module):
    """
    New version: dX/dt = MLP([X, LX])

    Only uses first Laplacian power.
    Much faster and simpler.
    """

    def __init__(self, num_edges, F_node,
                 hidden_dim_geom=32,
                 hidden_dim_mlp=64):
        super().__init__()

        # geometry for computing LX
        self.geom = SheafGeometry(
            num_edges=num_edges,
            F_node=F_node,
            hidden_dim=hidden_dim_geom
        )

        self.F_node = F_node

        # Input size is now 2*F_node (X and LX)
        self.mlp = nn.Sequential(
            nn.Linear(2 * F_node, hidden_dim_mlp),
            nn.Tanh(),
            nn.Linear(hidden_dim_mlp, F_node)
        )

        self.edge_index = None

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def forward(self, t, X):
        if self.edge_index is None:
            raise RuntimeError("edge_index not set")

        # Compute LX only
        LX = self.geom.incidence.laplacian_apply(X, self.edge_index)

        # Concatenate [X, LX]
        H = torch.cat([X, LX], dim=-1)   # (N, 2F)

        # dX/dt
        return self.mlp(H)


def integrate_sheaf_ode(odefunc, X0, tspan, **odeint_kwargs):
    return odeint(odefunc, X0, tspan, **odeint_kwargs)