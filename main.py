import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.ode import SheafMLPODEFunc, integrate_sheaf_ode
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------

def load_data():
    signals = np.load("data/full_state.npy", mmap_mode="r")   # (T, N, 3)
    adjacency = np.load("data/adjacency.npy")              # (N, N)
    return signals, adjacency


def build_edge_index(adjacency, device):
    adj = torch.tensor(adjacency, dtype=torch.float32, device=device)
    edge_index = torch.nonzero(adj, as_tuple=False).t()   # (2, E)
    return edge_index


# ----------------------------------------------------------
# 2. MAIN SCRIPT
# ----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    # -----------------
    # Load data
    # -----------------
    signals_np, adjacency_np = load_data()     # signals: (T, N, 3)
    T, N, F = signals_np.shape
    print(f"[main] Loaded signals: T={T}, N={N}")

    # Convert to torch
    signals = torch.tensor(signals_np, dtype=torch.float32, device=device)  # (T, N, F)

    # -----------------
    # Build edge index
    # -----------------
    edge_index = build_edge_index(adjacency_np, device)   # (2, E)
    E = edge_index.shape[1]
    print(f"[main] Edges: E={E}")

    # -----------------
    # Construct ODE function
    # -----------------
    F_node = F                 # Kuramoto-like 1D features
    hidden_geom = 32           # size of restriction-map MLPs
    hidden_mlp = 64            # MLP used in the ODE drift

    odefunc = SheafMLPODEFunc(
        num_edges=E,
        F_node=F_node,
        hidden_dim_geom=hidden_geom,
        hidden_dim_mlp=hidden_mlp
    ).to(device)

    odefunc.set_edge_index(edge_index)

    # -----------------
    # Optimizer
    # -----------------
    optimizer = optim.Adam(odefunc.parameters(), lr=1e-3)

    # -----------------
    # Training loop: match X(t+1) using ODE integration from X(t)
    # -----------------
    num_iters = 5
    print_every = 20

    # Time step used by data (assume Δt = 1)
    tspan = torch.tensor([0., 0.01], device=device)
    print("[main] Starting training...")
    for it in tqdm(range(num_iters)):

        # Randomly sample a time index t0 → predict t1
        idx0 = torch.randint(0, T-1, (1,)).item()

        X0 = signals[idx0]      # (N, 1)
        X1_true = signals[idx0+1]

        # Roll out the ODE for 1 unit of time
        X_pred = integrate_sheaf_ode(
            odefunc, X0, tspan, rtol=1e-4, atol=1e-4
        )[-1]   # take final state at t=1

        # Loss = MSE
        loss = ((X_pred - X1_true)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % print_every == 0:
            print(f"[iter {it:04d}] loss = {loss.item():.6f}")

    print("[main] Training done.")

    # -----------------
    # Final rollout from initial time
    # -----------------
    # tspan_full = torch.linspace(0., float(T-1), T, device=device)
    # X0_full = signals[0]  # initial state

    # X_t = integrate_sheaf_ode(
    #     odefunc, X0_full, tspan_full, rtol=1e-4, atol=1e-4
    # )   # (T, N, 1)

    # print("[main] Rollout done.")
    # print("X_t shape:", X_t.shape)

    # try:
    #     torch.save(X_t, "predicted_rollout.pt")
    # except Exception as e:
    #     print(f"[main] Failed to save rollout: {e}")

    # # Convert predicted rollout to CPU numpy
    # X_pred_np = X_t.squeeze(-1).detach().cpu().numpy()   # (T, N)
    # X_true_np = signals.squeeze(-1).detach().cpu().numpy()

    # # Plot a few representative nodes
    # num_plot = 6
    # nodes_to_plot = [0, 1, 2, 3, 4, 5]

    # plt.figure(figsize=(12, 6))
    # for i, node in enumerate(nodes_to_plot):
    #     plt.subplot(num_plot, 1, i+1)
    #     plt.plot(X_true_np[:, node], label="True", color="black", linewidth=1.0)
    #     plt.plot(X_pred_np[:, node], label="Predicted", color="red", alpha=0.7)
    #     plt.ylabel(f"Node {node}")
    #     if i == 0:
    #         plt.title("Predicted Sheaf-ODE Rollout vs Ground Truth")
    #     if i == num_plot - 1:
    #         plt.xlabel("Time step")

    # plt.tight_layout()
    # plt.savefig("rollout_vs_truth.png", dpi=200)
    # plt.close()
    # print("[main] Saved overlay plot: rollout_vs_truth.png")

if __name__ == "__main__":
    main()