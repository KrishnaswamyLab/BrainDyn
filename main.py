import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.ode import SheafMLPODEFunc, integrate_sheaf_ode
import matplotlib.pyplot as plt


def load_data():
    signals = np.load("data/full_state.npy", mmap_mode="r")   # (T, N, 3)
    adjacency = np.load("data/adjacency.npy")                 # (N, N)
    return signals, adjacency


def build_edge_index(adjacency, device):
    adj = torch.tensor(adjacency, dtype=torch.float32, device=device)
    edge_index = torch.nonzero(adj, as_tuple=False).t()       # (2, E)
    return edge_index


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    raw_signals_np, adjacency_np = load_data()     # raw_signals: (T, N, 3)

    # Build attenuated single-channel signal = feature0 * feature1
    attenuated = raw_signals_np[..., 0] * raw_signals_np[..., 1]  # (T, N)
    signals_np = attenuated[..., None]  # (T, N, 1)
    T, N, F = signals_np.shape
    print(f"[main] Loaded attenuated signals: T={T}, N={N}, F={F}")
    signals = torch.tensor(signals_np, dtype=torch.float32, device=device)

    edge_index = build_edge_index(adjacency_np, device)
    E = edge_index.shape[1]
    print(f"[main] Edges: E={E}")

    F_node = F
    hidden_geom = 32
    hidden_mlp = 64

    odefunc = SheafMLPODEFunc(
        num_edges=E,
        F_node=F_node,
        hidden_dim_geom=hidden_geom,
        hidden_dim_mlp=hidden_mlp
    ).to(device)

    odefunc.set_edge_index(edge_index)

    optimizer = optim.Adam(odefunc.parameters(), lr=1e-3)

    input_window = 20
    pred_horizon = 5
    num_epochs = 100
    print_every = 20

    dt = 1.0
    tspan = torch.linspace(0., pred_horizon * dt, pred_horizon + 1, device=device)

    print(f"[main] Starting sliding-window training (input={input_window}, pred={pred_horizon})")
    # Simple plot of the attenuated signal for a few nodes
    num_plot_nodes = min(6, N)
    nodes_to_plot = list(range(num_plot_nodes))

    plt.figure(figsize=(12, 8))
    for i, node in enumerate(nodes_to_plot):
        plt.subplot(num_plot_nodes, 1, i + 1)
        plt.plot(signals_np[:, node, 0], color='black')
        plt.ylabel(f'Node {node}')
        if i == 0:
            plt.title('Raw attenuated signal (feat0 * feat1) — all timesteps')
        if i == num_plot_nodes - 1:
            plt.xlabel('Timestep')

    plt.tight_layout()
    plt.savefig('signal_feat0x1.png', dpi=150)
    plt.close()
    print('[main] Saved plot: signal_feat0x1.png')
    losses = []
    sample_preds = []   # store full pred horizon
    sample_trues = []

    step = 0
    for epoch in range(num_epochs):
        for t in range(0, T - input_window - pred_horizon):

            X_window = signals[t : t + input_window]
            X0 = X_window[-1]
            X_target_seq = signals[t + input_window : t + input_window + pred_horizon]

            odefunc.set_history(X_window)

            X_pred_seq = integrate_sheaf_ode(
                odefunc, X0, tspan, rtol=1e-4, atol=1e-4
            )[1:]   # (pred_horizon, N, F)

            loss = ((X_pred_seq - X_target_seq) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # collect sample preds only in the final epoch to keep plots readable
            if epoch == num_epochs - 1:
                sample_preds.append(X_pred_seq.detach().cpu())     # (pred_horizon, N, F)
                sample_trues.append(X_target_seq.detach().cpu())   # (pred_horizon, N, F)
            if step % 100 == 0:
                print(f"[epoch {epoch} step {step}] loss = {loss.item():.6f}")

            step += 1

    print("[main] Sliding-window training done.")

  
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('Training step')
    plt.ylabel('MSE loss')
    plt.title('Training loss')
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.close()

    print('[main] Saved plot: training_loss.png')
   
    if len(sample_preds) > 0:
        pred_tensor = torch.stack(sample_preds, dim=0)   # (S, pred_horizon, N, F)
        true_tensor = torch.stack(sample_trues, dim=0)   # (S, pred_horizon, N, F)

        pred_tensor = torch.stack(sample_preds, dim=0)   # (S, pred_horizon, N, F)
        true_tensor = torch.stack(sample_trues, dim=0)   # (S, pred_horizon, N, F)

        # Extract single-channel attenuated values
        pred_vals = pred_tensor[..., 0].numpy()   # (S, pred_horizon, N)
        true_vals = true_tensor[..., 0].numpy()   # (S, pred_horizon, N)
        num_plot_nodes = 6
        nodes_to_plot = list(range(min(num_plot_nodes, N)))

        plt.figure(figsize=(14, 10))
        for i, node in enumerate(nodes_to_plot):
            plt.subplot(num_plot_nodes, 1, i+1)
            for k in range(pred_horizon):
                plt.plot(true_vals[:, k, node], color='black', alpha=0.5)
                plt.plot(pred_vals[:, k, node], color='red', alpha=0.5)
            plt.ylabel(f'Node {node}')
            if i == 0:
                plt.title('Training: multistep (attenuated signal)')
            if i == num_plot_nodes - 1:
                plt.xlabel('Training sample index')

        plt.tight_layout()
        plt.savefig('train_multistep_attenuated.png', dpi=150)
        plt.close()

        print('[main] Saved plot: train_multistep_attenuated.png')




    #Rollout evaluation
    rollout_start = 0
    rollout_steps = max(0, T - (rollout_start + input_window))
    print(f"[main] Running autoregressive rollout: {rollout_steps} steps")

    window_frames = signals[rollout_start : rollout_start + input_window].clone().to(device)

    preds = []
    with torch.no_grad():
        steps_remaining = rollout_steps
        while steps_remaining > 0:
            steps_to_pred = min(pred_horizon, steps_remaining)
            tspan_roll = torch.linspace(0., steps_to_pred * dt, steps_to_pred + 1, device=device)

            odefunc.set_history(window_frames)
            X0 = window_frames[-1]

            X_pred_seq = integrate_sheaf_ode(
                odefunc, X0, tspan_roll, rtol=1e-4, atol=1e-4
            )[1:]

            preds.append(X_pred_seq)

            window_frames = torch.cat([window_frames[steps_to_pred:], X_pred_seq], dim=0)
            window_frames = window_frames[-input_window:]

            steps_remaining -= steps_to_pred

    X_pred_rollout = torch.cat(preds, dim=0)
    X_true_rollout = signals[rollout_start + input_window :
                             rollout_start + input_window + rollout_steps].to(device)

    # amplitude-attenuated signals
    # Only plot the first 400 timesteps to focus on short-term behavior
    plot_horizon = min(400, X_pred_rollout.shape[0])
    X_pred_np = X_pred_rollout[..., 0][:plot_horizon].cpu().numpy()
    X_true_np = X_true_rollout[..., 0][:plot_horizon].cpu().numpy()

    num_plot = 6
    nodes_to_plot = list(range(min(num_plot, N)))

    plt.figure(figsize=(12, 8))
    for i, node in enumerate(nodes_to_plot):
        plt.subplot(num_plot, 1, i+1)
        plt.plot(X_true_np[:, node], label='True rollout (attenuated)', color='black')
        plt.plot(X_pred_np[:, node], label='Pred rollout (attenuated)', color='red', alpha=0.7)
        plt.ylabel(f'Node {node}')
        if i == 0:
            plt.title(f'Autoregressive rollout (amplitude-attenuated) — first {plot_horizon} steps')
        if i == num_plot - 1:
            plt.xlabel('Rollout step')

    plt.tight_layout()
    plt.savefig('rollout_attenuated.png', dpi=150)
    plt.close()

    print('[main] Saved plot: rollout_attenuated.png')

    # (no additional feature-product plots — signals are already attenuated)


if __name__ == "__main__":
    main()
