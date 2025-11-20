import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# =======================================================
# Multi-Rhythm Kuramoto with Strong Desync & Sync Cycles
# =======================================================

np.random.seed(42)

# -------------------------------------------------------
# Graph: small-world with modular structure
# -------------------------------------------------------
N = 100
G = nx.watts_strogatz_graph(N, k=8, p=0.15)
A = nx.to_numpy_array(G)

# -------------------------------------------------------
# Simulation length
# -------------------------------------------------------
steps = 500      # edit this 
dt = 0.1
T = steps * dt

# -------------------------------------------------------
# 1. MUCH STRONGER FREQUENCY DIVERSITY
# -------------------------------------------------------
# Frequencies in Hz-like units:
# Slow    = 0.3–0.7
# Medium  = 1–2.5
# Fast    = 4–7  (gamma-like)
#
# Much wider band separation + added jitter
n1, n2, n3 = 30, 40, 30
omega = np.zeros(N)

omega[:n1]          = np.random.uniform(0.3, 0.7, n1)
omega[n1:n1+n2]     = np.random.uniform(1.0, 2.5, n2)
omega[n1+n2:]       = np.random.uniform(4.0, 7.0, n3)

# Add jitter per node so no two frequencies match
omega += np.random.normal(0, 0.1, N)

# -------------------------------------------------------
# 2. Strongly time-varying coupling (sync → desync → sync)
# -------------------------------------------------------
# Pattern:
# K(t) = low @ start (fully desync)
#        ramps up to high (global sync)
#        drops again (partial desync)
#        increases again (late sync)
#
# This oscillatory modulation produces metastable bands.

K_low = 0.2     # almost no coupling
K_high = 8.0    # strong global coupling
cycles = 3      # number of sync/desync cycles over 150s
K_t_series = K_low + (K_high - K_low) * (
    0.5 * (1 + np.sin(2 * np.pi * cycles * np.arange(steps) / steps))
)

# -------------------------------------------------------
# 3. Amplitude with much larger variation (0.2–3.5)
# -------------------------------------------------------
amp = np.random.uniform(0.5, 1.5, N)   # random starting amplitudes
amp_mean = 1.5
amp_tau = 15.0
amp_sigma = 0.15                      # stronger amplitude variation
amp_min, amp_max = 0.2, 3.5

# -------------------------------------------------------
# 4. Initial phases = full randomization (desync start)
# -------------------------------------------------------
theta = np.random.uniform(0, 2*np.pi, N)

# -------------------------------------------------------
# 5. Higher-order nonlinear coupling
# -------------------------------------------------------
nonlinear_coeff = 0.5   # stronger harmonic effect

theta_record = np.zeros((steps, N))
amp_record = np.zeros((steps, N))

# =======================================================
# Simulation Loop
# =======================================================
for t in range(steps):

    # Time-varying coupling value
    K_t = K_t_series[t]

    # Phase differences
    phase_diff = theta[:, None] - theta[None, :]

    # Nonlinear coupling: sin(Δθ) + 0.5 sin(2Δθ)
    coupling = np.sum(
        A * (np.sin(-phase_diff) + nonlinear_coeff * np.sin(-2 * phase_diff)),
        axis=1
    )

    # Phase update
    dtheta = omega + (K_t / N) * coupling
    theta = (theta + dtheta * dt) % (2 * np.pi)
    theta_record[t] = theta

    # Amplitude update
    amp += (-(amp - amp_mean) / amp_tau) * dt + amp_sigma*np.sqrt(dt)*np.random.randn(N)
    amp = np.clip(amp, amp_min, amp_max)
    amp_record[t] = amp

# -------------------------------------------------------
# Observable signals - what we'll predict with RiTINI
# -------------------------------------------------------
signals = amp_record * np.sin(theta_record)  # (steps, N)

# Save all data
full_state = np.concatenate([
    amp_record[..., None],              # (steps, N, 1)
    np.sin(theta_record)[..., None],   # (steps, N, 1)
    np.cos(theta_record)[..., None],   # (steps, N, 1)
], axis=-1)                             # (steps, N, 3)

np.save("full_state.npy", full_state.astype(np.float32))
np.save("adjacency.npy", A.astype(np.float32))
np.save("signals.npy", signals.astype(np.float32))
print("Saved full_state.npy, adjacency.npy, and signals.npy")

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

time = np.linspace(0, T, steps)

# Plot 1: Observable signals (amplitude * sin(phase))
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(time, signals[:, i], label=f"Node {i}", alpha=0.9, linewidth=1.5)

plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Observable Signal (amp × sin(θ))", fontsize=12)
plt.title("Kuramoto Observable Signals - What RiTINI Will Predict", fontsize=14)
plt.legend(ncol=3, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

save_path = plots_dir / "kuramoto_observable_signals.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved plot: {save_path}")

# Plot 2: Amplitude over time
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(time, amp_record[:, i], label=f"Node {i}", alpha=0.9, linewidth=1.5)

plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.title("Kuramoto Amplitudes Over Time", fontsize=14)
plt.legend(ncol=3, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

save_path = plots_dir / "kuramoto_amplitudes.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved plot: {save_path}")

# Plot 3: Phase synchronization over time
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(time, theta_record[:, i], label=f"Node {i}", alpha=0.9, linewidth=1.5)

plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Phase (radians)", fontsize=12)
plt.title("Kuramoto Phases Over Time", fontsize=14)
plt.legend(ncol=3, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

save_path = plots_dir / "kuramoto_phases.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved plot: {save_path}")

print("\nAll plots saved in plots/ directory")