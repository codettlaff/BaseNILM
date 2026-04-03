import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from differential_privacy import differential_privacy
from power_flow import RadialNetwork


# ============================================================
# PARAMETERS
# ============================================================
EXPERIMENT_NAME = "data_optimality_power_flow"
EPSILON_VALUES = [50, 75, 80, 90, 95, 100, 150, 200, 250, 500, 750, 1000]
EPSILON_VALUES = [75, 100, 500, 1000] # For Testing
N_NODES = 6
V0 = 12.47e3
ROOT = 0

# -----------------------------
# Path Utilities (reuse style)
# -----------------------------
def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')

    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "experiment_results": os.path.join(base, "results", EXPERIMENT_NAME),
        "redd": os.path.join(base, "data", "redd")
    }

def get_redd_files(redd_path):
    return [
        os.path.join(redd_path, f)
        for f in os.listdir(redd_path)
        if f.endswith(".mat") and "HF" not in f
    ]

# ============================================================
# NETWORK CONSTRUCTION
# ============================================================
def build_radial_network(n_nodes, V0):
    nodes = list(range(n_nodes))
    edges = []

    for i in range(n_nodes - 1):
        r = 0.01
        x = 0.01
        edges.append((i, i + 1, r, x))

    return RadialNetwork(nodes, edges, root=ROOT, V0=V0)


# ============================================================
# SENSITIVITY: B_t = max appliance power at time t
# ============================================================
def compute_B_t(p_apps):
    return np.max(p_apps, axis=1)  # shape (T,)


# ============================================================
# APPLY DP:  p̃_i(t) = p_i(t) + η_i(t)
# ============================================================
def apply_dp_per_node(p_nodes, p_apps, epsilon):
    p_nodes_tilde = {}

    for i, signal in p_nodes.items():
        p_nodes_tilde[i] = differential_privacy(
            signal,
            p_apps,
            epsilon
        )

    return p_nodes_tilde


# ============================================================
# BUILD p_i(t) DICTIONARY
# ============================================================
def build_p_dict(p_nodes, t):
    return {i: p_nodes[i][t] for i in p_nodes}


# ============================================================
# ACCURACY METRICS (EMPIRICAL)
# ============================================================
# ============================================================
# ACCURACY METRICS (EMPIRICAL, TIME-AGGREGATED)
# ============================================================
def compute_accuracy_metrics(V_true_time, V_noisy_time, I_true_time, I_noisy_time):
    """
    Computes variance-based accuracy metrics over all time steps:

        Acc_V^(var) = 1 - (sum_{t,i} e_{v,i,t}^2) / (2 sum_{t,i} V_{i,t}^2)
        Acc_I^(var) = 1 - (sum_{t,ℓ} e_{ℓ,t}^2) / (2 sum_{t,ℓ} I_{ℓ,t}^2)

    Parameters
    ----------
    V_true_time, V_noisy_time : list of dicts
        Each element is V_{·,t}, indexed by node i
    I_true_time, I_noisy_time : list of dicts
        Each element is I_{·,t}, indexed by edge ℓ

    Returns
    -------
    acc_v, acc_l : floats
        Voltage and current accuracy
    """

    num_v = 0.0
    den_v = 0.0
    num_l = 0.0
    den_l = 0.0

    T = len(V_true_time)

    for t in range(T):

        V_true = V_true_time[t]
        V_noisy = V_noisy_time[t]

        I_true = I_true_time[t]
        I_noisy = I_noisy_time[t]

        # --- Voltage ---
        for i in V_true:
            e_v = V_noisy[i] - V_true[i]
            num_v += e_v**2
            den_v += V_true[i]**2

        # --- Line current ---
        for edge in I_true:
            e_l = I_noisy[edge] - I_true[edge]
            num_l += e_l**2
            den_l += I_true[edge]**2

    acc_v = 1 - num_v / (2 * den_v)
    acc_l = 1 - num_l / (2 * den_l)

    return acc_v, acc_l


# ============================================================
# THEORETICAL ACCURACY (FROM PAPER)
# ============================================================
def compute_v_acc_theory(V, D, B, epsilon):
    """
    Compute theoretical voltage accuracy Acc_V^(var).

    Parameters
    ----------
    V : array-like of shape (T, N)
        Voltage magnitudes V_{i,t}
    D : dict
        D[i] = set/list of nodes h in D(i) (downstream of node i)
    B : array-like of shape (N,)
        Appliance bounds B_h
    epsilon : float
        Privacy parameter ε

    Returns
    -------
    acc_v : float
        Theoretical voltage accuracy
    """
    import numpy as np

    V = np.array(V)
    T, N = V.shape

    # Denominator: sum_{t=1}^T sum_{i=1}^N V_{i,t}^2
    denom = np.sum(V**2)

    # Numerator: sum_{i=1}^N sum_{h in D(i)} B_h^2
    num_inner = 0.0
    for i in range(N):
        for h in D(i):
            num_inner += B[h]**2

    # Full expression: (4T / ε^2) * (num_inner / denom)
    acc_v = 1 - (4 * T / epsilon**2) * (num_inner / denom)

    return acc_v

def compute_i_acc_theory(I, V, P, D, C, beta, B, epsilon, edges):
    """
    Compute theoretical current accuracy Acc_I^(var).

    Parameters
    ----------
    I : array-like of shape (T, L)
        Current magnitudes I_{ij,t} for each line ℓ ≡ (i,j)
    V : array-like of shape (N,)
        Voltage magnitudes V_j (assumed time-invariant here)
    P : array-like of shape (L,)
        Line real power flows P_{ij}
    D : dict
        D[i] = set/list of nodes h in D(i)
    C : dict
        C[j] = set/list of nodes k in C(j)
    beta : 2D array-like of shape (N, N)
        β_{kj}
    B : array-like of shape (N,)
        Appliance bounds B_h
    epsilon : float
        Privacy parameter ε
    edges : list of tuples
        edges[ℓ] = (i, j)

    Returns
    -------
    acc_i : float
        Theoretical current accuracy
    """
    import numpy as np

    I = np.array(I)
    V = np.array(V)
    P = np.array(P)

    T, L = I.shape

    # Denominator: sum_{t=1}^T sum_{ℓ=1}^L I_{ij,t}^2
    denom = np.sum(I**2)

    # Numerator inner sum: sum_{ℓ=1}^L sum_{h ∈ D(i)} B_h^2 ( ... )^2
    num_inner = 0.0

    for ell in range(L):
        i, j = edges[ell]

        # Compute (1/V_j + (P_ij / V_j^2) * sum_{k ∈ C(j)} β_kj)
        beta_sum = sum(beta[k][j] for k in C[j])
        factor = (1 / V[j]) + (P[ell] / (V[j]**2)) * beta_sum

        # Sum over h ∈ D(i)
        for h in D(i):
            num_inner += B[h]**2 * (factor**2)

    # Full expression
    acc_i = 1 - (4 * T / epsilon**2) * (num_inner / denom)

    return acc_i

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment():

    paths = get_paths()
    results_folder = paths["experiment_results"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    redd_files = get_redd_files(paths["redd"])
    data = process_data(load_data(redd_files[0]), "redd")

    p_agg = data['Y']  # aggregate load
    p_apps = data['X']  # needed for DP mechanism

    T = p_agg.shape[0]

    # --- Network ---
    network = build_radial_network(N_NODES, V0)

    # --- Assign loads ---
    p_nodes = {i: p_agg.copy() for i in network.nodes}

    # --- Sensitivity ---
    B_t = compute_B_t(p_apps)   # B_h

    results = []

    for epsilon in EPSILON_VALUES:

        print(f"\nRunning epsilon = {epsilon}")

        # Apply DP
        p_nodes_tilde = apply_dp_per_node(p_nodes, p_apps, epsilon)

        # --- Accumulators for variance-based empirical metric ---
        num_v = 0.0
        den_v = 0.0
        num_l = 0.0
        den_l = 0.0

        # Store full trajectories
        V_i_true_time = []
        I_ij_true_time = []
        P_ij_true_time = []
        V_i_noisy_time = []
        P_ij_noisy_time = []
        I_ij_noisy_time = []

        # ---------------------------
        # TIME LOOP
        # ---------------------------
        for t in tqdm(range(T), desc="Computing Empirical Accuracy"):

            # True + noisy loads
            p_true = build_p_dict(p_nodes, t)
            p_tilde = build_p_dict(p_nodes_tilde, t)

            # Power flow
            V_i_true, P_ij_true, I_ij_true = network.solve_power_flow(p_true)
            V_i_noisy, P_ij_noisy, I_ij_noisy = network.solve_power_flow(p_tilde)

            V_i_true_time.append(V_i_true)
            P_ij_true_time.append(P_ij_true)
            I_ij_true_time.append(I_ij_true)
            V_i_noisy_time.append(V_i_noisy)
            P_ij_noisy_time.append(P_ij_noisy)
            I_ij_noisy_time.append(I_ij_noisy)

        # --- Empirical accuracy (variance-based) ---
        acc_v_emp, acc_i_emp = compute_accuracy_metrics(V_i_true_time, V_i_noisy_time, I_ij_true_time, I_ij_noisy_time)

        # --- Theoretical accuracy ---
        acc_v_th = compute_v_acc_theory(
            np.array([list(V.values()) for V in V_i_true_time]),
            network.D,
            B_t,
            epsilon
        )

        acc_i_th = compute_i_acc_theory(
            np.array([list(I.values()) for I in I_ij_true_time]),
            V_i_true_time,
            P_ij_true_time,
            network.D,
            network.C,
            network.beta,
            B_t,
            epsilon,
            network.edges
        )

        results.append({
            "epsilon": epsilon,
            "Acc_V_emp": acc_v_emp,
            "Acc_V_theory": acc_v_th,
            "Acc_l_emp": acc_i_emp,
            "Acc_l_theory": acc_i_th
        })

    results_filepath = os.path.join(results_folder, "results.csv")
    pd.DataFrame(results).to_csv(results_filepath, index=False)

def plot_results(show=False):
    """
    Reads results CSV and plots:

        1) Acc_V vs epsilon
        2) Acc_l vs epsilon

    Saves plots to experiment_results folder.
    """

    paths = get_paths()
    results_folder = paths["experiment_results"]
    results_filepath = os.path.join(results_folder, 'results.csv')

    # Ensure folder exists
    os.makedirs(results_folder, exist_ok=True)

    # ---------------------------
    # Load results
    # ---------------------------
    df = pd.read_csv(results_filepath)
    epsilon = df["epsilon"]

    # ---------------------------
    # Plot: Voltage Accuracy
    # ---------------------------
    plt.figure()

    plt.plot(epsilon, df["Acc_V_emp"], marker='o', label="Empirical")
    plt.plot(epsilon, df["Acc_V_theory"], linestyle='--', label="Theoretical")

    plt.xlabel(r"Privacy Budget $\epsilon$")
    plt.ylabel(r"$\mathrm{Acc}_V$")
    plt.title(r"Voltage Accuracy vs $\epsilon$")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Save figure
    save_path_v = os.path.join(results_folder, "acc_v_vs_epsilon.png")
    plt.savefig(save_path_v, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    # ---------------------------
    # Plot: Line Accuracy
    # ---------------------------
    plt.figure()

    plt.plot(epsilon, df["Acc_l_emp"], marker='o', label="Empirical")
    plt.plot(epsilon, df["Acc_l_theory"], linestyle='--', label="Theoretical")

    plt.xlabel(r"Privacy Budget $\epsilon$")
    plt.ylabel(r"$\mathrm{Acc}_\ell$")
    plt.title(r"Line Current Accuracy vs $\epsilon$")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Save figure
    save_path_l = os.path.join(results_folder, "acc_l_vs_epsilon.png")
    plt.savefig(save_path_l, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":

    run_experiment()
    plot_results(show=True)