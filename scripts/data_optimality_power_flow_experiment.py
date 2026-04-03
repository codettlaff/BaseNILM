import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.loadData import load_data, process_data
from differential_privacy import differential_privacy
from power_flow import RadialNetwork


# ============================================================
# PARAMETERS
# ============================================================
EXPERIMENT_NAME = "data_optimality_power_flow"
EPSILON_VALUES = [50, 75, 100, 150, 200, 300, 500, 1000]
N_NODES = 6
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
def build_radial_network(n_nodes):
    nodes = list(range(n_nodes))
    edges = []

    for i in range(n_nodes - 1):
        r = 0.01
        x = 0.01
        edges.append((i, i + 1, r, x))

    return RadialNetwork(nodes, edges, root=ROOT)


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
def compute_accuracy_metrics(V_true, V_noisy, I_true, I_noisy):
    """
    Computes:
        Acc_V = 1 - sum |e_v| / (2 sum |V|)
        Acc_l = 1 - sum |e_l| / (2 sum |I|)
    """

    # --- Voltage ---
    num_v = 0.0
    den_v = 0.0

    for i in V_true:
        e_v = V_noisy[i] - V_true[i]
        num_v += abs(e_v)
        den_v += abs(V_true[i])

    acc_v = 1 - num_v / (2 * den_v)

    # --- Line current ---
    num_l = 0.0
    den_l = 0.0

    for edge in I_true:
        e_l = I_noisy[edge] - I_true[edge]
        num_l += abs(e_l)
        den_l += abs(I_true[edge])

    acc_l = 1 - num_l / (2 * den_l)

    return acc_v, acc_l


# ============================================================
# THEORETICAL ACCURACY (FROM PAPER)
# ============================================================
def compute_acc_theory(network, V_true_time, I_true_time, B_t, epsilon):
    """
    Implements paper formulas:

    E|e| = (4 / (ε√π)) * sqrt( sum B_h^2 )
    """

    T = len(V_true_time)

    num_v = 0.0
    den_v = 0.0

    num_l = 0.0
    den_l = 0.0

    for t in range(T):

        V_t = V_true_time[t]
        I_t = I_true_time[t]
        B_sq = B_t[t] ** 2

        # ---------------------------
        # Voltage term
        # ---------------------------
        for i in V_t:

            # Path from root to node i
            path_edges = []
            node = i
            while node != network.root:
                parent = network.parent[node]
                path_edges.append((parent, node))
                node = parent

            # sum over path
            sum_Bh_sq = len(path_edges) * B_sq

            expected_ev = (4 / (epsilon * np.sqrt(np.pi))) * np.sqrt(sum_Bh_sq)

            num_v += expected_ev
            den_v += abs(V_t[i])

        # ---------------------------
        # Line current term
        # ---------------------------
        for (i, j), I_ij in I_t.items():

            subtree_nodes = network.D(j)
            sum_Bh_sq = len(subtree_nodes) * B_sq

            expected_el = (4 / (epsilon * np.sqrt(np.pi))) * np.sqrt(sum_Bh_sq)

            num_l += expected_el
            den_l += abs(I_ij)

    acc_v_th = 1 - num_v / (2 * den_v)
    acc_l_th = 1 - num_l / (2 * den_l)

    return acc_v_th, acc_l_th


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(p_agg, p_apps):

    T = p_agg.shape[0]

    # --- Network ---
    network = build_radial_network(N_NODES)

    # --- Assign loads ---
    p_nodes = {i: p_agg.copy() for i in network.nodes}

    # --- Sensitivity ---
    B_t = compute_B_t(p_apps)

    results = []

    for epsilon in EPSILON_VALUES:

        print(f"\nRunning epsilon = {epsilon}")

        # Apply DP
        p_nodes_tilde = apply_dp_per_node(p_nodes, p_apps, epsilon)

        acc_v_list = []
        acc_l_list = []

        V_true_time = []
        I_true_time = []

        # ---------------------------
        # TIME LOOP
        # ---------------------------
        for t in tqdm(range(T), desc=f"Epsilon Value {epsilon}: Timestep Loop"):

            # True + noisy loads
            p_true = build_p_dict(p_nodes, t)
            p_tilde = build_p_dict(p_nodes_tilde, t)

            # Power flow
            V_true, I_true = network.compute_voltage_and_current(p_true)
            V_noisy, I_noisy = network.compute_voltage_and_current(p_tilde)

            # Store for theory
            V_true_time.append(V_true)
            I_true_time.append(I_true)

            # Empirical accuracy
            acc_v, acc_l = compute_accuracy_metrics(
                V_true, V_noisy, I_true, I_noisy
            )

            acc_v_list.append(acc_v)
            acc_l_list.append(acc_l)

        # --- Empirical averages ---
        acc_v_emp = np.mean(acc_v_list)
        acc_l_emp = np.mean(acc_l_list)

        # --- Theoretical ---
        acc_v_th, acc_l_th = compute_acc_theory(
            network,
            V_true_time,
            I_true_time,
            B_t,
            epsilon
        )

        results.append({
            "epsilon": epsilon,
            "Acc_V_emp": acc_v_emp,
            "Acc_V_theory": acc_v_th,
            "Acc_l_emp": acc_l_emp,
            "Acc_l_theory": acc_l_th
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    paths = get_paths()
    results_folder = paths["experiment_results"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    redd_files = get_redd_files(paths["redd"])
    data_file_name = os.path.basename(redd_files[0]).replace(".mat", "")
    data = process_data(load_data(redd_files[0]), "redd")

    p_agg = data['Y']  # aggregate load
    p_apps = data['X']  # needed for DP mechanism

    results = run_experiment(p_agg, p_apps)