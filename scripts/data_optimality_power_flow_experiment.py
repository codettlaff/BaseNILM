import numpy as np
import pandas as pd
import os

from data.loadData import load_data, process_data
from differential_privacy import differential_privacy
from power_flow import RadialNetwork

from tqdm import tqdm # Progress bar for loops.

# -------------------------------
# Experiment Parameters
# -------------------------------
EXPERIMENT_NAME = "data_optimality_power_flow"
EPSILON_VALUES = [50, 75, 100, 150, 200, 300, 500, 1000]
N_NODES = 6   # small radial feeder
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

# -------------------------------
# Helper: Build Simple Radial Network
# -------------------------------
def build_radial_network(n_nodes):
    """
    Creates a simple chain:
    0 -> 1 -> 2 -> ... -> N
    """
    nodes = list(range(n_nodes))
    edges = []

    for i in range(n_nodes - 1):
        r = 0.01
        x = 0.01
        edges.append((i, i + 1, r, x))

    return RadialNetwork(nodes, edges, root=ROOT)


# -------------------------------
# Compute B_t (time-varying sensitivity)
# -------------------------------
def compute_B_t(p_apps):
    """
    B_t = max appliance power at each timestep
    """
    return np.max(p_apps, axis=1)  # shape (T,)


# -------------------------------
# Apply DP independently per node
# -------------------------------
def apply_dp_per_node(p_nodes, p_apps, epsilon):
    """
    p_nodes: dict {node: time series}
    """
    noisy = {}

    for node, signal in p_nodes.items():
        noisy[node] = differential_privacy(
            signal,
            p_apps,
            epsilon
        )

    return noisy


# -------------------------------
# Convert time-series → dict format
# -------------------------------
def build_power_dict(p_nodes, t):
    return {node: p_nodes[node][t] for node in p_nodes}

def compute_acc_l_theory(P_true_time, network, B_t, epsilon):
    """
    Compute theoretical branch flow accuracy Acc_l^theory.

    Parameters
    ----------
    P_true_time : list of dicts
        Each entry corresponds to time t and contains:
        { (i,j): P_ij,t } branch flows

    network : RadialNetwork
        Your network object (used to get subtree D(j))

    B_t : array-like, shape (T,)
        Time-varying sensitivity (max appliance power per timestep)

    epsilon : float
        Differential privacy parameter

    Returns
    -------
    acc_l_theory : float
    """

    T = len(P_true_time)

    numerator = 0.0
    denominator = 0.0

    for t in range(T):

        P_t = P_true_time[t]
        B_sq = B_t[t]**2

        for (i, j), P_ij in P_t.items():

            # --- sum over subtree D(j) ---
            subtree_nodes = network.get_subtree_nodes(j)
            subtree_size = len(subtree_nodes)

            sum_Bh_sq = subtree_size * B_sq  # since using same B_t for all nodes at time t

            # theoretical expected absolute error
            expected_error = (4 / (epsilon * np.sqrt(np.pi))) * np.sqrt(sum_Bh_sq)

            numerator += expected_error
            denominator += abs(P_ij)

    acc_l_theory = 1 - numerator / (2 * denominator)

    return acc_l_theory


# -------------------------------
# Main Experiment
# -------------------------------
def run_experiment(redd_filepath):

    data = process_data(load_data(redd_filepath), "redd")

    p_agg = data['Y']           # aggregated signal (T,)
    p_apps = data['X']          # appliance signals (T, M)

    T = p_agg.shape[0]

    # Build network
    network = build_radial_network(N_NODES)

    # Assign identical load to each node
    p_nodes = {i: p_agg.copy() for i in network.nodes}

    # Sensitivity per timestep
    B_t = compute_B_t(p_apps)

    results = []

    for epsilon in EPSILON_VALUES:

        # Apply DP independently at each node
        p_nodes_private = apply_dp_per_node(p_nodes, p_apps, epsilon)

        voltage_errors = []
        flow_errors = []

        # --- Time simulation ---
        for t in tqdm(range(T), desc="Timestep Loop"):

            # True power
            p_true = build_power_dict(p_nodes, t)

            # Noisy power
            p_noisy = build_power_dict(p_nodes_private, t)

            # True PF
            V_true, P_true = network.compute_voltages(p_true)

            # Noisy PF
            V_noisy, P_noisy = network.compute_voltages(p_noisy)

            # Voltage error
            for node in network.nodes:
                voltage_errors.append(V_noisy[node] - V_true[node])

            # Branch flow error
            for edge in P_true:
                flow_errors.append(P_noisy[edge] - P_true[edge])

        # --- Empirical RMSE ---
        voltage_errors = np.array(voltage_errors)
        flow_errors = np.array(flow_errors)

        rmse_v = np.sqrt(np.mean(voltage_errors**2))
        rmse_p = np.sqrt(np.mean(flow_errors**2))

        # -------------------------------
        # THEORETICAL ERRORS (from paper)
        # -------------------------------

        # Var(η_t) = 8 B_t^2 / ε^2
        var_eta = 8 * (B_t**2) / (epsilon**2)

        # Voltage variance: sum over path
        beta_sq_sum = 0
        for (i, j), r in network.r.items():
            x = network.x[(i, j)]
            beta = 2 * (r + x)
            beta_sq_sum += beta**2

        var_v = np.mean(var_eta) * beta_sq_sum
        rmse_v_theory = np.sqrt(var_v)

        # Branch flow variance: sum over subtree
        subtree_sizes = []

        for node in network.nodes:
            subtree_sizes.append(len(network.get_subtree_nodes(node)))

        avg_subtree = np.mean(subtree_sizes)

        var_p = np.mean(var_eta) * avg_subtree
        rmse_p_theory = np.sqrt(var_p)

        results.append({
            "epsilon": epsilon,
            "rmse_voltage": rmse_v,
            "rmse_voltage_theory": rmse_v_theory,
            "rmse_flow": rmse_p,
            "rmse_flow_theory": rmse_p_theory
        })

        print(f"Done epsilon={epsilon}")

    return pd.DataFrame(results)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":

    paths = get_paths()
    redd_path = os.path.join(paths["data"], "redd", "redd1.mat")

    df = run_experiment(redd_path)

    save_path = os.path.join("experiment_results", "results.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved results to {save_path}")