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
EPSILON_VALUES = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# EPSILON_VALUES = [75, 100, 500, 1000] # For Testing
NETWORK_NAME = "network"
N_NODES = 6
V0 = 12e3
ROOT = 0
T_set = 10 # Limit Timesteps

ALPHA = 0.0
R = 0.01
X = 0.01

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

def trim_data(data, T_new):
    data_trimmed = {}
    data_trimmed['Y'] = data['Y'][:T_new]
    data_trimmed['X'] = data['X'][:T_new]
    return data_trimmed

# ============================================================
# NETWORK CONSTRUCTION
# ============================================================
def build_radial_network(n_nodes, P, B):
    """
    Build a simple radial (chain) network.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    V0 : float
        Root voltage
    P : 2D array-like
        Active power injections, shape (n_nodes, T)
    B : 2D array-like
        Appliance bounds, shape (n_nodes, T)
    root : int
        Root node index
    alpha : float
        Power factor constant (Q = alpha P)
    epsilon : float or None
        Differential privacy parameter
    """

    # -----------------------------
    # Nodes (match required format)
    # -----------------------------
    nodes = {
        i: {
            "P": list(P[i]),
            "B": list(B[i])
        }
        for i in range(n_nodes)
    }

    # -----------------------------
    # Radial chain edges
    # -----------------------------
    edges = []
    for i in range(n_nodes - 1):
        r = R
        x = X
        edges.append((i, i + 1, r, x))

    # -----------------------------
    # Build network
    # -----------------------------
    return RadialNetwork(
        name=NETWORK_NAME,
        nodes=nodes,
        edges=edges,
        V0=V0
    )

def setup():
    paths = get_paths()
    results_folder = paths["experiment_results"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    redd_files = get_redd_files(paths["redd"])
    data = process_data(load_data(redd_files[0]), "redd")

    if T_set: data = trim_data(data, T_set)

    p_agg = data['Y']
    p_apps = data['X']
    T = p_agg.shape[0]

    P = np.tile(p_agg, (N_NODES, 1))

    B_i = np.max(p_apps, axis=1)
    B =  np.tile(B_i, (N_NODES, 1))

    return P, B, T


if __name__ == "__main__":

    P, B, T = setup()
    network = build_radial_network(N_NODES, P, B)
    network.power_flow()

    network.power_flow_results(t=2,display_results=True)

    network.do_differential_privacy = True
    network.epsilon = 0.1
    network.differential_privacy()
    network.noisy_power_flow()

    network.compute_theoretical_accuracy()
    network.compute_empirical_accuracy()

    acc_p_th = network.acc_p_th_exp
    acc_p_emp = network.acc_p

    print('')


