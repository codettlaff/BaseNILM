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
N_NODES = 6
V0 = 12.47e3
ROOT = 0
T_set = 10 # Limit Timesteps

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
def build_radial_network(n_nodes, V0, P, B, root=0, alpha=1.0, epsilon=None):
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
        r = 0.01
        x = 0.01
        edges.append((i, i + 1, r, x))

    # -----------------------------
    # Build network
    # -----------------------------
    return RadialNetwork(
        nodes=nodes,
        edges=edges,
        root=root,
        V0=V0,
        alpha=alpha,
        epsilon=epsilon
    )