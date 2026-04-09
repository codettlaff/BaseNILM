import os
import numpy as np
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from power_flow import RadialNetwork

# ============================================================
# PARAMETERS
# ============================================================
EXPERIMENT_NAME = "data_optimality_power_flow"
EPSILON_MIN = 50
EPSILON_MAX = 1000
N_EPSILON_VALUES = 10
EPSILON_VALUES = np.linspace(EPSILON_MIN, EPSILON_MAX, N_EPSILON_VALUES)
# EPSILON_VALUES = [75, 100, 500, 1000] # For Testing
NETWORK_NAME = "network"
N_NODES = 6
V0 = 12e3
ROOT = 0
T_set = 100 # Limit Timesteps

ALPHA = 0.0
R = 0.01
X = 0.01

N_TRIALS = 10 # Number of Times Noise is Sampled per Epsilon

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

def plot_accuracy_vs_epsilon(acc_th_bound, acc_th_exp, acc_emp,
                             plot_title, plot_logarithmic=False,
                             display_plot=False, save_plot=False, save_folderpath=None):

    plt.figure()

    plt.plot(EPSILON_VALUES, acc_th_bound, marker='o', label='Theoretical Bound')
    plt.plot(EPSILON_VALUES, acc_th_exp, marker='s', label='Theoretical Expected')
    plt.plot(EPSILON_VALUES, acc_emp, marker='^', label='Empirical')

    # ============================================================
    # LOG SCALE OPTION
    # ============================================================
    if plot_logarithmic:
        plt.xscale('log')

    plt.xlabel("Epsilon (Privacy Parameter)")
    plt.ylabel("Accuracy")
    plt.title(plot_title)

    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # ============================================================
    # SAVE PLOT
    # ============================================================
    if save_plot:
        if save_folderpath is None:
            save_folderpath = "."

        os.makedirs(save_folderpath, exist_ok=True)

        filename = plot_title.replace(" ", "_").replace("/", "_")
        if plot_logarithmic:
            filename += "_log"

        filepath = os.path.join(save_folderpath, f"{filename}.png")

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    if display_plot:
        plt.show()
    else:
        plt.close()

def experiment():

    P, B, T = setup()
    network = build_radial_network(N_NODES, P, B)
    network.power_flow()

    network.do_differential_privacy = True

    acc_p_th_bound = []
    acc_p_th_exp = []
    acc_p_emp = []

    acc_i_th_bound = []
    acc_i_th_exp = []
    acc_i_emp = []

    acc_v_th_bound = []
    acc_v_th_exp = []
    acc_v_emp = []

    for epsilon in EPSILON_VALUES:

        network.epsilon = epsilon

        acc_p_emp_epsilon = 0.0
        acc_i_emp_epsilon = 0.0
        acc_v_emp_epsilon = 0.0
        for n in range(N_TRIALS):
            network.differential_privacy()
            network.noisy_power_flow()
            network.compute_empirical_accuracy()
            acc_p_emp_epsilon += network.acc_p
            acc_i_emp_epsilon += network.acc_i
            acc_v_emp_epsilon += network.acc_v
        acc_p_emp_epsilon = acc_p_emp_epsilon / N_TRIALS
        acc_i_emp_epsilon = acc_i_emp_epsilon / N_TRIALS
        acc_v_emp_epsilon = acc_v_emp_epsilon / N_TRIALS

        network.compute_theoretical_accuracy()

        acc_p_th_bound.append(network.acc_p_th_bound)
        acc_p_th_exp.append(network.acc_p_th_exp)
        acc_p_emp.append(acc_p_emp_epsilon)

        acc_i_th_bound.append(network.acc_i_th_bound)
        acc_i_th_exp.append(network.acc_i_th_exp)
        acc_i_emp.append(acc_i_emp_epsilon)

        acc_v_th_bound.append(network.acc_V_th_bound)
        acc_v_th_exp.append(network.acc_V_th_exp)
        acc_v_emp.append(acc_v_emp_epsilon)

    results_folderpath = get_paths()["experiment_results"]
    if not os.path.exists(results_folderpath): os.makedirs(results_folderpath)

    plot_accuracy_vs_epsilon(acc_p_th_bound, acc_p_th_exp, acc_p_emp, "Power Flow Accuracy Versus Epsilon", plot_logarithmic=True, save_plot=True)
    plot_accuracy_vs_epsilon(acc_i_th_bound, acc_i_th_exp, acc_i_emp, "Current Flow Accuracy Versus Epsilon", plot_logarithmic=True, save_plot=True)
    plot_accuracy_vs_epsilon(acc_v_th_bound, acc_v_th_exp, acc_v_emp, "Node Voltage Accuracy Versus Epsilon", plot_logarithmic=True, save_plot=True)

    print('')

if __name__ == "__main__":

    experiment()


