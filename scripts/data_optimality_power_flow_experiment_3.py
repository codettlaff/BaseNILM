import os
import numpy as np
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from power_flow_with_dss_refactored import RadialNetwork

# ============================================================
# PARAMETERS
# ============================================================
EXPERIMENT_NAME = "data_optimality_power_flow_2"
EPSILON_VALUES = np.linspace(50, 1000, 10)

NETWORK_NAME = "network"
N_NODES = 6
V0 = 12e2
T_SET = 100

ALPHA = 0.0
R = 0.01
X = 0.01

N_TRIALS = 10


# ============================================================
# PATHS
# ============================================================
def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')
    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "experiment": os.path.join(base, "results", EXPERIMENT_NAME),
        "redd": os.path.join(base, "data", "redd"),
    }


# ============================================================
# DATA
# ============================================================
def load_redd_houses(n_houses=6):
    paths = get_paths()

    files = [
        os.path.join(paths["redd"], f)
        for f in os.listdir(paths["redd"])
        if f.endswith(".mat") and "HF" not in f
    ]

    assert len(files) >= n_houses, "Not enough REDD houses"

    P_list = []
    B_list = []
    lengths = []

    # -----------------------------
    # Load all houses first
    # -----------------------------
    raw_data = []
    for k in range(n_houses):
        data = process_data(load_data(files[k]), "redd")

        if T_SET:
            data = {
                "Y": data["Y"][:T_SET],
                "X": data["X"][:, :T_SET],
            }

        raw_data.append(data)
        lengths.append(len(data["Y"]))

    # -----------------------------
    # Use common minimum length
    # -----------------------------
    T_min = min(lengths)

    for data in raw_data:
        Y = data["Y"][:T_min]
        X = data["X"][:, :T_min]

        P_list.append(Y)
        B_list.append(np.max(X, axis=0))

    # -----------------------------
    # Force same length at stack time
    # -----------------------------
    T_min = min(arr.shape[-1] for arr in P_list + B_list)

    P = np.vstack([arr[:T_min] for arr in P_list])
    B = np.vstack([arr[:T_min] for arr in B_list])

    return P, B


# ============================================================
# NETWORK
# ============================================================
def build_network(P):
    nodes = {
        i: {"P": list(P[i-1])}
        for i in range(1,N_NODES+1)
    }

    edges = [(i, i + 1, R, X) for i in range(N_NODES)]

    return RadialNetwork(
        name=NETWORK_NAME,
        nodes=nodes,
        edges=edges,
        V0=V0,
    )


# ============================================================
# PLOTTING
# ============================================================
def plot_accuracy(eps, th_bound, th_exp, emp, title, save_path=None):
    plt.figure()

    plt.plot(eps, th_bound, 'o-', label="Bound")
    plt.plot(eps, th_exp, 's-', label="Theoretical")
    plt.plot(eps, emp, '^-', label="Empirical")

    plt.xscale('log')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = title.replace(" ", "_") + ".png"
        plt.savefig(os.path.join(save_path, fname), dpi=300)

    plt.close()


def plot_error_vs_distance(network, show=False, save=False, save_path=None):
    exp_by_d, norm_by_d = {}, {}

    for t in range(network.T):
        for (i, j) in network.lines:

            d = len(network.C(j)) - 1

            exp_by_d.setdefault(d, [])
            norm_by_d.setdefault(d, [])

            # Theoretical Results
            if (i, j, t) in network.e_p_exp:
                exp_by_d[d].append(network.e_p_exp[(i, j, t)])

            # Empirical Results
            if (i, j, t) in network.e_p_norm:
                norm_by_d[d].append(network.e_p_norm[(i, j, t)])

    distances = sorted(exp_by_d.keys())

    exp_avg = [np.mean(exp_by_d[d]) for d in distances]
    norm_avg = [np.mean(norm_by_d[d]) for d in distances]

    plt.figure()
    plt.plot(distances, exp_avg, 'o-', label="Theoretical Error")
    plt.plot(distances, norm_avg, 's-', label="Empirical Error")
    plt.xlabel("Distance from Root")
    plt.ylabel("Error")
    plt.title("Error vs Distance")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # Save option
    # -----------------------------
    if save:
        if save_path is None:
            save_path = "."

        os.makedirs(save_path, exist_ok=True)

        filename = "error_vs_distance.png"
        filepath = os.path.join(save_path, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    # -----------------------------
    # Show / close
    # -----------------------------
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# CORE COMPUTATION
# ============================================================
def run_trials(network, epsilon):
    network.epsilon = epsilon

    acc_p = acc_i = acc_v = 0.0

    for _ in range(N_TRIALS):
        network.differential_privacy()
        network.noisy_power_flow()
        network.compute_empirical_accuracy()

        acc_p += network.acc_p
        acc_i += network.acc_i
        acc_v += network.acc_v

    return acc_p / N_TRIALS, acc_i / N_TRIALS, acc_v / N_TRIALS


# ============================================================
# EXPERIMENT
# ============================================================
def experiment():
    paths = get_paths()
    results_path = paths["experiment"]
    os.makedirs(results_path, exist_ok=True)

    # -----------------------------
    # Setup
    # -----------------------------
    P, B = load_redd_houses()
    network = build_network(P)


    # IEEE 123 Test
    ieee123_dss_filepath = os.path.join(os.path.dirname(__file__), 'ieee_123bus_1ph.dss')
    network.dss_filepath = ieee123_dss_filepath
    network.build_from_dss_timeseries()

    network.export_to_dss_timeseries_single_phase()
    network.build_from_dss_timeseries()

    network.dss_power_flow_step_by_step()
    print("Step By Step Results:\n")
    network.power_flow_results(display_results=True)

    network.dss_power_flow_timeseries()
    print("Time Series Results:\n")
    network.power_flow_results(display_results=True)

    # -----------------------------
    # Storage
    # -----------------------------
    results = {
        "p": {"th_bound": [], "th_exp": [], "emp": []},
        "i": {"th_bound": [], "th_exp": [], "emp": []},
        "v": {"th_bound": [], "th_exp": [], "emp": []},
    }

    # -----------------------------
    # Sweep epsilon
    # -----------------------------
    for eps in EPSILON_VALUES:
        emp_p, emp_i, emp_v = run_trials(network, eps)

        network.compute_theoretical_accuracy()

        results["p"]["th_bound"].append(network.acc_p_th_bound)
        results["p"]["th_exp"].append(network.acc_p_th_exp)
        results["p"]["emp"].append(emp_p)

        results["i"]["th_bound"].append(network.acc_i_th_bound)
        results["i"]["th_exp"].append(network.acc_i_th_exp)
        results["i"]["emp"].append(emp_i)

        results["v"]["th_bound"].append(network.acc_V_th_bound)
        results["v"]["th_exp"].append(network.acc_V_th_exp)
        results["v"]["emp"].append(emp_v)

    # -----------------------------
    # Plot accuracy curves
    # -----------------------------
    plot_accuracy(EPSILON_VALUES, **results["p"], title="Power Accuracy", save_path=results_path)
    plot_accuracy(EPSILON_VALUES, **results["i"], title="Current Accuracy", save_path=results_path)
    plot_accuracy(EPSILON_VALUES, **results["v"], title="Voltage Accuracy", save_path=results_path)

    # -----------------------------
    # Plot error vs distance
    # -----------------------------
    plot_error_vs_distance(network, show=True, save=True, save_path=results_path)

if __name__ == "__main__":

    experiment()