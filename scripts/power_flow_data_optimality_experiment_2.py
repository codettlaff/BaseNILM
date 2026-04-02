import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from power_flow import run_power_flow_example

# -------------------------------
# Experiment Parameters
# -------------------------------
EPSILON_VALUES = [50, 75, 80, 90, 95, 100, 150, 200, 250, 500, 750, 1000]
EXPERIMENT_NAME = "power_flow_data_optimality_2"

# -------------------------------
# Path Utilities
# -------------------------------
def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')

    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "experiment": os.path.join(base, "results", EXPERIMENT_NAME),
        "redd": os.path.join(base, "data", "redd")
    }

def get_redd_files(redd_path):
    return [
        os.path.join(redd_path, f)
        for f in os.listdir(redd_path)
        if f.endswith(".mat") and "HF" not in f
    ]

# -------------------------------
# Error Metrics
# -------------------------------
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

# -------------------------------
# NEW: Node-level DP mechanism
# -------------------------------
def differential_privacy_per_node(p_apps, epsilon):
    """
    Apply independent Laplace noise to each appliance (node).

    p_apps : (T, N) array
    epsilon : float
    """
    B = np.max(p_apps)
    delta_f = 2 * B
    scale = delta_f / epsilon

    noise = np.random.laplace(loc=0, scale=scale, size=p_apps.shape)

    return p_apps + noise

# -------------------------------
# Theoretical Bounds (NOW MATCHING THEORY)
# -------------------------------
def theoretical_power_rmse(epsilon, B, D_size, N):
    return (np.sqrt(8) / epsilon) * B * np.sqrt(D_size * N)

def theoretical_voltage_rmse(epsilon, B, beta_sum_sq, N):
    return (np.sqrt(8) / epsilon) * B * np.sqrt(beta_sum_sq * N)

# -------------------------------
# Main Experiment
# -------------------------------
def run_experiment():

    paths = get_paths()
    if not os.path.exists(paths["experiment"]):
        os.makedirs(paths["experiment"])

    redd_files = get_redd_files(paths["redd"])

    for redd_file in redd_files:

        print(f"Processing: {redd_file}")

        data = process_data(load_data(redd_file), "redd")

        p_apps = data['X']  # (T, N)
        N = p_apps.shape[1]

        B = np.max(p_apps)

        # ---------------------------
        # Baseline Power Flow (TRUE node-level)
        # ---------------------------
        V_true, P_true = run_power_flow_example(p_apps)

        # ---------------------------
        # Network constants (approximate)
        # ---------------------------
        beta_01 = 2 * (0.01 + 0.02)
        beta_12 = 2 * (0.01 + 0.02)

        beta_sum_sq = beta_01**2 + beta_12**2

        # worst-case downstream size (chain feeder)
        D_size = N

        results = []

        for epsilon in EPSILON_VALUES:

            # ---------------------------
            # Apply DP per node
            # ---------------------------
            p_apps_private = differential_privacy_per_node(p_apps, epsilon)

            # ---------------------------
            # Power Flow
            # ---------------------------
            V_noisy, P_noisy = run_power_flow_example(p_apps_private)

            v_error = rmse(V_true, V_noisy)
            p_error = rmse(P_true, P_noisy)

            # ---------------------------
            # Theoretical bounds (UPDATED)
            # ---------------------------
            v_theory = (np.sqrt(8) / epsilon) * B * np.sqrt(N)
            p_theory = (np.sqrt(8) / epsilon) * B * np.sqrt(N)

            results.append({
                "epsilon": epsilon,
                "voltage_rmse": v_error,
                "powerflow_rmse": p_error,
                "voltage_theory": v_theory,
                "powerflow_theory": p_theory
            })

        df = pd.DataFrame(results)

        # ---------------------------
        # Save Results
        # ---------------------------
        name = os.path.basename(redd_file).replace(".mat", "")

        csv_path = os.path.join(paths["experiment"], f"{name}_pf_results.csv")
        df.to_csv(csv_path, index=False)

        # ---------------------------
        # Plot
        # ---------------------------
        plt.figure()

        plt.plot(df["epsilon"], df["voltage_rmse"], marker='o', label="Voltage RMSE (empirical)")
        plt.plot(df["epsilon"], df["powerflow_rmse"], marker='s', label="Branch Flow RMSE (empirical)")

        plt.plot(df["epsilon"], df["voltage_theory"], linestyle='--', label="Voltage RMSE (theory)")
        plt.plot(df["epsilon"], df["powerflow_theory"], linestyle='--', label="Branch Flow RMSE (theory)")

        plt.xscale("log")
        plt.xlabel("Epsilon")
        plt.ylabel("RMSE")
        plt.title(f"Power Flow Error vs Privacy ({name})")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(paths["experiment"], f"{name}_pf_plot.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved: {name}")

# -------------------------------
if __name__ == "__main__":
    run_experiment()