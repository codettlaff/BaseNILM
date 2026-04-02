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

# Aggregation factors (original data = 3 seconds)
AGG_FACTORS = [1, 5, 20, 100, 300]
# → 3s, 15s, 1min, 5min, 15min

EXPERIMENT_NAME = "power_flow_data_optimality_temporal_aggregation_experiment_1"

# -------------------------------
# Node-level DP
# -------------------------------
def differential_privacy_per_node(p_apps, epsilon):
    B = np.max(p_apps)
    delta_f = 2 * B
    scale = delta_f / epsilon

    noise = np.random.laplace(loc=0, scale=scale, size=p_apps.shape)
    return p_apps + noise

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
# Utilities
# -------------------------------
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def temporal_aggregate(data, agg_factor):
    data = np.asarray(data)
    n = len(data) // agg_factor * agg_factor
    data = data[:n]
    return data.reshape(-1, agg_factor).mean(axis=1)

def temporal_aggregate_matrix(X, agg_factor):
    return np.stack(
        [temporal_aggregate(X[:, i], agg_factor) for i in range(X.shape[1])],
        axis=1
    )

def format_resolution(agg_factor):
    seconds = agg_factor * 3
    if seconds < 60:
        return f"{seconds}s"
    else:
        return f"{seconds // 60}min"

# -------------------------------
# Main Experiment
# -------------------------------
def run_experiment():

    paths = get_paths()
    os.makedirs(paths["experiment"], exist_ok=True)

    redd_files = get_redd_files(paths["redd"])

    for redd_file in redd_files:

        print(f"\nProcessing: {redd_file}")

        data = process_data(load_data(redd_file), "redd")
        p_apps_full = data['X']  # (T, N)

        results = []

        for agg_factor in AGG_FACTORS:

            print(f"  Aggregation factor: {agg_factor}")

            # ---------------------------
            # Temporal aggregation (per node)
            # ---------------------------
            p_apps = temporal_aggregate_matrix(p_apps_full, agg_factor)

            # ---------------------------
            # Baseline power flow (NODE-LEVEL)
            # ---------------------------
            V_true, I_true = run_power_flow_example(p_apps)

            for epsilon in EPSILON_VALUES:

                # ---------------------------
                # DP per node
                # ---------------------------
                p_apps_private = differential_privacy_per_node(p_apps, epsilon)

                # ---------------------------
                # Power flow with noisy node loads
                # ---------------------------
                V_noisy, I_noisy = run_power_flow_example(p_apps_private)

                v_error = rmse(V_true, V_noisy)
                i_error = rmse(I_true, I_noisy)

                results.append({
                    "aggregation_factor": agg_factor,
                    "epsilon": epsilon,
                    "voltage_rmse": v_error,
                    "current_rmse": i_error
                })

        df = pd.DataFrame(results)

        # ---------------------------
        # Save CSV
        # ---------------------------
        name = os.path.basename(redd_file).replace(".mat", "")
        csv_path = os.path.join(paths["experiment"], f"{name}_results.csv")
        df.to_csv(csv_path, index=False)

        # ---------------------------
        # Plot 1: Voltage RMSE
        # ---------------------------
        plt.figure()

        for agg_factor in AGG_FACTORS:
            subset = df[df["aggregation_factor"] == agg_factor]
            label = format_resolution(agg_factor)

            plt.plot(
                subset["epsilon"],
                subset["voltage_rmse"],
                marker='o',
                label=label
            )

        plt.xscale("log")
        plt.xlabel("Epsilon")
        plt.ylabel("Voltage RMSE")
        plt.title(f"Voltage RMSE vs Privacy ({name})")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(paths["experiment"], f"{name}_voltage_plot.png"))
        plt.close()

        # ---------------------------
        # Plot 2: Current RMSE
        # ---------------------------
        plt.figure()

        for agg_factor in AGG_FACTORS:
            subset = df[df["aggregation_factor"] == agg_factor]
            label = format_resolution(agg_factor)

            plt.plot(
                subset["epsilon"],
                subset["current_rmse"],
                marker='s',
                label=label
            )

        plt.xscale("log")
        plt.xlabel("Epsilon")
        plt.ylabel("Current RMSE")
        plt.title(f"Current RMSE vs Privacy ({name})")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(paths["experiment"], f"{name}_current_plot.png"))
        plt.close()

        print(f"Saved results for: {name}")

# -------------------------------
if __name__ == "__main__":
    run_experiment()