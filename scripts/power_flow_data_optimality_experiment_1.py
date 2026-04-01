import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from differential_privacy import differential_privacy

from power_flow import run_power_flow_example

# -------------------------------
# Experiment Parameters
# -------------------------------
EPSILON_VALUES = [10, 25, 50, 75, 100, 200, 500, 1000]
EXPERIMENT_NAME = "power_flow_data_optimality"

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

        p_agg = data['Y'].flatten()   # ensure shape (T,)
        p_apps = data['X']

        # ---------------------------
        # Baseline Power Flow
        # ---------------------------
        V_true, P_true = run_power_flow_example(p_agg)

        results = []

        for epsilon in EPSILON_VALUES:

            # Apply DP
            p_private = differential_privacy(p_agg, p_apps, epsilon)

            # Ensure correct shape
            p_private = p_private.flatten()

            # Run power flow on noisy signal
            V_noisy, P_noisy = run_power_flow_example(p_private)

            v_error = rmse(V_true, V_noisy)
            p_error = rmse(P_true, P_noisy)

            results.append({
                "epsilon": epsilon,
                "voltage_rmse": v_error,
                "powerflow_rmse": p_error
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

        plt.plot(df["epsilon"], df["voltage_rmse"], marker='o', label="Voltage RMSE")
        plt.plot(df["epsilon"], df["powerflow_rmse"], marker='s', label="Branch Flow RMSE")

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