import os
import numpy as np
import pandas as pd

from data.loadData import load_data, process_data
from differential_privacy import differential_privacy
from data_optimality import (
    empirical_mean_analysis,
    empirical_energy_analysis,
    empirical_peak_analysis
)

# -----------------------------
# Experiment Parameters
# -----------------------------
EPSILON_VALUES = [50, 75, 80, 90, 95, 100, 150, 200, 250, 500, 750, 1000]
NUM_SAMPLES = 5
# NUM_SAMPLES = 1000   # Monte Carlo samples
EXPERIMENT_NAME = "data_optimality_experiment_1"


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


# -----------------------------
# Core Experiment
# -----------------------------
def run_experiment():

    paths = get_paths()

    results_folder = paths["experiment_results"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    redd_files = get_redd_files(paths["redd"])

    for redd_filepath in redd_files:

        data_file_name = os.path.basename(redd_filepath).replace(".mat", "")
        print(f"Running: {data_file_name}")

        # Load + process data
        data = process_data(load_data(redd_filepath), "redd")

        y = data['Y']  # aggregate load
        X = data['X']  # needed for DP mechanism

        results = []

        for epsilon in EPSILON_VALUES:

            # -----------------------------
            # Generate noisy samples (N, T)
            # -----------------------------
            y_tilde_samples = []

            for _ in range(NUM_SAMPLES):
                y_private = differential_privacy(y, X, epsilon)
                y_tilde_samples.append(y_private)

            y_tilde_samples = np.array(y_tilde_samples)

            # -----------------------------
            # Run empirical analyses
            # -----------------------------
            mean_res = empirical_mean_analysis(y, y_tilde_samples)
            energy_res = empirical_energy_analysis(y, y_tilde_samples)
            peak_res = empirical_peak_analysis(y, y_tilde_samples)

            # -----------------------------
            # Store results
            # -----------------------------
            results.append({
                "epsilon": epsilon,

                # Mean
                "mean_rmse": mean_res["rmse"],
                "mean_var": mean_res["variance"],

                # Energy
                "energy_rmse": energy_res["rmse"],
                "energy_var": energy_res["variance"],

                # Peak
                "peak_rmse": peak_res["rmse"],
                "peak_var": peak_res["variance"],
                "peak_bias": peak_res["bias"],
            })

        # -----------------------------
        # Save results
        # -----------------------------
        results_df = pd.DataFrame(results)

        results_filepath = os.path.join(
            results_folder,
            f"{data_file_name}_data_optimality.csv"
        )

        results_df.to_csv(results_filepath, index=False)

        print(f"Saved: {results_filepath}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_experiment()