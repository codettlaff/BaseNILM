import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_rmse_with_theoretical_bound(df, y, save_path=None):
    """
    Plots RMSE vs epsilon for mean, energy, and peak
    with theoretical bounds.

    results_filepath: path to CSV produced by experiment
    y: true aggregate signal (needed for T and B)
    """

    eps = df["epsilon"].values

    # -----------------------------
    # Extract empirical RMSE
    # -----------------------------
    mean_rmse = df["mean_rmse"].values
    energy_rmse = df["energy_rmse"].values
    peak_rmse = df["peak_rmse"].values

    # -----------------------------
    # Problem parameters
    # -----------------------------
    T = len(y)
    B = np.max(y)

    # -----------------------------
    # Theoretical bounds
    # -----------------------------
    # From your derivations:
    # Mean: sqrt(8B / (epsilon^2 T))
    mean_theory = np.sqrt(8 * B / (eps**2 * T))

    # Energy: sqrt(8B T / epsilon^2)
    energy_theory = np.sqrt(8 * B * T / (eps**2))

    # Peak: ~ sqrt(8B log(T)) / epsilon
    peak_theory = np.sqrt(8 * B * np.log(T)) / eps

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 4))

    # ---- Mean ----
    plt.subplot(1, 3, 1)
    plt.plot(eps, mean_rmse, 'o-', label="Empirical")
    plt.plot(eps, mean_theory, '--', label="Theory")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("RMSE")
    plt.title("Mean Load")
    plt.legend()
    plt.grid(True)

    # ---- Energy ----
    plt.subplot(1, 3, 2)
    plt.plot(eps, energy_rmse, 'o-', label="Empirical")
    plt.plot(eps, energy_theory, '--', label="Theory")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epsilon")
    plt.title("Daily Energy")
    plt.legend()
    plt.grid(True)

    # ---- Peak ----
    plt.subplot(1, 3, 3)
    plt.plot(eps, peak_rmse, 'o-', label="Empirical")
    plt.plot(eps, peak_theory, '--', label="Theory")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epsilon")
    plt.title("Peak Load")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # -----------------------------
    # Save / show
    # -----------------------------
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

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
                "mean_true": mean_res["true_mean"],
                "mean_rmse": mean_res["rmse"],
                "mean_cvrmse": mean_res["cvrmse"],
                "mean_var": mean_res["variance"],

                # Energy
                "energy_true": energy_res["true_energy"],
                "energy_rmse": energy_res["rmse"],
                "energy_cvrmse": energy_res["cvrmse"],
                "energy_var": energy_res["variance"],

                # Peak
                "peak_true": peak_res["true_peak"],
                "peak_rmse": peak_res["rmse"],
                "peak_cvrmse": peak_res["cvrmse"],
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
        # Plot results with theoretical bounds
        # -----------------------------
        plot_filepath = os.path.join(
            results_folder,
            f"{data_file_name}_rmse_plot.png"
        )

        plot_rmse_with_theoretical_bound(
            results_df,
            y,
            save_path=plot_filepath
        )

        print(f"Plotted: {plot_filepath}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_experiment()