import numpy as np
import matplotlib.pyplot as plt

def plot_results_with_theoretical_bound(
    results_df,
    epsilon_min,
    epsilon_max,
    T,
    B,
    sum_y,
    filepath,
    n_points=100,
    show_plot=False
):

    # --- Compute theoretical bound epsilon range ---
    epsilon_bound = np.logspace(
        np.log10(epsilon_min),
        np.log10(epsilon_max),
        n_points
    )

    # --- FIX: Use global B consistent with DP mechanism ---
    # B was previously per-appliance (vector). Convert to scalar:
    B_global = np.max(B)

    # Constant derived from formulation (matching implementation)
    C = (T * np.sqrt(8) * B_global) / (2 * sum_y)

    # Theoretical EACC bound
    eacc_bound = 1 - C / epsilon_bound

    # Optional (recommended): enforce valid range
    eacc_bound = np.clip(eacc_bound, 0, 1)

    # --- Compute experimental results ---
    grouped = results_df.groupby('epsilon').mean(numeric_only=True).reset_index()
    grouped = grouped.sort_values('epsilon')

    epsilon_exp = grouped['epsilon']
    eacc_exp = grouped['agg_EACC']

    # --- Plot ---
    plt.figure()

    plt.plot(
        epsilon_bound,
        eacc_bound,
        linestyle='--',
        color='black',
        label='Theoretical Bound'
    )

    plt.plot(
        epsilon_exp,
        eacc_exp,
        marker='o',
        label='Experimental EACC'
    )

    plt.xscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Energy Accuracy')
    plt.title('Experimental Accuracy vs Theoretical Bound')

    plt.grid(True)
    plt.legend()

    # --- Save plot ---
    plt.savefig(filepath, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()

    plt.close()

def plot_results_relative_accuracy(
    results_df,
    epsilon_min,
    epsilon_max,
    T,
    B,
    sum_y,
    filepath,
    n_points=100,
    show_plot=False
):

    # --- Compute theoretical bound epsilon range ---
    epsilon_bound = np.logspace(
        np.log10(epsilon_min),
        np.log10(epsilon_max),
        n_points
    )

    # --- FIX: Use global B consistent with DP mechanism ---
    # B was previously per-appliance (vector). Convert to scalar:
    B_global = np.max(B)

    # Constant derived from formulation (matching implementation)
    C = (T * np.sqrt(8) * B_global) / (2 * sum_y)

    # Theoretical EACC bound
    eacc_bound = 1 - C / epsilon_bound

    # Optional (recommended): enforce valid range
    eacc_bound = np.clip(eacc_bound, 0, 1)

    # --- Compute experimental results ---
    grouped = results_df.groupby('epsilon').mean(numeric_only=True).reset_index()
    grouped = grouped.sort_values('epsilon')

    epsilon_exp = grouped['epsilon']
    eacc_exp = grouped['agg_EACC']

    # --- Plot ---
    plt.figure()

    plt.plot(
        epsilon_bound,
        eacc_bound,
        linestyle='--',
        color='black',
        label='Theoretical Bound'
    )

    plt.plot(
        epsilon_exp,
        eacc_exp,
        marker='o',
        label='Experimental EACC'
    )

    plt.xscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Energy Accuracy')
    plt.title('Experimental Accuracy vs Theoretical Bound')

    plt.grid(True)
    plt.legend()

    # --- Save plot ---
    plt.savefig(filepath, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()

    plt.close()

