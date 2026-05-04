import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

    # ============================================================
    # IEEE FORMATTING
    # ============================================================
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # ============================================================
    # THEORETICAL BOUND
    # ============================================================
    epsilon_bound = np.logspace(
        np.log10(epsilon_min),
        np.log10(epsilon_max),
        n_points
    )

    B_global = np.max(B)
    C = (T * np.sqrt(8) * B_global) / (2 * sum_y)

    eacc_bound = 1 - C / epsilon_bound
    eacc_bound = np.clip(eacc_bound, 0, 1)

    # ============================================================
    # EXPERIMENTAL DATA
    # ============================================================
    grouped = results_df.groupby('epsilon').mean(numeric_only=True).reset_index()
    grouped = grouped.sort_values('epsilon')

    epsilon_exp = grouped['epsilon']
    eacc_exp = grouped['agg_EACC']

    # ============================================================
    # FIGURE (~4:3, single column)
    # ============================================================
    fig = plt.figure(figsize=(3.5, 2.6))

    # ============================================================
    # PLOT
    # ============================================================
    plt.plot(
        epsilon_bound,
        eacc_bound,
        linestyle='--',
        linewidth=1.5,
        color='black',
        label='Theoretical Bound'
    )

    plt.plot(
        epsilon_exp,
        eacc_exp,
        marker='o',
        linewidth=1.5,
        label='Experimental EACC'
    )

    plt.xscale('log')

    plt.xlabel('Epsilon (Privacy Parameter)')
    plt.ylabel('Energy Accuracy')

    # No title (IEEE convention)
    plt.legend(frameon=False, loc='best')

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    # ============================================================
    # SAVE (vector preferred)
    # ============================================================
    plt.savefig(filepath, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

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

