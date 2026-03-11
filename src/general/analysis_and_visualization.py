import numpy as np
import matplotlib.pyplot as plt

def plot_theoretical_eacc_bound(epsilon_min, epsilon_max, T, B, sum_y, n_points=100):

    # generate epsilon values on a log scale
    epsilon = np.logspace(np.log10(epsilon_min), np.log10(epsilon_max), n_points)

    B_sum = np.sum(B)

    # constant factor from bound
    C = (T * np.sqrt(8) * B_sum) / (2 * sum_y)

    # compute bound
    eacc_bound = 1 - C / epsilon

    plt.figure()
    plt.plot(epsilon, eacc_bound, linestyle='--', label='Theoretical Bound')

    plt.xscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Energy Accuracy Upper Bound')
    plt.title('Theoretical Accuracy Bound vs Epsilon')
    plt.grid(True)
    plt.legend()

    plt.show()

def plot_accuracy_versus_epsilon(results_dfs):

    plt.figure()

    for i, results_df in enumerate(results_dfs):

        # group by epsilon and average metrics across folds
        grouped = results_df.groupby('epsilon').mean(numeric_only=True).reset_index()

        # sort by epsilon
        grouped = grouped.sort_values('epsilon')

        epsilon = grouped['epsilon']
        eacc = grouped['agg_EACC']

        # plot each house
        plt.plot(epsilon, eacc, marker='o', label=f'House {i+1}')

    plt.xscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Energy Accuracy')
    plt.title('Accuracy vs Epsilon')
    plt.legend()
    plt.grid(True)

    plt.show()

