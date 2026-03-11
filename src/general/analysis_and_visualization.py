import numpy as np
import matplotlib.pyplot as plt

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