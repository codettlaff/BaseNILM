import pandas as pd
from data.loadData import load_data, process_data, split_data, window_data, unwindow_data
from pattern_matching import testMdlPM, trainMdlPM, evaluate_prediction, energy_accuracy, get_results
from general.analysis_and_visualization import plot_results_with_theoretical_bound
from differential_privacy import differential_privacy
import os
import numpy as np

# Experiment Parameters
EPSILON_VALUES = [50, 75, 80, 90, 95, 100, 150, 200, 250, 500, 750, 1000]
N_FOLDS = 5
WINDOW_LENGTH = 25
STRIDE = 10
FEATURE_SELECTION = ['all']

def copy_key_structure(d, fill_value=None):
    """
    Recursively copy dictionary key structure,
    replacing all leaf values with fill_value.
    """
    if isinstance(d, dict):
        return {k: copy_key_structure(v, fill_value) for k, v in d.items()}
    else:
        return fill_value

def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')

    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "mdl": os.path.join(base, "mdl"),
        "redd": os.path.join(base, "data", "redd")
    }

def get_redd_files(redd_path):
    return [
        os.path.join(redd_path, f)
        for f in os.listdir(redd_path)
        if f.endswith(".mat") and "HF" not in f
    ]

def run_fold(data, fold, mdl_path, data_file_name):

    data_split = split_data(data, 'k-fold', kfold=N_FOLDS, fold=fold)
    data_split_windowed = copy_key_structure(data_split)

    mdl_name = f"{data_file_name}_PM_fold{fold}.npz"
    mdl_filepath = os.path.join(mdl_path, mdl_name)

    X_train, Y_train = window_data(
        data_split['Train']['X'],
        data_split['Train']['Y'],
        WINDOW_LENGTH,
        stride=STRIDE
    )

    data_split_windowed['Train']['X'] = X_train
    data_split_windowed['Train']['Y'] = Y_train

    mdl = trainMdlPM(data_split_windowed['Train'], mdl_filepath, return_mdl=True)

    return mdl, data_split, data_split_windowed

def evaluate_epsilons(data_split, data_split_windowed, mdl, data, fold):

    results = []
    noisy_profiles = {}

    for epsilon in EPSILON_VALUES:

        data_split['Test']['Y_private'] = differential_privacy(
            data_split['Test']['Y'],
            data_split['Test']['X'],
            epsilon
        )

        data_split['Test']['X_private'] = data_split['Test']['X']

        noisy_profiles[f"epsilon_{epsilon}"] = data_split['Test']['Y_private']

        X_w, Y_w = window_data(
            data_split['Test']['X'],
            data_split['Test']['Y'],
            WINDOW_LENGTH,
            stride=STRIDE
        )

        Xp_w, Yp_w = window_data(
            data_split['Test']['X_private'],
            data_split['Test']['Y_private'],
            WINDOW_LENGTH,
            stride=STRIDE
        )

        X_pred_w = testMdlPM(Xp_w, Yp_w, mdl, FEATURE_SELECTION)

        X_pred = unwindow_data(X_pred_w, WINDOW_LENGTH, stride=STRIDE)
        X_true = unwindow_data(Y_w, WINDOW_LENGTH, stride=STRIDE)

        r = get_results(X_pred, X_true, data['X_labels'])
        r["fold"] = fold
        r["epsilon"] = epsilon

        results.append(r)

    return results, noisy_profiles

def run_experiment():

    paths = get_paths()
    redd_files = get_redd_files(paths["redd"])

    for redd_filepath in redd_files:

        data_file_name = os.path.basename(redd_filepath).replace(".mat","")

        results_filepath = os.path.join(paths["results"], f"{data_file_name}_results.csv")
        noisy_filepath = os.path.join(paths["results"], f"{data_file_name}_noisy_profiles.csv")

        data = process_data(load_data(redd_filepath), "redd")

        results_all = []
        noisy_all = []

        for fold in range(1, N_FOLDS + 1):

            mdl, data_split, data_split_windowed = run_fold(
                data, fold, paths["mdl"], data_file_name
            )

            results, noisy = evaluate_epsilons(
                data_split,
                data_split_windowed,
                mdl,
                data,
                fold
            )

            results_all.extend(results)
            noisy_all.append(pd.DataFrame(noisy))

        pd.DataFrame(results_all).to_csv(results_filepath, index=False)
        pd.concat(noisy_all).to_csv(noisy_filepath, index=False)

        print(f"Completed Experiment For {data_file_name}")

def plot_results_with_bound():

    paths = get_paths()
    redd_files = get_redd_files(paths["redd"])

    for redd_filepath in redd_files:

        data_file_name = os.path.basename(redd_filepath).replace(".mat","")

        results_filepath = os.path.join(
            paths["results"],
            f"{data_file_name}_results.csv"
        )

        noisy_profiles_filepath = os.path.join(
            paths["results"],
            f"{data_file_name}_noisy_profiles.csv"
        )

        # --- Load experiment results ---
        results_df = pd.read_csv(results_filepath)

        # Load noisy profiles (optional)
        noisy_profiles_df = pd.read_csv(noisy_profiles_filepath)

        # Filter EACC range
        results_df = results_df[
            (results_df['agg_EACC'] >= 0) &
            (results_df['agg_EACC'] <= 1)
        ]

        # --- Load original data for THIS house ---
        data = process_data(load_data(redd_filepath), "redd")

        T = data['Y'].shape[0]
        B = data['X'].max(axis=0)
        sum_y = np.sum(data['Y'])

        # epsilon range for plotting
        epsilon_min = results_df['epsilon'].min()
        epsilon_max = results_df['epsilon'].max()

        # --- Plot ---
        plot_filepath = os.path.join(
            paths["results"],
            f"{data_file_name}_eacc_plot.png"
        )

        plot_results_with_theoretical_bound(
            results_df,
            epsilon_min,
            epsilon_max,
            T,
            B,
            sum_y,
            plot_filepath,
            show_plot=False
        )

if __name__ == "__main__":

    run_experiment()
    plot_results_with_bound()