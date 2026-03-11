import pandas as pd

from data.loadData import load_data, process_data, split_data, window_data, unwindow_data
from model.testMdlPM import testMdlPM, evaluate_prediction, energy_accuracy, get_results
from model.trainMdlPM import trainMdlPM
from general.analysis_and_visualization import plot_accuracy_versus_epsilon, plot_theoretical_eacc_bound
from differential_privacy import differential_privacy
import os
import numpy as np
import scipy.io

def copy_key_structure(d, fill_value=None):
    """
    Recursively copy dictionary key structure,
    replacing all leaf values with fill_value.
    """
    if isinstance(d, dict):
        return {k: copy_key_structure(v, fill_value) for k, v in d.items()}
    else:
        return fill_value

def run_experiment_1():

    base_path = os.path.join(os.path.dirname(__file__),'..')
    data_path = os.path.join(base_path, 'data')
    results_path = os.path.join(base_path, 'results')
    mdl_path = os.path.join(base_path, 'mdl')

    redd_path = os.path.join(data_path, 'redd')
    redd_filepath_list = [os.path.join(redd_path, f) for f in os.listdir(redd_path) if f.endswith('.mat')]

    epsilon_values = [0.01,0.1,1,10,100,1000,10000,100000]

    n_folds = 5
    window_length = 25
    stride = 10
    feature_selection = ['all']

    # Run Experiment for Each REDD Filepath
    for redd_filepath in redd_filepath_list:

        data_file_name = os.path.basename(redd_filepath).replace('.mat','')
        results_filepath = os.path.join(results_path, data_file_name+'_results.csv')
        noisy_profiles_filepath = os.path.join(results_path, data_file_name+'_noisy_profiles.csv')
        results = pd.DataFrame()
        noisy_profiles = pd.DataFrame()

        # Prepare Data
        data = load_data(redd_filepath)
        data = process_data(data, 'redd')

        # Run Experiment for Each Fold
        for i in range(n_folds):
            data_split = split_data(data, 'k-fold', kfold=n_folds, fold=i+1)
            data_split_windowed = copy_key_structure(data_split)

            # Create Template Database
            mdl_name = f'{data_file_name}_PM_fold{i+1}.npz'
            mdl_filepath = os.path.join(mdl_path, mdl_name)
            data_split_windowed['Train']['X'], data_split_windowed['Train']['Y'] = window_data(data_split['Train']['X'], data_split['Train']['Y'], window_length, stride=stride)
            mdl = trainMdlPM(data_split_windowed['Train'], mdl_filepath, return_mdl=True)

            noisy_profiles_i = pd.DataFrame()
            noisy_profiles_i['fold'] = i + 1

            for epsilon in epsilon_values:

                # Apply Differential Privacy
                data_split['Test']['Y_private'] = differential_privacy(data_split['Test']['Y'], data_split['Test']['X'], epsilon)
                data_split['Test']['X_private'] = data_split['Test']['X']
                noisy_profiles_i[f'epsilon_{epsilon}'] = data_split['Test']['Y_private']
                data_split_windowed['Test']['X'], data_split_windowed['Test']['Y'] = window_data(data_split['Test']['X'], data_split['Test']['Y'], window_length, stride=stride)
                data_split_windowed['Test']['X_private'], data_split_windowed['Test']['Y_private'] = window_data(data_split['Test']['X_private'], data_split['Test']['Y_private'], window_length, stride=stride)

                # Predict Appliance Profiles
                X_pred_windowed = testMdlPM(data_split_windowed['Test']['X_private'], data_split_windowed['Test']['Y_private'], mdl, feature_selection)

                # Process Results
                X_pred = unwindow_data(X_pred_windowed, window_length, stride=stride)
                X_true = unwindow_data(data_split_windowed['Test']['Y'], window_length, stride=stride)

                results_dict = get_results(X_pred, X_true, data['X_labels'])
                results_dict['fold'] = i+1
                results_dict['epsilon'] = epsilon
                results = pd.concat([results, pd.DataFrame([results_dict])], ignore_index=True)

            noisy_profiles = pd.concat([noisy_profiles, noisy_profiles_i], ignore_index=True)

        results.to_csv(results_filepath, index=False)
        noisy_profiles.to_csv(noisy_profiles_filepath, index=False)
        print(f'Completed Experiment For {data_file_name}')

def plot_results():
    print('Plotting')
    base_path = os.path.join(os.path.dirname(__file__),'..')
    results_filepath_1 = os.path.join(base_path, 'results', 'redd1 _results.csv')
    results_filepath_2 = os.path.join(base_path, 'results', 'redd2 _results.csv')
    results_filepath_3 = os.path.join(base_path, 'results', 'redd3 _results.csv')
    results_df_1 = pd.read_csv(results_filepath_1)
    results_df_2 = pd.read_csv(results_filepath_2)
    results_df_3 = pd.read_csv(results_filepath_3)
    results_dfs = [results_df_1, results_df_2, results_df_3]
    plot_accuracy_versus_epsilon(results_dfs)
    print('Finished Plotting')

base_path = os.path.join(os.path.dirname(__file__),'..')
data_filepath_1 = os.path.join(base_path, 'data', 'redd', 'redd1.mat')
data_filepath_2 = os.path.join(base_path, 'data', 'redd', 'redd2.mat')
data_filepath_3 = os.path.join(base_path, 'data', 'redd', 'redd3.mat')
data1 = process_data(load_data(data_filepath_1), 'redd')
data2 = process_data(load_data(data_filepath_2), 'redd')
data3 = process_data(load_data(data_filepath_3), 'redd')

# T is the number of samples in data1['Y']
T = data1['Y'].shape[0]

# B is the maximum value in any column of data1['X']
B = data1['X'].max(axis=0)
B_max = data1['X'].max()

sum_y = np.sum(data1['Y'])

epsilon_min = 0.01
epsilon_max = 10000

epsilon_min = 25
epsilon_max = 600

plot_theoretical_eacc_bound(epsilon_min, epsilon_max, T, B_max, sum_y)