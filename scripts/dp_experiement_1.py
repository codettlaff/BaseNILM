from data.loadData import load_data, process_data, split_data, window_data, unwindow_data
from model.testMdlPM import testMdlPM, evaluate_prediction, energy_accuracy
from model.trainMdlPM import trainMdlPM
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

base_path = os.path.join(os.path.dirname(__file__),'..')
data_path = os.path.join(base_path, 'data')
results_path = os.path.join(base_path, 'results')
mdl_path = os.path.join(base_path, 'mdl')

redd_path = os.path.join(data_path, 'redd')
redd_filepath_list = [os.path.join(redd_path, f) for f in os.listdir(redd_path) if f.endswith('.mat')]

epsilon_values = [0,1,10,100,1000,10000,100000]

n_folds = 5
window_length = 25
stride = 10
feature_selection = ['all']

# Run Experiment for Each REDD Filepath
for redd_filepath in redd_filepath_list:

    # Prepare Data
    data = load_data(redd_filepath)
    data = process_data(data, 'redd')

    # Run Experiment for Each Fold
    for i in range(n_folds):
        data_split = split_data(data, 'k-fold', kfold=n_folds, fold=i)
        data_split_windowed = copy_key_structure(data_split)

        # Create Template Database
        mdl_name = f'REDD_PM_fold{i}.npz'
        mdl_filepath = os.path.join(mdl_path, mdl_name)
        data_split_windowed['Train']['X'], data_split_windowed['Train']['Y'] = window_data(data_split['Train']['X'], data_split['Train']['Y'], window_length, stride=stride)
        mdl = trainMdlPM(data_split_windowed['Train'], mdl_filepath, return_mdl=True)

    for epsilon in epsilon_values:

            # Apply Differential Privacy
            data_split['Test']['Y_private'] = differential_privacy(data_split['Test']['Y'], data_split['Test']['X'], epsilon)
            data_split['Test']['X_private'] = data_split['Test']['X']
            data_split_windowed['Test']['X'], data_split_windowed['Test']['Y'] = window_data(data_split['Test']['X'], data_split['Test']['Y'], window_length, stride=stride)
            data_split_windowed['Test']['X_private'], data_split_windowed['Test']['Y_private'] = window_data(data_split['Test']['X_private'], data_split['Test']['Y_private'], window_length, stride=stride)

            # Predict Appliance Profiles
            Y_pred_windowed = testMdlPM(data_split_windowed['Test']['X_private'], data_split_windowed['Test']['Y_private'], mdl, feature_selection)


# Create Template Database