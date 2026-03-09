from data.loadData import load_data, process_data, plot_data, trim_data, split_data, window_data, unwindow_data, data_table
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

basePath = os.path.dirname(__file__)

ampds_path = os.path.join(basePath, 'data', 'ampds', 'ampds2.mat')
eco_path = os.path.join(basePath, 'data', 'eco', 'eco1_1min.mat')
redd_path = os.path.join(basePath, 'data', 'redd', 'redd1.mat')

data = load_data(redd_path)
data = process_data(data, 'redd')

data_split = split_data(data, 'k-fold')

epsilon = 1000
data_split['Test']['X_private'] = differential_privacy(data_split['Test']['X'], data_split['Test']['Y'], epsilon)
data_split['Test']['Y_private'] = data_split['Test']['Y']

window_length = 25
stride = 10
data_split_windowed = copy_key_structure(data_split)
data_split_windowed['Train']['X'], data_split_windowed['Train']['Y'] = window_data(data_split['Train']['X'], data_split['Train']['Y'], window_length, stride=stride)
data_split_windowed['Test']['X'], data_split_windowed['Test']['Y'] = window_data(data_split['Test']['X'], data_split['Test']['Y'], window_length, stride=stride)
data_split_windowed['Test']['X_private'], data_split_windowed['Test']['Y_private'] = window_data(data_split['Test']['X_private'], data_split['Test']['Y_private'], window_length, stride=stride)

mdl_name = 'eco1_PM_mdl'
mdl_filepath = trainMdlPM(data_split_windowed['Train'], mdl_name, basePath)

feature_selection = ['all']
Y_pred_windowed = testMdlPM(data_split_windowed['Test']['X_private'], data_split_windowed['Test']['Y_private'], mdl_filepath, feature_selection)

Y_pred = unwindow_data(Y_pred_windowed, window_length, stride=stride)
Y_true = unwindow_data(data_split_windowed['Test']['Y'], window_length, stride=stride)

metrics = evaluate_prediction(Y_pred, Y_true)
energy_accuracy = energy_accuracy(Y_pred, Y_true)

print('')
