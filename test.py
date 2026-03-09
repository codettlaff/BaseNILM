from data.loadData import load_data, process_data, plot_data, trim_data, split_data, window_data, unwindow_data, data_table
from model.testMdlPM import testMdlPM, evaluate_prediction, energy_accuracy
from model.trainMdlPM import trainMdlPM
from differential_privacy import differential_privacy
import os
import numpy as np
import scipy.io

# Settings

# Code

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

'''
data_split = split_data(data, 'k-fold')
data_split_windowed = copy_key_structure(data_split)
data_split_windowed['Train']['X'], data_split_windowed['Train']['Y'] = window_data(data_split['Train']['X'], data_split['Train']['Y'], window_length)
data_split_windowed['Test']['X'], data_split_windowed['Test']['Y'] = window_data(data_split['Test']['X'], data_split['Test']['Y'], window_length)

save_path = os.path.join(basePath, 'data', f'{dataset_name}_kfold_windowed_5000sample.npz')
np.savez_compressed(
    save_path,
    Train_X=data_split_windowed['Train']['X'],
    Train_Y=data_split_windowed['Train']['Y'],
    Test_X=data_split_windowed['Test']['X'],
    Test_Y=data_split_windowed['Test']['Y'],
    window_length=window_length
)
print(save_path)

loaded = np.load(savepath)
data_split_windowed = {
    'Train': {
        'X': loaded['Train_X'],
        'Y': loaded['Train_Y']
    },
    'Test': {
        'X': loaded['Test_X'],
        'Y': loaded['Test_Y']
    }
}
window_length = int(loaded['window_length'])

mdl = {}
mdl['mdl'] = 'DTW'
# feat = ['Mean', 'Std']
feat = None
# mdl_filepath = trainMdlPM(data_split_windowed['Train'], mdl, basePath)
# 'C:\\Users\\codett\\PycharmProjects\\Differentially_Private_Smart_Meter\\mdl\\mdl_PM_DTW.npz'
mdl_filepath = r'C:\\Users\\codett\\PycharmProjects\\Differentially_Private_Smart_Meter\\mdl\\mdl_PM_DTW.npz'

Y_pred_windowed = testMdlPM(data_split_windowed['Test']['X'], data_split_windowed['Test']['Y'], mdl_filepath, feat)
Y_pred = unwindow_data(Y_pred_windowed, window_length, stride=1)
Y_true = unwindow_data(data_split_windowed['Test']['Y'], window_length, stride=1)
metrics = evaluate_prediction(Y_pred, Y_true)

results_csv_filepath = os.path.join(basePath, 'results', 'results.csv')

print('')
'''

# TO DO
# Have good dataset with ECO
# Not good results for correlation maximization. Try DTW next.

# If stuck on pattern-matching, move on to another method.
