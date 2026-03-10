# Casey Dettlaff

import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd

def load_data(filepath):

    raw = scipy.io.loadmat(filepath)

    if 'labelInp' not in raw or 'labelOut' not in raw:
        raise ValueError('Missing Input or Output Labels')

    app_power_labels = [l.strip() for l in raw['labelInp'][2:]]
    agg_power_labels = [l.strip() for l in raw['labelOut'][2:]]
    agg_power_units = [l.strip() for l in raw['unitInp'][2:]]
    app_power_units = [l.strip() for l in raw['unitOut'][2:]]
    datetimes = raw['input'][:,0]
    sampling_period = np.mean(np.diff(datetimes))
    agg_power = raw['input'][:, 2:]
    app_powers = raw['output'][:, 2:]

    return{
        'X': app_powers,
        'Y': agg_power,
        'sampling_period': sampling_period,
        'Y_labels': app_power_labels,
        'X_labels': agg_power_labels,
        'Y_units': app_power_units,
        'X_units': agg_power_units,
    }

def device_type(profile, tol=1e-3, max_states=5):

    profile = np.asarray(profile).flatten()

    # Round values slightly to collapse small noise
    rounded = np.round(profile / tol) * tol
    unique_vals = np.unique(rounded)

    # Check one-state (ON/OFF)
    if np.all(np.isin(unique_vals, [0, 1])):
        return "one-state"

    # Few discrete values → multi-state
    if len(unique_vals) <= max_states:
        return "multi-state"

    # Otherwise continuous
    return "continuous"

def ghost_data(p_agg, p_appliances):

    # Sum appliance power at each timestep
    appliance_sum = np.sum(p_appliances, axis=1)

    # Ghost power = aggregated - sum of appliances
    ghost_power = p_agg - appliance_sum

    # Add ghost column to appliance matrix
    p_appliances_with_ghost = np.column_stack((ghost_power, p_appliances))

    # Total energy calculations
    total_energy = np.sum(p_agg)
    ghost_energy = np.sum(np.abs(ghost_power))

    ghost_percent = 100 * ghost_energy / total_energy

    return p_appliances_with_ghost, ghost_percent

# Work in progress.
def process_data(data, dataset_name):

    # For AMPDS, only want Input and Output Power
    # Appliance Powers do not sum to the aggregate power.
    if dataset_name == 'ampds':
        data['Y'] = data['Y'][:, 0]  # keep column at index 0
        data['in_labels'] = data['in_labels'][0]  # keep entry at index 0
        data['in_units'] = data['in_units'][0]  # keep entry at index 0
        data['X'] = data['X'][:, 0]  # keep column at index 0

    if dataset_name == 'eco':
        data['Y'] = data['Y'][:, 0] + data['Y'][:, 5] + data['Y'][:, 10] # sum P1, P2, P3
        data['Y_labels'] = 'P_agg'  # keep entry at index 0
        data['Y_units'] = 'W'  # keep entry at index 0

    if dataset_name == 'redd':
        data['Y'] = data['Y'][:, 0]
        data['Y_labels'] = 'P_agg'

        device_types = []
        for i in range(data['X'].shape[1]):
            profile = data['X'][:, i]
            device_types.append(device_type(profile))
        device_types = list(set(device_types))
        data['device_types'] = device_types

        # add 'GHOST' as oth entry to data['Y']['out_labels']
        data['X_labels'].insert(0, 'GHOST')
        data['X'], data['ghost_percent'] = ghost_data(data['Y'], data['X'])

    return data

def data_table(X_true, Y_true, output_labels, csv_filepath, X_pred=None):

    if X_true.ndim == 2:              # Not Windowed (T, num_apps)
        T, num_apps = X_true.shape
    elif X_true.ndim == 3:            # Windowed (N, T, num_apps)
        _, _, num_apps = X_true.shape
    else:
        raise ValueError("Y_true must be 2D or 3D.")

    if len(output_labels) != num_apps:
        raise ValueError("Length of output_labels must match number of appliances.")

    # Flatten (undo windowing if present)
    Y_flat = Y_true.reshape(-1)
    X_true_flat = X_true.reshape(-1, num_apps)

    if X_pred is not None:
        if X_pred.shape[-1] != num_apps:
            raise ValueError("Y_pred appliance dimension mismatch.")
        X_pred_flat = X_pred.reshape(-1, num_apps)

    # ---------------------------
    # Build dataframe
    # ---------------------------
    data = {"aggregate_power": Y_flat}

    # True appliance columns
    for i, label in enumerate(output_labels):
        data[f"true_{label}"] = X_true_flat[:, i]

    # Predicted appliance columns (optional)
    if X_pred is not None:
        for i, label in enumerate(output_labels):
            data[f"pred_{label}"] = X_pred_flat[:, i]

    df = pd.DataFrame(data)

    # ---------------------------
    # Write to CSV
    # ---------------------------
    df.to_csv(csv_filepath, index=False)

    return df

def trim_data(data, n_samples):

    X = data['X']
    Y = data['Y']

    if n_samples > X.shape[0]: return data

    trimmed_data = {}
    trimmed_data['X'] = X[:n_samples]
    trimmed_data['Y'] = Y[:n_samples]
    trimmed_data['X_labels'] = data['X_labels']
    trimmed_data['Y_labels'] = data['Y_labels']

    return trimmed_data

def split_data(data, method='1-fold', rT=0.7, rV=0.15, kfold=5, fold=1, shuffle=False, random_state=42):
    """
    :param data: Dict, output of load_dataset()
    :param method: '1-fold' or 'k-fold'
    :param rT: Train ratio
    :param rV: Validation ratio
    :param kfold: Number of folds
    :param fold: Which fold to use
    :param shuffle: Whether to shuffle
    :return: data_split
    """

    X = data['X']
    Y = data['Y']
    N = X.shape[0]

    data_split = {}

    # 1-Fold Split
    if method == '1-fold':

        if shuffle:
            idx = np.random.permutation(N)
            X = X[idx]
            Y = Y[idx]

        n_train = int(rT * N)
        n_val = int(rV * N)

        X_train = X[:n_train]
        Y_train = Y[:n_train]

        X_val = X[n_train:n_train+n_val]
        Y_val = Y[n_train:n_train+n_val]

        X_test = X[n_train+n_val:]
        Y_test = Y[n_train+n_val:]

    elif method == 'k-fold':

        if shuffle: kf = KFold(n_splits=kfold, shuffle=shuffle, random_state=random_state)
        else: kf = KFold(n_splits=kfold, shuffle=shuffle)

        fold_idx = 1
        for train_idx, test_idx in kf.split(X):

            if fold_idx == fold:
                X_train_full = X[train_idx]
                Y_train_full = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                break

            fold_idx += 1

        n_train_full = X_train_full.shape[0]
        n_val = int(rV * n_train_full)

        X_val = X_train_full[:n_val]
        Y_val = Y_train_full[:n_val]

        X_train = X_train_full[n_val:]
        Y_train = Y_train_full[n_val:]

    else:
        raise ValueError('Invalid method')

    data_split['Train'] = {'X': X_train, 'Y': Y_train}
    data_split['Val'] = {'X': X_val, 'Y': Y_val}
    data_split['Test'] = {'X': X_test, 'Y': Y_test}
    data_split['sampling_period'] = 'sampling_period'
    data_split['X_labels'] = data['X_labels']
    data_split['Y_labels'] = data['Y_labels']
    data_split['X_units'] = data['X_units']
    data_split['Y_units'] = data['Y_units']

    return data_split

def window_data(X, Y, window_length, stride=1):

    # N = number of windows
    # Before windowing, X shape = (T,).
    # Before windowing, Y shape = (T, numApp).
    # After windowing, X shape = (N, window_length).
    # After windowing, Y shape = (N, window_length, numApp).

    T = Y.shape[0]
    numApp = X.shape[1] # Number of Appliances

    N = (T - window_length) // stride + 1 # Number of Windows

    Y_win = np.zeros((N, window_length), dtype=np.float32)
    X_win = np.zeros((N, window_length, numApp), dtype=np.float32)

    idx = 0
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        Y_win[idx] = Y[start:end]
        X_win[idx] = X[start:end]
        idx += 1

    return Y_win, X_win

def unwindow_data(X_win, window_length, stride):
    """
    Inverse of window_data for Y-type input
    (N, window_length, numApp) → (T_original, numApp)
    """

    N, T, numApp = X_win.shape

    if T != window_length:
        raise ValueError("window_length mismatch.")

    # Recover original length T
    T_original = (N - 1) * stride + window_length

    X_recon = np.zeros((T_original, numApp))
    counts = np.zeros(T_original)

    for i in range(N):
        start = i * stride
        end = start + window_length

        X_recon[start:end] += X_win[i]
        counts[start:end] += 1

    counts[counts == 0] = 1
    X_recon /= counts[:, None]

    return X_recon