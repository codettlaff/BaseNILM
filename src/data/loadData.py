# Casey Dettlaff
# 20260217

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

    in_labels = [l.strip() for l in raw['labelInp'][2:]]
    out_labels = [l.strip() for l in raw['labelOut'][2:]]
    in_units = [l.strip() for l in raw['unitInp'][2:]]
    out_units = [l.strip() for l in raw['unitOut'][2:]]
    datetimes = raw['input'][:,0]
    sampling_period = np.mean(np.diff(datetimes))
    inp = raw['input'][:, 2:]
    out = raw['output'][:, 2:]

    return{
        'X': inp,
        'Y': out,
        'sampling_period': sampling_period,
        'in_labels': in_labels,
        'out_labels': out_labels,
        'in_units': in_units,
        'out_units': out_units,
    }

def device_type(profile, tol=1e-3, max_states=5):
    """
    Classify appliance profile as 'one-state', 'multi-state', or 'continuous'.

    Parameters
    ----------
    profile : 1D numpy array
        Power time series of the appliance.
    tol : float
        Tolerance for grouping similar values.
    max_states : int
        Maximum number of discrete levels to consider multi-state.

    Returns
    -------
    str
        'one-state', 'multi-state', or 'continuous'
    """
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
    """
    p_agg: 1D numpy array (N,)
        Aggregated power signal

    p_appliances: 2D numpy array (N, M)
        Each column corresponds to an appliance

    Returns
    -------
    p_appliances_with_ghost : 2D numpy array (N, M+1)
        Original appliance data with ghost power added as last column

    ghost_percent : float
        Percentage of total energy that is unaccounted for
    """

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


def process_data(data, dataset_name):

    # For AMPDS, only want Input and Output Power
    # Appliance Powers do not sum to the aggregate power.
    if dataset_name == 'ampds':
        data['X'] = data['X'][:, 0]  # keep column at index 0
        data['in_labels'] = data['in_labels'][0]  # keep entry at index 0
        data['in_units'] = data['in_units'][0]  # keep entry at index 0
        data['Y'] = data['Y'][:, 0]  # keep column at index 0

    if dataset_name == 'eco':
        data['X'] = data['X'][:, 0] + data['X'][:, 5] + data['X'][:, 10] # sum P1, P2, P3
        data['in_labels'] = 'P_agg'  # keep entry at index 0
        data['in_units'] = 'W'  # keep entry at index 0

    if dataset_name == 'redd':
        data['X'] = data['X'][:, 0]
        data['in_labels'] = 'P_agg'

        device_types = []
        for i in range(data['Y'].shape[1]):
            profile = data['Y'][:, i]
            device_types.append(device_type(profile))
        device_types = list(set(device_types))
        data['device_types'] = device_types

        # add 'GHOST' as oth entry to data['Y']['out_labels']
        data['out_labels'].insert(0, 'GHOST')
        data['Y'], data['ghost_percent'] = ghost_data(data['X'], data['Y'])

    return data


def load_dataset_old(basePath, dataset_name):

    folderpath = os.path.join(basePath, 'data', dataset_name)
    filepath_list = sorted([
        os.path.join(folderpath, f)
        for f in os.listdir(folderpath)
        if f.endswith(".mat")
    ])
    valid_filepath_list = []

    save_filepath = os.path.join(basePath, 'data', dataset_name+'_processed.npz')

    all_input_features = set()
    all_appliances = set()

    for filepath in filepath_list:
        raw = scipy.io.loadmat(filepath)

        # Skip REDD HF files
        if dataset_name == 'redd' and raw['input'].ndim == 3:
            continue

        # Skip files without input labels
        if 'labelInp' not in raw or 'labelOut' not in raw:
            continue

        valid_filepath_list.append(filepath)

        in_labels = [l.strip() for l in raw['labelInp'][2:]]
        out_labels = [l.strip() for l in raw['labelOut'][2:]]

        all_input_features.update(in_labels)
        all_appliances.update(out_labels)

    all_input_features = sorted(list(all_input_features))
    all_appliances = sorted(list(all_appliances))

    input_index = {feat: idx for idx, feat in enumerate(all_input_features)}
    appliance_index = {app: idx for idx, app in enumerate(all_appliances)}

    X_list = []
    Y_list = []

    for filepath in valid_filepath_list:

        raw = scipy.io.loadmat(filepath)
        inp = raw['input']
        out = raw['output']

        # Build X - Always 1D
        if inp.ndim == 3: X_raw = inp[:, :, 2]  # (N, T) → active power only
        else: X_raw = inp[:, 2]  # (N,) or (N, features)
        X_i = X_raw.astype(np.float32)
        N = len(X_raw)
        X_list.append(X_i)

        # Build padded Y - Always 2D. Dimensions depend on # Appliances
        Y_raw = out[:, 2:]
        if Y_raw.ndim == 3: Y_raw = Y_raw[:, :, 2] # Keep active power only.
        Y_raw = Y_raw.astype(np.float32)
        current_output_labels = [l.strip() for l in raw['labelOut'][2:]]

        Y_i = np.zeros((N, len(all_appliances)), dtype=np.float32)
        for j, label in enumerate(current_output_labels):
            col_idx = appliance_index[label]
            Y_i[:, col_idx] = Y_raw[:, j]

        Y_list.append(Y_i)

    # Normalize
    '''
    for i in range(len(X_list)):
        data = X_list[i]
        peak = np.max(np.abs(data))
        X_list[i] = data / peak

    for i in range(len(Y_list)):
        data = Y_list[i]
        Y_list[i] = data / peak
    '''

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list, axis=0)

    np.savez_compressed(save_filepath, X=X, Y=Y, output_labels=all_appliances)

    return {
        'X': X,
        'Y': Y,
        'output_labels': all_appliances
    }

# Post-Load Processing
# Compute total energy per appliance, keep only top contributers.
# Limit samples
# Remove constant columns
# split training / testing (1-fold split, k-fold split, transfer learning)
# rolling feature engineering
# normalization / statistics
# sampling time computation

def plot_data(data):

    X = data['X']
    Y = data['Y']
    labels = data['output_labels']
    N = len(X)
    t = np.arange(N)

    plt.figure(figsize=(14, 6))

    plt.plot(t, X, linewidth=2.5, label="Aggregate (X)") # Plot aggregate

    # Plot appliances (thin + transparent)
    for i in range(Y.shape[1]):
        plt.plot(
            t,
            Y[:, i],
            linewidth=1,
            alpha=0.7,
            label=labels[i]
        )

    plt.title("Aggregate and Appliance Power")
    plt.ylabel("Normalized Power")
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show()

def data_table(X_true, Y_true, output_labels, csv_filepath, Y_pred=None):

    # ---------------------------
    # Determine number of appliances
    # ---------------------------
    if Y_true.ndim == 2:              # (T, num_apps)
        T, num_apps = Y_true.shape
    elif Y_true.ndim == 3:            # (N, T, num_apps)
        _, _, num_apps = Y_true.shape
    else:
        raise ValueError("Y_true must be 2D or 3D.")

    if len(output_labels) != num_apps:
        raise ValueError("Length of output_labels must match number of appliances.")

    # ---------------------------
    # Flatten (undo windowing if present)
    # ---------------------------
    X_flat = X_true.reshape(-1)
    Y_true_flat = Y_true.reshape(-1, num_apps)

    if Y_pred is not None:
        if Y_pred.shape[-1] != num_apps:
            raise ValueError("Y_pred appliance dimension mismatch.")
        Y_pred_flat = Y_pred.reshape(-1, num_apps)

    # ---------------------------
    # Build dataframe
    # ---------------------------
    data = {"aggregate_power": X_flat}

    # True appliance columns
    for i, label in enumerate(output_labels):
        data[f"true_{label}"] = Y_true_flat[:, i]

    # Predicted appliance columns (optional)
    if Y_pred is not None:
        for i, label in enumerate(output_labels):
            data[f"pred_{label}"] = Y_pred_flat[:, i]

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
    trimmed_data['output_labels'] = data['output_labels']

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
    data_split['in_labels'] = data['in_labels']
    data_split['out_labels'] = data['out_labels']
    data_split['in_units'] = data['in_units']
    data_split['out_units'] = data['out_units']

    return data_split

def window_data(X, Y, window_length, stride=1):

    # N = number of windows
    # Before windowing, X shape = (T,).
    # Before windowing, Y shape = (T, numApp).
    # After windowing, X shape = (N, window_length).
    # After windowing, Y shape = (N, window_length, numApp).

    T = X.shape[0]
    numApp = Y.shape[1] # Number of Appliances

    N = (T - window_length) // stride + 1 # Number of Windows

    X_win = np.zeros((N, window_length), dtype=np.float32)
    Y_win = np.zeros((N, window_length, numApp), dtype=np.float32)

    idx = 0
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        X_win[idx] = X[start:end]
        Y_win[idx] = Y[start:end]
        idx += 1

    return X_win, Y_win

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