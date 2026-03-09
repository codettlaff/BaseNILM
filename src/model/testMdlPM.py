#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         testMdlPM
# Date:         23.05.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function implements the testing case of the pattern matching based energy disaggregation.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.features1D import features2D

# ==============================================================================
# External
# ==============================================================================
import dtw
import numpy as np
from numpy import load
from tslearn import metrics
from dtw import *
from tqdm import tqdm
import time
from sys import getsizeof
import inspect

def dtw_distance(x,y):
    # Compute DTW distance between two 1D sequences.

    x = np.asarray(x)
    y = np.asarray(y)

    T_x = len(x)
    T_y = len(y)

    dist_func = lambda a, b: abs(a - b)

    # Cost Matrix
    D = np.zeros((T_x + 1, T_y + 1))
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0

    # Fill Matrix
    for i in range(1, T_x + 1):
        for j in range(1, T_y + 1):
            cost = dist_func(x[i - 1], y[j - 1])

            D[i, j] = cost + min(
                D[i - 1, j],  # insertion
                D[i, j - 1],  # deletion
                D[i - 1, j - 1]  # match
            )

    distance = D[T_x, T_y]

    # Backtrack to recover path
    i, j = T_x, T_y
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        steps = [
            D[i - 1, j],
            D[i, j - 1],
            D[i - 1, j - 1]
        ]

        argmin = np.argmin(steps)

        if argmin == 0:
            i -= 1
        elif argmin == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()

    return distance, path

# Build Matrix D(i,j) which stores cumulative cost to align x[0:i] and y[0:j]
# Warping path is perfect diagonal - no warping.
# DTW is not giving any benefit over simple L1 distance.
# DTW is dominated by total energy difference, causing low-energy templates to win.

# Option 1: Normalize per window before matching.
# Option 2: Use Correlation instead of L1
# --- Shape-based and scale-invariant.

def correlation(x, y):

    eps = 1e-12  # Avoid divide-by-zero errors.

    x = np.asarray(x)
    y = np.asarray(y)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    x_norm = np.linalg.norm(x_centered) + eps
    y_norm = np.linalg.norm(y_centered) + eps
    corr = np.dot(x_centered, y_centered) / (x_norm * y_norm)

    return corr

def hybrid_score(x, y):

    eps = 1e-12 # Avoid divide-by-zero errors.
    alpha = 10 # Shape score scaling.
    beta = 1 # amplitude score scaling.

    x = np.asarray(x)
    y = np.asarray(y)

    # Shape term (correlation)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    x_norm = np.linalg.norm(x_centered) + eps
    y_norm = np.linalg.norm(y_centered) + eps

    corr = np.dot(x_centered, y_centered) / (x_norm * y_norm)
    corr = np.clip(corr, -1.0, 1.0)
    shape_term = 1.0 - corr

    # Magnitude Term
    Ex = np.sum(x)
    Ey = np.sum(y)
    energy_term = abs(Ex - Ey) / (abs(Ex) + eps)

    # Combined Score
    score = alpha * shape_term + beta * energy_term
    return score

def testMdlPM(X_test, Y_test, mdl_filepath, feature_selection=None, C=0.01):

    method = 'correlation_maximization'
    # method = 'dtw_minimization'

    # Saved Model (Database) Shape
    mdl = load(mdl_filepath)['arr_0']

    # Mdl Shape
    N_mdl, T_mdl, numApp_mdl = mdl.shape  # N,T = num_samples, num_timesteps
    numApp_mdl = numApp_mdl - 1

    C = int(np.floor(C * N_mdl))  # number of candidate model patterns to compare using DTW

    # Test Data Shape
    N_Xtest, T_Xtest = X_test.shape
    N_Ytest, T_Ytest, numApp_Ytest = Y_test.shape

    Y_pred = np.zeros((N_Ytest, T_Ytest, numApp_Ytest))

    # Check mdl and test data are formatted the same
    if not (T_mdl == T_Xtest == T_Ytest): raise ValueError(f'Timestep mismatch: mdl={T_mdl}, X_test={T_Xtest}, Y_test={T_Ytest}')
    if numApp_mdl != numApp_Ytest : raise ValueError(f'numApp mismatch: mdl={numApp_mdl}, Y_test={numApp_Ytest}')

    # Feature Extraction
    if feature_selection:
        features_mdl = features2D(mdl[:, :, 0], feature_selection)
        features_X = features2D(X_test, feature_selection)

    sel_list = [] # For debugging

    # Find Top Candidates
    for i in tqdm(range(N_Xtest)):

        if feature_selection:
            feature_diff = abs(features_X[i, :] - features_mdl)
            feature_diff = np.sum(feature_diff, axis=1)  # Reduce over feature dimension
            C_eff = min(C, N_mdl)
            idx = np.argpartition(feature_diff, C_eff)[:C_eff]  # Indices of C smallest feature distances
        else: # Disable feature filtering for debugging
            C_eff = N_mdl
            idx = np.arange(N_mdl)

        tempMdl = mdl[idx, :, :]  # Top C candidates

        # For top candidates, Compute Correlation / Distance
        dist = np.zeros(C_eff)

        x = X_test[i, :]
        x_norm = x / np.sum(x)

        for ii in range(C_eff):

            template = tempMdl[ii, :, 0]
            template_norm = template / np.sum(template)

            if method.split('_')[0] == 'correlation': dist_ii = correlation(x_norm, template_norm)
            elif method.split('_')[0] == 'dtw': dist_ii, path = dtw_distance(x_norm, template_norm)
            else: raise ValueError(f'Unknown Method {method}.')

            dist[ii] = dist_ii

        # Remove all 'None' entries
        dist = dist[~np.isnan(dist)]
        if method.split('_')[1] == 'maximization': sel = np.argmax(np.abs(dist))
        elif method.split('_')[1] == 'minimization': sel = np.argmin(np.abs(dist))
        else: raise ValueError(f'Unknown Method {method}.')

        sel_list.append(sel) # For debugging

        best_template_agg = tempMdl[sel, :, 0]
        best_template_apps = tempMdl[sel, :, 1:]
        # scale = np.sum(x) / np.sum(best_template_agg) # If best_template is all 0, this will cause error
        # scale = np.sum(x) / np.sum(np.sum(best_template_apps, axis=1))
        Y_pred[i, :, :] = best_template_apps

    return Y_pred

def evaluate_prediction(Y_pred, Y_test):

    if Y_pred.shape != Y_test.shape:
        raise ValueError(f"Shape mismatch: Y_pred {Y_pred.shape}, Y_test {Y_test.shape}")

    if Y_test.ndim != 2:
        raise ValueError("Expected unwindowed data with shape (T, numApp).")

    T, numApp = Y_test.shape

    # Overall metrics
    mae = np.mean(np.abs(Y_pred - Y_test))
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    denom = np.sum(Y_test ** 2)
    nde = np.sum((Y_pred - Y_test) ** 2) / denom if denom != 0 else 0.0

    # Per-appliance metrics
    mae_per_app = np.mean(np.abs(Y_pred - Y_test), axis=0)
    rmse_per_app = np.sqrt(np.mean((Y_pred - Y_test) ** 2, axis=0))

    metrics = {
        "MAE_overall": mae,
        "RMSE_overall": rmse,
        "NDE_overall": nde,
        "MAE_per_appliance": mae_per_app,
        "RMSE_per_appliance": rmse_per_app,
    }

    return metrics

def energy_accuracy(Y_pred, Y_true):
    Y_pred = np.asarray(Y_pred)
    Y_true = np.asarray(Y_true)

    numerator = np.sum(np.abs(Y_true - Y_pred))
    denominator = 2 * np.sum(Y_true)

    acc = 1 - numerator / denominator
    return acc

# Pattern-matching Predictor
# 1. Database of past windows (mdl) shaped like (N,T,F,numApp+1)
# --- T = Timesteps per window
# --- F = Num features per timestep
# --- 4th Channel (numApp+1)
# --- --- Channel 0 -  Input Signal (Aggregate)
# --- --- Channel 1: - Outputs Signals (Appliances)
# 2. For each test window X_test[i] (shape (T,F)):
# --- --- Cheap feature distance to all stored patterns -> keep only top C candidates
# --- --- Expensive DTW distance to those C candidates
# --- --- Pick best match
# --- --- Copy stored output channels into Y_pred

# Results
# Error is extremely small, but this is only because most values in Y are near 0
# Most appliances are OFF most of the time
# NDE is almost 1 - this is what would happen if we predicted near-zero for everything.
# np.mean(Y_pred) = 7.3e-05
# np.mean(Y_test) = 0.00209
# Percentage of zeros in Y_test = 0.2 = 20%
# Percentage of zeros in Y_pred = 0.4 = 40%
# Not a mostly zero dataset
# The model is selecting templates with low aggregate energy, minimizing DTW distance by favoring low-amplitude signals.
# If two signals differ in amplitude, DTW often prefers the smaller amplitude signal

# Suggested solution.
# Normalize globally, not per-house.
# Compute one global peak across training data.
# Example
# House A peak = 10 kW
# House B peak = 3 kW
# After normalization:
# 10 kW -> 1.0
# 3 kW -> 1.0
# Both look equally large, even though house A appliances are higher power
# A 2kW kettle in a small house and a 2kW kettle in a large house end up scaled differently relative to other loads.

# Removed normalization
# np.mean(Y_pred) = 0.6133
# np.mean(Y_test) = 17.58
# underestimation problem did not come from per-house normalization

# Problem
# Matching strategy is choosing low-energy templates.
# I am matching only on aggregate signal.
# Implemented energy penatly

# Observation: Energy penalty is always exactly equal to the distance from dtw (temp[0])
# energy mismatch is already dominating DTW distance
# rather than minimizing absolute difference, want to minimize relative distance or shape similarity independent of scale.
# separate into shape similarity and magnitude alignment.

# Feature filtering is always filtering to the same windows.
# Turned off feature filtering, model is free to search all templates.
# using maximizing correlation only
# Now have overestimation
# -- mean(Y_test) = 17.6
# --- mean(Y_pred) = 41.5
# -- NDE = 34
# -- MAE = 31

# Want to try normalizing windows.

# Normalizing Windows
# want to remove magnitude bias
# for each window x:
# x_norm = x / (sum(x)+epsilon) -> converts the window into a distribution of energy over time.
# after selecting the best template, we can restore magnitude: Y_pred = Y_template * (sum(x_test)/sum(x_template))

# New results:
# mean(Y_test) = 17.58
# mean(Y_pred) = 16.31
# magnitude bias is gone
# MAE = 14.88
# RMSE = 61.05
# NDE = 2.58
# remaining error is true modeling error, not optimization bias

# remaining error is coming from:
# template mismatch - database diversity
# appliance composition ambiguity
# similar aggregate shapes from different appliance combinations
# window length choice
# correlation not capturing meaningful shape similarity

# Shoot for :
# NDE 0.5 - 1.0 (reasonable)
# MAE / mean(Y_test) = 20-40%

# DTW probably won't work any better than correlation because DTW path is perfect diagonal.

# DTW with window normalization, and with feature filtering disabled:
# Underestimation
# Mean(Y_pred) = 5.38
# Mean(Y_test) = 17.58
# MAE_per_appliance: [0.3469, 62.1490, 0.3564, 0.3469, 0.]: One appliance dominates total error (2nd Appliance)

# Results are actually better for DTW than for correlation, but has bad magnitude bias.