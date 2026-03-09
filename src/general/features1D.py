#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         features1D
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
This function calculates one dimensional statistical features based on the feature selection in the model parameter
file.
Inputs:     1) data:    input data
            2) feat:    feature selection
Outputs:    1) out:     output feature vector
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy import stats


#######################################################################################################################
# Function
#######################################################################################################################

# Computes statistical features along the time axis and returns them as a feature matrix.
# Input Data Formatting
# Case 1 2D Input: Shape (N, T)
# Case 2 3D Input: Shape (N, T, F_in)
# Feature Selection: List of Features
# --- keys: Mean, Std, RMS, Peak2Rms, Median, Min, Max, Per25, Per75, Energy, Var, Range, 3rdMoment, 4thMoment
# Case 1: Output Shape (N, F)
# Case 2: Output Shape (N, F_in, F)

def features2D(data, feature_selection):

    if 'all' in feature_selection: F = 14
    else: F = len(feature_selection)
    N,T = data.shape
    out = np.zeros((N, F))
    idx = 0

    if 'Mean' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.mean(data, axis=1)
        idx = idx + 1
    if 'Std' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.std(data, axis=1)
        idx = idx + 1
    if 'RMS' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.sqrt(np.mean(data ** 2, axis=1))
        idx = idx + 1
    if 'Peak2Rms' in feature_selection or 'all' in feature_selection:
        temp = np.max(data, axis=1)
        temp2 = np.sqrt(np.mean(data ** 2, axis=1))
        out[:, idx] = np.divide(temp, temp2)
        idx = idx + 1
    if 'Median' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.median(data, axis=1)
        idx = idx + 1
    if 'Min' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.min(data, axis=1)
        idx = idx + 1
    if 'Max' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.max(data, axis=1)
        idx = idx + 1
    if 'Per25' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.percentile(data, 25, axis=1)
        idx = idx + 1
    if 'Per75' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.percentile(data, 75, axis=1)
        idx = idx + 1
    if 'Energy' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.mean(data, axis=1)
        idx = idx + 1
    if 'Var' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.var(data, axis=1)
        idx = idx + 1
    if 'Range' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.ptp(data, axis=1)
        idx = idx + 1
    if '3rdMoment' in feature_selection or 'all' in feature_selection:
        out[:, idx] = stats.skew(data, axis=1)
        idx = idx + 1
    if '4thMoment' in feature_selection or 'all' in feature_selection:
        out[:, idx] = stats.kurtosis(data, axis=1)
        idx = idx + 1

    # Post-processing
    out = np.nan_to_num(out)
    out[out == np.inf] = 0

    return out
