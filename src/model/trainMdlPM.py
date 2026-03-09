#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlPM
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
This function implements the training case of the pattern matching based energy disaggregation.
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
from numpy import savez_compressed
import numpy as np
import time
from sys import getsizeof
import os

# The trained PM model is the database of stored windowed patterns.
# Unlike NN, PM does not learn parameters, it stores examples.
# There is no loss minimization.
# Full memory of training data.

#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlPM(data_train, mdl_name, basePath):

    N = data_train['X'].shape[0]
    window_length = data_train['X'].shape[1]
    numApp = data_train['Y'].shape[2]

    mdl = np.zeros((N, window_length, numApp+1))
    mdl_filepath = os.path.join(basePath, 'mdl', mdl_name+'.npz')

    start = time.time()

    # Model Input and Output
    # mdl Shape: {n_windows, n_samples_per_window, n_columns}
    # X is first column, Y is subsequent columns
    mdl[:, :, 0] = data_train['X']
    mdl[:, :, 1:] = data_train['Y']

    savez_compressed(mdl_filepath, mdl)

    return mdl_filepath

# Final Saved Model (mdl)
# Stores [ X | Y₁ | Y₂ | ... | Y_numApp ]
# mdl[:,:,0] = X
# mdl[:,:,1] = Y appliance 1
# mdl[:,:,2] = Y appliance 2
# Saved file contains a single numpy array named automatically as 'arr_0'.
# Persistence model dataset container.


