# Casey Dettlaff
# Reference: BaseNILM toolkit for energy disaggregation, Dr. Pascal A. Schirmer

from numpy import savez_compressed
import numpy as np
import os

def trainMdlPM(data_train, mdl_name, basePath):

    N = data_train['X'].shape[0]
    window_length = data_train['X'].shape[1]
    numApp = data_train['Y'].shape[2]

    mdl = np.zeros((N, window_length, numApp+1))
    mdl_filepath = os.path.join(basePath, 'mdl', mdl_name+'.npz')

    mdl[:, :, 0] = data_train['X']
    mdl[:, :, 1:] = data_train['Y']

    savez_compressed(mdl_filepath, mdl)

    return mdl_filepath

# Final Saved Model (mdl)
# Stores [ Y | X₁ | X₂ | ... | X_numApp ]
# mdl[:,:,0] = Y
# mdl[:,:,1] = X₁ appliance 1
# mdl[:,:,2] = X₂ appliance 2


