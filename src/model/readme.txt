Casey Dettlaff
20260216

SCRIPT          PARADIGM            FRAMEWORK           TYPE
testMdlPM       Pattern Matching    DTW / tslearn        Non-parametric
testMdlSS       Source Separation   NMF / Sparse Coding  Linear Factorization
testMdlSK       Classical ML        sklearn              Shallow Parametric
testMdlPT       Deep Learning       PyTorch              Deep Parametric
testMdlTF       Deep Learning       TensorFlow           Deep Parametric

These Functions, as taken from BaseNILM, only returned predictions. They did not compute error metrics.

What must be true for fair comparison:
1. Same test dataset.
2. Same preprocessing.
3. Same output shape.
4. Same postprocessing.
5. Same evaluation metric.

#===== testMdlPM ======#
Pattern-matching (PM) time-series disaggregation model.
-- Non-Parametric Time-Series Nearest Neighbor Model.
Key Parameters:
-- numApp: Number of output channels (e.g., number of appliances).
-- C: how many candidate templates to compare against.
-- mdl: The model is a .npz file containing stored template time-series (database of example patterns).
Feature Extraction (Preselection Step)
-- Computes feature vectors for Test Signals, Stored Templates
-- Statistical features may include:
-- -- Mean
-- -- Variance
-- -- Energy
-- -- Peaks
Core Loop (For each Test Sample):
-- 1. Candidate selection: compute distance between test signal and all templates, four methods:
-- -- -- DTW (Dynamic time-warping)
-- -- -- GAK (Global Alignment Kernel)
-- -- -- soft-DTW
-- -- -- MVM (Modified DTW with custom step patterns)
-- 2. Keep only C closest matches.

Data format required.
data = {
    'T': {
        'X': ...,
        'y': ...
    }
}
data = {
    'T': {...},   # training
    'V':  {...},   # validation
    'T':  {...}    # test
}
Function assumes:
data['T']['X'].shape[0]   # number of samples
data['T']['X'].shape[1]   # window length
X is not raw time-series. It is already windowed.

For each test window, extract features, compare to stored model patterns.

Univariate
X_test: (N_windows, window_length)
y_test: (N_windows, window_length, numApp)

Multivariate
X_test: (N_windows, window_length, nDim)
y_test: (N_windows, window_length, numApp)

# ===== testMdlSS ===== #
Matrix Factorization / Sparse Coding (unsupervised source separation).
-- Aggregated signal is a linear combination of basis signals.
Two supported models:
-- NMF Non-negative Matrix Factorization
-- DSC Dictionary Sparse Coding
1. Initialization
2. Load Model (model saved as .npz)
3. Case 1 NMF, Case 2 DSC

Feature             NMF             DSC
Constraint          Non-negative    Sparse
Optimization        NNLS            Lasso
Interpretability    High            High
Flexibility         Lower           Higher

# ===== testMdlSK ===== #
Scikit-Learn / Traditional ML
1. Reshapes input data.
2. Loads a trained sklearn-style model.
3. Runs .predict().
4. Measures inference time.
5. Returns predictions.

# ===== testMdlPT ===== #
PyTorch DNN/CNN Inference
1. Reshapes test data.
2. Builds PyTorch Model (DNN or CNN).
3. Loads trained weights.
4. Runs forward pass.
5. Measures inference time.
6. Returns predictions.

# ===== testMdlTF ===== #
TensorFlow Deep Learning
1. Reshapes test data.
2. Builds TensorFlow model (DNN, CNN, Transformer, ect..).
3. Loads saved weights.
4. Compiles model.
5. Runs inference.
6. Measures inference time.
7. Returns prediction.