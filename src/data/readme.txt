
# ===== loadData ===== #
1. Loads different file formats (CSV, Matlab, Pickle, H5).
2. Selects input/output features.
--- --- All outputs, explicit list, or Energy-based selection.
--- --- Energy-based selection:
--- --- --- 1. Sums energy per appliance.
--- --- --- 2. Sorts by contribution.
--- --- --- 3. Keeps appliance until threshold energy reached.
3. Splits data (1-fold, k-fold, transfer, ID-based).
--- --- Removes constant features: if a column never changes, remove it.
4. Cleans data.
5. Rolls features (time-windowing).
6. Computes normalization statistics.
--- --- Max, Min, Mean, Variance, Quartiles
--- --- Does not normalize, computes statistics and stores them.
--- --- Calculates sampling time from time column.
7. Stores metadata for the model.
Final Output: Returns [data, setupDat, setupExp]
--- data['X'], data['Y'] are ready for training.