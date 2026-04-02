import numpy as np

def temporal_aggregate(data, agg_factor):
    """
    Temporal aggregation via simple block averaging.

    Parameters
    ----------
    data : array-like
        Input time series
    agg_factor : int
        Number of samples per block

    Returns
    -------
    aggregated : np.ndarray
    """

    data = np.asarray(data)

    n = len(data) // agg_factor * agg_factor
    data = data[:n]

    return data.reshape(-1, agg_factor).mean(axis=1)