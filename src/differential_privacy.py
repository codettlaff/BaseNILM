import numpy as np

def differential_privacy(p_agg, p_appliances, epsilon):
    """
    p_agg : 1D numpy array
        Aggregated power signal

    p_appliances : 2D numpy array
        Appliance power signals (columns = appliances)

    epsilon : float
        Differential privacy parameter
    """

    # Maximum power value of any appliance
    B = np.max(p_appliances)

    # L1 sensitivity
    delta_f = 2 * B

    # Laplace noise scale
    scale = delta_f / epsilon

    # Draw Laplace noise (same length as p_agg)
    eta = np.random.laplace(loc=0, scale=scale, size=p_agg.shape)

    # Add noise to aggregated signal
    p_agg_private = p_agg + eta

    return p_agg_private