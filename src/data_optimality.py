import numpy as np

def calculate_rmse(estimate, true):
    return np.sqrt(np.mean((estimate - true)**2))

def empirical_mean_analysis(y, y_tilde):
    """
    y: shape (T,)
    noise_samples: shape (N, T)
    """
    T = len(y)
    true_mean = np.mean(y)
    noisy_means = np.mean(y_tilde, axis=1)

    variance = np.var(noisy_means)
    rmse = calculate_rmse(noisy_means, true_mean)

    return {
        "true_mean": true_mean,
        "variance": variance,
        "rmse": rmse
    }

def empirical_energy_analysis(y, y_tilde):
    """
    y: shape (T,)
    y_tilde: shape (N, T)
    """
    true_energy = np.sum(y)

    # Energy per realization
    noisy_energy = np.sum(y_tilde, axis=1)

    variance = np.var(noisy_energy)
    rmse = np.sqrt(np.mean((noisy_energy - true_energy)**2))

    return {
        "true_energy": true_energy,
        "variance": variance,
        "rmse": rmse
    }

def empirical_peak_analysis(y, y_tilde):
    """
    y: shape (T,)
    y_tilde: shape (N, T)
    """
    true_peak = np.max(y)

    # Peak per realization
    noisy_peaks = np.max(y_tilde, axis=1)

    bias = np.mean(noisy_peaks) - true_peak
    variance = np.var(noisy_peaks)
    rmse = np.sqrt(np.mean((noisy_peaks - true_peak)**2))

    return {
        "true_peak": true_peak,
        "bias": bias,
        "variance": variance,
        "rmse": rmse
    }