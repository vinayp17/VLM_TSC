import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import entropy, linregress
import pywt

# Function to extract features from a single beat and return them as a concatenated string in scientific notation
def extract_features(ecg_signal):
    features = {}

    # Basic statistics
    features['mean'] = np.mean(ecg_signal)
    features['std_dev'] = np.std(ecg_signal)
    features['max_amplitude'] = np.max(ecg_signal)
    features['min_amplitude'] = np.min(ecg_signal)
    features['median'] = np.median(ecg_signal)

    # Peak features
    peaks, _ = find_peaks(ecg_signal)
    features['peak_count'] = len(peaks)
    if len(peaks) > 0:
        features['mean_peak_amplitude'] = np.mean(ecg_signal[peaks])
        features['max_peak_amplitude'] = np.max(ecg_signal[peaks])
    else:
        features['mean_peak_amplitude'] = 0
        features['max_peak_amplitude'] = 0

    # Range
    features['range'] = features['max_amplitude'] - features['min_amplitude']

    # Skewness and Kurtosis
    features['skewness'] = pd.Series(ecg_signal).skew()
    features['kurtosis'] = pd.Series(ecg_signal).kurtosis()

    # Energy of the signal
    features['energy'] = np.sum(np.square(ecg_signal))

    # Entropy of the signal
    histogram, bin_edges = np.histogram(ecg_signal, bins=10, density=True)
    features['entropy'] = entropy(histogram)

    # Zero Crossing Rate
    zero_crossings = np.where(np.diff(np.sign(ecg_signal)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_signal)

    # Signal Slope
    slope, _, _, _, _ = linregress(range(len(ecg_signal)), ecg_signal)
    features['signal_slope'] = slope

    # Autocorrelation (lag 1)
    features['autocorrelation'] = np.corrcoef(ecg_signal[:-1], ecg_signal[1:])[0, 1]

    # Power Spectral Density (PSD)
    freqs, psd = welch(ecg_signal, fs=1000)
    features['psd_mean'] = np.mean(psd)
    features['psd_max'] = np.max(psd)

    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(ecg_signal ** 2)
    noise_power = np.var(ecg_signal)
    features['snr'] = 10 * np.log10(signal_power / noise_power)

    # Wavelet Transform Coefficients (using Daubechies wavelet)
    coeffs = pywt.wavedec(ecg_signal, 'db1', level=4)
    features['wavelet_mean'] = np.mean(coeffs[0])
    features['wavelet_std'] = np.std(coeffs[0])

    # Hurst Exponent
    def hurst_exponent(ts):
        lags = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    features['hurst_exponent'] = hurst_exponent(ecg_signal)

    # Concatenate all feature values into a string in scientific notation
    feature_string = ', '.join(f"{key}: {value:.4e}" for key, value in features.items())

    return feature_string
