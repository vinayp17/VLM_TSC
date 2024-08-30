import numpy as np
from scipy.ndimage import uniform_filter1d

def calculate_variability(signal, window_size):
    num_windows = len(signal) // window_size
    remainder = len(signal) % window_size

    variability = []
    for i in range(num_windows):
        window = signal[i*window_size:(i+1)*window_size]
        variability.append(np.std(window))
    if remainder > 0:
        variability.append(np.std(signal[-remainder:]))

    return np.array(variability)

def adaptive_downsampling_impl(signal, variability, min_rate, max_rate):
    norm_variability = (variability - np.min(variability)) / (np.max(variability) - np.min(variability))
    downsampling_rates = min_rate + (1 - norm_variability) * (max_rate - min_rate)

    downsampled_signal = []
    for i, rate in enumerate(downsampling_rates):
        start_idx = i * len(signal) // len(variability)
        end_idx = min((i+1) * len(signal) // len(variability), len(signal))
        segment = signal[start_idx:end_idx]
        factor = int(np.ceil(rate))

        if factor > 1:
            smoothed_segment = uniform_filter1d(segment, size=factor)[::factor]
        else:
            smoothed_segment = segment

        downsampled_signal.extend(smoothed_segment)

    return np.array(downsampled_signal)

def intelligently_remove_points(signal, target_length):
    current_length = len(signal)
    if current_length <= target_length:
        return signal

    points_to_remove = int(current_length - target_length)
    if points_to_remove == 0:
        return signal
    print(f'Had to remove {points_to_remove} points')

    # Remove points at regular intervals
    interval = current_length / points_to_remove
    indices_to_remove = np.round(np.arange(0, current_length, interval)).astype(int)

    # Ensure indices_to_remove is within the bounds of the signal length
    indices_to_remove = np.clip(indices_to_remove, 0, len(signal) - 1)

    # Remove these indices from the signal
    signal = np.delete(signal, indices_to_remove[:points_to_remove])

    return signal


'''
def intelligently_remove_points(signal, target_length):
    current_length = len(signal)
    if current_length <= target_length:
        return signal

    points_to_remove = int(current_length - target_length)

    # Calculate interval for removing points
    interval = current_length / points_to_remove
    indices_to_remove = np.round(np.arange(0, current_length, interval)).astype(int)

    # Ensure indices_to_remove is within the bounds of the signal length
    indices_to_remove = np.clip(indices_to_remove, 0, len(signal) - 1)

    # Average out points with their neighbors before removal
    for idx in indices_to_remove[:points_to_remove]:
        if 0 < idx < len(signal) - 1:  # Ensure index is within bounds
            signal[idx] = (signal[idx - 1] + signal[idx + 1]) / 2
        elif idx == 0:  # Edge case for the first element
            signal[idx] = signal[idx + 1]
        elif idx == len(signal) - 1:  # Edge case for the last element
            signal[idx] = signal[idx - 1]

    # Remove the points
    signal = np.delete(signal, indices_to_remove[:points_to_remove])

    return signal

def intelligently_remove_points(signal, target_length):
    current_length = len(signal)
    if current_length <= target_length:
        return signal

    points_to_remove = current_length - target_length

    # Calculate interval for removing points
    interval = current_length / points_to_remove
    indices_to_remove = np.round(np.arange(0, current_length, interval)).astype(int)

    # Average out points with their neighbors before removal
    for idx in indices_to_remove[:points_to_remove]:
        if 0 < idx < len(signal) - 1:  # Ensure index is within bounds
            signal[idx] = (signal[idx - 1] + signal[idx + 1]) / 2
        elif idx == 0:  # Edge case for the first element
            signal[idx] = signal[idx + 1]
        elif idx == len(signal) - 1:  # Edge case for the last element
            signal[idx] = signal[idx - 1]

    # Remove the points
    signal = np.delete(signal, indices_to_remove[:points_to_remove])

    return signal
'''


def final_downsampling(signal, variability, target_length, min_rate, max_rate):
    downsampled_signal = adaptive_downsampling_impl(signal, variability, min_rate, max_rate)

    # Intelligent removal of points to match target length
    downsampled_signal = intelligently_remove_points(downsampled_signal, target_length)

    return downsampled_signal

def find_optimal_rates(signal, variability, target_length, tolerance=0.01):
    L = len(signal)
    F = L / target_length

    min_rate = 1
    max_rate = 1.5 * F  # Initial guess

    for iteration in range(100):  # Limit iterations to avoid infinite loops
        downsampled_signal = adaptive_downsampling_impl(signal, variability, min_rate, max_rate)
        downsampled_length = len(downsampled_signal)

        if np.abs(downsampled_length - target_length) / target_length < tolerance:
            break

        if downsampled_length > target_length:
            min_rate += 0.1
            max_rate += 0.1
        else:
            max_rate -= 0.1
            if max_rate < min_rate:
                min_rate -= 0.1

    return min_rate, max_rate

def adaptive_downsample(X, factor):
    signal = X
    window_size = 100
    target_length = len(signal) / factor  # Desired downsampled length
    variability = calculate_variability(signal, window_size)
    min_rate, max_rate = find_optimal_rates(signal, variability, target_length)

    downsampled_signal = final_downsampling(signal, variability, target_length, min_rate, max_rate)

    print("Original signal length:", len(signal))
    print("Downsampled signal length:", len(downsampled_signal))
    print("Optimal min_rate:", min_rate)
    print("Optimal max_rate:", max_rate)
    return downsampled_signal
