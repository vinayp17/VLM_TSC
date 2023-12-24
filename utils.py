from aeon.datasets import load_classification

import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.signal import decimate, medfilt, gaussian
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d


def downsample(X):
    # Downsampling factor
    factor = 5

    # 1. Simple Subsampling (Decimation)
    simple_subsampling = X[::factor]

    # 2. Averaging
    average_downsampled = uniform_filter1d(X, size=factor)[::factor]

    # 3. Low-pass Filtering followed by Decimation
    lpf_downsampled = decimate(X, factor, ftype='fir')

    # 4. Max Pooling
    max_pooling = np.max(X.reshape(-1, factor), axis=1)

    # 5. Median Filtering
    median_filtered = medfilt(X, kernel_size=factor)[::factor]

    # 6. Gaussian Downsampling
    gaussian_filter = gaussian(factor, std=2)
    gaussian_downsampled = np.convolve(X, gaussian_filter, mode='same')[::factor]

    # 7. Resampling with Interpolation
    # interp = interp1d(np.arange(len(X)), original_series, kind='cubic')
    # resampled_t = np.linspace(0, length-1, length // factor)
    # interpolated_downsampled = interp(resampled_t)


def generate_llava_entry(question, target, image_filename_id, image_filename_path):

    entry = {
        "id": image_filename_id,
        "image": image_filename_path,
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": target
            }
        ]
    }
    return entry

def generate_line_graph(X, image_filename_path):

    plt.figure(figsize=(4,4))
    plt.plot(X)

    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='y', length=0)
    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()

    # Hide the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(image_filename_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_ucr_data(dataset, split, image_path, data_path, extract_path='~/Downloads/UCRArchive_2018/'):

    X, y, meta = load_classification(
        dataset, extract_path=extract_path, split=split, return_metadata=True
    )

    entries = []

    for n in range(0, len(y)):
        image_filename_id = f"image_{split}_{n}"
        image_filename_path = f"{image_path}/image_{split}_{n}.png"

        combined_signal_string = [str(i) for i in X[n][0].tolist()]
        question = f"Which class is the following signal? {combined_signal_string}".replace("\'", "")
        target = int(y[n])

        ### CREATE DATA ENTRY
        entries.append(generate_llava_entry(question, target, image_filename_id, image_filename_path))

        ### CREATE IMAGE ENTRY
        generate_line_graph(X[n][0], image_filename_path)

    json_output = json.dumps(entries, indent=2)

    # Write to a file
    with open(f'{data_path}/output.json', 'w') as file:
        file.write(json_output)



generate_ucr_data('TwoLeadECG', 'TRAIN', '.', '.', extract_path='~/Downloads/UCRArchive_2018/')




