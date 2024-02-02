from aeon.datasets import load_classification

import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
import argparse

from scipy.signal import decimate, medfilt, gaussian
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

import pandas as pd

def generate_llava_qa_entry(question, target, image_filename_id, image_filename_path):

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

def generate_llava_eval_entry(question, target, image_filename_id, image_filename_path):

    entry = {
        "question_id": image_filename_id,
        "image": image_filename_path,
        "text": question,
        "output": target,
    }
    return entry

def generate_vicuna_qa_entry(question, target, image_filename_id, image_filename_path):

    entry = {
        "id": image_filename_id,
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

def generate_vicuna_eval_entry(question, target, image_filename_id, image_filename_path):

    entry = {
        "question_id": image_filename_id,
        "text": question,
        "output": target,
    }
    return entry

def generate_qwen_vl_entry(question, target, image_filename_id, image_filename_path):

    entry = {
      "id": image_filename_id,
      "conversations": [
        {
          "from": "user",
          "value": f"Picture 1: <img>{image_filename_path}</img>\n{question}"
        },
        {
          "from": "assistant",
          "value": target
        }
      ]
    }
    return entry

def generate_data_entry(split, model, question, target, image_filename_id, image_filename_path):
    if model == "llava":
        if split == "train": 
            return generate_llava_qa_entry(question, target, image_filename_id, image_filename_path)
        else:
            return generate_llava_eval_entry(question, target, image_filename_id, image_filename_path)
    elif model == "vicuna":
        if split == "train": 
            return generate_vicuna_qa_entry(question, target, image_filename_id, image_filename_path)
        else:
            return generate_vicuna_eval_entry(question, target, image_filename_id, image_filename_path)
    elif model == "qwen":
        return generate_qwen_vl_entry(question, target, image_filename_id, image_filename_path)
    else:
        print("MODEL NOT FOUND")

def generate_graph(X, image_filename_path, style="line"):

    plt.figure(figsize=(4,4))


    for dimension in range(0, len(X)):
        if style == "line":
            plt.plot(X[dimension])
        elif style == "area":
            plt.stackplot(range(0, len(X)), X)

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


def downsample(X, factor=2):
    # # 1. Simple Subsampling (Decimation) -- This method selects every kth element from the original array,
    # simple_subsampling = X[::factor]

    # 2. Averaging -- pplies a uniform filter (sliding window average) over the data and then selects every kth element
    if factor is not None:
        average_downsampled = uniform_filter1d(X, size=factor)[::factor]
        return average_downsampled
    else:
        return X

    # # 3. Low-pass Filtering followed by Decimation -- decimate function in SciPy applies a low-pass filter before downsampling by the specified factor.
    # lpf_downsampled = decimate(X, factor, ftype='fir')

    # # 4. Max Pooling -- Max pooling involves dividing the array into non-overlapping segments of length k and taking the maximum value in each segment.
    # max_pooling = np.max(X.reshape(-1, factor), axis=1)

    # # 5. Median Filtering -- Median filtering involves applying a median filter with a specified kernel size and then picking every kth element
    # median_filtered = medfilt(X, kernel_size=factor)[::factor]

    # # 6. Gaussian Downsampling -- This method first applies a Gaussian filter to the data and then performs subsampling. The Gaussian filter, created with a specified standard deviation (here, std=2), smooths the data by giving more weight to nearby points and less to distant ones.
    # gaussian_filter = gaussian(factor, std=2)
    # gaussian_downsampled = np.convolve(X, gaussian_filter, mode='same')[::factor]

    # # 7. Resampling with Interpolation
    # interp = interp1d(np.arange(len(X)), original_series, kind='cubic')
    # resampled_t = np.linspace(0, length-1, length // factor)
    # interpolated_downsampled = interp(resampled_t)


def format_numbers_combined(numbers, max_precision, padded=False, round_to=None):
    if round_to:
        numbers = [np.round(i, round_to) for i in numbers]

    formatted_numbers = [str(num) for num in numbers]

    return formatted_numbers

def generate_data(X, y, index, image_path, max_y, max_precision, padded, round_to, downsample_to):
    
    df = pd.DataFrame(columns=['question', 'target', 'image_filename_id', 'image_filename_path'])

    for n in range(0, len(y)):
        image_filename_id = f"image_{index[n]}"
        image_filename_path = f"{image_path}/image_{index[n]}.png"

        question = f"Which class is the following signal from?\n"
        
        
        for dimension in range(0, len(X[n])):

            if downsample_to is not None:
                combined_signal_string = format_numbers_combined(downsample(X[n][dimension], downsample_to), max_precision, padded=padded, round_to=round_to)
            else:
                combined_signal_string = format_numbers_combined(X[n][dimension], max_precision, padded=padded, round_to=round_to)

            #options = "A" + "".join([f" or {chr(number + 64)}" for number in range(2, int(max_y)+1)]) + "?"
            if len(X[n]) == 1:
                question = question + f"{combined_signal_string}\n"
            else:
                question = question + f"Dimension {dimension}: {combined_signal_string}\n"
            
        question = (question + "Class: ").replace("\'", "")
        target = chr(int(y[n]) + 64) #+ " " + y[n]

        generate_graph(X[n], image_filename_path)

        df.loc[n] = [question, target, image_filename_id, image_filename_path]
    

    return df









