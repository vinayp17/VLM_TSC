from aeon.datasets import load_classification

import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
import argparse

from scipy.signal import decimate, medfilt, gaussian
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

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
    else:
        return generate_qwen_vl_entry(question, target, image_filename_id, image_filename_path)

def generate_graph(X, image_filename_path, style="line"):

    plt.figure(figsize=(4,4))

    if style == "line":
      plt.plot(X)
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


def format_numbers_combined(numbers, round_to=None):
    if round_to:
        numbers = [np.round(i, round_to) for i in numbers]

    max_decimal_places = max(len(str(num).split('.')[1]) if '.' in str(num) else 0 for num in numbers)
    intermediate_numbers = [f"{num:.{max_decimal_places}f}" for num in numbers]
    max_length = max(len(num) for num in intermediate_numbers if "-" not in num)
    formatted_numbers = [f"{num:0>{max_length}}" for num in intermediate_numbers] # formatted_numbers = [f"+{num:0>{max_length}}" if "-" not in num else f"{num:0>{max_length}}" for num in intermediate_numbers]

    return formatted_numbers

def generate_data(X, y, image_path, data_path, split, model, round_to, downsample_to):

    entries = []

    for n in range(0, len(y)):
        image_filename_id = f"image_{split}_{n}"
        image_filename_path = f"{image_path}/image_{split}_{n}.png"

        combined_signal_string = format_numbers_combined(downsample(X[n][0], factor=downsample_to), round_to=round_to)
        question = f"Which class is the following signal from? {combined_signal_string}".replace("\'", "")
        target = str(y[n])

        entries.append(generate_data_entry(split, model, question, target, image_filename_id, image_filename_path))
        generate_graph(X[n][0], image_filename_path)

    return entries



# def generate_ucr_data(dataset, image_path, data_path, model, round_to, downsample_to, extract_path='~/Downloads/UCRArchive_2018/'):
#     X, y, meta = load_classification(dataset, return_metadata=True)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(len(X))
    # generate_data(X_train, y_train, image_path, data_path, "train", model, round_to, downsample_to)
    # generate_data(X_test, y_test, image_path, data_path, "test", model, round_to, downsample_to)


from multiprocessing import Pool

# Define process_chunk at the module level
def process_chunk(chunk_data, image_path, data_path, model, round_to, downsample_to):
    data_subset, label_subset, subset_name = chunk_data
    return generate_data(data_subset, label_subset, image_path, data_path, subset_name, model, round_to, downsample_to)

def generate_ucr_data(dataset, image_path, data_path, model, round_to, downsample_to, extract_path='~/Downloads/UCRArchive_2018/'):
    X, y, meta = load_classification(dataset, return_metadata=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating chunks
    chunk_size = len(X_train)//5  # Adjust based on your data
    train_chunks = [(X_train[i:i + chunk_size], y_train[i:i + chunk_size], 'train') for i in range(0, len(X_train), chunk_size)]
    test_chunks = [(X_test[i:i + chunk_size], y_test[i:i + chunk_size], 'test') for i in range(0, len(X_test), chunk_size)]
    all_chunks = train_chunks + test_chunks

    # Using Pool to parallelize

    with Pool() as pool:
        # Create a partial function to include additional arguments
        from functools import partial
        func = partial(process_chunk, image_path=image_path, data_path=data_path, model=model, round_to=round_to, downsample_to=downsample_to)
        results = pool.map(func, all_chunks)

    flattened_results = [item for sublist in results for item in sublist]

        

# Rest of your code...



def main():
    parser = argparse.ArgumentParser(description='Generate UCR Data')
    parser.add_argument('dataset', type=str, help='Name of the dataset')
    parser.add_argument('image_path', type=str, help='Path to save images')
    parser.add_argument('data_path', type=str, help='Path to save data')
    parser.add_argument('model', type=str, help='Model to use')
    parser.add_argument('--round_to', type=int, default=None, help='Round To')
    parser.add_argument('--downsample_to', type=int, default=None, help='Downsample To')
    parser.add_argument('--extract_path', type=str, default='~/Downloads/UCRArchive_2018/', help='Path to extract data')


    args = parser.parse_args()

    generate_ucr_data(args.dataset, args.image_path, args.data_path, args.model, args.round_to, args.downsample_to, args.extract_path)

if __name__ == '__main__':
    main()







