from aeon.datasets import load_classification

import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
import argparse

from scipy.signal import decimate, medfilt, gaussian
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

from utils import format_numbers_combined, downsample, generate_graph, generate_data, generate_data_entry
from multiprocessing import Pool

import pandas as pd

import yaml


def process_chunk(chunk_data, image_path, max_precision, padded, round_to, downsample_to):
    data_subset, label_subset, index_subset, = chunk_data
    return generate_data(data_subset, label_subset, index_subset, image_path, max_precision, padded, round_to, downsample_to)

class UCRDataSet():
    def __init__(self, dataset, image_path, data_path):
        self.dataset = dataset
        self.image_path = image_path
        self.data_path  = data_path

        self.round_to = None
        self.downsample_to = None
        self.padded = False

        self.X, self.y, meta = load_classification(dataset, return_metadata=True)

    def multiprocessing(self, X, y):
        max_precision = max(len(str(num).split('.')[1]) if '.' in str(num) else 0 for num in X.flatten().tolist())

        chunk_size = len(X)//6

        chunks = [(X[i:i + chunk_size], y[i:i + chunk_size], range(i, i+chunk_size, 1)) for i in range(0, len(X), chunk_size)]

        with Pool() as pool:
            from functools import partial
            func = partial(process_chunk, image_path=self.image_path, max_precision=max_precision, padded=self.padded, round_to=self.round_to, downsample_to=self.downsample_to)
            results = pool.map(func, chunks)

        return pd.concat(results, axis=0)
    
    def generate_data_splits(self, model):
        # mean = np.mean(self.X, axis=0)
        # std = np.std(self.X, axis=0)
        # self.X = (self.X - mean) / std

        df = self.multiprocessing(self.X, self.y)

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_entries = train_set.apply(lambda row: generate_data_entry('train', model, row['question'], row['target'], row['image_filename_id'], row['image_filename_path']), axis=1).tolist()
        test_entries = test_set.apply(lambda row: generate_data_entry('test', model, row['question'], row['target'], row['image_filename_id'], row['image_filename_path']), axis=1).tolist()

        json_output = json.dumps(train_entries, indent=2)
        with open(f'{self.data_path}/train.json', 'w') as file:
            file.write(json_output)

        with open(f'{self.data_path}/test.jsonl', 'w') as file:
            for dictionary in test_entries:
                json_string = json.dumps(dictionary)
                file.write(json_string + '\n')

        test_entries_for_eval = test_set.apply(lambda row: generate_data_entry('train', model, row['question'], row['target'], row['image_filename_id'], row['image_filename_path']), axis=1).tolist()
        json_output = json.dumps(test_entries_for_eval[:10], indent=2)
        with open(f'{self.data_path}/test.json', 'w') as file:
            file.write(json_output)


def main():
    # parser = argparse.ArgumentParser(description='Generate UCR Data')
    # parser.add_argument('dataset', type=str, help='Name of the dataset')
    # parser.add_argument('image_path', type=str, help='Path to save images')
    # parser.add_argument('data_path', type=str, help='Path to save data')
    # parser.add_argument('model', type=str, help='Model to use')
    # parser.add_argument('--padded', type=bool, default=False, help='Pad Numbers')
    # parser.add_argument('--round_to', type=int, default=None, help='Round To')
    # parser.add_argument('--downsample_to', type=int, default=None, help='Downsample To')

    # args = parser.parse_args()

    # dataset = UCRDataSet(args.dataset, args.image_path, args.data_path)
    # dataset.round_to = args.round_to
    # dataset.downsample_to = args.downsample_to
    # dataset.padded = args.padded

    
    # dataset.generate_data_splits(model=args.model)

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dataset_name = config['dataset']['name']
    image_path = config['image_path']
    data_path = config['data_path']
    model = config['model']
    padded = config['options']['padded']
    round_to = config['options']['round_to']
    downsample_to = config['options']['downsample_to']

    dataset = UCRDataSet(dataset_name, image_path, data_path)
    dataset.round_to = round_to
    dataset.downsample_to = downsample_to
    dataset.padded = padded

    dataset.generate_data_splits(model=model)


if __name__ == '__main__':
    main()
