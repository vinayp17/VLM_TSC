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

from preprocess import load_birds
import sys


def process_chunk(chunk_data, image_path,  round_to, downsample_to):
    data_subset, label_subset, index_subset, = chunk_data
    return generate_data(data_subset, label_subset, index_subset, image_path, round_to, downsample_to)

class UCRDataSet():
    def __init__(self, dataset, image_path, data_path):
        self.dataset = dataset
        self.image_path = image_path
        self.data_path  = data_path

        self.round_to = None
        self.downsample_to = None
        self.padded = False

        if dataset == "birds":
            self.X, self.y = load_birds('../birds.arff')
        else:
            self.X, self.y, meta = load_classification(dataset, return_metadata=True)

        if len(self.X.shape) == 2:
            self.X = self.X.reshape(self.X.shape[0], 1, self.X.shape[1])

        self.y[self.y == '0'] = '2'
        self.y[self.y == '-1'] = '2'


    def multiprocessing(self, X, y, split):
        chunk_size = len(X)//6

        chunks = [(X[i:i + chunk_size], y[i:i + chunk_size], [f'{split}_{f}' for f in range(i, i+chunk_size, 1)]) for i in range(0, len(X), chunk_size)]

        with Pool() as pool:
            from functools import partial
            func = partial(process_chunk, image_path=self.image_path, round_to=self.round_to, downsample_to=self.downsample_to)
            results = pool.map(func, chunks)

        return pd.concat(results, axis=0)

    def generate_data_splits(self, model):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=21)

        # Apply multiprocessing to the training set
        train_set = self.multiprocessing(X_train, y_train, split='train')

        # Apply multiprocess to the validation set
        validation_set = self.multiprocessing(X_validation, y_validation, split='validation')

        # Apply multiprocessing to the testing set
        test_set = self.multiprocessing(X_test, y_test, split='test')

        train_entries = train_set.apply(lambda row: generate_data_entry('train', model, row['question'], row['target'], row['image_filename_id'], row['image_filename_path']), axis=1).tolist()
        validation_entries = validation_set.apply(lambda row: generate_data_entry('validation', model, row['question'], row['target'], row['image_filename_id'], row['image_filename_path']), axis=1).tolist()
        test_entries = test_set.apply(lambda row: generate_data_entry('test', model, row['question'], row['target'], row['image_filename_id'], row['image_filename_path']), axis=1).tolist()

        json_output = json.dumps(train_entries, indent=2)
        with open(f'{self.data_path}/train.json', 'w') as file:
            file.write(json_output)

        json_output = json.dumps(validation_entries, indent=2)
        with open(f'{self.data_path}/validation.json', 'w') as file:
            file.write(json_output)

        with open(f'{self.data_path}/test.json', 'w') as file:
            for dictionary in test_entries:
                json_string = json.dumps(dictionary)
                file.write(json_string + '\n')

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

    with open(sys.argv[1], 'r') as file:
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
