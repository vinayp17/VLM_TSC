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
from parameter_finder import compute_downsample_setting_new
from data_templates import DataRepresentation
from multiprocessing import Pool

import pandas as pd

import yaml

from preprocess import load_birds
import sys

def process_chunk(chunk_data, image_path,  round_to, downsample_to, data_repr, split, dataset, use_adaptive_downsampling, plot_downsampled_graph):
    data_subset, label_subset, index_subset, = chunk_data
    return generate_data(dataset, data_subset, label_subset, index_subset, image_path, round_to, downsample_to, data_repr, split, use_adaptive_downsampling, plot_downsampled_graph)

class UCRDataSet():
    def __init__(self, dataset, image_path, data_path, context_length, data_repr):
        self.dataset = dataset
        self.image_path = image_path
        self.data_path  = data_path
        self.context_length = context_length
        self.data_repr = data_repr

        self.round_to = None
        self.downsample_to = None
        self.use_adaptive_downsampling = None
        self.plot_downsampled_graph = None
        self.padded = False

        if dataset == "birds":
            self.X, self.y = load_birds('../birds.arff')
        else:
            self.X, self.y, meta = load_classification(dataset, return_metadata=True)

        if len(self.X.shape) == 2:
            self.X = self.X.reshape(self.X.shape[0], 1, self.X.shape[1])

        self.y[self.y == '0'] = '2'
        self.y[self.y == '-1'] = '2'


    def multiprocessing(self,X, y, split):
        chunk_size = len(X)//6

        chunks = [(X[i:i + chunk_size], y[i:i + chunk_size], [f'{split}_{f}' for f in range(i, i+chunk_size, 1)]) for i in range(0, len(X), chunk_size)]

        with Pool() as pool:
            from functools import partial
            func = partial(process_chunk, image_path=self.image_path, round_to=self.round_to, downsample_to=self.downsample_to, data_repr=self.data_repr,split=split, dataset=self.dataset, use_adaptive_downsampling=self.use_adaptive_downsampling, plot_downsampled_graph=self.plot_downsampled_graph)
            results = pool.map(func, chunks)

        return pd.concat(results, axis=0)

    def generate_data_splits(self, model):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=21)

        #self.downsample_to = compute_downsample_setting(X_train[0], 'liuhaotian/llava-v1.5-7b', self.context_length, self.round_to, 300 if self.data_repr != DataRepresentation.BASELINE else 0, self.data_repr)
        #Remove split as an arg, all splits should get the same data
        '''
        self.downsample_to = compute_downsample_setting_new(self.X[0], self.y[0], self.round_to, self.dataset, "train", self.data_repr, 'liuhaotian/llava-v1.5-7b', self.context_length)
        for i in range(1, len(self.X)):
            potential_downsample = compute_downsample_setting_new(self.X[i], self.y[i], self.round_to, self.dataset, "train", self.data_repr, 'liuhaotian/llava-v1.5-7b', self.context_length)
            if potential_downsample is None:
                continue
            if self.downsample_to is None:
                self.downsample_to = potential_downsample
                continue
            assert self.downsample_to is not None
            assert potential_downsample is not None
            if potential_downsample > self.downsample_to:
                print(f"Switching downsample from:{self.downsample_to} to:{potential_downsample}")
                self.downsample_to = potential_downsample
        '''
        self.downsample_to = compute_downsample_setting_new(X_train[0], y_train[0], self.round_to, self.dataset, "train", self.data_repr, 'liuhaotian/llava-v1.5-7b', self.context_length, self.use_adaptive_downsampling)

        #For a given training sample
        #Check which mode are we in : BaseLine, Rationale, Signal
        #Check what is the maximum token length for the original timeseries in the given mode
        #Based on the above, come up with an appropriate downsampling parameter to use
        #While downsampling, downsample both image and text

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
    with open(sys.argv[1], 'r') as file:
        config = yaml.safe_load(file)

    dataset_name = config['dataset']['name']
    image_path = config['image_path']
    data_path = config['data_path']
    model = config['model']
    padded = config['options']['padded']
    round_to = config['options']['round_to']
    use_adaptive_downsampling = config['options']['use_adaptive_downsampling']
    plot_downsampled_graph = config['options']['plot_downsampled_graph']
    downsample_to = config['options']['downsample_to']
    context_length = config['options']['context_length']
    data_repr = config['options']['data_repr']
    if data_repr == "BASELINE":
        data_repr = DataRepresentation.BASELINE
    elif data_repr == "WITH_RATIONALE":
        data_repr = DataRepresentation.WITH_RATIONALE
    elif data_repr == "WITH_SIGNAL_ANALYSIS":
        data_repr = DataRepresentation.WITH_SIGNAL_ANALYSIS
    elif data_repr == "WITH_STATS":
        data_repr = DataRepresentation.WITH_STATS

    print( data_repr )

    dataset = UCRDataSet(dataset_name, image_path, data_path, context_length, data_repr)
    dataset.round_to = round_to
    dataset.use_adaptive_downsampling = use_adaptive_downsampling
    dataset.plot_downsampled_graph = plot_downsampled_graph
    dataset.downsample_to = downsample_to
    dataset.padded = padded

    dataset.generate_data_splits(model=model)


if __name__ == '__main__':
    main()
