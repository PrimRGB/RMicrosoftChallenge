from random import random
import pandas as pd
from .columns_types import dtypes
import random

train_file_path = "../data/train.csv"
test_file_path = "../data/test.csv"
target_column_name = "HasDetections"


def _load_data(data_file_name, dtypes, sample_size):
    data = pd.read_csv(
        data_file_name,
        header=0,
        skiprows=lambda i: i>0 and random.random() > sample_size,
        dtype=dtypes
    )
    return data


def get_train_data(sample_size=0.1):
    data = _load_data(train_file_path, dtypes, sample_size)
    target = data[target_column_name]
    data = data.drop(target_column_name, axis=1)
    return data, target


def get_real_test_data(sample_size=0.1):
    data = _load_data(test_file_path, dtypes, sample_size)
    return data

