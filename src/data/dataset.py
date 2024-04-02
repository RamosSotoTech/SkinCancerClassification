# src/data/dataset.py
import tensorflow as tf
from typing import Tuple
import numpy as np
import pandas as pd
import math
from datasets import DatasetDict, Dataset, load_dataset
import functools
import warnings

from src.training.utils import preprocess_fn

huggingface_path = "marmal88/skin_cancer"


@functools.lru_cache(maxsize=1)
def get_dataset_dict():
    return load_dataset(huggingface_path)


@functools.lru_cache(maxsize=1)
def get_train_split():
    """Get the raw train data."""
    return load_dataset(huggingface_path, split='train')


@functools.lru_cache(maxsize=1)
def get_test_split():
    """Get the raw test data."""
    return load_dataset(huggingface_path, split='test')


@functools.lru_cache(maxsize=1)
def get_val_split():
    """Get the raw validation data."""
    return load_dataset(huggingface_path, split='validation')


def get_default_localization():
    return set(get_train_split()['localization'])


def get_default_classes():
    return set(get_train_split()['dx'])


def create_tf_datasets(dataset: DatasetDict, image_size: Tuple[int, int], batch_size: int) -> Tuple[Dataset, Dataset]:
    if dataset is None:
        dataset = []
        dataset['train'] = get_train_split()
        dataset['validation'] = get_val_split()
        warnings.warn("Dataset was not provided. Using the default dataset.", UserWarning)

    ages = np.concatenate([dataset['train']['age'], dataset['validation']['age']])
    ages = [age for age in ages if age is not None]
    age_mean, age_std = np.mean(ages), np.std(ages)

    all_localizations = dataset['train']['localization'] + dataset['validation']['localization']

    all_dx = dataset['train']['dx'] + dataset['validation']['dx']

    unique_localizations = pd.unique(all_localizations)
    unique_dx = pd.unique(all_dx)

    localization_to_code = {loc: code for code, loc in enumerate(unique_localizations)}
    dx_to_code = {dx: code for code, dx in enumerate(unique_dx)}

    def pre_tensor_transform(sample):
        sample['age'] = age_mean if sample['age'] is None else sample['age']
        sample['sex'] = int(sample['sex'] == 'male')
        sample['localization'] = localization_to_code[sample['localization']]
        sample['dx'] = dx_to_code[sample['dx']]
        return sample

    def filter_None_Nan_samples(example):
        for value in example.values():
            if value is None:
                return False
            elif isinstance(value, float) and math.isnan(value):
                return False
        return True

    train_ds = dataset['train'].map(pre_tensor_transform).filter(filter_None_Nan_samples)
    val_ds = dataset['validation'].map(pre_tensor_transform).filter(filter_None_Nan_samples)

    train_tf_ds = train_ds.to_tf_dataset(columns=['image', 'sex', 'age', 'localization', 'dx'], label_cols=['dx'],
                                         shuffle=True, batch_size=batch_size)
    val_tf_ds = val_ds.to_tf_dataset(columns=['image', 'sex', 'age', 'localization', 'dx'], label_cols=['dx'],
                                     shuffle=False, batch_size=batch_size)

    def wrap_preprocess_image_and_metadata(training):
        preprocess_function = preprocess_fn(image_size, unique_localizations, unique_dx)
        return lambda features, labels: preprocess_function(features, labels, training)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_tf_ds = train_tf_ds.with_options(options).map(wrap_preprocess_image_and_metadata(training=True),
                                                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_tf_ds = val_tf_ds.with_options(options).map(wrap_preprocess_image_and_metadata(training=False),
                                                    num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(
        tf.data.AUTOTUNE)

    return train_tf_ds, val_tf_ds

def create_test_dataset(dataset: DatasetDict, image_size: Tuple[int, int], batch_size: int) -> Dataset:
    if dataset is None or 'test' not in dataset:
        raise ValueError("Dataset must be provided and must contain a 'test' split.")

    ages = np.concatenate([dataset['train']['age'], dataset['validation']['age'], dataset['test']['age']])
    ages = [age for age in ages if age is not None]
    age_mean, age_std = np.mean(ages), np.std(ages)

    all_localizations = dataset['train']['localization'] + dataset['validation']['localization'] + dataset['test']['localization']

    all_dx = dataset['train']['dx'] + dataset['validation']['dx'] + dataset['test']['dx']

    unique_localizations = pd.unique(all_localizations)
    unique_dx = pd.unique(all_dx)

    localization_to_code = {loc: code for code, loc in enumerate(unique_localizations)}
    dx_to_code = {dx: code for code, dx in enumerate(unique_dx)}

    def pre_tensor_transform(sample):
        sample['age'] = age_mean if sample['age'] is None else sample['age']
        sample['sex'] = int(sample['sex'] == 'male')
        sample['localization'] = localization_to_code[sample['localization']]
        sample['dx'] = dx_to_code[sample['dx']]
        return sample

    def filter_None_Nan_samples(example):
        for value in example.values():
            if value is None:
                return False
            elif isinstance(value, float) and math.isnan(value):
                return False
        return True

    test_ds = dataset['test'].map(pre_tensor_transform).filter(filter_None_Nan_samples)

    test_tf_ds = test_ds.to_tf_dataset(columns=['image', 'sex', 'age', 'localization', 'dx'], label_cols=['dx'],
                                       shuffle=False, batch_size=batch_size)

    def wrap_preprocess_image_and_metadata(training):
        preprocess_function = preprocess_fn(image_size, unique_localizations, unique_dx)
        return lambda features, labels: preprocess_function(features, labels, training)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    test_tf_ds = test_tf_ds.with_options(options).map(wrap_preprocess_image_and_metadata(training=False),
                                                      num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

    return test_tf_ds


def preprocess_dataset(dataset, image_size, batch_size):
    train_ds, val_ds = create_tf_datasets(dataset, image_size=image_size, batch_size=batch_size)
    return train_ds, val_ds


if __name__ == "__main__":
    dataset = get_dataset_dict()
    image_size = (256, 256)
    batch_size = 32
    train_ds, val_ds = create_tf_datasets(dataset, image_size=image_size, batch_size=batch_size)
    print(train_ds)
    print(val_ds)
