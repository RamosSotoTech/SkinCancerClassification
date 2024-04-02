# src/training/utils.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))

import tensorflow as tf
import tensorflow_addons as tfa
from typing import Tuple
import warnings
from importlib import import_module

# from src.data.dataset import get_default_localization, get_default_classes
import math


@tf.function
def adaptive_resize(image: tf.Tensor, target_size: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    current_size = tf.cast(tf.shape(image)[:2], tf.float32)
    target_size_float = tf.cast(target_size, tf.float32)
    max_scale_factor = tf.reduce_max(target_size_float / current_size)

    resized_image = tf.cond(max_scale_factor < 1,
                            lambda: resize_area(image, target_size),
                            lambda: resize_bicubic(image, target_size))

    return resized_image


@tf.function
def resize_area(image: tf.Tensor, target_size: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.AREA)


@tf.function
def resize_bicubic(image: tf.Tensor, target_size: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BICUBIC)


@tf.function
def process_training(image, image_size=(256, 256)):
    img = tf.image.random_flip_left_right(image)
    img = tf.image.random_flip_up_down(img)
    img = tf.cast(img, tf.float32)
    img = adaptive_resize(img, image_size)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tfa.image.rotate(img, tf.random.uniform(shape=[], minval=-math.pi / 8.0, maxval=math.pi / 8.0),
                           fill_mode='reflect')
    return img


@tf.function
def process_validation(image, image_size=(256, 256)):
    img = adaptive_resize(image, image_size)
    return img


def preprocess_fn(image_size=(256, 256), unique_localizations=None, unique_classes=None):
    if unique_localizations is None or len(unique_localizations) == 0:
        warnings.warn(
            "Unique_localization was not provided or was empty. The program will continue, Using the default "
            "localization set.",
            UserWarning)
        module = import_module('src.data.dataset')
        if module is None:
            raise ImportError("Module src.data.dataset not found.")
        if not hasattr(module, 'get_default_localization'):
            raise AttributeError("Function get_default_localization not found in module src.data.dataset.")
        get_default_localization = getattr(module, 'get_default_localization')
        unique_localizations = get_default_localization()
        warnings.warn(f"Using the default localization set: {unique_localizations}", UserWarning)

    if unique_classes is None or len(unique_classes) == 0:
        warnings.warn(
            "Unique_classes was not provided or was empty. The program will continue, Using the default classes set.",
            UserWarning)
        module = import_module('src.data.dataset')
        if module is None:
            raise ImportError("Module src.data.dataset not found.")
        if not hasattr(module, 'get_default_classes'):
            raise AttributeError("Function get_default_classes not found in module src.data.dataset.")
        get_default_classes = getattr(module, 'get_default_classes')
        unique_classes = get_default_classes()
        warnings.warn(f"Using the default classes set: {unique_classes}", UserWarning)

    if image_size is None or not isinstance(image_size, tuple):
        warnings.warn(
            "Image_size was not provided or was not a tuple of length 2. The program will continue, Using the default "
            "image size.",
            UserWarning)
        image_size = (256, 256)

    num_localizations = len(unique_localizations)
    num_classes = len(unique_classes)

    @tf.function
    def preprocess_image_and_metadata(features, labels, training):
        image = features['image']

        processed_images = tf.cond(
            tf.equal(training, True),
            lambda: process_training(image, image_size),
            lambda: process_validation(image, image_size)
        )

        localization_one_hot = tf.one_hot(features['localization'], depth=num_localizations)

        dx_one_hot = tf.one_hot(labels, depth=num_classes)

        sex = tf.expand_dims(tf.cast(features['sex'], tf.float32), -1)
        age = tf.expand_dims(tf.cast(features['age'], tf.float32), -1)

        metadata_input = tf.concat([localization_one_hot, sex, age], axis=-1)

        processed_features = {
            'image_input': processed_images,
            'metadata_input': metadata_input
        }
        return processed_features, dx_one_hot

    return preprocess_image_and_metadata
