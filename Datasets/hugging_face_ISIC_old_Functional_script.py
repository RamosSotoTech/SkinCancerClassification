import data
from data import load_dataset

# Load the dataset
dataset = load_dataset("marmal88/skin_cancer")

# Explore the format of the first example in the training set
classes = set(dataset['train']['dx'])
unique_classes = list(classes)
num_classes = len(unique_classes)

localizations = set(dataset['train']['localization'])
unique_localizations = list(localizations)
num_localizations = len(unique_localizations)

# Print the unique classes
print(unique_classes)
print(dataset['train'][0])

# Fit the LabelEncoder on the unique classes (sex, localization, dx)
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
le_localization = LabelEncoder()
le_dx = LabelEncoder()

le_localization.fit(dataset['train']['localization'])
le_dx.fit(dataset['train']['dx'])


import os
from tensorflow.keras.layers import Permute, Reshape, Multiply, GlobalAveragePooling2D, Conv2D

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import numpy as np
import tensorflow as tf

# Get the number of available CPU cores
import multiprocessing
num_cores = multiprocessing.cpu_count()

# Configuring TensorFlow session to use all available cores
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 10)]
        )
    except RuntimeError as e:
        print(e)

from keras.src.utils import to_categorical

tf.experimental.numpy.experimental_enable_numpy_behavior()

from keras import Model
from keras.layers import BatchNormalization, Activation, Dropout
from keras.losses import CategoricalFocalCrossentropy
from keras.optimizers import AdamW
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna_integration import TFKerasPruningCallback
import gc
from tensorflow.keras import backend as K

from sklearn.utils import compute_class_weight

from tensorflow.keras.layers import Dense, Flatten

import matplotlib
matplotlib.use('TkAgg')
import math

image_size = (256, 256)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

import tensorflow_addons as tfa


def adaptive_resize(image, target_size=(224, 224)):
    current_size = tf.cast(tf.shape(image)[:2], tf.float32)
    target_size_float = tf.cast(target_size, tf.float32)

    scale_factor = target_size_float / current_size
    max_scale_factor = tf.reduce_max(scale_factor)

    if max_scale_factor < 1:
        resized_image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.AREA)
    else:
        resized_image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BICUBIC)

    return resized_image


def shift_image(image, width_shift, height_shift):
    original_dims = tf.shape(image)

    padded_image = tf.image.pad_to_bounding_box(image, height_shift, width_shift, original_dims[0] + height_shift,
                                                original_dims[1] + width_shift)

    shifted_image = tf.image.crop_to_bounding_box(padded_image, 0, 0, original_dims[0], original_dims[1])

    return shifted_image


# def preprocess_function(training=False, return_original=False, image_size=(256, 256)):
#     def process_path(file_path, label):
#         img = tf.io.read_file(file_path)
#         img = tf.image.decode_jpeg(img, channels=3)
#
#         if training:
#             img = tf.image.random_flip_left_right(img)
#             img = tf.image.random_flip_up_down(img)
#             img = tf.cast(img, tf.float32)
#         img = adaptive_resize(img, image_size)
#         if training:
#             img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
#             img = tfa.image.rotate(img, tf.random.uniform(shape=[], minval=-math.pi/8.0, maxval=math.pi/8.0), fill_mode='reflect')
#
#         if return_original:
#             original_image = tf.io.read_file(file_path)
#             original_image = tf.image.decode_jpeg(original_image, channels=3)
#             original_image = adaptive_resize(original_image, image_size)
#             return original_image, img, label
#         else:
#             return img, label
#
#     return process_path


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = BatchNormalization()(se)
    se = Activation('relu')(se)
    se = Dense(filters, kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation('sigmoid')(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])
    return x

def cbam_block(input_tensor, ratio=16, kernel_size=7):
    channel = GlobalAveragePooling2D()(input_tensor)
    channel = Dense(channel.shape[-1] // ratio, kernel_initializer='he_normal', use_bias=True)(channel)
    channel = BatchNormalization()(channel)
    channel = Activation('relu')(channel)
    channel = Dense(input_tensor.shape[-1], kernel_initializer='he_normal', use_bias=True)(channel)
    channel = Activation('sigmoid')(channel)

    if K.image_data_format() == 'channels_first':
        channel = Reshape((input_tensor.shape[1], 1, 1))(channel)
    else:  # 'channels_last'
        channel = Reshape((1, 1, input_tensor.shape[-1]))(channel)

    channel_attention = Multiply()([input_tensor, channel])

    spatial = Conv2D(1, kernel_size, padding='same', use_bias=True)(channel_attention)
    spatial = Activation('sigmoid')(spatial)
    output = Multiply()([channel_attention, spatial])

    return output


def reestimate_batch_norm(model, data, num_batches=100):
    """
    Re-estimates the Batch Normalization statistics of the model.

    Parameters:
    - model: The Keras model with Batch Normalization layers.
    - data: The dataset to use for the re-estimation. Should yield batches of (input, target).
    - num_batches: The number of batches from `data` to use for re-estimation.
    """

    def reset_bn_layers(model):
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.moving_mean.assign(layer.moving_mean * 0)
                layer.moving_variance.assign(layer.moving_variance * 0 + 1)
                if layer.scale:
                    layer.gamma.assign(layer.gamma * 0 + 1)
                if layer.center:
                    layer.beta.assign(layer.beta * 0)
        return model

    # Reset BN layers
    model = reset_bn_layers(model)

    # Iterate over the specified number of batches and update BN statistics
    for i, (images, _) in enumerate(data.take(num_batches)):
        # Set training=True to update BN statistics
        _ = model(images, training=True)

    print(f"BatchNorm statistics updated using {num_batches} batches.")


class BestValueTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestValueTracker, self).__init__()
        self.best_val_accuracy = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get("val_accuracy")
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.best_epoch = epoch

from collections import defaultdict
import random
# def get_train_val_paths(dataset):
#     lesion_group_train = defaultdict(list)
#     for item in dataset['train']:
#         lesion_group_train[item['lesion_id']].append((item['image_path'], item['dx']))
#
#     lesion_group_validation = defaultdict(list)
#     for item in dataset['validation']:
#         lesion_group_validation[item['lesion_id']].append((item['image_path'], item['dx']))
#
#     # Shuffle lesion groups in the training set to ensure a random distribution
#     lesion_ids_train = list(lesion_group_train.keys())
#     random.shuffle(lesion_ids_train)
#
#     train_images_labels = [image_label for lesion_id in lesion_ids_train for image_label in
#                            lesion_group_train[lesion_id]]
#     val_images_labels = [image_label for lesion_group in lesion_group_validation.values() for image_label in
#                          lesion_group]
#
#     return train_images_labels, val_images_labels


age_by_diagnosis = {
    'melanocytic_nevi': 30,
    'basal_cell_carcinoma': 50,
    'vascular_lesions': 40,
    'actinic_keratoses': 60,
    'melanoma': 50,
    'benign_keratosis-like_lesions': 50,
    'dermatofibroma': 40
}


# def preprocess_data(item, image_size=(256, 256), default_age=None):
#     # Image preprocessing
#     image = tf.keras.preprocessing.image.img_to_array(item['image'])
#     image = tf.image.resize(image, image_size)
#     image /= 255.0
#
#     # Additional feature preprocessing
#     sex = tf.cast(tf.equal(item['sex'], 'male'), tf.float32)  # 1 for male, 0 for female
#     localization_index = le_localization.transform([item['localization']])[0]
#     localization = tf.one_hot(localization_index, depth=len(le_localization.classes_))
#
#     # Handle missing age
#     if item['age'] is None or item['age'] == 0:
#         age = default_age
#     else:
#         age = item['age']
#
#     age /= 100.0  # Normalize age
#
#     # Combine additional features into a single tensor
#     sex = tf.cast(sex, tf.float32)
#     localization = tf.cast(localization, tf.float32)
#     age = tf.cast(age, tf.float32)
#
#     # Now you can concatenate them without a datatype issue
#     additional_features = tf.concat([tf.reshape(sex, (-1,)), localization, tf.reshape(age, (-1,))], axis=0)
#
#     # Encode the 'dx' label using le_dx
#     label_encoded = tf.cast(le_dx.transform([item['dx']])[0], tf.int32)
#     label_hot = tf.one_hot(label_encoded, depth=len(le_dx.classes_))
#
#     return image, additional_features, label_hot
#
#
# def create_tf_datasets(dataset, image_size=(256, 256), batch_size=32):
#     # Function to generate batches of data
#
#     train_ages = [item['age'] for item in dataset['train'] if item['age'] is not None]
#     median_age = np.median(train_ages)
#
#     def gen(ds):
#         for item in ds:
#             image_tensor, metadata_tensor, label_int = preprocess_data(item, image_size, default_age=median_age)
#             yield (image_tensor, metadata_tensor), label_int
#
#     unique_localizations = set(
#         [item['localization'] for item in dataset['train']] + [item['localization'] for item in dataset['validation']])
#     # Correct metadata_input_shape
#     metadata_input_shape = (len(unique_localizations) + 2,)
#
#     # Create TensorFlow data
#     train_ds = tf.data.Dataset.from_generator(
#         lambda: gen(dataset['train']),
#         output_signature=(
#             (
#                 tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
#                 tf.TensorSpec(shape=metadata_input_shape, dtype=tf.float32)
#             ),
#             tf.TensorSpec(shape=(len(le_dx.classes_),), dtype=tf.int32)
#         )
#     ).map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
#
#     val_ds = tf.data.Dataset.from_generator(
#         lambda: gen(dataset['validation']),
#         output_signature=(
#             (
#                 tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
#                 tf.TensorSpec(shape=metadata_input_shape, dtype=tf.float32)
#             ),
#             tf.TensorSpec(shape=(len(le_dx.classes_),), dtype=tf.int32)
#         )
#     ).map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
#     # Batch and prefetch
#     train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#     return train_ds, val_ds

localization_lookup_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(unique_localizations),
        values=tf.constant(list(range(num_localizations)), dtype=tf.int64),
    ),
    default_value=-1  # Use -1 or any appropriate value for unknown classes
)

dx_lookup_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(unique_classes),
        values=tf.constant(list(range(num_classes)), dtype=tf.int64),
    ),
    default_value=-1
)

def preprocess_metadata(sex, age, localization):
    sex = tf.where(tf.equal(sex, "male"), 1.0, 0.0)
    age = tf.cast(age, tf.float32) / 100.0 if age is not None else 0.5
    localization_tensor = tf.convert_to_tensor(localization)
    localization_index = localization_lookup_table.lookup(localization_tensor)
    localization = tf.one_hot(localization_index, depth=num_localizations)

    return tf.concat([[sex, age], localization], axis=0)

def tf_preprocess(sample):
    # Define how to preprocess a sample here
    img = sample['image']  # Or any preprocessing you need to apply
    metadata = preprocess_metadata(sample['sex'], sample['age'], sample['localization'])
    label = sample['dx']  # Or preprocess the label as necessary

    # Return the final processed sample...
    return img, metadata, label

from PIL import Image
from io import BytesIO


def preprocess_function(training=False, return_original=False, image_size=(256, 256)):
    def process_data(features):
        # Convert numpy array to TensorFlow Tensor and normalize pixel values
        img = features['image']

        if return_original:
            original_img = adaptive_resize(img, image_size)

        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.cast(img, tf.float32)
        img = adaptive_resize(img, image_size)
        if training:
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            img = tfa.image.rotate(img, tf.random.uniform(shape=[], minval=-math.pi/8.0, maxval=math.pi/8.0), fill_mode='reflect')

        # Metadata processing (similar to your preprocess_metadata function)

        metadata = preprocess_metadata(features['sex'], features['age'], features['localization'])

        # Label processing
        dx_tensor = tf.convert_to_tensor(features['dx'])
        dx_index = dx_lookup_table.lookup(dx_tensor)
        label_hot = tf.one_hot(dx_index, depth=num_classes)

        if return_original:
            return (original_img, img, metadata), label_hot
        else:
            return (img, metadata), label_hot

    return process_data

def to_tf_dataset(hf_dataset):
    # Create a TensorFlow dataset directly from the Hugging Face dataset
    return tf.data.Dataset.from_generator(
        lambda: ({
            'image': tf.keras.preprocessing.image.img_to_array(sample['image']),
            'sex': sample['sex'],
            'age': sample['age'],
            'localization': sample['localization'],
            'dx': sample['dx']
        } for sample in hf_dataset),
        output_types={
            'image': tf.uint8,
            'sex': tf.string,
            'age': tf.float32,
            'localization': tf.string,
            'dx': tf.string
        },
        output_shapes={
            'image': tf.TensorShape([None, None, 3]),  # Flexible dimensions for the image
            'sex': tf.TensorShape([]),
            'age': tf.TensorShape([]),
            'localization': tf.TensorShape([]),
            'dx': tf.TensorShape([])
        }
    )

def create_tf_datasets(image_size=(256, 256), batch_size=32):
    hf_dataset = load_dataset("marmal88/skin_cancer")

    # Define preprocessing functions
    train_preprocess = preprocess_function(training=True, image_size=image_size)
    val_preprocess = preprocess_function(training=False, image_size=image_size)

    # Convert Hugging Face data to TensorFlow data
    train_dataset = to_tf_dataset(hf_dataset['train'])
    val_dataset = to_tf_dataset(hf_dataset['validation'])

    # Apply preprocessing with parallelization
    train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch, shuffle, and prefetch
    train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset
# def create_tf_datasets(image_size=(256, 256), batch_size=32):
#     hf_dataset = load_dataset("marmal88/skin_cancer")
#
#     # Convert Hugging Face data to TensorFlow data
#     train_dataset = to_tf_dataset(hf_dataset['train'])
#     val_dataset = to_tf_dataset(hf_dataset['validation'])
#
#     # Apply preprocessing with parallelization
#     train_dataset = train_dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#     val_dataset = val_dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#
#     # Batch, shuffle, and prefetch
#     train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#     return train_dataset, val_dataset

    # # Load the dataset
    # hf_dataset = load_dataset("marmal88/skin_cancer")
    #
    # # Define the preprocessing function with the specified parameters
    # train_preprocess = preprocess_function(training=True, image_size=image_size)
    # val_preprocess = preprocess_function(training=False, image_size=image_size)
    #
    # # Convert Hugging Face dataset to TensorFlow dataset
    # def hf_to_tf_dataset(hf_dataset, preprocess_fn):
    #     def gen():
    #         for sample in hf_dataset:
    #             yield preprocess_fn(sample)
    #
    #     return tf.data.Dataset.from_generator(
    #         gen,
    #         output_signature=(
    #             (
    #                 tf.TensorSpec(shape=image_size + (3,), dtype=tf.float32),  # Adjust depending on your preprocess_function
    #                 tf.TensorSpec(shape=[None], dtype=tf.float32)  # Adjust for metadata
    #             ),
    #             tf.TensorSpec(shape=[None], dtype=tf.float32)  # Adjust for labels
    #         )
    #     )
    #
    # # Apply preprocessing and create TensorFlow data
    # train_dataset = hf_to_tf_dataset(hf_dataset['train'], train_preprocess)
    # val_dataset = hf_to_tf_dataset(hf_dataset['validation'], val_preprocess)
    #
    # # Batch, shuffle, and prefetch the data
    # train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #
    # return train_dataset, val_dataset

    # lesion_group_train = defaultdict(list)
    # lesion_group_validation = defaultdict(list)
    #
    # # Note the order: image, sex, age, localization, label (dx)
    # for item in dataset['train']:
    #     lesion_group_train[item['lesion_id']].append(
    #         (item['image'], item['sex'], item['age'], item['localization'], item['dx']))
    #
    # for item in dataset['validation']:
    #     lesion_group_validation[item['lesion_id']].append(
    #         (item['image'], item['sex'], item['age'], item['localization'], item['dx']))
    #
    # # Select a random sample from each lesion group
    # train_samples = [random.choice(images_labels) for images_labels in lesion_group_train.values()]
    # val_samples = [random.choice(images_labels) for images_labels in lesion_group_validation.values()]
    #
    # # Unpack the samples, ensuring the order matches how they were appended
    # train_images, train_sexes, train_ages, train_localizations, train_labels = zip(*train_samples)
    # val_images, val_sexes, val_ages, val_localizations, val_labels = zip(*val_samples)
    #
    # # Create TensorFlow data
    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (train_images, train_sexes, train_ages, train_localizations, train_labels))
    # val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_sexes, val_ages, val_localizations, val_labels))
    #
    # # Define preprocessing functions
    # training_preprocess_fn = preprocess_function(training=True, return_original=False, image_size=image_size)
    # validation_preprocess_fn = preprocess_function(training=False, return_original=False, image_size=image_size)
    #
    # # Map the preprocessing functions
    # train_ds = train_ds.map(lambda img, sex, age, loc, lbl: training_preprocess_fn(img, sex, age, loc, lbl),
    #                         num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.map(lambda img, sex, age, loc, lbl: validation_preprocess_fn(img, sex, age, loc, lbl),
    #                     num_parallel_calls=tf.data.AUTOTUNE)
    #
    # # Batch and prefetch for optimal performance
    # train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #
    # return train_ds, val_ds

def objective(trial):
    batch_size = trial.suggest_int('batch', 16, 64)
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 32, 1024) for i in range(num_dense_layers)]
    batch_normalization_layers = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                                  range(num_dense_layers)]
    dropout_rate_layers = [trial.suggest_float(f'dropout_rate_{i}', 0.3, 0.7) for i in range(num_dense_layers)]
    loss_type = trial.suggest_categorical('loss_type', ['focal_loss', 'categorical_crossentropy'])
    alpha_phase1 = trial.suggest_float('alpha_phase1', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase1 = trial.suggest_float('gamma_phase1', 1.0, 5.0) if loss_type == 'focal_loss' else None
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    weight_decay_phase1 = trial.suggest_float('weight_decay_phase1', 1e-6, 1e-2, log=True)
    learning_rate_phase1 = trial.suggest_float('learning_rate_phase1', 1e-5, 1e-3, log=True)
    learning_rate_phase2 = trial.suggest_float('learning_rate_phase2', 1e-6, 1e-3, log=True)
    alpha_phase2 = trial.suggest_float('alpha_phase2', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase2 = trial.suggest_float('gamma_phase2', 1.0, 5.0) if loss_type == 'focal_loss' else None
    weight_decay_phase2 = trial.suggest_float('weight_decay_phase2', 1e-6, 1e-2, log=True)
    base_model_architecture = trial.suggest_categorical('base_model_architecture', ['VGG16', 'VGG19', 'ResNet101V2',
                                                                                    'InceptionResNetV2',
                                                                                    'Xception', 'MobileNetV2'])
    attention_mechanism = trial.suggest_categorical('attention_mechanism', ['SENet', 'CBAM', 'None'])
    use_amsgrad = trial.suggest_categorical('use_amsgrad', [True, False])
    pre_trained_weights = trial.suggest_categorical('pre_trained_weights', [True, False])


    weight = 'imagenet' if pre_trained_weights else None
    # Define the base model
    if base_model_architecture == 'VGG16':
        base_model = tf.keras.applications.VGG16(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'VGG19':
        base_model = tf.keras.applications.VGG19(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'ResNet101V2':
        base_model = tf.keras.applications.ResNet101V2(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'InceptionResNetV2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'Xception':
        base_model = tf.keras.applications.Xception(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weight, input_shape=image_size + (3,))
    else:
        raise ValueError(f"Invalid base model architecture: {base_model_architecture}")

    # unfreeze layers for phase 2 (1 to total layers)
    if pre_trained_weights:
        num_layers_unfreeze = trial.suggest_int('num_layers_unfreeze', 1, len(base_model.layers))
        for layer in base_model.layers[-num_layers_unfreeze:]:
            layer.trainable = False
    else:
        num_layers_unfreeze = len(base_model.layers)
        for layer in base_model.layers:
            layer.trainable = True

    def print_params():
        print(f"{batch_size=}")
        print(f"{num_dense_layers=}")
        print(f"{dense_units=}")
        print(f"{batch_normalization_layers=}")
        print(f"{dropout_rate_layers=}")
        print(f"{loss_type=}")
        print(f"{alpha_phase1=}")
        print(f"{gamma_phase1=}")
        print(f"{use_class_weights=}")
        print(f"{weight_decay_phase1=}")
        print(f"{learning_rate_phase1=}")
        print(f"{learning_rate_phase2=}")
        print(f"{alpha_phase2=}")
        print(f"{gamma_phase2=}")
        print(f"{weight_decay_phase2=}")
        print(f"{base_model_architecture=}")
        print(f"{attention_mechanism=}")
        print(f"{use_amsgrad=}")
        print(f"{num_layers_unfreeze=}")
        print(f"{pre_trained_weights=}")

    print_params()

    attention_func = None
    if attention_mechanism == 'SENet':
        attention_func = squeeze_excite_block
    elif attention_mechanism == 'CBAM':
        attention_func = cbam_block

    # Prepare the dataset

    # Model definition
    metadata_input_shape = (len(dataset['train'][0]['localization']) + 2,)
    metadata_input = tf.keras.layers.Input(shape=metadata_input_shape, name='metadata_input')
    image_input = tf.keras.layers.Input(shape=image_size + (3,), name='image_input')

    # Apply the base model
    x = base_model(image_input)

    if pre_trained_weights:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True

    # Apply the attention mechanism if any
    if attention_func:
        x = attention_func(x)

    x = Flatten()(x)
    x = tf.keras.layers.concatenate([x, metadata_input])

    for i in range(num_dense_layers):
        x = Dense(dense_units[i])(x)
        if batch_normalization_layers[i]:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate_layers[i])(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=[image_input, metadata_input], outputs=predictions)

    train_ds, val_ds = create_tf_datasets(image_size=image_size, batch_size=batch_size)

    # Phase 1
    opt = AdamW(weight_decay=weight_decay_phase1, learning_rate=learning_rate_phase1, amsgrad=use_amsgrad)
    if loss_type == 'focal_loss':
        loss_fn = CategoricalFocalCrossentropy(alpha=alpha_phase1, gamma=gamma_phase1)
        class_weights = None
    else:
        loss_fn = 'categorical_crossentropy'
        if use_class_weights:
            train_labels = [item['dx'] for item in dataset['train']]
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = dict(enumerate(weights))
        else:
            class_weights = None

    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    reestimate_batch_norm(model, train_ds, num_batches=100)

    model.fit(train_ds
              , validation_data=val_ds
              , epochs=3
              , class_weight=class_weights
              , callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=2)
                            ]
              , workers=num_cores
              , use_multiprocessing=True
              )

    # Phase 2
    opt = AdamW(weight_decay=weight_decay_phase2, learning_rate=learning_rate_phase2, amsgrad=use_amsgrad, clipnorm=1.0)
    if loss_type == 'focal_loss':
        loss_fn = CategoricalFocalCrossentropy(alpha=alpha_phase2, gamma=gamma_phase2)
        class_weights = None
    else:
        loss_fn = 'categorical_crossentropy'
        if use_class_weights:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = dict(enumerate(weights))
        else:
            class_weights = None
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    for layer in base_model.layers[-num_layers_unfreeze:]:
        layer.trainable = True
    # Re-estimate BatchNorm statistics
    reestimate_batch_norm(model, train_ds, num_batches=1000)

    best_value_tracker = BestValueTracker()

    try:
        model.fit(train_ds
                  , validation_data=val_ds
                  , epochs=100
                  , class_weight=class_weights
                  , callbacks=[
                               TFKerasPruningCallback(trial, 'val_accuracy'),
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,
                                                                start_from_epoch=70),
                               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2),
                                  best_value_tracker
                               ]
                    , workers=num_cores
                    , use_multiprocessing=True
                  )
    except optuna.exceptions.TrialPruned as e:
        best_accuracy = best_value_tracker.best_val_accuracy
        best_epoch = best_value_tracker.best_epoch
        print(f"Best validation accuracy before pruning: {best_accuracy} at epoch {best_epoch}")
        if best_accuracy < 0.50:
            # Prune if best accuracy is less than 60%
            raise e
        else:
            return best_accuracy
    # Evaluate
    accuracy = model.evaluate(val_ds, return_dict=True)['accuracy']
    print(f"accuracy: {accuracy}")

    return accuracy

if __name__ == '__main__':
    # optuna callbacks to free up keras backend memory
    def clear_session_callback(study, trial):
        K.clear_session()
        gc.collect()

    study_name = 'skin_lesion_classification_with_HAM10000_dataset'
    storage_name = f"sqlite:///{study_name}.db"

    from optuna.pruners import PercentilePruner

    # optuna.delete_study(study_name, storage_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True,
                                # add Patient pruner with Median pruner as the base pruner
                                pruner=optuna.pruners.PatientPruner(
                                    wrapped_pruner=PercentilePruner(percentile=50, n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
                                    , patience=10, min_delta=0.0001))

    study.optimize(objective, n_trials=1, callbacks=[clear_session_callback])

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)