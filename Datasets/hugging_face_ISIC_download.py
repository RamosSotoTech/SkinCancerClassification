# Python built-in modules
import os
import math
import multiprocessing

# Third-party modules
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, DatasetDict
import matplotlib
import optuna
import gc

# Annotation Libraries
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

# TensorFlow and Keras
# set the environment variables

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf

num_cores = multiprocessing.cpu_count()

# Configuring TensorFlow session to use all available cores
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 20)]
        )
    except RuntimeError as e:
        print(e)

tf.experimental.numpy.experimental_enable_numpy_behavior()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

from tensorflow.keras.layers import (Permute, Reshape, Multiply, Conv2D, Dense, Flatten,
                                     BatchNormalization, Activation, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from tensorflow.keras.applications import VGG16, VGG19, ResNet101V2, InceptionResNetV2, Xception, MobileNetV2
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K

# Supporting libraries
import tensorflow_addons as tfa
from sklearn.utils import compute_class_weight

# Set Global Variables
# Load the dataset
dataset: Dataset = load_dataset("marmal88/skin_cancer")

# Get the unique classes (labels) and localizations
classes = set(dataset['train']['dx'])
unique_classes = list(classes)
num_classes = len(unique_classes)

localizations = set(dataset['train']['localization'])
unique_localizations = list(localizations)
num_localizations = len(unique_localizations)

image_size = (256, 256)

matplotlib.use('TkAgg')


@tf.function
def adaptive_resize(image: tf.Tensor, target_size: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    """
    Resizes an image using either area or bicubic interpolation based on the scale factor between the current size and the target size.

    :param image: A tensor representing the image to be resized.
    :type image: tf.Tensor
    :param target_size: A tuple specifying the target size for the resized image. (Default: (256, 256))
    :type target_size: Tuple[int, int]
    :return: A tensor representing the resized image.
    :rtype: tf.Tensor

    The function first calculates the scale factor between the current size of the image and the target size. If the scale factor is less than 1, the function resizes the image using the area interpolation method. Otherwise, it uses the bicubic interpolation method.

    Example usage:

    image = tf.io.read_file('image.jpg')
    image = tf.image.decode_jpeg(image, channels=3)
    resized_image = adaptive_resize(image, target_size=(256, 256))
    """
    current_size = tf.cast(tf.shape(image)[:2], tf.float32)
    target_size_float = tf.cast(target_size, tf.float32)
    max_scale_factor = tf.reduce_max(target_size_float / current_size)

    resized_image = tf.cond(max_scale_factor < 1,
                            lambda: resize_area(image, target_size),
                            lambda: resize_bicubic(image, target_size))

    return resized_image


@tf.function
def resize_area(image: tf.Tensor, target_size: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    """
    Resize the input image using the area method.

    :param image: A `tf.Tensor` representing the image to be resized.
    :type image: tf.Tensor
    :param target_size: A tuple of two integers representing the target size of the image after resizing. Default is (256, 256).
    :type target_size: Tuple[int, int]

    :return: A `tf.Tensor` representing the resized image.
    :rtype: tf.Tensor
    """
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.AREA)


@tf.function
def resize_bicubic(image: tf.Tensor, target_size: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    """
    Resize the input image using the bicubic interpolation method.

    :param image: A tensor representing the input image.
    :type image: tf.Tensor
    :param target_size: A tuple representing the target size for the resized image. The default value is (256, 256).
    :type target_size: Tuple[int, int]
    :return: A tensor representing the resized image.
    :rtype: tf.Tensor

    """
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BICUBIC)


def squeeze_excite_block(input: tf.Tensor, ratio: int = 16) -> tf.Tensor:
    """
    Apply a Squeeze and Excitation block to the input tensor.

    :param input: The input tensor to apply the Squeeze and Excitation block to.
    :type input: tf.Tensor
    :param ratio: The reduction factor used to reduce the number of filters in the block. The default value is 16.
    :type ratio: int
    :return: The output tensor after applying the Squeeze and Excitation block.
    :rtype: tf.Tensor

    The Squeeze and Excitation block consists of the following steps:
    1. Apply a Global Average Pooling layer to the input tensor.
    2. Apply a Dense layer with a ReLU activation function and the specified reduction factor.
    3. Apply another Dense layer with a sigmoid activation function to obtain the channel-wise attention weights.
    4. Multiply the input tensor with the channel-wise attention weights to obtain the final output tensor.

    Example usage:

    input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))
    output_tensor = squeeze_excite_block(input_tensor, ratio=8)
    model = tf.keras.models.Model(input_tensor, output_tensor)
    """
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = BatchNormalization()(se)
    se = Activation(tf.keras.activations.gelu)(se)
    se = Dense(filters, kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation('sigmoid')(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])
    return x


def cbam_block(input_tensor: tf.Tensor, ratio: int = 16, kernel_size: int = 7) -> tf.Tensor:
    """
    :param input_tensor: input tensor to apply the Convolutional Block Attention Module (CBAM) to
    :type input_tensor: tf.Tensor
    :param ratio: ratio parameter used to reduce the number of filters in the block (default: 16)
    :type ratio: int
    :param kernel_size: kernel size for the Conv2D layer in the spatial attention block (default: 7)
    :type kernel_size: int
    :return: tensor after applying the CBAM block
    :rtype: tf.Tensor

    This method implements a Convolutional Block Attention Module (CBAM), which is a type of attention mechanism used to enhance the representational power of CNNs. It combines
    * both channel-wise attention and spatial attention mechanisms to selectively amplify or suppress features within an input tensor.

    The CBAM block consists of the following steps:
    1. Apply the channel-wise squeeze-excite block to the input tensor.
    2. Compute the spatial attention of the input tensor by applying a 2D convolutional layer with a sigmoid activation function to the output of the channel-wise squeeze-excite block.
    3. Multiply the input tensor element-wise with the spatial attention tensor to obtain the final output.

    Example usage:

    input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))
    output_tensor = cbam_block(input_tensor, ratio=8, kernel_size=7)
    model = tf.keras.models.Model(input_tensor, output_tensor)
    """
    channel = GlobalAveragePooling2D()(input_tensor)
    channel = Dense(channel.shape[-1] // ratio, kernel_initializer='he_normal', use_bias=True)(channel)
    channel = BatchNormalization()(channel)
    channel = Activation(tf.keras.activations.gelu)(channel)
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


class BestValueTracker(tf.keras.callbacks.Callback):
    """
    Tracks the best validation accuracy during training.

    This class is a subclass of tf.keras.callbacks.Callback and is used to track the monitoring metric (i.e. validation
    accuracy) during training. It stores the best value of the monitoring metric and the epoch number at which the best
    value was achieved.

    Attributes:
        best_value (float): The best monitored achieved so far.
        best_epoch (int): The epoch number at which the best value accuracy was achieved.

    Methods:
        on_epoch_end(epoch, logs): Called at the end of each epoch to update the best value accuracy and epoch.

    Example:
        tracker = BestValueTracker()
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[tracker])
    """
    def __init__(self, monitor='val_accuracy'):
        super(BestValueTracker, self).__init__()
        self.best_value = 0
        self.best_epoch = 0
        self._monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if self._monitor not in logs:
            raise ValueError(f"Monitoring metric '{self._monitor}' not found in logs.")

        current_best = logs.get(self._monitor)
        if current_best > self.best_value:
            self.best_value = current_best
            self.best_epoch = epoch


@tf.function
def process_training(image):
    """
    Process a training image. This function applies random vertical flips, horizontal flips, and rotations to the input image for data augmentation.

    :param image: The input training image.
    :type image: tf.Tensor
    :return: The processed image.
    :rtype: tf.Tensor

    """
    img = tf.image.random_flip_left_right(image)
    img = tf.image.random_flip_up_down(img)
    img = tf.cast(img, tf.float32)
    img = adaptive_resize(img, image_size)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tfa.image.rotate(img, tf.random.uniform(shape=[], minval=-math.pi / 8.0, maxval=math.pi / 8.0),
                           fill_mode='reflect')

    return img


@tf.function
def process_validation(image):
    """
    Processes the validation image by adaptive resizing it based on the `image_size` parameter.

    :param image: The input validation image.
    :type image: tf.Tensor
    :return: The processed validation image.
    :rtype: tf.Tensor
    """
    img = adaptive_resize(image, image_size)
    return img


@tf.function
def preprocess_image_and_metadata(features, labels, training):
    """
    Preprocesses the image and metadata features.

    :param features: A dictionary containing the features. The dictionary should contain the following:
        - 'image': A tensor containing the image.
        - 'sex': A tensor containing the patient sex (male = 1, female = 0).
        - 'age': A tensor containing the patient age.
        - 'localization': A tensor containing the lesion localization, one-hot encoded.
    :param labels: A tensor containing the labels. The labels are the skin lesion one-hot encoded.
    :param training: A boolean indicating if the method is called during training.
    :return: Returns a tuple containing the processed features and one-hot encoded labels.
    """
    # Image preprocessing

    image = features['image']

    processed_images = tf.cond(
        tf.equal(training, True),
        lambda: process_training(image),
        lambda: process_validation(image)
    )

    # Localization one-hot encoding using lookup table
    localization_one_hot = tf.one_hot(features['localization'], depth=len(unique_localizations))

    # Dx one-hot encoding using lookup table
    dx_one_hot = tf.one_hot(labels, depth=len(unique_classes))

    # Sex and age processing
    sex = tf.expand_dims(tf.cast(features['sex'], tf.float32), -1)
    age = tf.expand_dims(tf.cast(features['age'], tf.float32), -1)

    # Concatenate metadata features
    metadata_input = tf.concat([localization_one_hot, sex, age], axis=-1)

    processed_features = {
        'image_input': processed_images,
        'metadata_input': metadata_input
    }
    return processed_features, dx_one_hot


def create_tf_datasets(dataset: DatasetDict, image_size: Tuple[int, int], batch_size: int) -> Tuple[Dataset, Dataset]:
    """
    Create TensorFlow datasets for training and validation.

    :param dataset: The dataset containing the training and validation splits.
    :type dataset: DatasetDict
    :param image_size: The target image size for resizing.
    :type image_size: Tuple[int, int]
    :param batch_size: The batch size for the datasets.
    :type batch_size: int
    :return: Returns a tuple containing the training and validation TensorFlow datasets.
    :rtype: Tuple[Dataset, Dataset]

    The function first computes the mean and standard deviation of the ages in the dataset, to normalize them.
    It then creates mappings for the localization and dx values. The function then applies these mappings to the
    dataset and creates TensorFlow datasets for training and validation. The function also applies image preprocessing
    and metadata processing to the datasets.

    Note: The function assumes that the dataset contains the following keys: 'train' and 'validation'.
    Each of these keys should contain the following: 'image', 'age', 'sex', 'localization', and 'dx'.

    * The returned datasets are TensorFlow datasets which contains three tensors: 2 input tensors ('image_input' and 'metadata_input') and one output tensor ('dx').

    """
    # Compute age scaling parameters
    ages = np.concatenate([dataset['train']['age'], dataset['validation']['age']])
    ages = [age for age in ages if age is not None]
    age_mean, age_std = np.mean(ages), np.std(ages)

    # Define pre-tensor transformation function

    all_localizations = dataset['train']['localization'] + dataset['validation']['localization']

    # Similarly, concatenate 'dx' values directly if needed
    all_dx = dataset['train']['dx'] + dataset['validation']['dx']

    # Now you can proceed with creating your unique category codes
    unique_localizations = pd.unique(all_localizations)
    unique_dx = pd.unique(all_dx)

    # Create mappings for localization and dx
    localization_to_code = {loc: code for code, loc in enumerate(unique_localizations)}
    dx_to_code = {dx: code for code, dx in enumerate(unique_dx)}

    # Step 3: Define a function to apply these mappings
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

    # Convert to TensorFlow data
    train_tf_ds = train_ds.to_tf_dataset(columns=['image', 'sex', 'age', 'localization', 'dx'], label_cols=['dx'],
                                         shuffle=True, batch_size=batch_size)
    val_tf_ds = val_ds.to_tf_dataset(columns=['image', 'sex', 'age', 'localization', 'dx'], label_cols=['dx'],
                                     shuffle=False, batch_size=batch_size)

    # Wrap the preprocessing function to include the training flag
    def wrap_preprocess_image_and_metadata(training):
        return lambda features, labels: preprocess_image_and_metadata(features, labels, training)

    # Apply image preprocessing
    train_tf_ds = train_tf_ds.map(wrap_preprocess_image_and_metadata(training=True),
                                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_tf_ds = val_tf_ds.map(wrap_preprocess_image_and_metadata(training=False),
                              num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

    return train_tf_ds, val_tf_ds


def objective(trial):
    batch_size = trial.suggest_int('batch', 32, 64)
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
        base_model = VGG16(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'VGG19':
        base_model = VGG19(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'ResNet101V2':
        base_model = ResNet101V2(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'InceptionResNetV2':
        base_model = InceptionResNetV2(include_top=False, weights=weight,
                                                             input_shape=image_size + (3,))
    elif base_model_architecture == 'Xception':
        base_model = Xception(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'MobileNetV2':
        base_model = MobileNetV2(include_top=False, weights=weight, input_shape=image_size + (3,))
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

    attention_func = None
    if attention_mechanism == 'SENet':
        attention_func = squeeze_excite_block
    elif attention_mechanism == 'CBAM':
        attention_func = cbam_block

    print(f"Trial parameters: {trial.params}")

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
        x = Activation('swish')(x)
        x = Dropout(dropout_rate_layers[i])(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=[image_input, metadata_input], outputs=predictions)

    train_ds, val_ds = create_tf_datasets(dataset, image_size=image_size, batch_size=batch_size)

    # Phase 1
    opt = AdamW(weight_decay=weight_decay_phase1, learning_rate=learning_rate_phase1, amsgrad=use_amsgrad, clipnorm=1.0)
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
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(unique_classes), y=train_labels)
            class_weights = dict(enumerate(weights))
        else:
            class_weights = None
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    for layer in base_model.layers[-num_layers_unfreeze:]:
        layer.trainable = True

    best_value_tracker = BestValueTracker()

    try:
        model.fit(train_ds
                  , validation_data=val_ds
                  , epochs=100
                  , class_weight=class_weights
                  , callbacks=[
                # TFKerasPruningCallback(trial, 'val_accuracy'),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2),
                # best_value_tracker
            ]
                  , workers=num_cores
                  , use_multiprocessing=True
                  )
    except optuna.exceptions.TrialPruned as e:
        best_accuracy = best_value_tracker.best_val_accuracy
        best_epoch = best_value_tracker.best_epoch
        print(f"Best validation accuracy before pruning: {best_accuracy} at epoch {best_epoch}")
        if best_accuracy < 0.70:
            # Prune if best accuracy is less than 70%
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
    # storage_name = f"sqlite:///{study_name}_testing.db"

    from optuna.pruners import PercentilePruner

    # optuna.delete_study(study_name, storage_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True,
                                # add Patient pruner with Median pruner as the base pruner
                                pruner=optuna.pruners.PatientPruner(
                                    wrapped_pruner=PercentilePruner(percentile=25, n_startup_trials=5,
                                                                    n_warmup_steps=10, interval_steps=1)
                                    , patience=5, min_delta=0.0001)
                                # pruner=optuna.pruners.NopPruner()
                                )

    # Check which model architecture has fewer trials and enqueue a trial for that model
    from collections import Counter
    import operator

    trials = study.get_trials(states=[optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING, optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED])

    if len(trials) == 0:
        study.enqueue_trial({'base_model_architecture': 'VGG16'})
        study.optimize(objective, n_trials=1, callbacks=[clear_session_callback])
        exit()

    num_trials = Counter(trial.params['base_model_architecture'] for trial in trials)

    # find the architecture with minimum trials
    architecture_min_trials = min(num_trials.items(), key=operator.itemgetter(1))[0]

    # enqueue the trial for that architecture
    study.enqueue_trial({'base_model_architecture': architecture_min_trials})

    study.optimize(objective, n_trials=1, callbacks=[clear_session_callback])
