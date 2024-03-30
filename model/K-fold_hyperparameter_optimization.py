import warnings
import os
from typing import Optional, Dict, Any

from pip._internal.utils.misc import ask
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
        # To prevent TensorFlow from allocating all memory upfront, enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Optionally, set a hard limit on GPU memory
        # Replace <memory_limit_in_MB> with the desired limit
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 10)]
        )
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from keras.src.utils import to_categorical

tf.experimental.numpy.experimental_enable_numpy_behavior()

from keras import Model, Input
from keras.layers import BatchNormalization, Activation, Dropout
from keras.losses import CategoricalFocalCrossentropy
from keras.optimizers import AdamW
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna_integration import TFKerasPruningCallback
import gc
from tensorflow.keras import backend as K
import cv2

from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dotenv import load_dotenv
import pathlib

from tensorflow.keras.layers import Dense, Flatten
import keras_cv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

# Get the directory where the current script is located
script_dir = pathlib.Path(__file__).parent.resolve()

# Define the path to the .env file
config_env_path = script_dir / '../config.env'  # Assuming the .env file is in the project root

load_dotenv(dotenv_path=config_env_path)

assert 'DATASET_DIR' in os.environ, "Please set 'DATASET_DIR' environment variable"


# absolute paths
dataset_path_str: str = os.getenv('DATASET_DIR')
dataset_path: str = os.path.abspath(dataset_path_str)
train_path: str = os.path.join(dataset_path, 'Train')
test_path: str = os.path.join(dataset_path, 'Test')

# Parameters
k = 5  # Number of folds
image_size = (256, 256)

classes = os.listdir(train_path)

# Prepare file paths and labels
file_paths = []
labels = []
for class_name in classes:
    class_dir = os.path.join(train_path, class_name)
    class_files = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
    file_paths.extend(class_files)
    labels.extend([class_name] * len(class_files))

# Convert labels to numerical format
label_to_index = dict((name, index) for index, name in enumerate(classes))
numerical_labels = np.array([label_to_index[label] for label in labels])

# Stratified K-Fold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


# def preprocess_function(func, training=False):
#     image_gen = ImageDataGenerator(
#         rotation_range=180,
#         # width_shift_range=0.02,
#         # height_shift_range=0.02,
#         horizontal_flip=True,
#         vertical_flip=True,
#         # zoom_range=0.01,
#         fill_mode='constant',
#
#         # reflect seems to be the best fill mode for most images
#         cval=0, # Use a constant value for fill mode: 0 (black)
#         # cval=255,  # Use a constant value for fill mode: 255 (white)
#         # preprocessing_function=func
#     ) if training else None
#
#     def process_path(file_path, label):
#         # Load the raw data from the file as a string
#         img = tf.io.read_file(file_path)
#         img = tf.image.decode_jpeg(img, channels=3)
#
#         # Divide transformation process into two distinct phases
#         if training:
#             # Use a py_function to allow the use of RandomTransform within a map function
#             img = tf.py_function(func=lambda img: image_gen.random_transform(img.numpy()),
#                                  inp=[img],
#                                  Tout=tf.float32)
#             img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
#
#         img = tf.image.resize(img, image_size)
#
#         # else:
#         #     img = func(img)
#
#         # img = preprocess_input(img) was removed from here since it modifies the image in a way that
#         # alters the features needed for the model to learn.
#         # When removing the preprocess_input, the model will learn the features of the images as they are,
#         # which improves the model's performance by at least 30%, for the dataset used in this example.
#         # This is because the preprocess_input function is designed to normalize the input data for the VGG16 model,
#         # VGG19 model, and ResNet50 model, from object detection models to image classification models which highlights
#         # the features of the images. However, the ISIC dataset and skin lesions are too sensitive to the
#         # normalization process, which causes the model to learn the wrong features.
#         return img, label
#
#     return process_path

import tensorflow_addons as tfa


def adaptive_resize(image, target_size=(224, 224)):
    # Ensure target_size is a 1-D int32 Tensor
    # target_size = tf.convert_to_tensor(target_size, dtype=tf.int32)

    # Get the current size of the image
    current_size = tf.cast(tf.shape(image)[:2], tf.float32)
    target_size_float = tf.cast(target_size, tf.float32)

    # Determine if the image is larger or smaller than the target size
    scale_factor = target_size_float / current_size
    max_scale_factor = tf.reduce_max(scale_factor)

    # If the image is larger than the target, downscale using 'area' interpolation
    if max_scale_factor < 1:
        resized_image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.AREA)
    else:
        # If the image is smaller, upscale using 'bicubic' interpolation
        resized_image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BICUBIC)

    return resized_image


def shift_image(image, width_shift, height_shift):
    # Image dimensions
    original_dims = tf.shape(image)

    # Pad the image with zeros on the top and left
    padded_image = tf.image.pad_to_bounding_box(image, height_shift, width_shift, original_dims[0] + height_shift,
                                                original_dims[1] + width_shift)

    # Crop the padded image to maintain the original dimensions
    shifted_image = tf.image.crop_to_bounding_box(padded_image, 0, 0, original_dims[0], original_dims[1])

    return shifted_image


def preprocess_function(training=False, return_original=False, image_size=(256, 256)):
    def process_path(file_path, label):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Transformation during training
        if training:
            # Use a py_function to allow the use of RandomTransform within a map function
            # apply random vertical and horizontal flips
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            # img.set_shape([None, None, 3])
            # apply random rotations
            # img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
        img = adaptive_resize(img, image_size)
        if training:
            # rotate 90 degrees
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            # minimal rotation to avoid large distortions
            img = tfa.image.rotate(img, tf.random.uniform(shape=[], minval=-math.pi/8.0, maxval=math.pi/8.0), fill_mode='reflect')

        if return_original:
            original_image = tf.io.read_file(file_path)
            original_image = tf.image.decode_jpeg(original_image, channels=3)
            original_image = adaptive_resize(original_image, image_size)
            return original_image, img, label
        else:
            return img, label

    return process_path


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
    # Channel attention module
    channel = GlobalAveragePooling2D()(input_tensor)
    channel = Dense(channel.shape[-1] // ratio, kernel_initializer='he_normal', use_bias=True)(channel)
    channel = BatchNormalization()(channel)
    channel = Activation('relu')(channel)
    channel = Dense(input_tensor.shape[-1], kernel_initializer='he_normal', use_bias=True)(channel)
    # channel = BatchNormalization()(channel) # This layer is not necessary, as the BatchNormalization will be
    # normalized by the sigmoid activation function
    channel = Activation('sigmoid')(channel)

    # Reshape channel to be broadcastable over the spatial dimensions of input_tensor
    if K.image_data_format() == 'channels_first':
        channel = Reshape((input_tensor.shape[1], 1, 1))(channel)
    else:  # 'channels_last'
        channel = Reshape((1, 1, input_tensor.shape[-1]))(channel)

    channel_attention = Multiply()([input_tensor, channel])

    # Spatial attention module
    spatial = Conv2D(1, kernel_size, padding='same', use_bias=True)(channel_attention)
    # spatial = BatchNormalization()(spatial) # same reason as the BatchNormalization layer above
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
                # Here you need to reset layer weights.
                # Reset mode and moving mean and variance.
                layer.moving_mean.assign(layer.moving_mean * 0)
                layer.moving_variance.assign(layer.moving_variance * 0 + 1)
                # Reset the weights to original state
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


# class CustomPruningCallback(TFKerasPruningCallback):
#
#     def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
#         super().__init__(trial, monitor)
#
#     def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
#         logs = logs or {}
#         current_score = logs.get(self._monitor)
#
#         if current_score is None:
#             message = (
#                 "The metric '{}' is not in the evaluation logs for pruning. "
#                 "Please make sure you set the correct metric name.".format(self._monitor)
#             )
#             warnings.warn(message)
#             return
#
#         # Report current score and epoch to Optuna's trial.
#         self._trial.report(float(current_score), step=epoch)
#
#         # Prune trial if needed
#         if self._trial.should_prune():
#             print("Trial was \"pruned\" (early stopped) at epoch {}.".format(epoch))
#             self.model.stop_training = True

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
                                                                                    'InceptionResNetV2', 'EfficientNetV2B0',
                                                                                    'Xception', 'MobileNetV2'])
    attention_mechanism = trial.suggest_categorical('attention_mechanism', ['SENet', 'CBAM', 'None'])
    # incremental_learning = trial.suggest_categorical('incremental_learning', [True, False])
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
    elif base_model_architecture == 'EfficientNetV2B0':
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False, weights=weight, input_shape=image_size + (3,))
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
        # print(f"{incremental_learning=}")
        print(f"{use_amsgrad=}")
        print(f"{num_layers_unfreeze=}")
        print(f"{pre_trained_weights=}")

    print_params()

    # Assuming 'squeeze_excite_block' and 'cbam_block' are properly defined elsewhere
    attention_func = None
    if attention_mechanism == 'SENet':
        attention_func = squeeze_excite_block
    elif attention_mechanism == 'CBAM':
        attention_func = cbam_block

    average_accuracy = 0
    current_step = 0 # epoch for pruning

    # K-Fold Stratified Training
    for fold, (train_index, val_index) in enumerate(skf.split(file_paths, numerical_labels)):
        print(f"Starting fold {fold + 1} of {k}")

        # Data split for training and validation
        train_paths = np.array(file_paths)[train_index]
        train_labels = numerical_labels[train_index]

        val_paths = np.array(file_paths)[val_index]
        val_labels = numerical_labels[val_index]

        # Create tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, to_categorical(train_labels)))
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, to_categorical(val_labels)))

        # Process paths
        train_ds = train_ds.map(preprocess_function(training=True), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocess_function(), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        # The cache() method is not used here because the augmented images should not be reused
        # If cache were used, the same augmented images would be shown in each epoch
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Model definition
        if pre_trained_weights:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = True

        # Apply the attention mechanism
        if attention_func:
            x = attention_func(base_model.output)
        else:
            # Proceed without attention mechanism
            x = base_model.output
        x = Flatten()(x)
        for i in range(num_dense_layers):
            x = Dense(dense_units[i])(x)
            if batch_normalization_layers[i]:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout_rate_layers[i])(x)
        predictions = Dense(len(classes), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Phase 1
        opt = AdamW(weight_decay=weight_decay_phase1, learning_rate=learning_rate_phase1, amsgrad=use_amsgrad)
        if loss_type == 'focal_loss':
            loss_fn = CategoricalFocalCrossentropy(alpha=alpha_phase1, gamma=gamma_phase1)
            class_weights = None
        else:
            loss_fn = 'categorical_crossentropy'
            if use_class_weights:
                weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
                class_weights = dict(enumerate(weights))
            else:
                class_weights = None

        model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

        reestimate_batch_norm(model, train_ds, num_batches=100)

        model.fit(train_ds
                  , validation_data=val_ds
                  , steps_per_epoch=len(train_paths) // batch_size + (1 if len(train_paths) % batch_size != 0 else 0)
                  , validation_steps=len(val_paths) // batch_size + (1 if len(val_paths) % batch_size != 0 else 0)
                  , epochs=3
                  , class_weight=class_weights
                  , callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=2)
                                ]
                  , workers=num_cores
                  , use_multiprocessing=True
                  )

        # Phase 2
        opt = AdamW(weight_decay=weight_decay_phase2, learning_rate=learning_rate_phase2, amsgrad=use_amsgrad)
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
                      , steps_per_epoch=len(train_paths) // batch_size + (1 if len(train_paths) % batch_size != 0 else 0)
                      , validation_steps=len(val_paths) // batch_size + (1 if len(val_paths) % batch_size != 0 else 0)
                      , epochs=100
                      , class_weight=class_weights
                      , callbacks=[
                                   TFKerasPruningCallback(trial, 'val_accuracy'),
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,
                                                                    start_from_epoch=70),
                                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2),
                                      best_value_tracker
                                     # , unfreeze_callback
                                   ]
                        , workers=num_cores
                        , use_multiprocessing=True
                      )
        except optuna.exceptions.TrialPruned as e:
            print(f"Trial was pruned at fold {fold + 1}")
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
        average_accuracy += accuracy
        print(f"Fold {fold + 1} accuracy: {accuracy}")
        break

    average_accuracy
    print(f"Average accuracy: {average_accuracy}")

    return average_accuracy

import itertools

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    #
    # def show_batch(original_images, augmented_images, labels):
    #     plt.figure(figsize=(20, 20))  # Increased figure size
    #     for n in range(len(original_images)):
    #         # Original image
    #         ax = plt.subplot(len(original_images), 2, 2 * n + 1)
    #         plt.imshow(original_images[n] / 255)
    #         plt.title("Original")
    #         plt.axis('off')
    #
    #         # Augmented image
    #         ax = plt.subplot(len(original_images), 2, 2 * n + 2)
    #         plt.imshow(augmented_images[n] / 255)
    #         plt.title("Augmented")
    #         plt.axis('off')
    #
    #
    # train_ds = tf.data.Dataset.from_tensor_slices((file_paths, to_categorical(numerical_labels)))
    # train_ds = train_ds.map(preprocess_function(training=True, return_original=True),
    #                         num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000)
    # train_ds_batch = train_ds.batch(8)
    # # Fetch a batch of original and augmented images
    # original_image_batch, augmented_image_batch, label_batch = next(iter(train_ds_batch))
    #
    # # Display the original and augmented images
    # show_batch(original_image_batch.numpy(), augmented_image_batch.numpy(), label_batch.numpy())
    #
    # plt.show()
    # pass


    # def custom_preprocess_func(image):
    #     # Convert the TensorFlow tensor to a NumPy array
    #     # image = image.numpy()
    #
    #     # Define padding (example uses constant padding, but you can customize this)
    #     top, bottom, left, right = [10, 10, 10, 10]  # Adjust padding values as needed
    #     image_padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #
    #     # Define rotation (customize the angle as needed)
    #     angle = np.random.uniform(-180, 180)  # Random rotation angle
    #     center = (image_padded.shape[1] // 2, image_padded.shape[0] // 2)
    #     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     image_rotated = cv2.warpAffine(image_padded, rotation_matrix, (image_padded.shape[1], image_padded.shape[0]))
    #
    #     image_resized = cv2.resize(image_rotated, image_size)
    #
    #     # Convert back to a TensorFlow tensor
    #     return tf.convert_to_tensor(image_resized, dtype=tf.float32)
    #
    # # Define your ImageDataGenerator with the specified augmentations
    # # image_gen = ImageDataGenerator(
    # #     rotation_range=180,
    # #     width_shift_range=0.05,
    # #     height_shift_range=0.05,
    # #     horizontal_flip=True,
    # #     vertical_flip=True,
    # #     zoom_range=0.05,
    # #     fill_mode='reflect',
    # #     # cval=0,
    # #     # preprocessing_function=custom_preprocess_func  # Uncomment and define 'func' if you have a specific preprocessing function
    # # )
    # image_gen = ImageDataGenerator(
    #     # rotation_range=180,
    #     width_shift_range=0.02,
    #     height_shift_range=0.02,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     zoom_range=0.05,
    #     fill_mode='reflect',
    #     # reflect seems to be the best fill mode for most images
    #     # cval=0, # Use a constant value for fill mode: 0 (black)
    #     # cval=255,  # Use a constant value for fill mode: 255 (white)
    #     # preprocessing_function=func
    # )
    #
    # # Assuming 'train_path' is the path to your training images directory
    # # This directory should be structured in subdirectories for each class
    # # For visualization, we'll load a small subset of images
    # generator = image_gen.flow_from_directory(
    #     train_path,
    #     target_size=image_size,  # Assuming 'image_size' is defined (e.g., (224, 224))
    #     batch_size=16,  # Adjust based on how many images you want to visualize at once
    #     class_mode='categorical',  # Adjust based on your problem ('binary', 'categorical', 'input', etc.)
    #     shuffle=True  # Shuffle to get random images
    # )
    #
    # # Get a batch of images
    # images, labels = next(generator)
    #
    # # Display the images
    # plt.figure(figsize=(10, 10))
    # for i in range(len(images)):
    #     plt.subplot(4, 4, i + 1)  # Adjust the grid size based on the batch size
    #     img = images[i]  # Get the i-th image in the batch
    #     img = img.astype('uint8')  # Convert to uint8 type if not already
    #     plt.imshow(img)
    #     plt.axis('off')
    # plt.show()
    # sample_files = tf.data.Dataset.list_files(train_path + '/*/*.jpg').take(16)  # Adjust the path and number of images as needed
    #
    # # Define your preprocessing and augmentation function
    # def preprocess_and_augment(file_path):
    #     img = tf.io.read_file(file_path)
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     img = tf.image.resize(img, image_size)
    #     img = tf.keras.applications.vgg19.preprocess_input(img)  # Adjust preprocessing according to your base model
    #
    #     # Apply your augmentation logic here (adjust according to your actual augmentation pipeline)
    #     img = tf.image.random_flip_left_right(img)
    #     img = tf.image.random_flip_up_down(img)
    #     # img = tf.image.random_brightness(img, max_delta=0.3)  # Example augmentation
    #
    #     return img
    #
    # # Apply preprocessing and augmentation to each image in the sample set
    # augmented_images = sample_files.map(preprocess_and_augment)
    #
    # # Display the images
    # plt.figure(figsize=(10, 10))
    # for i, img in enumerate(augmented_images):
    #     plt.subplot(4, 4, i + 1)  # Adjust based on the number of images you're displaying
    #     plt.imshow(img.numpy().astype("uint8"))
    #     plt.axis('off')
    # plt.show()


    # # study_name_1 = 'k_fold_hyperparameter_optimization_with_only_one_phase'
    # study_name_2 = 'k_fold_hyperparameter_optimization'
    # # study_name_3 = 'k_fold_hyperparameter_optimization_with_attention_mechanisms'
    # study_name_4 = 'k_fold_hyperparameter_optimization_with_attention_mechanisms_without_augmentation'
    # study_name_5 = 'k_fold_hyperparameter_optimization_testing'
    # study_name = 'k_fold_hyperparameter_optimization_with_attention_mechanisms_with_simple_augmentation'
    #
    # previous_storages = [f"sqlite:///{study_name}.db", f"sqlite:///{study_name_2}.db",
    #                         f"sqlite:///{study_name_4}.db", f"sqlite:///{study_name_5}.db",
    #                      f"sqlite:///identifier.sqlite", f"sqlite:///model/k_fold_hyperparameter_optimization_with_attention_mechanisms_with_simple_augmentation.db",
    #                      f"sqlite:///model/identifier.sqlite"]
    #
    # studies_missing_params_with_value = ['base_model_architecture', 'attention_mechanism', 'use_amsgrad']
    # categorical_distribution_of_missing_params = {'base_model_architecture': ['VGG16', 'VGG19', 'ResNet50'],
    #                                                 'attention_mechanism': ['SENet', 'CBAM', 'None'],
    #                                                 'use_amsgrad': [True, False]}
    #
    # previous_best_params = []
    #
    # study_number = 0
    # # total of 8 studies
    # queue_previous_best_params = [] # queue to store the previous best params, with missing values to be filled
    # # adding combinations of missing values to the queue

    study_name = 'skin_lesion_hyperparameter_optimization_with_detail_augmentation'
    storage_name = f"sqlite:///{study_name}.db"

    # optuna callbacks to free up keras backend memory
    def clear_session_callback(study, trial):
        K.clear_session()
        gc.collect()
        # pass


    class WindowPercentilePruner(optuna.pruners.BasePruner):
        def __init__(self, percentile, window_size, n_startup_trials=5, n_warmup_steps=10, interval_steps=1):
            super(WindowPercentilePruner, self).__init__()
            self.percentile = percentile
            self.window_size = window_size
            self.n_startup_trials = n_startup_trials
            self.n_warmup_steps = n_warmup_steps
            self.interval_steps = interval_steps

        def prune(self, study, trial):
            n_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

            # Check if trial is within the startup trials or warmup steps
            if n_trials < self.n_startup_trials or trial.last_step < self.n_warmup_steps:
                return False

            step = trial.last_step
            if step is None or step % self.interval_steps != 0:
                return False

            # Check if the trial should be pruned at each step in the window
            for window_step in range(max(step - self.window_size + 1, self.n_warmup_steps), step + 1,
                                     self.interval_steps):
                all_intermediate_values = [t.intermediate_values.get(window_step) for t in study.trials if
                                           window_step in t.intermediate_values and t.state == optuna.trial.TrialState.COMPLETE]

                if not all_intermediate_values:
                    continue

                # Calculate the performance threshold based on the specified percentile
                performance_threshold = np.percentile(all_intermediate_values, self.percentile)

                # Get the current trial's performance at the window step
                current_performance = trial.intermediate_values.get(window_step)

                # If the current trial's performance is below the threshold, prune
                if current_performance is not None and current_performance < performance_threshold:
                    return True

            return False

    study_name = 'skin_lesion_hyperparameter_optimization_with_detail_augmentation_and_patient_pruner'

    from optuna.pruners import PercentilePruner

    # optuna.delete_study(study_name, storage_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True,
                                # add Patient pruner with Median pruner as the base pruner
                                pruner=optuna.pruners.PatientPruner(
                                    wrapped_pruner=PercentilePruner(percentile=50, n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
                                    , patience=10, min_delta=0.0001))
    #
    # init_num_trials = len(study.get_trials(deepcopy=False))
    # print(f"Initial number of trials: {init_num_trials}")
    #
    # added_trial = False
    # if init_num_trials < 21 + (62 * 5):
    #     for storage in previous_storages:
    #         # load all studies from previous storages, and get the best parameters
    #         study_names = optuna.get_all_study_names(storage)
    #         for i, study_name in enumerate(study_names):
    #             study = optuna.load_study(study_name=study_name, storage=storage)
    #             if len(study.trials) > 0:  # Check if the study has any trials
    #                 try:
    #                     base_params = study.best_params
    #                     missing_params = [param for param in studies_missing_params_with_value if
    #                                       param not in base_params.keys()]
    #                     if missing_params:
    #                         missing_params_values = [
    #                             categorical_distribution_of_missing_params[param] for param in missing_params]
    #                         for values in itertools.product(*missing_params_values):
    #                             new_params = base_params.copy()
    #                             new_params.update(dict(zip(missing_params, values)))
    #                             previous_best_params.append(new_params)
    #                     else:
    #                         previous_best_params.append(base_params)
    #                 except ValueError:
    #                     print(f"No completed trial found for {study_name}")
    #
    #     for param in previous_best_params:
    #         study.enqueue_trial(params=param, skip_if_exists=True)
    #         # Check the updated number of trials
    #         updated_num_trials = len(study.get_trials(deepcopy=False))
    #         if updated_num_trials > init_num_trials:
    #             print(f"A new trial has been enqueued. Number of trials now: {updated_num_trials}")
    #             added_trial = True
    #             break
    #         else:
    #             print("The trial already existed, moving to the next trial")
    #
    #     if not added_trial:
    #         print("No new trials have been added in this run")

    study.optimize(objective, n_trials=1, callbacks=[clear_session_callback])
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
