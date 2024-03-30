import numpy as np
import tensorflow as tf
from keras.src.layers import Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.debugging.experimental.enable_dump_debug_info('logs', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

import optuna
from keras.src.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications import VGG19, VGG16, ResNet50
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from dataset_reader import train_dataset_size, validation_dataset_size, num_classes, \
    class_weights, config_env_path, classes, image_size
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc
from memory_profiler import profile
from sklearn.model_selection import StratifiedKFold


def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_with_probs(probs, targets):
        '''
        References :Lee, J.-W., & Kang, H.-S. (2024). Three-Stage Deep Learning Framework for Video Surveillance. Applied Sciences (2076-3417), 14(1), 408. https://doi-org.lopes.idm.oclc.org/10.3390/app14010408

        :param probs: y_pred from the model (predicted probabilities)
        :param targets: y_true from the model (true labels)
        :return: Focal loss
        '''

        eps = tf.keras.backend.epsilon()
        loss = targets * (-alpha * tf.pow((1 - probs), gamma) * tf.math.log(probs + eps)) + \
               (1 - targets) * (-alpha * tf.pow(probs, gamma) * tf.math.log(1 - probs + eps))
        return tf.reduce_mean(loss)

    return focal_loss_with_probs


import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=config_env_path)

assert 'DATASET_DIR' in os.environ, "Please set 'DATASET_DIR' environment variable"

# absolute paths
dataset_path_str: str = os.getenv('DATASET_DIR')
dataset_path: str = os.path.abspath(dataset_path_str)
train_path: str = os.path.join(dataset_path, 'Train')
test_path: str = os.path.join(dataset_path, 'Test')

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

# Load the dataset
train_set, validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_path,
    labels='inferred',
    label_mode='categorical',
    class_names=classes,
    color_mode='rgb',
    image_size=image_size,
    batch_size=None,  # Adjust the batch size as needed
    shuffle=True,  # Shuffling before splitting
    seed=42,
    validation_split=0.2,
    subset='both',  # Requesting both training and validation data
    interpolation='bilinear'
)

train_set = train_set.with_options(options)
validation_set = validation_set.with_options(options)

full_dataset = train_set.concatenate(validation_set)
labels = np.argmax(full_dataset.map(lambda image, label: label).as_numpy_iterator(), axis=-1)
n_split = 10

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                 mode='min')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True,
                                                  start_from_epoch=10)


def load_and_preprocess_training(preprocess_input=None):
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        if preprocess_input is not None:
            image = preprocess_input(image)
        return image, label

    return preprocess


def load_and_preprocess(preprocess_input=None):
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        if preprocess_input is not None:
            image = preprocess_input(image)
        return image, label

    return preprocess


def objective_resnet50(trial):
    global train_set, validation_set

    preprocess_input = resnet_preprocess_input
    # Load and configure the ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    batch_size = trial.suggest_int('batch', 16, 64)

    train_set_size = train_dataset_size // batch_size + (1 if train_dataset_size % batch_size != 0 else 0)
    validation_set_size = validation_dataset_size // batch_size + (
        1 if validation_dataset_size % batch_size != 0 else 0)

    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 32, 256) for i in range(num_dense_layers)]
    batch_normalization_layers = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                                  range(num_dense_layers)]
    dropout_rate_layers = [trial.suggest_float(f'dropout_rate_{i}', 0.3, 0.7) for i in range(num_dense_layers)]

    # Add custom layers
    x = Flatten()(base_model.output)
    for i in range(num_dense_layers):
        x = Dense(dense_units[i])(x)
        if batch_normalization_layers[i]:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=dropout_rate_layers[i])(x)

    # Suggest the type of loss function and whether to use class weights
    loss_type = trial.suggest_categorical('loss_type', ['focal_loss', 'categorical_crossentropy'])

    # Only suggest these if Focal Loss is chosen
    alpha_phase1 = trial.suggest_float('alpha_phase1', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase1 = trial.suggest_float('gamma_phase1', 1.0, 5.0) if loss_type == 'focal_loss' else None

    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    weight_decay_phase1 = trial.suggest_float('weight_decay_phase1', 1e-6, 1e-2, log=True)

    # Other hyperparameters
    learning_rate_phase1 = trial.suggest_float('learning_rate_phase1', 1e-5, 1e-3, log=True)

    learning_rate_phase2 = trial.suggest_float('learning_rate_phase2', 1e-6, 1e-4, log=True)
    num_layers_unfreeze = trial.suggest_int('num_layers_unfreeze', 1, 10)
    alpha_phase2 = trial.suggest_float('alpha_phase2', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase2 = trial.suggest_float('gamma_phase2', 1.0, 5.0) if loss_type == 'focal_loss' else None
    weight_decay_phase2 = trial.suggest_float('weight_decay_phase2', 1e-6, 1e-2, log=True)

    prediction = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)


    # Training dataset setup
    train_preprocessed_set = (train_set
                              .map(load_and_preprocess_training(), num_parallel_calls=2)
                              .shuffle(buffer_size=1000)
                              .batch(batch_size)
                              .repeat()
                              .prefetch(2))

    # Validation dataset setup
    validation_preprocessed_set = (validation_set
                                   .map(load_and_preprocess(), num_parallel_calls=2)
                                   .batch(batch_size)
                                   .prefetch(2))
    # Define the loss function based on the trial's suggestion
    if loss_type == 'focal_loss':
        loss_function = CategoricalFocalCrossentropy(alpha=alpha_phase1, gamma=gamma_phase1)
    else:
        loss_function = 'categorical_crossentropy'

    class_weight_arg = class_weights if use_class_weights and loss_type != 'focal_loss' else None

    # Compile the model
    model.compile(optimizer=AdamW(learning_rate=learning_rate_phase1,
                                  weight_decay=weight_decay_phase1),  # included weight_decay in optimizer
                  loss=loss_function, metrics=['accuracy'])

    # Train the model
    model.fit(
        train_preprocessed_set,
        steps_per_epoch=train_set_size,
        validation_data=validation_preprocessed_set,
        validation_steps=validation_set_size,
        epochs=50,
        verbose=1,
        class_weight=class_weight_arg,
        callbacks=[early_stopping, reduce_lr]
    )

    # Unfreeze the last num_layers_unfreeze layers
    for layer in base_model.layers[-num_layers_unfreeze:]:
        layer.trainable = True

    if loss_type == 'focal_loss':
        loss_function = CategoricalFocalCrossentropy(alpha=alpha_phase2, gamma=gamma_phase2)

    # Compile the model with a smaller learning rate
    model.compile(optimizer=AdamW(learning_rate=learning_rate_phase2,
                                  weight_decay=weight_decay_phase2),
                  loss=loss_function, metrics=['accuracy'])

    # Train the model again
    history = model.fit(
        train_preprocessed_set,
        steps_per_epoch=train_set_size,
        validation_data=validation_preprocessed_set,
        validation_steps=validation_set_size,
        epochs=100,
        verbose=1,
        class_weight=class_weight_arg,
        callbacks=[early_stopping, reduce_lr]
    )

    # return the validation accuracy of the lower validation loss epoch
    return 1.0 - history.history['val_accuracy'][np.argmin(history.history['val_loss'])]


def objective_vgg16(trial):
    global train_set, validation_set

    preprocess_input = resnet_preprocess_input
    # Load and configure the ResNet50 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    batch_size = trial.suggest_int('batch', 16, 64)

    train_set_size = train_dataset_size // batch_size + (1 if train_dataset_size % batch_size != 0 else 0)
    validation_set_size = validation_dataset_size // batch_size + (
        1 if validation_dataset_size % batch_size != 0 else 0)

    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 32, 256) for i in range(num_dense_layers)]
    batch_normalization_layers = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                                  range(num_dense_layers)]
    dropout_rate_layers = [trial.suggest_float(f'dropout_rate_{i}', 0.3, 0.7) for i in range(num_dense_layers)]

    # Add custom layers
    x = Flatten()(base_model.output)
    for i in range(num_dense_layers):
        x = Dense(dense_units[i])(x)
        if batch_normalization_layers[i]:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=dropout_rate_layers[i])(x)

    # Suggest the type of loss function and whether to use class weights
    loss_type = trial.suggest_categorical('loss_type', ['focal_loss', 'categorical_crossentropy'])

    # Only suggest these if Focal Loss is chosen
    alpha_phase1 = trial.suggest_float('alpha_phase1', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase1 = trial.suggest_float('gamma_phase1', 1.0, 5.0) if loss_type == 'focal_loss' else None

    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    weight_decay_phase1 = trial.suggest_float('weight_decay_phase1', 1e-6, 1e-2, log=True)

    # Other hyperparameters
    learning_rate_phase1 = trial.suggest_float('learning_rate_phase1', 1e-5, 1e-3, log=True)

    learning_rate_phase2 = trial.suggest_float('learning_rate_phase2', 1e-6, 1e-4, log=True)
    num_layers_unfreeze = trial.suggest_int('num_layers_unfreeze', 1, 10)
    alpha_phase2 = trial.suggest_float('alpha_phase2', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase2 = trial.suggest_float('gamma_phase2', 1.0, 5.0) if loss_type == 'focal_loss' else None
    weight_decay_phase2 = trial.suggest_float('weight_decay_phase2', 1e-6, 1e-2, log=True)

    prediction = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    # Training dataset setup
    train_preprocessed_set = (train_set
                              .map(load_and_preprocess_training(preprocess_input=preprocess_input), num_parallel_calls=2)
                              .cache()  # Cache the data after preprocessing
                              .shuffle(buffer_size=1000)
                              .batch(batch_size)
                              .repeat()
                              .prefetch(2))

    # Validation dataset setup
    validation_preprocessed_set = (validation_set
                                   .map(load_and_preprocess(preprocess_input=preprocess_input),
                                        num_parallel_calls=2)
                                   .cache()  # Cache the data after preprocessing
                                   .batch(batch_size)
                                   .prefetch(2))

    # Define the loss function based on the trial's suggestion
    if loss_type == 'focal_loss':
        loss_function = CategoricalFocalCrossentropy(alpha=alpha_phase1, gamma=gamma_phase1)
    else:
        loss_function = 'categorical_crossentropy'

    class_weight_arg = class_weights if use_class_weights and loss_type != 'focal_loss' else None

    # Compile the model
    model.compile(optimizer=AdamW(learning_rate=learning_rate_phase1,
                                  weight_decay=weight_decay_phase1),  # included weight_decay in optimizer
                  loss=loss_function, metrics=['accuracy'])

    # Train the model
    model.fit(
        train_preprocessed_set,
        steps_per_epoch=train_set_size,
        validation_data=validation_preprocessed_set,
        validation_steps=validation_set_size,
        epochs=50,
        verbose=1,
        class_weight=class_weight_arg,
        callbacks=[early_stopping, reduce_lr]
    )

    # Unfreeze the last num_layers_unfreeze layers
    for layer in base_model.layers[-num_layers_unfreeze:]:
        layer.trainable = True

    if loss_type == 'focal_loss':
        loss_function = CategoricalFocalCrossentropy(alpha=alpha_phase2, gamma=gamma_phase2)

    # Compile the model with a smaller learning rate
    model.compile(optimizer=AdamW(learning_rate=learning_rate_phase2,
                                  weight_decay=weight_decay_phase2),
                  loss=loss_function, metrics=['accuracy'])

    # Train the model again
    history = model.fit(
        train_preprocessed_set,
        steps_per_epoch=train_set_size,
        validation_data=validation_preprocessed_set,
        validation_steps=validation_set_size,
        epochs=100,
        verbose=1,
        class_weight=class_weight_arg,
        callbacks=[early_stopping, reduce_lr]
    )

    # return the validation accuracy of the lower validation loss epoch
    return 1.0 - history.history['val_accuracy'][np.argmin(history.history['val_loss'])]


def objective_vgg19(trial):
    global train_set, validation_set

    preprocess_input = resnet_preprocess_input
    # Load and configure the ResNet50 model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    batch_size = trial.suggest_int('batch', 16, 64)

    train_set_size = train_dataset_size // batch_size + (1 if train_dataset_size % batch_size != 0 else 0)
    validation_set_size = validation_dataset_size // batch_size + (
        1 if validation_dataset_size % batch_size != 0 else 0)

    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 32, 256) for i in range(num_dense_layers)]
    batch_normalization_layers = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                                  range(num_dense_layers)]
    dropout_rate_layers = [trial.suggest_float(f'dropout_rate_{i}', 0.3, 0.7) for i in range(num_dense_layers)]

    # Add custom layers
    x = Flatten()(base_model.output)
    for i in range(num_dense_layers):
        x = Dense(dense_units[i])(x)
        if batch_normalization_layers[i]:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=dropout_rate_layers[i])(x)

    # Suggest the type of loss function and whether to use class weights
    loss_type = trial.suggest_categorical('loss_type', ['focal_loss', 'categorical_crossentropy'])

    # Only suggest these if Focal Loss is chosen
    alpha_phase1 = trial.suggest_float('alpha_phase1', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase1 = trial.suggest_float('gamma_phase1', 1.0, 5.0) if loss_type == 'focal_loss' else None

    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    weight_decay_phase1 = trial.suggest_float('weight_decay_phase1', 1e-6, 1e-2, log=True)

    # Other hyperparameters
    learning_rate_phase1 = trial.suggest_float('learning_rate_phase1', 1e-5, 1e-3, log=True)

    learning_rate_phase2 = trial.suggest_float('learning_rate_phase2', 1e-6, 1e-4, log=True)
    num_layers_unfreeze = trial.suggest_int('num_layers_unfreeze', 1, 10)
    alpha_phase2 = trial.suggest_float('alpha_phase2', 0.2, 0.8) if loss_type == 'focal_loss' else None
    gamma_phase2 = trial.suggest_float('gamma_phase2', 1.0, 5.0) if loss_type == 'focal_loss' else None
    weight_decay_phase2 = trial.suggest_float('weight_decay_phase2', 1e-6, 1e-2, log=True)

    prediction = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)


    # Training dataset setup
    train_preprocessed_set = (train_set
                              .map(load_and_preprocess_training(preprocess_input=preprocess_input), num_parallel_calls=2)
                              .shuffle(buffer_size=1000)
                              .batch(batch_size)
                              .repeat()
                              .prefetch(2))

    # Validation dataset setup
    validation_preprocessed_set = (validation_set
                                   .map(load_and_preprocess(preprocess_input=preprocess_input),
                                        num_parallel_calls=2)
                                   .batch(batch_size)
                                   .prefetch(2))

    # Define the loss function based on the trial's suggestion
    if loss_type == 'focal_loss':
        loss_function = CategoricalFocalCrossentropy(alpha=alpha_phase1, gamma=gamma_phase1)
    else:
        loss_function = 'categorical_crossentropy'

    class_weight_arg = class_weights if use_class_weights and loss_type != 'focal_loss' else None

    # Compile the model
    model.compile(optimizer=AdamW(learning_rate=learning_rate_phase1,
                                  weight_decay=weight_decay_phase1),  # included weight_decay in optimizer
                  loss=loss_function, metrics=['accuracy'])

    # Train the model
    model.fit(
        train_preprocessed_set,
        steps_per_epoch=train_set_size,
        validation_data=validation_preprocessed_set,
        validation_steps=validation_set_size,
        epochs=50,
        verbose=1,
        class_weight=class_weight_arg,
        callbacks=[early_stopping, reduce_lr]
    )

    # Unfreeze the last num_layers_unfreeze layers
    for layer in base_model.layers[-num_layers_unfreeze:]:
        layer.trainable = True

    if loss_type == 'focal_loss':
        loss_function = CategoricalFocalCrossentropy(alpha=alpha_phase2, gamma=gamma_phase2)

    # Compile the model with a smaller learning rate
    model.compile(optimizer=AdamW(learning_rate=learning_rate_phase2,
                                  weight_decay=weight_decay_phase2),
                  loss=loss_function, metrics=['accuracy'])

    # Train the model again
    history = model.fit(
        train_preprocessed_set,
        steps_per_epoch=train_set_size,
        validation_data=validation_preprocessed_set,
        validation_steps=validation_set_size,
        epochs=100,
        verbose=1,
        class_weight=class_weight_arg,
        callbacks=[early_stopping, reduce_lr]
    )

    # return the validation accuracy of the lower validation loss epoch
    return 1.0 - history.history['val_accuracy'][np.argmin(history.history['val_loss'])]


if __name__ == '__main__':
    storage_location = "sqlite:///identifier.sqlite"
    study_name_vgg19 = "pretrained_models_skin_condition_study_vgg19_#1"
    study_name_vgg16 = "pretrained_models_skin_condition_study_vgg16_#1"
    study_name_resnet50 = "pretrained_models_skin_condition_study_resnet50_#1"

    # load study if it exists
    if not optuna.study.get_all_study_names(storage=storage_location).__contains__(study_name_vgg19):
        # optuna.delete_study(study_name_vgg19, storage=storage_location)
        optuna.create_study(study_name=study_name_vgg19, storage=storage_location, load_if_exists=True)
    if not optuna.study.get_all_study_names(storage=storage_location).__contains__(study_name_vgg16):
        # optuna.delete_study(study_name_vgg16, storage=storage_location)
        optuna.create_study(study_name=study_name_vgg16, storage=storage_location, load_if_exists=True)
    if not optuna.study.get_all_study_names(storage=storage_location).__contains__(study_name_resnet50):
        # optuna.delete_study(study_name_resnet50, storage=storage_location)
        optuna.create_study(study_name=study_name_resnet50, storage=storage_location, load_if_exists=True)

    study_vgg19 = optuna.load_study(study_name=study_name_vgg19, storage=storage_location)
    study_vgg16 = optuna.load_study(study_name=study_name_vgg16, storage=storage_location)
    study_resnet50 = optuna.load_study(study_name=study_name_resnet50, storage=storage_location)


    def clear_session_callback(study, trial):
        K.clear_session()
        gc.collect()


    # select study with the least number of trials
    if len(study_vgg19.trials) <= len(study_vgg16.trials) and len(study_vgg19.trials) <= len(study_resnet50.trials):
        study = optuna.create_study(direction="minimize", study_name=study_name_vgg19, storage=storage_location,
                                    load_if_exists=True)
        study.optimize(objective_vgg19, n_trials=1, show_progress_bar=True, callbacks=[clear_session_callback])
        prefix = 'VGG19'
    elif len(study_vgg16.trials) <= len(study_vgg19.trials) and len(study_vgg16.trials) <= len(study_resnet50.trials):
        study = optuna.create_study(direction="minimize", study_name=study_name_vgg16, storage=storage_location,
                                    load_if_exists=True)
        study.optimize(objective_vgg16, n_trials=1, show_progress_bar=True, callbacks=[clear_session_callback])
        prefix = 'VGG16'
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name_resnet50, storage=storage_location,
                                    load_if_exists=True)
        study.optimize(objective_resnet50, n_trials=1, show_progress_bar=True, callbacks=[clear_session_callback])
        prefix = 'ResNet50'

    # Create a new study or load an existing one

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

    # save the best hyperparameters as a pickle file
    import pickle

    with open(f'best_hyperparameters_{prefix}.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)
    with open(f'best_value_{prefix}.pkl', 'wb') as f:
        pickle.dump(study.best_value, f)
    with open(f'best_trial_{prefix}.pkl', 'wb') as f:
        pickle.dump(study.best_trial, f)
    #
    # # Visualization: Optimization History
    # optuna.visualization.plot_optimization_history(study)
    #
    # # Visualization: Hyperparameter Importance
    # optuna.visualization.plot_param_importances(study)

    print(f'Best hyperparameters for {prefix}: {study.best_params}')
