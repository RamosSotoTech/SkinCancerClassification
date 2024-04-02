# src/main.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))

import os
import multiprocessing
import numpy as np
from datasets import load_dataset
import matplotlib
from optuna.integration import TFKerasPruningCallback
import optuna
import gc

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

from tensorflow.data import Dataset
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K

from sklearn.utils import compute_class_weight

matplotlib.use('TkAgg')

from src.models.architectures import build_model
from src.data.dataset import create_tf_datasets
from src.training.callbacks import BestValueTracker
from src.data.dataset import get_dataset_dict


def objective(trial):
    batch_size = trial.suggest_int('batch', 32, 64)
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

    print(f"Trial parameters: {trial.params}")

    model, num_layers_unfreeze = build_model(trial, base_model_architecture, pre_trained_weights, attention_mechanism,
                                             dataset, image_size)

    train_ds, val_ds = create_tf_datasets(dataset, image_size=image_size, batch_size=batch_size)

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

    opt = AdamW(weight_decay=weight_decay_phase2, learning_rate=learning_rate_phase2, amsgrad=use_amsgrad, clipnorm=1.0)
    if loss_type == 'focal_loss':
        loss_fn = CategoricalFocalCrossentropy(alpha=alpha_phase2, gamma=gamma_phase2)
        class_weights = None

    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    for layer in model.layers[-num_layers_unfreeze:]:
        layer.trainable = True

    best_value_tracker = BestValueTracker()

    try:
        model.fit(train_ds
                  , validation_data=val_ds
                  , epochs=100
                  , class_weight=class_weights
                  , callbacks=[TFKerasPruningCallback(trial, 'val_accuracy'),
                               tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                                restore_best_weights=True),
                               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                                                    cooldown=2),
                               best_value_tracker
            ]
                  , workers=num_cores
                  , use_multiprocessing=True
                  )
    except optuna.exceptions.TrialPruned as e:
        best_accuracy = best_value_tracker.best_value
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


def main():
    # optuna callbacks to free up keras backend memory
    def clear_session_callback(study, trial):
        K.clear_session()
        gc.collect()

    study_name = 'skin_lesion_classification_with_HAM10000_dataset'
    storage_name = f"sqlite:///../experiments/{study_name}.db"
    # storage_name = f"sqlite:///"+ str(project_root) + "/experiments/{study_name}_testing.db"

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

    trials = study.get_trials(
        states=[optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING, optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED])

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


if __name__ == "__main__":
    import src.data.dataset as dat
    dataset: Dataset = get_dataset_dict()
    unique_classes = dat.get_default_classes()
    num_classes = len(unique_classes)
    localizations = dat.get_default_localization()
    num_localizations = len(localizations)
    image_size = (256, 256)

    main()
