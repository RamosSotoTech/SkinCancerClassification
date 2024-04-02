# src/models/fit_best_models.py
import argparse
import optuna
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
import pickle
import tensorflow as tf

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))
sys.path.append(str(project_root))

# Import the required modules using the absolute path
from src.models.architectures import build_model_v2
from src.data.dataset import create_tf_datasets, get_dataset_dict

def fit_best_model_for_architecture(study_name, storage_name, base_architecture, output_dir):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    trials = study.get_trials(deepcopy=False)

    best_trial = None
    best_value = -1.0

    for trial in trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and \
                trial.params['base_model_architecture'] == base_architecture and \
                trial.value > best_value:
            best_trial = trial
            best_value = trial.value

    if best_trial is None:
        print(f"No completed trials found for base architecture: {base_architecture}")
        return

    dataset = get_dataset_dict()
    train_ds, val_ds = create_tf_datasets(dataset)
    labels = dataset['train'].features['dx'].names

    print(f"Fitting model for base architecture: {base_architecture}")
    model, opt, class_weights = build_model_v2(best_trial, dataset['train'].features['localization'].names, labels)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Evaluate the model
    print("Evaluating the model...")
    y_true = []
    y_pred = []
    for batch in val_ds:
        X, y = batch
        y_true.extend(np.argmax(y.numpy(), axis=-1))
        y_pred.extend(np.argmax(model.predict(X), axis=-1))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("ROC AUC Score:")
    print(roc_auc_score(y_true, y_pred, multi_class='ovr'))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save the model
    model_filename = f"{output_dir}/model_{base_architecture}.h5"
    model.save(model_filename)

    # Save the hyperparameters
    hyperparams_filename = f"{output_dir}/hyperparams_{base_architecture}.pkl"
    with open(hyperparams_filename, 'wb') as f:
        pickle.dump(best_trial.params, f)

    print(f"Model saved to {model_filename}")
    print(f"Hyperparameters saved to {hyperparams_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit the best model for a specific base architecture from an Optuna '
                                                 'study.')
    parser.add_argument('--study-name', type=str, required=True, help='Name of the Optuna study')
    parser.add_argument('--storage-name', type=str, required=True, help='Storage name for the Optuna study')
    parser.add_argument('--base-architecture', type=str, required=True, help='Base architecture of the model to fit')
    parser.add_argument('--output-dir', type=str, default=str(project_root) + 'best_models', help='Output directory for saving models and '
                                                                              'hyperparameters')

    args = parser.parse_args()

    fit_best_model_for_architecture(args.study_name, args.storage_name, args.base_architecture, args.output_dir)
