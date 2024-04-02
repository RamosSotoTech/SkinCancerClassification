# src/reporting/comparative_analysis.py
import pickle
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.data.dataset import create_tf_datasets, get_dataset_dict
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd


import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))

def load_model_and_hyperparams(model_filename, hyperparams_filename):
    model = tf.keras.models.load_model(model_filename)
    with open(hyperparams_filename, 'rb') as f:
        hyperparams = pickle.load(f)
    return model, hyperparams


def evaluate_model(model, dataset):
    y_true = []
    y_pred = []
    for batch in dataset:
        X, y = batch
        y_true.extend(np.argmax(y.numpy(), axis=-1))
        y_pred.extend(np.argmax(model.predict(X), axis=-1))

    return y_true, y_pred


def compare_models(models_dir):
    dataset = get_dataset_dict()
    _, val_ds = create_tf_datasets(dataset)
    labels = dataset['train'].features['dx'].names

    models = []
    hyperparams = []
    for filename in os.listdir(models_dir):
        if filename.endswith(".h5"):
            model_filename = os.path.join(models_dir, filename)
            hyperparams_filename = os.path.join(models_dir, f"hyperparams_{filename[:-3]}.pkl")
            model, hyperparams_dict = load_model_and_hyperparams(model_filename, hyperparams_filename)
            models.append(model)
            hyperparams.append(hyperparams_dict)

    results = []
    for i, model in enumerate(models):
        y_true, y_pred = evaluate_model(model, val_ds)
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        cm = confusion_matrix(y_true, y_pred)
        results.append({
            'model_id': i + 1,
            'hyperparams': hyperparams[i],
            'classification_report': report,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        })

    return results


def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def compare_hyperparams(results):
    # Compare hyperparameters and their impact on model performance

    for result in results:
        print(f"Model ID: {result['model_id']}")
        for key, value in result['hyperparams'].items():
            print(f"{key}: {value}")
        print(f"ROC AUC score: {result['roc_auc']}")
        print("\n")


def compare_classification_reports(results):
    # Compare classification reports of different models
    for result in results:
        print(f"\n\nModel ID: {result['model_id']}")
        print(pd.DataFrame(result['classification_report']).transpose())


def compare_roc_auc_scores(results):
    # Compare ROC AUC scores of different models

    model_ids = [result['model_id'] for result in results]
    roc_aucs = [result['roc_auc'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.bar(model_ids, roc_aucs, color='blue')
    plt.xlabel('Model ID')
    plt.ylabel('ROC AUC score')
    plt.title('Comparison of ROC AUC scores')
    plt.show()


def compare_confusion_matrices(results):
    # Compare confusion matrices of different models
    for result in results:
        print(f"\n\nModel ID: {result['model_id']}")
        plot_confusion_matrix(result['confusion_matrix'], title=f"Model {result['model_id']} Confusion Matrix")


if __name__ == "__main__":
    models_dir = str(project_root) + "best_models"
    results = compare_models(models_dir)

    # Perform relevant comparisons and analysis
    compare_hyperparams(results)
    compare_classification_reports(results)
    compare_roc_auc_scores(results)
    compare_confusion_matrices(results)
