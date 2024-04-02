# src/reporting/report.py
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.data.dataset import create_tf_datasets, get_dataset_dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import tensorflow as tf


def load_model_and_hyperparams(model_filename, hyperparams_filename):
    model = tf.keras.models.load_model(model_filename)
    with open(hyperparams_filename, 'rb') as f:
        hyperparams = pickle.load(f)
    return model, hyperparams


def evaluate_model(model, dataset, labels):
    y_true = []
    y_pred = []
    for batch in dataset:
        X, y = batch
        y_true.extend(np.argmax(y.numpy(), axis=-1))
        y_pred.extend(np.argmax(model.predict(X), axis=-1))

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    cm = confusion_matrix(y_true, y_pred)

    return report, roc_auc, cm


def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


def generate_report(models_dir, output_file):
    dataset = get_dataset_dict()
    _, val_ds = create_tf_datasets(dataset, image_size=(256, 256), batch_size=None)
    labels = set(dataset['train']['dx'])

    models = []
    hyperparams = []
    for filename in os.listdir(models_dir):
        if filename.endswith(".h5") or filename.endswith(".keras"):
            model_filename = os.path.join(models_dir, filename)
            hyperparams_filename = os.path.join(models_dir, f"hyperparams_{filename[:-3]}.pkl")
            model, hyperparams_dict = load_model_and_hyperparams(model_filename, hyperparams_filename)
            models.append(model)
            hyperparams.append(hyperparams_dict)

    results = []
    for i, model in enumerate(models):
        report, roc_auc, cm = evaluate_model(model, val_ds, labels)
        results.append({
            'model_id': i + 1,
            'base_architecture': hyperparams[i]['base_model_architecture'],
            'hyperparams': hyperparams[i],
            'classification_report': report,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        })
        plot_confusion_matrix(cm, labels, f"Confusion Matrix - Model {i + 1}")

    with open(output_file, 'w') as f:
        f.write("# Skin Lesion Classification Report\n\n")

        f.write("## Model Performance\n\n")
        for result in results:
            f.write(f"### Model {result['model_id']} - {result['base_architecture']}\n\n")
            f.write(f"#### Hyperparameters\n")
            f.write(tabulate(list(result['hyperparams'].items()), headers=['Hyperparameter', 'Value'], tablefmt='pipe'))
            f.write("\n\n")
            f.write(f"#### Classification Report\n")
            f.write(tabulate(pd.DataFrame(result['classification_report']).T, headers='keys', tablefmt='pipe'))
            f.write("\n\n")
            f.write(f"#### ROC AUC Score\n")
            f.write(f"{result['roc_auc']:.4f}\n\n")
            f.write(f"#### Confusion Matrix\n")
            f.write(
                f"![Confusion Matrix - Model {result['model_id']}](Confusion Matrix - Model {result['model_id']}.png)\n\n")

        f.write("## Comparison of Models\n\n")
        f.write("### ROC AUC Scores\n\n")
        roc_auc_scores = [result['roc_auc'] for result in results]
        f.write(tabulate(list(zip([result['base_architecture'] for result in results], roc_auc_scores)),
                         headers=['Model', 'ROC AUC Score'], tablefmt='pipe'))
        f.write("\n\n")

        f.write("### Classification Reports\n\n")
        for label in labels:
            f.write(f"#### {label}\n\n")
            f.write(tabulate([[result['base_architecture'], result['classification_report'][label]['precision'],
                               result['classification_report'][label]['recall'],
                               result['classification_report'][label]['f1-score'],
                               result['classification_report'][label]['support']] for result in results],
                             headers=['Model', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='pipe'))
            f.write("\n\n")

        f.write("## Conclusion\n\n")
        best_model_idx = np.argmax([result['roc_auc'] for result in results])
        best_model = results[best_model_idx]
        f.write(
            f"The best-performing model is Model {best_model['model_id']} with the {best_model['base_architecture']} "
            f"architecture, achieving an ROC AUC score of {best_model['roc_auc']:.4f}.\n\n")
        f.write(
            "The model's performance can be further improved by fine-tuning the hyperparameters, increasing the "
            "training data, and exploring different model architectures.\n\n")

    print(f"Report generated and saved to {output_file}")

    return output_file

def generate_model_report(model):
    report = f"Model Summary:\n{model.summary()}\n\n"

    report += "Layers:\n"
    for layer in model.layers:
        report += f"  - {layer.name}: {type(layer).__name__}\n"

    report += "\nModel Configuration:\n"
    config = model.get_config()
    for key, value in config.items():
        report += f"  - {key}: {value}\n"

    report += "\nModel Weights:\n"
    for layer in model.layers:
        weights = layer.get_weights()
        report += f"  - {layer.name}:\n"
        for i, weight in enumerate(weights):
            report += f"    - Weight {i}: Shape={weight.shape}, Min={weight.min()}, Max={weight.max()}, Mean={weight.mean()}\n"

    return report


if __name__ == "__main__":

    import os
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get the project root directory from the environment variable
    project_root = Path(os.getenv('PROJECT_ROOT'))
    models_dir = project_root / "models/best_models"
    output_file = project_root / "reports/skin_lesion_classification_report.md"

    generate_report(str(models_dir), str(output_file))

