# src/reporting/optimization_analysis.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import argparse


def get_trials_dataframe(study):
    trials = study.trials_dataframe()
    trials = trials.join(pd.DataFrame([t.params for t in study.get_trials()], index=trials.index))
    return trials


def plot_optimization_history(study):
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()


def plot_param_importances(study):
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


def plot_parallel_coordinate(study):
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.show()


def plot_slice(study, params):
    fig = optuna.visualization.plot_slice(study, params=params)
    fig.show()


def plot_contour(study, params):
    fig = optuna.visualization.plot_contour(study, params=params)
    fig.show()


def plot_intermediate_values(study):
    trials = study.get_trials(states=[optuna.trial.TrialState.PRUNED])

    if len(trials) == 0:
        print("No pruned trials found. Skipping intermediate value plot.")
        return

    fig = make_subplots(rows=len(trials), cols=1, subplot_titles=[f"Trial {t.number}" for t in trials])

    for i, t in enumerate(trials):
        data = optuna.visualization._get_intermediate_plot_data(t)
        if data is None:
            fig.add_trace(go.Scatter(), row=i + 1, col=1)
        else:
            for step, value in zip(data.step, data.intermediate_values):
                fig.add_trace(
                    go.Scatter(
                        x=[step],
                        y=[value],
                        mode="markers",
                        marker={"size": 5},
                        name=f"Trial {t.number}",
                    ),
                    row=i + 1,
                    col=1,
                )

    fig.update_layout(height=500 * len(trials), title_text="Intermediate Values", showlegend=False)
    fig.show()


def compare_models(trials_df):
    model_architectures = trials_df['base_model_architecture'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in model_architectures:
        model_trials = trials_df[trials_df['base_model_architecture'] == model]
        sns.kdeplot(model_trials['value'], label=model, ax=ax)

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Density')
    ax.set_title('Model Architecture Comparison')
    ax.legend()
    plt.show()


def compare_attention_mechanisms(trials_df):
    attention_mechanisms = trials_df['attention_mechanism'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    for mechanism in attention_mechanisms:
        mechanism_trials = trials_df[trials_df['attention_mechanism'] == mechanism]
        sns.kdeplot(mechanism_trials['value'], label=mechanism, ax=ax)

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Density')
    ax.set_title('Attention Mechanism Comparison')
    ax.legend()
    plt.show()


def get_default_study_trials():
    study_name = 'skin_lesion_classification_with_HAM10000_dataset'
    storage_name = f"sqlite:///" + str(project_root) + "experiments/skin_lesion_classification_with_HAM10000_dataset.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    trials = study.trials_dataframe()
    trials = trials.join(pd.DataFrame([t.params for t in study.get_trials()], index=trials.index))
    return trials

def main():
    global project_root
    parser = argparse.ArgumentParser(description="Analyze optimization results.")
    parser.add_argument("--study-name", type=str, required=False, help="Name of the study.",
                        default='skin_lesion_classification_with_HAM10000_dataset')
    parser.add_argument("--storage-name", type=str, required=False, help="Name of the storage.",
                        default=f"sqlite:///" + str(project_root) + "experiments/skin_lesion_classification_with_HAM10000_dataset.db")

    args = parser.parse_args()

    if 'PROJECT_ROOT' in os.environ:
        project_root = Path(os.environ['PROJECT_ROOT'])

        storage_name = project_root / args.storage_name
    else:
        print("Environment variable 'PROJECT_ROOT' is not set.")
        storage_name = args.storage_name

    study_name = args.study_name

    study = optuna.load_study(study_name=study_name, storage=storage_name)
    trials_df = get_trials_dataframe(study)

    print("Optimization History:")
    plot_optimization_history(study)

    print("Parameter Importances:")
    plot_param_importances(study)

    print("Parallel Coordinate Plot:")
    plot_parallel_coordinate(study)

    print("Slice Plot:")
    plot_slice(study, params=['base_model_architecture', 'attention_mechanism'])

    print("Contour Plot:")
    plot_contour(study, params=['learning_rate_phase1', 'learning_rate_phase2'])

    print("Intermediate Values:")
    plot_intermediate_values(study)

    print("Model Architecture Comparison:")
    compare_models(trials_df)

    print("Attention Mechanism Comparison:")
    compare_attention_mechanisms(trials_df)


if __name__ == "__main__":
    main()
