# src/reporting/generic_report.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))


import pandas as pd
import matplotlib.pyplot as plt
from src.reporting.report import generate_report
from src.reporting.optimization_analysis import plot_optimization_history, plot_param_importances
import optuna


def generate_generic_report(study_name, output_dir):
    # Load the optimization study
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{str(project_root)}/experiments/{study_name}.db")

    # Generate the report
    report = f"# Skin Lesion Classification Study\n\n"

    # Introduction
    report += "## Introduction\n"
    report += "...\n\n"

    # Methodology
    report += "## Methodology\n"
    report += "...\n\n"

    # Results
    report += "## Results\n"
    report += "...\n\n"

    # Generate model evaluation report
    model_report = generate_report("best_models", "skin_lesion_classification_report.md")
    report += model_report

    # Optimization analysis
    report += "### Hyperparameter Optimization\n"
    plot_optimization_history(study)
    plt.savefig(f"{output_dir}/optimization_history.png")
    report += f"![Optimization History]({output_dir}/optimization_history.png)\n\n"

    plot_param_importances(study)
    plt.savefig(f"{output_dir}/param_importances.png")
    report += f"![Parameter Importances]({output_dir}/param_importances.png)\n\n"

    # Conclusion
    report += "## Conclusion\n"
    report += "...\n\n"

    # References
    report += "## References\n"
    report += "...\n\n"

    # Save the report
    with open(project_root / f"{output_dir}/{study_name}_report.md", "w") as f:
        f.write(report)

    print(f"Generic report generated and saved to {output_dir}/{study_name}_report.md")


if __name__ == "__main__":
    study_name = "skin_lesion_classification_with_HAM10000_dataset"
    output_dir = str(project_root) + "reports"
    generate_generic_report(study_name, output_dir)