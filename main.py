import os
from pathlib import Path
from dotenv import load_dotenv
import argparse
from datasets import load_dataset
from src.data.dataset import preprocess_dataset
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.reporting.report import generate_report, generate_model_report
import optuna

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_path)

    image_size = (args.image_size, args.image_size)

    # Preprocess the dataset
    train_ds, val_ds = preprocess_dataset(dataset, image_size, args.batch_size)

    # Train the model
    model, history = train_model(dataset, args.study_name, args.database_config,
                                 image_size, args.batch_size)

    # Evaluate the model
    report, roc_auc, cm = evaluate_model(model, dataset)

    # Generate the report
    report_output_dir = str(project_root / args.output_dir)

    # Generate the generic report
    model_report = generate_model_report(model)

    # Save the model report
    with open(f"{report_output_dir}/model_report.txt", "w") as f:
        f.write(model_report)

    print(f"Model training completed. Report generated at: {report_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skin Lesion Classification')
    parser.add_argument('--dataset-path', type=str, default='marmal88/skin_cancer', help='Path to the dataset')
    parser.add_argument('--image-size', type=int, default=256, help='Image size for preprocessing')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--study-name', type=str, default='skin_lesion_classification_with_HAM10000_dataset',
                        help='Name of the Optuna study')
    parser.add_argument('--database-config', type=str, default="sqlite:///" + str(
        project_root / 'experiments/skin_lesion_classification_with_HAM10000_dataset.db'),
                        help='Path to the Optuna study database')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save the outputs')

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)