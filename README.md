# Skin Lesion Classification with Hyperparameter Optimization

This project focuses on the classification of skin lesions using deep learning techniques with TensorFlow. It incorporates various state-of-the-art architectures and techniques, including attention mechanisms, data augmentation, and hyperparameter optimization using Optuna.

## Project Structure

The project is organized into the following directories:

- `src/`: Contains the source code for the project.
  - `data/`: Contains scripts for data loading and preprocessing.
  - `models/`: Contains scripts for model architecture and training.
  - `reporting/`: Contains scripts for generating reports and visualizations.
  - `training/`: Contains utility scripts for training and callbacks.
- `experiments/`: Contains Optuna study databases for hyperparameter optimization.
- `reports/`: Contains generated reports and visualizations.
- `models/`: Contains saved models and their hyperparameters.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Optuna
- Pandas, NumPy, Matplotlib, scikit-learn
- Datasets library (Hugging Face)

## Setup

1. Clone the repository:
git clone https://github.com/RamosSotoTech/SkinCancerClassification.git

2. Install the required Python packages:
pip install -r requirements.txt

3. Set up the environment variables:
- Create a `.env` file in the project root directory.
- Add the following line to the `.env` file:
  ```
  PROJECT_ROOT=.
  ```

4. Download the HAM10000 dataset and place it in the `data/` directory.

## Usage

1. Preprocess the dataset by running the following command:
python src/data/dataset.py

2. Run the main script to start the training and hyperparameter optimization process:
python main.py --study-name <study_name> --database-config <database_config> --output-dir <output_directory>

- `<study_name>`: Name of the Optuna study (e.g., "skin_lesion_classification_study").
- `<database_config>`: Path to the Optuna study database (e.g., "experiments/study.db").
- `<output_directory>`: Directory to save the trained models and generated reports (e.g., "output/").

3. The script will perform hyperparameter tuning using Optuna and train models based on the specified configurations. The training process includes custom callbacks to monitor validation accuracy and adjust learning rates.

4. After the training is complete, the script will generate reports and visualizations in the `reports/` directory.

## Jupyter Notebook

An outdated Jupyter notebook is provided with the project. Please note that the notebook may not be fully aligned with the current scripts and may require updates to work correctly.
To convert the notebook to a Word document, run the following command:

## Results

The trained models and their hyperparameters will be saved in the `models/` directory. The generated reports and visualizations will be available in the `reports/` directory.

## Limitations

- The project assumes the use of the HAM10000 dataset. If you want to use a different dataset, you may need to modify the data loading and preprocessing scripts accordingly.
- The Jupyter notebook provided is outdated and may not be fully aligned with the current scripts. It may require updates to work correctly.

## Contributing

Contributions to improve the project are welcome. Please adhere to the project's coding standards and submit pull requests for any enhancements.

## License

Distributed under the MIT License. See `LICENSE` for more information.