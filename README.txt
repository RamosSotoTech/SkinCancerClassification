# Project Title: Advanced Skin Lesion Classification with TensorFlow
This project is an advanced implementation for classifying skin lesions using deep learning techniques with TensorFlow. It incorporates various state-of-the-art architectures and techniques, including attention mechanisms, data augmentation, and hyperparameter optimization using Optuna.

## Key Features
1. **Data Preprocessing**: Adaptive resizing, data augmentation (random flips, rotations), and metadata inclusion.
2. **Deep Learning Models**: Utilizes pre-trained models like VGG16, VGG19, ResNet101V2, InceptionResNetV2, Xception, and MobileNetV2 with options for attention mechanisms (SENet, CBAM).
3. **Hyperparameter Optimization**: Uses Optuna for optimizing model parameters, architectures, and training strategies.
4. **Custom Callbacks**: Implements custom TensorFlow callbacks for tracking best validation metrics and managing training phases.
5. **Environment Configuration**: Configures TensorFlow to efficiently use GPU resources and prevent memory overflow issues.

## Architecture
The system is designed to be modular, with distinct components for data preprocessing, model construction, training, and hyperparameter tuning:

- **Data Handling**: Utilizes the datasets library for loading and managing the dataset, with preprocessing functions to prepare images for training.
- **Model Building**: Dynamically constructs models based on selected architectures, attention mechanisms, and fully connected layers, integrating image and metadata inputs.
- **Training Pipeline**: Employs a two-phase training strategy, first fine-tuning the top layers and then unfreezing and training more layers for improved accuracy.
- **Hyperparameter Tuning**: Leverages Optuna to explore a vast hyperparameter space, aiming to find the optimal configuration for the best classification performance.

## Setup
### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Optuna
- Pandas, NumPy, Matplotlib, scikit-learn
- Datasets library

### Installation
1. Clone the repository:
2. Install required Python packages:

```bash
pip install -r requirements.txt
```

### Configuration
Set up the TensorFlow environment to optimize GPU usage by setting environment variables as specified in the script. Ensure your system has the appropriate CUDA and cuDNN versions installed to match the TensorFlow version you're using.

Notes:
* The script is configured for TensorFlow 2.x. If you're using a different version, adjust the environment variables accordingly.
* The script assumes you have a GPU with CUDA and cuDNN installed. If you're using a CPU-only setup, remove the GPU-related environment variables.

## Usage
### Data Preparation
Load your dataset using the datasets' library. The project is configured to use the "marmal88/skin_cancer" dataset as an example. Adapt the data loading logic if you're using a different dataset.

### Model Training
Run the main script to start the training and hyperparameter optimization process:
This script will perform hyperparameter tuning using Optuna and train models based on the specified configurations. The training process includes a custom callback to monitor validation accuracy and adjust learning rates.

### Evaluation
The script logs the performance of each trial, including accuracy metrics. The best performing models and their parameters are saved for further analysis and deployment.

## Contributing
Contributions to improve the project are welcome. Please adhere to the project's coding standards and submit pull requests for any enhancements.

## License

Distributed under the MIT License. See `LICENSE` for more information.