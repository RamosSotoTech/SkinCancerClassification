# src/models/architectures.py
import os
from pathlib import Path

from tensorflow.keras.applications import VGG16, VGG19, ResNet101V2, InceptionResNetV2, Xception, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from src.models.layers import squeeze_excite_block, cbam_block
from sklearn.utils import compute_class_weight
import numpy as np
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalFocalCrossentropy
import argparse

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = Path(os.getenv('PROJECT_ROOT'))

def build_model(trial, base_model_architecture, pre_trained_weights, attention_mechanism, dataset, image_size,
                unique_classes=None):
    if unique_classes is None:
        import warnings
        from src.data.dataset import get_default_classes
        warnings.warn("Unique classes were not provided. The program will continue, Using the default classes set.",
                      UserWarning)
        unique_classes = get_default_classes()

    weight = 'imagenet' if pre_trained_weights else None
    # Define the base model
    if base_model_architecture == 'VGG16':
        base_model = VGG16(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'VGG19':
        base_model = VGG19(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'ResNet101V2':
        base_model = ResNet101V2(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'InceptionResNetV2':
        base_model = InceptionResNetV2(include_top=False, weights=weight,
                                       input_shape=image_size + (3,))
    elif base_model_architecture == 'Xception':
        base_model = Xception(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'MobileNetV2':
        base_model = MobileNetV2(include_top=False, weights=weight, input_shape=image_size + (3,))
    else:
        raise ValueError(f"Invalid base model architecture: {base_model_architecture}")

    if pre_trained_weights:
        num_layers_unfreeze = trial.suggest_int('num_layers_unfreeze', 1, len(base_model.layers))
        for layer in base_model.layers[-num_layers_unfreeze:]:
            layer.trainable = False
    else:
        num_layers_unfreeze = len(base_model.layers)
        for layer in base_model.layers:
            layer.trainable = True

    attention_func = None
    if attention_mechanism == 'SENet':
        attention_func = squeeze_excite_block
    elif attention_mechanism == 'CBAM':
        attention_func = cbam_block

    # Model definition
    metadata_input_shape = (len(dataset['train'][0]['localization']) + 2,)
    metadata_input = tf.keras.layers.Input(shape=metadata_input_shape, name='metadata_input')
    image_input = tf.keras.layers.Input(shape=image_size + (3,), name='image_input')

    x = base_model(image_input)

    if pre_trained_weights:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True

    # Apply the attention mechanism if any
    if attention_func:
        x = attention_func(x)

    x = Flatten()(x)
    x = tf.keras.layers.concatenate([x, metadata_input])

    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    for i in range(num_dense_layers):
        dense_units = trial.suggest_int(f'dense_units_{i}', 32, 1024)
        batch_normalization = trial.suggest_categorical(f'batch_normalization_{i}', [True, False])
        dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.3, 0.7)

        x = Dense(dense_units)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(dropout_rate)(x)

    predictions = Dense(len(unique_classes), activation='softmax')(x)

    model = Model(inputs=[image_input, metadata_input], outputs=predictions)

    return model, num_layers_unfreeze


def build_model_v2(hyperparameters, localizations, labels, image_size=(256, 256)):

    batch_size = hyperparameters.params['batch']
    num_dense_layers = hyperparameters.params['num_dense_layers']
    dense_units = [hyperparameters.params[f'dense_units_{i}'] for i in range(num_dense_layers)]
    batch_normalization_layers = [hyperparameters.params[f'batch_normalization_{i}'] for i in range(num_dense_layers)]
    dropout_rate_layers = [hyperparameters.params[f'dropout_rate_{i}'] for i in range(num_dense_layers)]
    loss_type = hyperparameters.params['loss_type']
    use_class_weights = hyperparameters.params['use_class_weights']
    learning_rate_phase2 = hyperparameters.params['learning_rate_phase2']
    alpha_phase2 = hyperparameters.params['alpha_phase2'] if loss_type == 'focal_loss' else None
    gamma_phase2 = hyperparameters.params['gamma_phase2'] if loss_type == 'focal_loss' else None
    weight_decay_phase2 = hyperparameters.params['weight_decay_phase2']
    base_model_architecture = hyperparameters.params['base_model_architecture']
    attention_mechanism = hyperparameters.params['attention_mechanism']
    use_amsgrad = hyperparameters.params['use_amsgrad']
    pre_trained_weights = hyperparameters.params['pre_trained_weights']
    unique_classes = set(labels)

    weight = 'imagenet' if pre_trained_weights else None
    # Define the base model
    if base_model_architecture == 'VGG16':
        base_model = tf.keras.applications.VGG16(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'VGG19':
        base_model = tf.keras.applications.VGG19(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'ResNet101V2':
        base_model = tf.keras.applications.ResNet101V2(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'InceptionResNetV2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights=weight,
                                                             input_shape=image_size + (3,))
    elif base_model_architecture == 'Xception':
        base_model = tf.keras.applications.Xception(include_top=False, weights=weight, input_shape=image_size + (3,))
    elif base_model_architecture == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weight, input_shape=image_size + (3,))
    else:
        raise ValueError(f"Invalid base model architecture: {base_model_architecture}")

        # unfreeze layers for phase 2 (1 to total layers)
    if pre_trained_weights:
        num_layers_unfreeze = hyperparameters.params['num_layers_unfreeze']
        for layer in base_model.layers[-num_layers_unfreeze:]:
            layer.trainable = False
    else:
        num_layers_unfreeze = len(base_model.layers)
        for layer in base_model.layers:
            layer.trainable = True

    attention_func = None
    if attention_mechanism == 'SENet':
        attention_func = squeeze_excite_block
    elif attention_mechanism == 'CBAM':
        attention_func = cbam_block

    # Prepare the dataset

    # Model definition
    metadata_input_shape = (len(localizations) + 2,)
    metadata_input = tf.keras.layers.Input(shape=metadata_input_shape, name='metadata_input')
    image_input = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3), name='image_input')

    # Apply the base model
    x = base_model(image_input)

    if pre_trained_weights:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True

    # Apply the attention mechanism if any
    if attention_func:
        x = attention_func(x)

    x = Flatten()(x)
    x = tf.keras.layers.concatenate([x, metadata_input])

    for i in range(num_dense_layers):
        x = Dense(dense_units[i])(x)
        if batch_normalization_layers[i]:
            x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(dropout_rate_layers[i])(x)
    predictions = Dense(len(labels), activation='softmax')(x)

    model = Model(inputs=[image_input, metadata_input], outputs=predictions)

    # Phase 2
    opt = AdamW(weight_decay=weight_decay_phase2, learning_rate=learning_rate_phase2, amsgrad=use_amsgrad, clipnorm=1.0)
    if loss_type == 'focal_loss':
        loss_fn = CategoricalFocalCrossentropy(alpha=alpha_phase2, gamma=gamma_phase2)
        class_weights = None
    else:
        loss_fn = 'categorical_crossentropy'
        if use_class_weights:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(unique_classes),
                                           y=[labels])
            class_weights = dict(enumerate(weights))
        else:
            class_weights = None
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    for layer in base_model.layers[-num_layers_unfreeze:]:
        layer.trainable = True

    return model, opt, class_weights, batch_size


if __name__ == '__main__':
    import optuna
    from src.data.dataset import get_default_classes, get_default_localization, get_train_split
    import pickle
    parser = argparse.ArgumentParser(description='Build the model for the specified study and database.')
    parser.add_argument('--study', type=str, required=True, help='The name of your study.')
    parser.add_argument('--database', type=str, default=f'{str(project_root)}/experiments/skin_lesion_db.sqlite',
                        help='The specific database configuration you\'d want to use.')

    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{args.database}")
    trials = study.best_trial
    hyperparameters = trials.params

    labels = get_train_split()['dx']
    localizations = get_default_localization()

    model, optimizer, class_weights, batch_size = build_model_v2(hyperparameters=hyperparameters, localizations=localizations, labels=labels)
    model.summary()

    if 'PROJECT_ROOT' in os.environ:
        project_root = Path(os.environ['PROJECT_ROOT'])
        models_dir = project_root / "models/best_models"
        output_file = project_root / "reports/skin_lesion_classification_report.md"

        # save model
        model.save(f"{models_dir}/{args.study}.h5")

        # sometimes the optimizer is not saved correctly with .keras format
        with open(f"{models_dir}/{args.study}_optimizer.pkl", "wb") as f:
            pickle.dump(optimizer, f)

        with open(f"{models_dir}/{args.study}_class_weights.pkl", "wb") as f:
            pickle.dump(class_weights, f)

        with open(f"{models_dir}/{args.study}_batch_size.pkl", "wb") as f:
            pickle.dump(batch_size, f)

        print("Model built successfully.")
    else:
        print("Environment variable 'PROJECT_ROOT' is not set.")
