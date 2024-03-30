import numpy as np
import tensorflow as tf
import os

from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
import pathlib
from collections import Counter

# Get the directory where the current script is located
script_dir = pathlib.Path(__file__).parent.resolve()

# Define the path to the .env file
config_env_path = script_dir / 'config.env'  # Assuming the .env file is in the project root

# Make sure the .env file exists
assert os.path.exists(config_env_path), f"Can't find .env file at {config_env_path}"

# Load the .env file
load_dotenv(dotenv_path=config_env_path)

assert 'DATASET_DIR' in os.environ, "Please set 'DATASET_DIR' environment variable"

# absolute paths
dataset_path_str: str = os.getenv('DATASET_DIR')
dataset_path: str = os.path.abspath(dataset_path_str)
train_path: str = os.path.join(dataset_path, 'Train')
test_path: str = os.path.join(dataset_path, 'Test')

image_size: tuple[int, int] = (224, 224)
batch_size: int = 32

train_datagen: ImageDataGenerator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=180,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2)

test_datagen: ImageDataGenerator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)


train_generator: tf.keras.preprocessing.image.DirectoryIterator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=None,
    class_mode='categorical',
    classes=os.listdir(train_path),
    subset='training',
    shuffle=True)

validation_generator: tf.keras.preprocessing.image.DirectoryIterator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=None,
    class_mode='categorical',
    classes=os.listdir(train_path),
    subset='validation',
    shuffle=True)

test_generator: tf.keras.preprocessing.image.DirectoryIterator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=None,
    class_mode='categorical',
    classes=os.listdir(test_path),
    shuffle=True)

classes = list(train_generator.class_indices.keys())
num_classes = len(classes)
counts = list(train_generator.class_indices.values())
y_train = train_generator.classes

# Derive the unique classes directly from your dataset
unique_classes = np.unique(y_train)

# Compute class weights using the unique classes derived from your dataset
weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)

# Create a dictionary mapping class indices to their respective weights
class_weights = dict(enumerate(weights))

classes = os.listdir(train_path)

train_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_path,
    labels='inferred',
    label_mode='categorical',
    class_names=classes,
    color_mode='rgb',
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training',
    interpolation='bilinear',
)
train_dataset_size = train_generator.samples

validation_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_path,
    labels='inferred',
    label_mode='categorical',
    class_names=classes,
    color_mode='rgb',
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation',
    interpolation='bilinear',
)
validation_dataset_size = validation_generator.samples

test_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_path,
    labels='inferred',
    label_mode='categorical',
    class_names=classes,
    color_mode='rgb',
    image_size=image_size,
    shuffle=True,
    seed=42,
    interpolation='bilinear',
)
test_dataset_size = test_generator.samples


if __name__ == '__main__':
    print('Number of classes:', len(classes))
    print('Classes:', classes)
    print('Number of training images:', train_generator.samples)
    print('Number of test images:', test_generator.samples)
    print('Number of batches in trainloader:', len(train_generator))
    print('Number of batches in testloader:', len(test_generator))

    # Get the count of each class in the training set
    train_counter = Counter(train_generator.classes)
    validation_counter = Counter(validation_generator.classes)
    test_counter = Counter(test_generator.classes)
    all_counter = train_counter + validation_counter + test_counter

    # Output results
    class_counts_train = {classes[k]: v for k, v in train_counter.items()}
    class_counts_val = {classes[k]: v for k, v in validation_counter.items()}
    class_counts_test = {classes[k]: v for k, v in test_counter.items()}
    class_counts_all = {classes[k]: v for k, v in all_counter.items()}

    print('Count of each class in the training set:', class_counts_train)
    print('Count of each class in the validation set:', class_counts_val)
    print('Count of each class in the test set:', class_counts_test)
    print('Count of each class in the all set:', class_counts_all)

    print('Class weights:', class_weights)
