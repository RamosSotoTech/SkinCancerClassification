from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
from src.data.dataset import create_test_dataset


def evaluate_model(model, test_dataset, image_size=(256, 256), batch_size=32):
    test_ds = create_test_dataset(test_dataset, image_size=image_size, batch_size=batch_size)

    test_image_inputs = []
    test_metadata_inputs = []
    test_labels = []

    for features, labels in test_ds:
        test_image_inputs.append(features['image_input'].numpy())
        test_metadata_inputs.append(features['metadata_input'].numpy())
        test_labels.append(np.argmax(labels.numpy(), axis=-1))

    test_image_inputs = np.concatenate(test_image_inputs, axis=0)
    test_metadata_inputs = np.concatenate(test_metadata_inputs, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    predictions = model.predict([test_image_inputs, test_metadata_inputs])
    predicted_classes = np.argmax(predictions, axis=1)

    report = classification_report(test_labels, predicted_classes)
    roc_auc = roc_auc_score(test_labels, predictions, multi_class='ovr')
    cm = confusion_matrix(test_labels, predicted_classes)

    return report, roc_auc, cm