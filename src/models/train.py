import optuna
from src.models.architectures import build_model_v2
from src.data.dataset import create_tf_datasets, get_default_classes, get_default_localization


def train_model(dataset, study_name, database_config, image_size, batch_size):
    study = optuna.load_study(study_name=study_name, storage=database_config)
    best_trial = study.best_trial
    model, opt, class_weight, _ = build_model_v2(best_trial, get_default_localization(), get_default_classes(), image_size)
    train_ds, val_ds = create_tf_datasets(dataset, image_size=image_size, batch_size=batch_size)

    history = model.fit(train_ds, validation_data=val_ds, epochs=1, class_weight=class_weight)

    return model, history