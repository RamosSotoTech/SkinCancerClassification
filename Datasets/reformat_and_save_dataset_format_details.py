import numpy as np
import pandas as pd
from data import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import h5py
import cv2
import os

def save_image(image, image_id, lession_id, split):
    img_path = os.path.join("data", split, "images", f"{lession_id}_{image_id}.jpg")
    cv2.imwrite(img_path, image)
    return img_path

def main():
    dataset = load_dataset("marmal88/skin_cancer")
    # Convert data to pandas DataFrames
    train_ds = pd.DataFrame.from_dict(dataset['train'])
    val_ds = pd.DataFrame.from_dict(dataset['validation'])
    test_ds = pd.DataFrame.from_dict(dataset['test'])
    train_val_ds = pd.concat([train_ds, val_ds])

    # Use pandas DataFrame instead of list of dicts
    le_dx = LabelEncoder()
    le_dx.fit(train_val_ds['dx'])

    le_dx_type = LabelEncoder()
    le_dx_type.fit(train_val_ds['dx_type'])

    le_localization = LabelEncoder()
    le_localization.fit(train_val_ds['localization'])

    age_scaler = StandardScaler()
    train_val_ds['age'] = train_val_ds['age'].fillna(train_val_ds['age'].median())
    age_scaler.fit(np.array(train_val_ds['age']).reshape(-1, 1))

    median_age = age_scaler.mean_

    for split in ['train', 'validation', 'test']:
        os.makedirs(f'data/{split}/images', exist_ok=True)

    # Convert each dataset to Pandas dataframes and save them as CSV files
    for split in dataset:
        df = pd.DataFrame.from_dict(dataset[split])
        df['dx'] = le_dx.transform(df['dx'])
        df['dx_type'] = le_dx_type.transform(df['dx_type'])
        df['localization'] = le_localization.transform(df['localization'])
        df['age'] = age_scaler.transform(df['age'])

        # set index to lesion_id
        df.set_index('lesion_id', inplace=True)

        # encode sex as 0 or 1
        df['sex'] = df['sex'].map({'male': 1, 'female': 0})

        df['age'].fillna(median_age, inplace=True)

        img_paths = []
        for _, row in df.iterrows():
            img_path = save_image(row['image'], row['image_id'], row['lesion_id'], split)
            img_paths.append(img_path)
        df['image_path'] = img_paths

        df = df.drop(columns=['image'])
        df.to_csv(f"data/{split}_data.csv")


if __name__ == "__main__":
    main()
