import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


from typing import Tuple


@joblib.delayed()
def load_raw_csv(path: str) -> pd.DataFrame:
return pd.read_csv(path)


# We'll implement a simple loader + preprocess


def load_and_preprocess(csv_paths, features=None, label_col='Label', save_path=None):
"""Load a list of csv file paths, concatenate, filter attacks, scale features.
- csv_paths: list of file paths to CIC-IDS2017 CSVs
- features: list of feature names to keep (None = auto select numeric)
- label_col: column marking attack/benign
"""
dfs = []
for p in csv_paths:
df = pd.read_csv(p)
dfs.append(df)
df = pd.concat(dfs, ignore_index=True)


# Simplify: drop columns with object dtype (non-numeric) except label
if features is None:
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
if label_col in df.columns and label_col not in numeric:
# if label isn't numeric keep original and handle separately
pass
features = numeric


# Keep label if present
if label_col in df.columns:
labels = df[label_col]
X = df[features].fillna(0.0)
else:
labels = None
X = df[features].fillna(0.0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


if save_path:
os.makedirs(save_path, exist_ok=True)
pd.DataFrame(X, columns=features).to_csv(os.path.join(save_path, 'features_raw.csv'), index=False)
joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))


return X_scaled, labels, features




if __name__ == '__main__':
# example usage
import glob
csvs = glob.glob('data/raw/*.csv')
X, labels, features = load_and_preprocess(csvs, save_path='data/processed')
print('Done', X.shape)
