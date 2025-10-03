import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse


def clean_and_concat(raw_dir, use_small=False):
    # Use raw_dir from argument, no need to reassign
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if use_small:
        files = files[:2]  # subselect for small mode

    print("CSV files found:", files)

    frames = []
    for fname in files:
        df = pd.read_csv(os.path.join(raw_dir, fname), low_memory=False)
        df.columns = df.columns.str.strip()  # remove leading/trailing spaces

        if 'Label' not in df.columns or df.shape[1] < 10:
            print(f"Skipping {fname}: missing 'Label' or too few columns")
            continue

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['Label'])
        frames.append(df)  # <-- THIS WAS MISSING

    if not frames:
        raise ValueError(f"No valid CSVs found in {raw_dir}. Check your files and column names.")

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    print(f"Loaded {len(frames)} CSVs with total shape: {df_all.shape}")
    return df_all


def preprocess(df):
    # Remove non-feature columns
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port", "Destination Port"]
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    # Categorical encoding
    cat_cols = [col for col in df.columns if df[col].dtype=='O' and col != 'Label']
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
    # Feature scaling
    feat_cols = [c for c in df.columns if c != 'Label']
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols].astype(np.float32))
    return df

def split_and_save(df, out_dir):
    X, y = df.drop("Label", axis=1), df["Label"]
    y = (y != "BENIGN").astype(int)  # Binary: 1=attack, 0=benign
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(out_dir, "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(out_dir, "test.npz"), X=X_test, y=y_test)
    # Save CSVs for sample inspection
    X_train.assign(Label=y_train).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    X_val.assign(Label=y_val).to_csv(os.path.join(out_dir, "val.csv"), index=False)
    X_test.assign(Label=y_test).to_csv(os.path.join(out_dir, "test.csv"), index=False)
    # Class distribution print
    print("Train dist:", np.bincount(y_train))
    print("Val dist:", np.bincount(y_val))
    print("Test dist:", np.bincount(y_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=r"C:\Users\veena\Green GAN\green-gan\data\raw\MachineLearningCVE")
    parser.add_argument("--output_dir", default=r"C:\Users\veena\Green GAN\green-gan\data\processed")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    df = clean_and_concat(args.input_dir, use_small=args.small)
    df = preprocess(df)
    split_and_save(df, args.output_dir)

