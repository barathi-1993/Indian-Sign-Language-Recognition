#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess collected MediaPipe keypoints into train/test/validation arrays.

Expected input:
data/keypoints/<gesture>/<sequence>/<frame>.npy

Output:
data/processed_dataset.npz
models/labelencoder.pkl
"""

import argparse
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_sequences(input_dir: str, sequence_length: int = 30):
    input_dir = Path(input_dir)
    actions = sorted([p.name for p in input_dir.iterdir() if p.is_dir()])

    sequences = []
    labels = []

    for action in actions:
        sequence_dirs = sorted(
            [p for p in (input_dir / action).iterdir() if p.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        )

        for seq_dir in sequence_dirs:
            frames = []
            valid = True

            for frame_num in range(sequence_length):
                frame_path = seq_dir / f"{frame_num}.npy"
                if not frame_path.exists():
                    valid = False
                    break
                x = np.load(frame_path)
                if x.shape[0] != 1662:
                    valid = False
                    break
                frames.append(x)

            if valid:
                sequences.append(frames)
                labels.append(action)

    if not sequences:
        raise ValueError(f"No valid sequences found in {input_dir}")

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels), actions


def main():
    parser = argparse.ArgumentParser(description="Preprocess ISL keypoint sequences.")
    parser.add_argument("--input_dir", default="data/keypoints", type=str)
    parser.add_argument("--output_file", default="data/processed_dataset.npz", type=str)
    parser.add_argument("--labelencoder_path", default="models/labelencoder.pkl", type=str)
    parser.add_argument("--sequence_length", default=30, type=int)
    parser.add_argument("--test_size", default=0.15, type=float)
    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    X, labels, actions = load_sequences(args.input_dir, args.sequence_length)

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y = to_categorical(y_int).astype(np.float32)

    # 70/15/15 split.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=y_int
    )

    tmp_int = np.argmax(y_tmp, axis=1)
    val_fraction_of_tmp = args.val_size / (args.test_size + args.val_size)
    X_test, X_val, y_test, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction_of_tmp, random_state=args.seed, stratify=tmp_int
    )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        actions=le.classes_,
    )

    labelencoder_path = Path(args.labelencoder_path)
    labelencoder_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labelencoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"Saved dataset to {output_file}")
    print(f"Saved label encoder to {labelencoder_path}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")


if __name__ == "__main__":
    main()
