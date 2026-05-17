#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained MOPGRU/GRU Keras model."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOPGRU-ISL model.")
    parser.add_argument("--model", default="models/mopgru_model.h5", type=str)
    parser.add_argument("--data", default="data/processed_dataset.npz", type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    actions = data["actions"]

    model = load_model(args.model, compile=False)

    pred_prob = model.predict(X_test)
    y_pred = np.argmax(pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    with open(output_dir / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    report = classification_report(y_true, y_pred, target_names=actions)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(metrics)
    print(report)


if __name__ == "__main__":
    main()
