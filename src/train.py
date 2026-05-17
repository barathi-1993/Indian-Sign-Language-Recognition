#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train MOPGRU-ISL on preprocessed MediaPipe landmark sequences."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from mopgru_cell import build_mopgru_model


def plot_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["categorical_accuracy"], label="Train Accuracy")
    if "val_categorical_accuracy" in history.history:
        plt.plot(history.history["val_categorical_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train MOPGRU-ISL.")
    parser.add_argument("--data", default="data/processed_dataset.npz", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=13, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--save_path", default="models/mopgru_model.h5", type=str)
    parser.add_argument("--results_dir", default="results", type=str)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    actions = data["actions"]

    model = build_mopgru_model(
        sequence_length=X_train.shape[1],
        feature_dim=X_train.shape[2],
        num_classes=y_train.shape[1],
        lr=args.lr,
    )

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        TensorBoard(log_dir="logs/mopgru"),
        ModelCheckpoint(args.save_path, monitor="val_categorical_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_categorical_accuracy", patience=20, mode="max", restore_best_weights=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    plot_history(history, args.results_dir)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=actions, output_dict=False)
    with open(Path(args.results_dir) / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(Path(args.results_dir) / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(report)


if __name__ == "__main__":
    main()
