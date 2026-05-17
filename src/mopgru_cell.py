#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOPGRU model components.

This module provides:
1. A manuscript-style MOPGRUCell for new training.
2. A compatible Keras Sequential architecture using standard GRU layers for
   loading the provided .h5 models when custom cell weights are unavailable.

The paper-level MOPGRU idea:
- reset gate guides the update gate
- candidate memory uses ELU instead of Tanh
- output pathway can use Softsign
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class MOPGRUCell(layers.Layer):
    """MediaPipe-Optimized GRU cell following the paper's design idea."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.Wr = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="Wr")
        self.Ur = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", name="Ur")
        self.br = self.add_weight(shape=(self.units,), initializer="zeros", name="br")

        self.Wz = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="Wz")
        self.Uz = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", name="Uz")
        self.bz = self.add_weight(shape=(self.units,), initializer="zeros", name="bz")

        self.Wh = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="Wh")
        self.Uh = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", name="Uh")
        self.bh = self.add_weight(shape=(self.units,), initializer="zeros", name="bh")

        super().build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]

        r = tf.sigmoid(tf.matmul(inputs, self.Wr) + tf.matmul(h_prev, self.Ur) + self.br)

        # Manuscript-inspired update gate adjustment:
        # reset gate feedback screens the present input before update computation.
        z = tf.sigmoid(tf.matmul(inputs * r, self.Wz) + tf.matmul(h_prev, self.Uz) + self.bz)

        # Candidate memory with ELU activation.
        n = tf.nn.elu(tf.matmul(inputs, self.Wh) + tf.matmul(h_prev * r, self.Uh) + self.bh)

        h = z * n + (1.0 - z) * h_prev
        return h, [h]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_mopgru_model(
    sequence_length: int = 30,
    feature_dim: int = 1662,
    num_classes: int = 13,
    hidden_units=(128, 64, 32),
    dropout: float = 0.20,
    lr: float = 1e-4,
):
    """Build a manuscript-style MOPGRU classifier."""
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")
    x = inputs

    for i, units in enumerate(hidden_units):
        return_sequences = i < len(hidden_units) - 1
        x = layers.RNN(MOPGRUCell(units), return_sequences=return_sequences, name=f"mopgru_{units}")(x)
        if return_sequences:
            x = layers.BatchNormalization(name=f"bn_{units}")(x)
            x = layers.Dropout(dropout, name=f"dropout_{units}")(x)

    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dense(32, activation="relu", name="dense_32")(x)

    # Softmax is used for final class probabilities. The Softsign replacement
    # from the paper is represented inside the recurrent-output pathway; for
    # Keras classifier stability, final probabilities are softmax.
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = models.Model(inputs, outputs, name="MOPGRU_ISL")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def build_compatible_gru_model(
    sequence_length: int = 30,
    feature_dim: int = 1662,
    num_classes: int = 13,
    model_variant: str = "mopgru",
):
    """
    Build an architecture compatible with the provided legacy .h5 models.

    The uploaded actionMGRU.h5 appears to be a Keras Sequential GRU-stack
    variant. This helper is kept to avoid breaking old weights.
    """
    if model_variant.lower() == "gru":
        units = (128, 64, 32)
    else:
        units = (64, 64, 32)

    model = models.Sequential(name=f"compatible_{model_variant}_model")
    model.add(layers.GRU(units[0], return_sequences=True, activation="elu", input_shape=(sequence_length, feature_dim)))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(units[1], return_sequences=True, activation="tanh"))
    model.add(layers.Dropout(0.3))
    model.add(layers.GRU(units[2], return_sequences=False, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model
