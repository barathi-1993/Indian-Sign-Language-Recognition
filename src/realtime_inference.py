#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-time webcam inference for MOPGRU-ISL."""

import argparse
from collections import deque
import pickle

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from mediapipe_utils import draw_styled_landmarks, extract_keypoints, mediapipe_detection


def main():
    parser = argparse.ArgumentParser(description="Run real-time ISL recognition.")
    parser.add_argument("--model", default="models/mopgru_model.h5", type=str)
    parser.add_argument("--labelencoder", default="models/labelencoder.pkl", type=str)
    parser.add_argument("--sequence_length", default=30, type=int)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--threshold", default=0.50, type=float)
    args = parser.parse_args()

    model = load_model(args.model, compile=False)
    with open(args.labelencoder, "rb") as f:
        label_encoder = pickle.load(f)

    sequence = deque(maxlen=args.sequence_length)
    cap = cv2.VideoCapture(args.camera)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == args.sequence_length:
                X = np.expand_dims(np.asarray(sequence, dtype=np.float32), axis=0)
                probs = model.predict(X, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]

                if confidence >= args.threshold:
                    text = f"{pred_label} ({confidence:.2f})"
                else:
                    text = "..."

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("MOPGRU-ISL Real-Time Inference", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
