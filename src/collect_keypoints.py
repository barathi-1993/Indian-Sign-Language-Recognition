#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect MediaPipe Holistic keypoints from webcam for ISL gestures.

Output layout:
data/keypoints/<gesture>/<sequence>/<frame>.npy
"""

import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe_utils import draw_styled_landmarks, extract_keypoints, mediapipe_detection


DEFAULT_ACTIONS = [
    "fail", "friend", "good", "hello", "iloveyou", "like", "location",
    "meet", "phonecall", "takecare", "thanks", "think", "you"
]


def parse_actions(actions_arg):
    if actions_arg:
        return [a.strip() for a in actions_arg.split(",") if a.strip()]
    return DEFAULT_ACTIONS


def main():
    parser = argparse.ArgumentParser(description="Collect ISL landmark keypoints using webcam.")
    parser.add_argument("--output_dir", default="data/keypoints", type=str)
    parser.add_argument("--actions", default=None, type=str, help="Comma-separated action names.")
    parser.add_argument("--num_sequences", default=30, type=int)
    parser.add_argument("--num_frames", default=30, type=int)
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()

    actions = parse_actions(args.actions)
    output_dir = Path(args.output_dir)

    for action in actions:
        for seq in range(args.num_sequences):
            (output_dir / action / str(seq)).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(args.num_sequences):
                for frame_num in range(args.num_frames):
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("Could not read frame from webcam.")

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(
                            image,
                            "STARTING COLLECTION",
                            (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            4,
                            cv2.LINE_AA,
                        )
                        cv2.waitKey(500)

                    cv2.putText(
                        image,
                        f"Collecting {action} | Sequence {sequence} | Frame {frame_num}",
                        (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    keypoints = extract_keypoints(results)
                    npy_path = output_dir / action / str(sequence) / f"{frame_num}.npy"
                    import numpy as np
                    np.save(npy_path, keypoints)

                    cv2.imshow("MOPGRU-ISL Collection", image)
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
