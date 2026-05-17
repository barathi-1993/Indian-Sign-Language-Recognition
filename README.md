<h1 align="center">
  An integrated mediapipe-optimized GRU model for Indian sign language recognition
</h1>

<p align="center">
  📄 <b>Published in Scientific Reports, Nature Portfolio</b> 📄
</p>

<p align="center">
  <b>🖐️ Indian Sign Language</b> &nbsp;|&nbsp;
  <b>🎥 Real-Time Gesture Recognition</b> &nbsp;|&nbsp;
  <b>🧠 MOPGRU</b> &nbsp;|&nbsp;
  <b>📍 MediaPipe Holistic</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Scientific%20Reports-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Indian%20Sign%20Language%20Recognition-teal?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Feature%20Extractor-MediaPipe-red?style=for-the-badge" />
</p>

<p align="center">
  Official implementation-style repository for <b>An Integrated MediaPipe-Optimized GRU Model for Indian Sign Language Recognition</b>.
</p>

<p align="center">
  <img src="assets/mopgru_pipeline.jpg" width="850">
</p>

---

## 📚 Table of Contents

- [📌 Overview](#-overview)
- [🧠 Key Idea](#-key-idea)
- [🏗️ System Architecture](#️-system-architecture)
- [📍 MediaPipe Feature Extraction](#-mediapipe-feature-extraction)
- [🧬 MOPGRU Model](#-mopgru-model)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Installation](#️-installation)
- [🗂️ Dataset](#️-dataset)
- [🚀 Usage](#-usage)
- [🏋️ Training](#️-training)
- [🎥 Real-Time Inference](#-real-time-inference)
- [📊 Results](#-results)
- [📦 Pretrained Model](#-pretrained-model)
- [📚 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)

---

## 📌 Overview

This repository implements a real-time **Indian Sign Language Recognition** system based on the published paper:

> **An integrated MediaPipe-optimized GRU model for Indian sign language recognition**  
> Scientific Reports, 2022.

The system recognizes isolated Indian Sign Language gestures from webcam/video input by extracting body, face, and hand landmarks using **MediaPipe Holistic**, followed by sequence classification using a modified GRU-based model called **MOPGRU**.

Unlike standard GRU models, MOPGRU improves the recurrent cell structure to enhance learning efficiency, reduce redundant temporal information, and improve gesture classification performance.

---

## 🧠 Key Idea

Sign language recognition is challenging because of:

- hand occlusion
- gesture similarity
- temporal dependency across frames
- high computational cost
- difficulty tracking hand, pose, and facial landmarks simultaneously

This project addresses these challenges using:

1. **MediaPipe Holistic** for lightweight landmark extraction
2. **Sequence-based modeling** of gesture frames
3. **MOPGRU**, a modified GRU cell designed for faster convergence and improved recognition
4. **Real-time webcam-based inference**

The paper reports that MOPGRU achieves better learning efficiency and prediction accuracy than Simple RNN, LSTM, Standard GRU, BiGRU, and BiLSTM baselines.

---

## 🏗️ System Architecture

<p align="center">
  <img src="assets/mopgru_architecture.jpg" width="850">
</p>

The complete pipeline has three stages:

### Stage 1: Data Preprocessing and Feature Extraction

Input video frames are captured from a webcam. MediaPipe Holistic extracts landmark coordinates from:

- face
- pose/body
- left hand
- right hand

### Stage 2: Data Cleaning and Labelling

Extracted keypoints are stored, cleaned, and labelled. Null or failed landmark detections are removed before training.

### Stage 3: Gesture Recognition

The cleaned landmark sequences are passed to the MOPGRU model for classification into Indian Sign Language gesture classes.

---

## 📍 MediaPipe Feature Extraction

The model uses the **MediaPipe Holistic** pipeline to extract landmark keypoints from each frame.

For each frame, the following landmarks are extracted:

| Component | Number of Landmarks | Coordinates |
|---|---:|---|
| Left hand | 21 | x, y, z |
| Right hand | 21 | x, y, z |
| Pose | 33 | x, y, z, visibility |
| Face | 468 | x, y, z |

The total feature size per frame is:

```text
(21 × 3) + (21 × 3) + (33 × 4) + (468 × 3) = 1662
```

Each gesture video is represented as a sequence:

```text
30 frames × 1662 keypoints
```

So the model input shape is:

```text
(30, 1662)
```

---

## 🧬 MOPGRU Model

MOPGRU stands for **MediaPipe-Optimized Gated Recurrent Unit**.

The proposed model modifies the standard GRU cell in three main ways:

### 1. Reset-Gate-Guided Update Gate

In the standard GRU, the update gate controls how much past information should be retained.

In MOPGRU, the update gate is adjusted using feedback from the reset gate. This helps reduce redundant past information and gives more attention to the current input.

### 2. ELU Activation for Candidate Memory

The standard GRU uses Tanh for candidate memory. MOPGRU replaces Tanh with **ELU** to improve convergence and reduce vanishing-gradient behavior.

### 3. Softsign Output Activation

MOPGRU replaces SoftMax with **Softsign** in the GRU output pathway to reduce computational complexity and improve training efficiency.

---

## 🧩 Model Architecture

The model uses a sequential recurrent structure:

```text
Input: 30 × 1662 landmark sequence

GRU/MOPGRU layer: 128 hidden units
Batch Normalization
Dropout

GRU/MOPGRU layer: 64 hidden units
Batch Normalization
Dropout

GRU/MOPGRU layer: 32 hidden units

Dense layers
Output layer for gesture classification
```

The original paper used:

| Parameter | Value |
|---|---:|
| Input sequence length | 30 frames |
| Feature dimension per frame | 1662 |
| GRU hidden units | 128, 64, 32 |
| Dropout | 0.20 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs | 100 |
| Train/Test/Validation split | 70/15/15 |
| Number of ISL gestures | 13 |

---

## 📁 Repository Structure

Recommended final repository structure:

```text
MOPGRU-ISL/
├── assets/
│   ├── mopgru_pipeline.jpg
│   ├── mopgru_architecture.jpg
│   ├── standard_gru_cell.jpg
│   ├── proposed_gru_cell.jpg
│   └── results_comparison.jpg
│
├── data/
│   ├── raw_videos/
│   ├── keypoints/
│   └── labels/
│
├── models/
│   └── mopgru_model.h5
│
├── notebooks/
│   └── demo.ipynb
│
├── src/
│   ├── collect_keypoints.py
│   ├── preprocess_keypoints.py
│   ├── mopgru_cell.py
│   ├── train.py
│   ├── evaluate.py
│   └── realtime_inference.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── classification_report.csv
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/barathi-1993/An-integrated-mediapipe-optimized-GRU-model-for-Indian-sign-language-recognition-.git
cd An-integrated-mediapipe-optimized-GRU-model-for-Indian-sign-language-recognition-
```

Create a virtual environment:

```bash
conda create -n mopgru_isl python=3.8 -y
conda activate mopgru_isl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal dependencies:

```text
numpy
pandas
opencv-python
mediapipe
tensorflow
scikit-learn
matplotlib
seaborn
```

---

## 🗂️ Dataset

The original study used a real-time Indian Sign Language dataset collected using a webcam.

The dataset contains **13 isolated ISL gestures**:

| Label | Gesture |
|---:|---|
| 0 | Fail |
| 1 | Friend |
| 2 | Good |
| 3 | Hello |
| 4 | I love you |
| 5 | Like |
| 6 | Location |
| 7 | Meet |
| 8 | Phone call |
| 9 | Take care |
| 10 | Thank you |
| 11 | Think |
| 12 | You |

Each gesture was recorded as:

```text
30 videos per gesture
30 frames per video
640 × 480 webcam resolution
```

The data split used in the paper was:

```text
Training   : 70%
Testing    : 15%
Validation : 15%
```

---

## 🚀 Usage

### 1. Collect Landmark Keypoints

Use webcam input to collect gesture sequences:

```bash
python src/collect_keypoints.py \
  --output_dir data/keypoints \
  --num_frames 30
```

### 2. Preprocess Keypoints

Clean null entries and prepare labelled sequences:

```bash
python src/preprocess_keypoints.py \
  --input_dir data/keypoints \
  --output_file data/processed_dataset.npz
```

### 3. Train MOPGRU

```bash
python src/train.py \
  --data data/processed_dataset.npz \
  --epochs 100 \
  --batch_size 13 \
  --lr 0.0001 \
  --save_path models/mopgru_model.h5
```

### 4. Evaluate Model

```bash
python src/evaluate.py \
  --model models/mopgru_model.h5 \
  --data data/processed_dataset.npz
```

### 5. Run Real-Time Inference

```bash
python src/realtime_inference.py \
  --model models/mopgru_model.h5
```

---

## 🏋️ Training

The model is trained using landmark sequences extracted from MediaPipe Holistic.

Input format:

```text
X shape: (num_samples, 30, 1662)
y shape: (num_samples,)
```

The training script saves:

```text
models/mopgru_model.h5
results/training_accuracy.png
results/training_loss.png
results/classification_report.csv
results/confusion_matrix.png
```

---

## 🎥 Real-Time Inference

During real-time inference:

1. The webcam captures frames.
2. MediaPipe Holistic extracts face, hand, and pose landmarks.
3. A rolling sequence of 30 frames is maintained.
4. The trained MOPGRU model predicts the gesture.
5. The predicted English word is displayed on the screen.

Example output:

```text
Predicted gesture: Hello
Confidence: 0.95
```

---

## 📊 Results

### Regression-Style Error Metrics

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| Simple RNN | 4.10 | 28.90 | -1.38 |
| LSTM | 0.75 | 4.95 | 0.59 |
| Standard GRU | 0.44 | 1.38 | 0.83 |
| BiGRU | 0.40 | 2.50 | 0.79 |
| BiLSTM | 0.85 | 5.35 | 0.56 |
| **MOPGRU** | **0.22** | **1.34** | **0.88** |

### Test Accuracy Comparison

| Model | Test Accuracy |
|---|---:|
| Simple RNN | 80% |
| LSTM | 85% |
| BiLSTM | 85% |
| Standard GRU | 90% |
| BiGRU | 90% |
| **MOPGRU** | **95%** |

### Benchmark Dataset Results

| Model | Dataset | Accuracy |
|---|---|---:|
| Pose-TGCN | WLASL100 | 55.43% |
| Pose-GRU | WLASL100 | 46.51% |
| **MOPGRU** | WLASL100 | **63.18%** |
| I3D | LSA64 | 98.91% |
| **MOPGRU** | LSA64 | **99.92%** |

---

## 📦 Pretrained Model

The pretrained model file is not included by default.

If you have the trained model, place it under:

```text
models/mopgru_model.h5
```

If the dataset or trained weights are not publicly shareable, clearly state:

```text
Dataset and pretrained weights are available from the authors upon reasonable request.
```

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{subramanian2022mopgru,
  title   = {An integrated mediapipe-optimized GRU model for Indian sign language recognition},
  author  = {Subramanian, Barathi and Olimov, Bekhzod and Naik, Shraddha M. and Kim, Sangchul and Park, Kil-Houm and Kim, Jeonghong},
  journal = {Scientific Reports},
  volume  = {12},
  number  = {11964},
  year    = {2022},
  publisher = {Nature Portfolio},
  doi     = {10.1038/s41598-022-15998-7}
}
```

---

## 🙏 Acknowledgements

This project uses:

- **MediaPipe Holistic** for face, hand, and pose landmark extraction
- **TensorFlow/Keras** for recurrent neural network modeling
- **OpenCV** for webcam-based video processing
- **Scientific Reports / Nature Portfolio** publication support

---

## ⚠️ Notes

This repository is intended for research and educational use.

Real-time sign language recognition performance may vary depending on:

- lighting conditions
- camera quality
- signer distance from camera
- hand occlusion
- background clutter
- gesture similarity

For robust deployment, the model should be trained on larger and more diverse signers, backgrounds, and lighting conditions.

---

## 📬 Contact

For questions, please open an issue in this repository.

---

<p align="center">
  ⭐ If this project is useful, please consider starring the repository. ⭐
</p>
