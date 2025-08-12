📌 Overview
This project is a deep learning pipeline for detecting shoplifting behavior from surveillance videos. It combines a Convolutional Neural Network (CNN) for frame-level feature extraction with a Recurrent Neural Network (RNN) using LSTM layers for temporal sequence modeling.

The workflow:
Load training and testing video datasets.
Preprocess and normalize video frames.
Extract frame-level features using a CNN.
Pass extracted features to an LSTM-based classifier.
Train and evaluate the model.

📂 Project Structure
Data Loading & Preparation
Reads videos from predefined train/test directories.
Saves file paths and labels into CSV files.

Preprocessing
Resizes frames to 224x224 pixels.
Normalizes pixel values to [0,1].
Pads videos to a fixed number of frames (MAX_FRAMES = 20).

Label Encoding

Encodes "non shop lifters" → 0 and "shop lifters" → 1.

Feature Extraction (CNN)

A CNN extracts spatial features from individual frames.

Sequence Modeling (RNN)

An LSTM processes the temporal sequence of CNN features.

Outputs a class probability (shoplifter / non-shoplifter).

🧠 Model Architecture
CNN Feature Extractor
Conv2D → MaxPooling layers (32, 64, 128 filters)

Flatten → Dense(128) features

LSTM Classifier
LSTM(64) → Dense(32) → Dense(2, softmax)
