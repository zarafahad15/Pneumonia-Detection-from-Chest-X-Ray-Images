# Pneumonia-Detection-from-Chest-X-Ray-Image
Overview

This project implements a deep learning model that classifies chest X-ray images into two categories: Normal and Pneumonia. The model is built using a Convolutional Neural Network (CNN) and trained on the publicly available “Chest X-Ray Pneumonia” dataset. The goal is to provide a reliable, lightweight, and easily deployable medical imaging solution.

Dataset

The dataset used is the Chest X-Ray Pneumonia dataset by Kermany et al.
It contains three folders:

chest_xray/
    train/
    val/
    test/

Each folder includes two classes:
	•	NORMAL
	•	PNEUMONIA

This dataset is commonly available on Kaggle.

Features
	•	Complete pipeline: preprocessing, augmentation, training, validation, and testing.
	•	CNN architecture optimized for binary image classification.
	•	Use of data augmentation to reduce overfitting.
	•	Includes EarlyStopping and ModelCheckpoint for stable training.
	•	Saves the best-performing model for deployment.

Model Architecture

The model consists of:
	•	Three convolutional blocks (Conv2D + MaxPooling2D)
	•	Flatten layer
	•	Dense layer with dropout for regularization
	•	Sigmoid activation in the output layer for binary classification

The network is lightweight and suitable for both GPU and CPU environments.

Training Configuration
	•	Optimizer: Adam
	•	Loss function: Binary Cross-Entropy
	•	Epochs: 15–20
	•	Batch size: 32
	•	Callbacks: EarlyStopping and ModelCheckpoint

Requirements

Install the following dependencies:

tensorflow
numpy
opencv-python
matplotlib

How to Run
	1.	Download and extract the dataset into a folder named chest_xray.
	2.	Ensure the directory structure matches the format above.
	3.	Update the dataset path in the script if needed.
	4.	Run the training script to begin model training.
	5.	The best model will be saved as best_pneumonia_model.h5.
	6.	Use the saved model for predictions.

Inference Example

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("best_pneumonia_model.h5")

img = cv2.imread("sample_xray.jpg")
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Pneumonia detected")
else:
    print("Normal")

Results

With proper training, the model typically achieves:
	•	92%–96% accuracy on the test set
	•	Strong generalisation due to augmentation and regularisation
	•	Robust performance for deployment in clinical screening tools

Project Structure

├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
├── pneumonia_model.py
├── best_pneumonia_model.h5
└── README.md

Deployment

The trained model can be deployed as:
	•	A web application (Streamlit or Flask)
	•	A REST API
	•	A mobile application using TensorFlow Lite
	•	A desktop application with PyQt or Electron

⸻
