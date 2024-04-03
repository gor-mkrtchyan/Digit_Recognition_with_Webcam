# Real-time digit classification

# MNIST Digit Recognition with Real-Time Webcam Inference

This repository contains code for a simple digit recognition system using the MNIST dataset and a Convolutional Neural Network (CNN) implemented with Keras. Additionally, it includes a real-time webcam inference script that utilizes the pre-trained model for digit recognition.

## Files:

### 1. MNIST.ipynb

This Jupyter Notebook file contains the code for:

- Loading the MNIST dataset
- Preprocessing the data
- Building and training a CNN model
- Evaluating the model
- Plotting the training history
- Saving the trained model

### 2. real_time_recognition.py

This Python script uses OpenCV to access the webcam, capture frames in real-time, and perform digit recognition using the pre-trained model. The script draws a bounding box around a specified region of interest (ROI) in the webcam feed and predicts the digit within that region.

## Instructions:

1. Run `MNIST.ipynb` to train the CNN model on the MNIST dataset.
2. Save the trained model as 'digit_recognition_model_MNIST.h5'.
3. Execute `real_time_recognition.py` to run the real-time digit recognition on your webcam feed.

## Dependencies:

- matplotlib
- keras
- scikit-learn
- seaborn
- pandas
- numpy
- OpenCV
- imutils

## Usage:

- Ensure you have all the dependencies installed by running `pip install -r requirements.txt`.
- Run `MNIST.ipynb` in a Jupyter Notebook environment to train the model.
- Save the trained model as 'digit_recognition_model_MNIST.h5'.
- Execute `real_time_recognition.py` to open a webcam window with real-time digit recognition.

## Notes:

- Press 'q' to exit the real-time recognition script.