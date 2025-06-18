# Android App for Traffic Signs Classification using CNN

This repository provides an Android application for classifying traffic signs using a pre-trained Convolutional Neural Network (CNN) model. The app is designed for real-time traffic sign recognition and can be used in applications such as driver assistance and autonomous vehicle systems.


For a detailed description of the CNN model and the underlying project, please refer to the main repository: [Traffic Signs Classification Using Convolutional Neural Networks](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolution-Neural-Networks).

### 📥 Download the Mobile Application for Android

Download the app by clicking the button below.

### [![Download Button](https://img.shields.io/badge/Download-APK-blue?style=for-the-badge&logo=android)](https://github.com/nishatrhythm/OpenCV-Back-up-for-Traffic-Signs-Classification-Android-App/raw/main/Release%20App/Traffic%20Sign.apk)

_[**Note**: It may take a few moments for the download to begin, as GitHub generates a direct download link for large files.]_

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [App Screenshots](#app-screenshots)
5. [Usage](#usage)
6. [Model Integration](#model-integration)
7. [Technical Details](#technical-details)

---

## Introduction

The Android app provided in this repository is an extension of the Traffic Signs Classification project that uses a CNN model for detecting and classifying traffic signs into 43 different categories. This app is designed for real-time classification and provides an easy-to-use interface for users to test and experience how AI can assist in understanding traffic signs.

---

## Features

- **Real-time traffic sign recognition** using a pre-trained CNN model.
- **User-friendly interface** designed for intuitive navigation.
- **Mobile-optimized performance** with efficient image processing.

---

## Installation

### Prerequisites
The app requires OpenCV for image processing functionalities. The OpenCV folder has been kept in a separate repository to reduce the size of this Android app repository.

- **Download the OpenCV library** from the dedicated repository: [OpenCV Back-up for Traffic Signs Classification Android App](https://github.com/nishatrhythm/OpenCV-Back-up-for-Traffic-Signs-Classification-Android-App)

  Or download directly from [here](https://github.com/nishatrhythm/OpenCV-Back-up-for-Traffic-Signs-Classification-Android-App/raw/main/opencv.zip).
  
  _[**Note**: It may take a few moments for the download to begin, as GitHub generates a direct download link for large files.]_

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/nishatrhythm/Android-App-of-Traffic-Signs-Classification-using-CNN.git
   cd Android-App-of-Traffic-Signs-Classification-using-CNN
   ```
2. Extract the downloaded `opencv.zip` into your Android project directory.
3. Open the project in Android Studio.
4. Ensure that your development environment is set up with the necessary SDKs and tools.
5. Build and run the app on an Android device or emulator.

---

## App Screenshots

| ![App Screenshot 1](https://github.com/nishatrhythm/Android-App-of-Traffic-Signs-Classification-using-CNN/blob/main/readme%20images/App_Screenshot_1.PNG) | ![App Screenshot 2](https://github.com/nishatrhythm/Android-App-of-Traffic-Signs-Classification-using-CNN/blob/main/readme%20images/App_Screenshot_2.PNG) | ![App Screenshot 3](https://github.com/nishatrhythm/Android-App-of-Traffic-Signs-Classification-using-CNN/blob/main/readme%20images/App_Screenshot_3.PNG) |
| --- | --- | --- |

---

## Usage

1. Launch the app on your Android device.
2. Point your device’s camera at a traffic sign.
3. The app will detect and classify the sign, displaying the classification result in real-time.

---

## Model Integration

The app uses a TensorFlow Lite model (`model_trained.tflite`) that was converted from the original trained CNN model. This enables the app to perform efficient on-device inference, ensuring fast and responsive user interactions.

### Model Conversion

The original CNN model, trained using Keras, was converted to a TensorFlow Lite format for use in mobile applications.

---

## Technical Details

- **Frameworks Used**: Android SDK, TensorFlow Lite
- **Languages**: Java, XML
- **Dependencies**:
  - TensorFlow Lite library for on-device ML processing.
  - CameraX for real-time camera feed integration.
  - OpenCV for image processing.

### File Structure

- `app/src/main/java/`: Contains the main source code, including activities and helper classes.
- `assets/model_trained.tflite`: The pre-trained TensorFlow Lite model for traffic sign classification.
- `res/layout/`: XML files defining the UI of the app.

