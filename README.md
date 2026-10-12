# 0-1-Handwriting-Recog

A Deep Learning project that recognizes handwritten digits (0-9) using Neural Networks and Convolutional Neural Networks (CNN).

This project uses the **MNIST dataset** to train models capable of recognizing handwritten digits from 0 to 9. It also provides an interactive GUI where users can draw a digit and receive a real-time prediction along with the confidence score for each class.

The project includes two different neural network approaches:

- **CNN (Convolutional Neural Network)** for image-based digit recognition
- **Dense Neural Network** using fully connected layers

The CNN training process also includes an interactive neural network visualization that displays neuron activity, prediction results, training accuracy, and loss during training.

---

## Demo

Watch the video demo of this application in action on Instagram:

[![Instagram Demo](https://img.shields.io/badge/Instagram-View%20Demo-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/reel/DNE-5umAStt/?xtok=N2dxcmV0ZXo4bTV2)

---

## Features

- **Handwritten Digit Recognition**
  
  Recognizes handwritten digits from **0 to 9** using a trained neural network.

- **CNN-Based Recognition**
  
  Uses a custom Convolutional Neural Network designed for image classification on the MNIST dataset.

- **Dense Neural Network**
  
  Includes an alternative fully connected neural network architecture for comparison with the CNN model.

- **Interactive Drawing GUI**
  
  Provides a simple Tkinter-based canvas where users can draw a handwritten digit directly using the mouse.

- **Real-Time Prediction**
  
  The application predicts the drawn digit and displays the confidence percentage for every digit from 0 to 9.

- **Neural Network Visualization**
  
  The CNN training script includes a visualization of neural network activity, including layer activations, neuron connections, predictions, training accuracy, and loss.

- **Data Augmentation**
  
  Training data is augmented using rotation, zoom, width shifting, and height shifting to improve model generalization.

- **Early Stopping**
  
  EarlyStopping is used to stop training when validation performance stops improving.

- **Learning Rate Adjustment**
  
  ReduceLROnPlateau dynamically reduces the learning rate when validation loss stops improving.

---

## Tech Stack

| Technology | Usage |
|---|---|
| **Python** | Main programming language |
| **TensorFlow / Keras** | Deep Learning and Neural Networks |
| **NumPy** | Numerical and array processing |
| **Matplotlib** | Training and neural network visualization |
| **Tkinter** | Graphical User Interface |
| **Pillow (PIL)** | Drawing and image processing |
| **OpenCV** | Image processing support |
| **scikit-learn** | Machine learning utilities |
| **MNIST** | Handwritten digit dataset |

---

## Project Structure

```text
0-1-Handwriting-Recog/
│
├── digit_recognition_app.py
├── neural_cnn.py
├── neural_dense.py
├── requirements.txt
├── README.md
│
└── trained_digit_model_cnn.keras
   └── Generated after training
