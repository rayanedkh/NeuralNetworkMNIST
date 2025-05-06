# RayaNN: a Neural Network for MNIST Digit Recognition from Scratch 

This repository contains a simple Multi-Layer Perceptron (MLP) used to classify handwritten digits from the MNIST dataset. The goal was to implement, train, save, and load a neural network from scratch using only fundamental Python libraries.

## 📋 Project Overview

* **Objective**: Build and train a neural network from scratch to predict digits (0–9) from MNIST grayscale images (28×28 pixels).
* **Dataset**: MNIST training set (`data_set/train.csv`, 42,000 samples) and test set (`data_set/test.csv`).
* **Model**: Two-layer MLP with ReLU activation and LogSoftmax output for numerical stability.
* **Training**: Gradient descent for 5,000 epochs with batch size 100.

## 🚀 Technologies & Dependencies
* **Programming Language**: Python

* **Core Libraries**:

  * `numpy` (array operations & saving/loading weights)
  * `pandas` (data handling)
  * `scipy.special.logsumexp` (stable softmax implementation)
  * `tqdm` (training progress bar)
  * `matplotlib` (visualizing predictions)

## 📂 Project Structure

```bash
├── README.md                  # Project documentation (this file)
├── save_model.npz             # Binary file with learned weights (generated after training)
└── RayaNN/        
    ├── rayane_nn.py           # Training script
    ├── main.py                # Used to test the model visually           
    └── data_set/              # CSV files for training and test data
        ├── train.csv
        └── test.csv

```
## 🧩 How the Neural Network Works

1. **Data Preprocessing**:

   * Load CSV, shuffle, normalize pixel values to `[0,1]`, split into training and dev sets.
2. **Model Architecture**:

   * **Layer 1**: Fully-connected (784→10) + ReLU.
   * **Layer 2**: Fully-connected (10→10) + LogSoftmax.
3. **Training Loop**:

   * Trained the model for 5,000 epochs with a batch size of 100.
   * Used a **loss function** and an **optimization function** implementing the **steepest descent** algorithm.
4. **Saving & Loading**:

   * Weights are saved in `save_model.npz` via `numpy.savez`.
   * Inference script loads `.npz`, rebuilds the model, and predicts a random test image.

## 📊 Results

* **Development accuracy**: \~92–93% after 5,000 epochs.
* **Inference**: Visual display of a randomly selected test image with predicted digit.

---

*Feel free to explore, modify the architecture, or extend to deeper networks!*
