# Digit Recognition using Convolutional Neural Networks (CNNs)

This repository contains code and resources for building and training a digit recognition system using Convolutional Neural Networks (CNNs). CNNs are particularly well-suited for image-based tasks, making them ideal for recognizing handwritten or printed digits.

## Project Overview

This project aims to provide a clear and concise implementation of a digit recognition system using CNNs. It covers the essential steps from data loading and preprocessing to model training and evaluation.

## Key Features

* **CNN Implementation:** Utilizes TensorFlow/Keras or PyTorch (specify which you're using) to build a CNN architecture.
* **MNIST Dataset:** Leverages the widely used MNIST dataset for training and testing.
* **Data Preprocessing:** Includes image normalization and reshaping to prepare data for the CNN.
* **Model Training:** Demonstrates how to train the CNN model using the prepared data.
* **Model Evaluation:** Provides metrics and visualizations to evaluate the model's performance.
* **Model Saving and Loading:** Shows how to save and load the trained model for future use.
* **Prediction Visualization:** Visualizes model predictions on test images, highlighting correct and incorrect classifications.

## Technologies Used

* **Python 3.x:** The primary programming language.
* **TensorFlow/Keras or PyTorch:** For building and training the CNN.
* **NumPy:** For numerical operations.
* **Matplotlib:** For data visualization.
* **Scikit-learn (optional):** For evaluation metrics.

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow/Keras or PyTorch (install using `pip install tensorflow` or `pip install torch torchvision`)
* NumPy (install using `pip install numpy`)
* Matplotlib (install using `pip install matplotlib`)
* Scikit-learn (optional, install using `pip install scikit-learn`)

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/digit_recognition_cnn.git](https://www.google.com/search?q=https://www.google.com/search%3Fq%3Dhttps://github.com/your-username/digit_recognition_cnn.git)
    cd digit_recognition_cnn
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt #if you create a requirements file, or install the above libraries manually
    ```

### Usage

1.  **Run the Python script/Jupyter notebook:** Execute the `digit_recognition_cnn.py` script or the provided Jupyter notebook.
2.  **Model Training:** The script will automatically download the MNIST dataset, preprocess it, and train the CNN model.
3.  **Model Evaluation:** After training, the script will evaluate the model's performance on the test dataset.
4.  **View Results:** The script will display evaluation metrics and visualize predictions.
5.  **Use the trained model:** The script will save the trained model, which can be loaded and used for digit recognition in other applications.

**Future Enhancements**

-Improve recognition accuracy for various digit styles and image qualities.

-Implement real-time digit recognition using a camera feed.

-Develop a user interface for easy data input and CNC control.

-Expand the project to recognize letters and other symbols.

-Implement various methods of image capture, such as capturing from a webcam, or from a dedicated camera mounted to the CNC machine.

-Implement data augmentation to improve model robustness.

Output-

![Screenshot 2025-03-13 142840](https://github.com/user-attachments/assets/b3657450-34e5-4681-8f75-ff0a8343248a)

![Screenshot 2025-03-13 142917](https://github.com/user-attachments/assets/c0526b86-c554-4c18-a588-27f8cebf0ec6)

![Screenshot 2025-03-13 142940](https://github.com/user-attachments/assets/42bea94c-98c8-47e5-a5eb-4b0949ca7a29)

![Screenshot 2025-03-13 143002](https://github.com/user-attachments/assets/61eb18df-a048-4c80-b9fd-741d4f583ec5)

![Screenshot 2025-03-13 143018](https://github.com/user-attachments/assets/7dd1c0bc-04d8-475a-b31a-3b54428c4a66)

![Screenshot 2025-03-13 143049](https://github.com/user-attachments/assets/f3439e45-f3d4-45e6-bcaa-871e5e47c319)
