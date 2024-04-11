# Brain Tumor Classification Project

This project utilizes Convolutional Neural Networks (CNNs) implemented with TensorFlow and Keras to classify brain tumor images.
The dataset used for this project is sourced from Kaggle, and you can access it [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).

[Link to Jupyter Notebook](https://github.com/abhishekiiitbh2903/BRAIN_TUMOR/blob/main/Brain%20Tumor%20Detection.ipynb)

## Overview

- **Training Model**: Custom CNN
- **Training Batch Size**: 32
- **Image Size**: 256x256x3
- **Epochs**: 20
- **Dataset Bifurcation**:
  - Training Dataset: 75%
  - Testing Dataset: 15%
  - Validation Dataset: 10%

## Data Preprocessing

To optimize training efficiency, several preprocessing techniques were applied:

- **Caching**: Caching the dataset in memory to avoid re-loading the data during each epoch.
- **Shuffling**: Randomly shuffling the dataset to prevent the model from learning the order of the samples.
- **Prefetching**: Prefetching batches of data to reduce latency.

## Data Augmentation

Data augmentation techniques were employed to augment the dataset, helping to prevent overfitting and improve model generalization.

## Model Architecture

### Custom Model

- Convolution layer with 8 filters and maxpooling with size of (2,2) and stride=2
- Convolution layer with 16 filters and maxpooling with size of (2,2) and stride=2
- Convolution layer with 32 filters and maxpooling with size of (2,2) and stride=2
- Convolution layer with 64 filters and maxpooling with size of (2,2) and stride=2
- Convolution layer with 128 filters and maxpooling with size of (2,2) and stride=2
- Dropout of 0.2
- Flattening
- Fully Connected (FC) layer with 64 nodes

### Other Models
- AlexNet
- LeNet
- ResNet

The custom model outperformed the other three models, achieving a test accuracy of 97.85%.
![Comparison of Different Models](https://github.com/abhishekiiitbh2903/BRAIN_TUMOR/blob/main/Models_Performance_Comparisons.png)


## Displaying Predictions and Confidence

The model displays predictions along with their confidence levels during the prediction phase.

## Author

- Abhishek Singh
- Undergraduate in Computer Science and Engineering, IIITBH

For any inquiries or feedback, please contact Abhishek Singh at [abhishekrathore1806@gmail.com](mailto:abhishekrathore1806@gmail.com).
