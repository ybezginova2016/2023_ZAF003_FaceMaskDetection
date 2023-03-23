# 2023_ZAF003_FaceMaskDetection
## Overview
Face mask detection is an image processing and computer vision task that involves detecting whether a person in an image or video is wearing a face mask or not. With the ongoing COVID-19 pandemic, face masks have become a crucial tool in preventing the spread of the virus. Face mask detection systems can be used in public spaces such as airports, malls, and hospitals to ensure that individuals are following safety protocols.

## Methodology
- Import Libraries: First, you need to import the necessary libraries, including PyTorch, NumPy, OpenCV, and Matplotlib.

- Load the Data: Load the dataset of images with labeled classes (mask or no mask) and split it into training, validation, and testing sets. You can use PyTorch's DataLoader to load and preprocess the data.

- Load Pretrained Model: Load the pre-trained ResNet model from PyTorch's model zoo. ResNet is a popular choice for image classification tasks due to its superior performance on large datasets.

- Freeze Layers: Freeze the weights of the convolutional layers in the pre-trained model to prevent them from being updated during training.

- Replace the Classifier: Replace the final fully connected layer of the pre-trained model with a new classifier that has the appropriate number of output classes (2 for mask and no mask).

- Define Loss Function: Define the loss function for the binary classification problem, such as binary cross-entropy loss.

- Define Optimizer: Define the optimizer, such as stochastic gradient descent (SGD), to update the weights of the new classifier.

- Train the Model: Train the new classifier using the training set and validate it on the validation set. Use PyTorch's training loop to perform iterations of forward and backward propagation, and update the model's weights based on the loss and gradients.

- Test the Model: Test the final trained model on the testing set to evaluate its performance.

- Fine-Tune the Model: Fine-tune the model by unfreezing some of the earlier layers in the pre-trained model and continuing to train the model with a lower learning rate.

- Deploy the Model: Deploy the trained model for inference on new images or videos. Use the trained model to detect face masks in real-time scenarios.

## Business Segments
1. Healthcare
2. Safety

## Data
Face Mask Detection Datset - [Link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## Papers
1. Resnet Architecture - [Link](https://arxiv.org/abs/1512.03385)

## Instructions
In order to use this model, follow the below steps:
1. Clone the repository into your local system.
```
git clone https://github.com/ybezginova2016/ZAF003_CV_FaceMaskDetection.git
```
2. Install the required dependencies using pip.
```
pip install -r requirements.txt
```

3. Run the 'load_predict.py' to launch live face detection model. 

## Demo Link
**Click below for demo**
[![image](face_mask_detection_preview.jpg)](https://1drv.ms/v/s!AuQ0zVSghQNegtFGIAS2p4cdmRhj4A?e=EKSQ5a)

## Contributors:
- Shubham - shubhamwankar
