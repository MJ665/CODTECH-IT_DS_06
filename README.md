---

# Deep Learning Model for Image Recognition using CNNs and Transfer Learning

## Overview
This project involves building a deep learning model for image recognition using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. The model leverages transfer learning with the pre-trained VGG16 architecture to improve performance. The dataset is augmented to increase diversity and improve generalization.

## Tasks
### Task Six:
- **Build a deep learning model for image recognition tasks using CNNs**.
- **Preprocess image data** and augment the dataset to increase diversity.
- **Design a CNN architecture** with multiple layers, including convolutional, pooling, and fully connected layers.
- **Train the model on a large dataset (CIFAR-10)**.
- **Employ transfer learning techniques**, using the VGG16 model pre-trained on ImageNet for improved performance.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- Numpy
- Other dependencies as listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/deep-learning-image-recognition.git
    cd deep-learning-image-recognition
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data
- The model uses the CIFAR-10 dataset, which is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into a training set of 50,000 images and a test set of 10,000 images.

## Approach

1. **Data Preprocessing**:
   - The images are normalized to the range [0, 1] for better training performance.
   - Image augmentation techniques, such as rotation, shifting, shearing, zooming, and flipping, are applied to diversify the dataset.

2. **Transfer Learning with VGG16**:
   - The VGG16 model, pre-trained on ImageNet, is used as the base for feature extraction. The top layers are removed to prevent overfitting.
   - The VGG16 layers are frozen, and new layers (dense, dropout) are added to perform classification for the CIFAR-10 dataset.

3. **Model Architecture**:
   - **Base model**: VGG16 with pre-trained weights, without the top layers.
   - **Fully connected layers**: Flatten layer, Dense layer with 128 units, Dropout layer (to prevent overfitting), and a final Dense layer with 10 units for classification.

4. **Training**:
   - The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
   - The model is trained for 3 epochs, with batch size 64, using the augmented training dataset.
   - The performance of the model is evaluated on the test dataset.

## Training and Results
After training the model, the accuracy and loss for both the training and validation datasets are plotted for each epoch.

- **Training Accuracy and Loss**: These metrics are plotted for each epoch to visualize the learning process.
- **Test Accuracy**: The final accuracy of the model on the CIFAR-10 test set is displayed.

## Example of Running the Code
To run the model, execute the following:

```bash
python train_model.py
```

This will start the training process on the CIFAR-10 dataset using the VGG16 pre-trained model.

## Evaluation

After training, the model is evaluated on the test dataset, and the test accuracy is printed.

```bash
Test Accuracy: 0.7840
```

## Visualizing Results
The training and validation accuracy, as well as the training and validation loss, are plotted using Matplotlib for further analysis of the model's performance over the epochs.

## Conclusion
By utilizing transfer learning with VGG16, we were able to achieve a good level of performance on the CIFAR-10 dataset. The model was trained for 3 epochs and evaluated on the test set, providing a solid foundation for further improvements, such as fine-tuning the VGG16 model or experimenting with other architectures like ResNet.

