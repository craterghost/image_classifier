# Classifier for the CIFAR100 Dataset

This code is an implementation of a deep learning classifier for the CIFAR100 dataset, a collection of tiny color images that are commonly used for benchmarking the performance of computer vision models. The classifier uses the best trade-offs from three popular deep neural network architectures: VGG16, ResNet, and MobileNet.
Requirements

   - TensorFlow
   - Numpy

# Data Preprocessing

The CIFAR100 dataset is loaded using the keras.datasets.cifar100.load_data() method and is normalized by dividing every pixel value by 255. The target labels are one-hot encoded using the keras.utils.to_categorical function.
Residual Block

# Residual Block
The residual block is a building block used in ResNet architecture. It implements the tradeoff of allowing a network to learn the residual mapping between its inputs and outputs, instead of learning the direct mapping.
VGG16 Tradeoff

# VGG16 Tradeoff
The VGG16 tradeoff is to use multiple convolutional layers with small filters. In this code, this is achieved by stacking two convolutional layers with a filter size of 3x3, followed by a batch normalization and activation layer. The output of these layers is then max-pooled to reduce the spatial dimensions of the feature maps. This process is repeated with increasing filter sizes and is used to extract increasingly complex features from the input images.
ResNet Tradeoff

# ResNet Tradeoff
The ResNet tradeoff is to use residual blocks. In this code, the residual block is implemented as a function that takes in the input tensor and the number of filters as arguments. The function uses two convolutional layers with a filter size of 3x3, followed by a batch normalization and activation layer. The output of these layers is then added to the input tensor and passed through another activation layer.
MobileNet Tradeoff

# MobileNet Tradeoff
The MobileNet tradeoff is to use depthwise separable convolutions. This tradeoff allows for a significant reduction in the number of parameters and computations in the network, making it well-suited for deployment on resource-constrained devices. In this code, the tradeoff is not implemented, but can be added as an extension if desired.
Running the Code

# Running the Code
To run the code, simply execute it in a Python environment with the required libraries installed. The model will be trained on the CIFAR100 dataset and the accuracy on the test set will be printed at the end of the training process.

Note: The model performance may vary depending on the hyperparameters used, such as the batch size, learning rate, and number of epochs.
