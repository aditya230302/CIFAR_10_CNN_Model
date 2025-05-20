# CIFAR_10_CNN_Model

---
The CIFAR-10 and CIFAR-100 datasets are labeled subsets of the 80 million tiny images dataset. CIFAR-10 and CIFAR-100 were created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

---
# **The CIFAR-10 Dataset**
- CIFAR : Canadian Institute For Advance Research
- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

- The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

- The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

- Here are the classes in the dataset as well as 10 random images from each class:
  ![image](https://github.com/user-attachments/assets/26d13664-794a-4871-9ff8-eac8f4ddfffd)

- `Conv1D` is used for the text

- **`Conv2D` is used for the images**

- `Conv3D` 3D is used for imaging, video processing

### Parameters of the `Conv2D` layers

*   `filters` - value of the filters show that `number of filters` from which CNN model and the Convolutional layer will learn from
*   `kernel-size` - a filter which will move through the image and extract the features of the part using a `dot` product. It basically means the dimension of the filter aka kernel which is `height` X `width`

---
# Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

"""
datasets : CIFAR-10 and CIFAR-100
models   : helps build Sequential or functional models
layers   : provides Conv2D, MaxPooling@d, Flatten, Dense, etc.
"""
```

---
# Model Building
```python
model = models.Sequential()
```
---
## Base CNN Model

### FIRST CONVOLUTION + POOLING LAYER
1. Added a 2D convolutional layer with 32 filters, each having filter size 3X3
        no. of filters = 32
        kernal size = (3,3)
        imput shape = Height x Width x No. of Channels = (32,32,3)
2. Adds a max pooling layer having pool size of 2 by 2 grid --> to reduce the spatial dimension by half
```python
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
```
### SECOND CONVOLUTION + POOLING LAYER
1. Added a 2D convolutional layer with 64 filters, each having filter size 3X3
2. Adds a max pooling layer having pool size of 2 by 2 grid
```python
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
```
### THIRD CONVOLUTION (JUST FILTERING)
1. Added a 2D convolutional layer with 64 filters, each having filter size 3X3
```python
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
```
### ANN LAYER
1. Convert a multi-dimensional input coming from convolutional layer into a 1D vector
2. A layer with 64 neurons and applying ReLU activation function
3. Final fully connected layer with 10 neurons as its the output layer to predict 10 classes or categories
```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```
### Model Summary and visualisation
```python
model.summary()

import pydot
import graphviz
from tensorflow.keras.utils import plot_model

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True
    )
```
### Compile and Train the Model
```python
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics= ['accuracy'])
history = model.fit(train_images, train_labels, epochs=25, validation_data=(test_images, test_labels))
```
### plot training vs validation loss
```python
plt.figure(figsize=(8,3))
# plt.subplots(1,2,1)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
```
### plot training vs validation Accuracy
```python
plt.figure(figsize=(8,3))
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Acuuracy")
plt.legend()
plt.grid(True)
```
----
## CNN model with Batch Normalization + Regularisation + DropOut Layer
```python
from tensorflow.keras.regularizers import l2

model = models.Sequential()

### Convolutional Layer
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32, 3)))
### Batch Normalization Layer
model.add(layers.BatchNormalization())
### Max Pooling Layer
model.add(layers.MaxPool2D((2,2)))
"""
Dropout Layer
"""
model.add(layers.Dropout(0.25))
"""
Convolutional layer with regularization
"""
model.add(layers.Conv2D(64, (3,3), activation = 'relu', kernel_regularizer = l2(0.01)))

### Batch Normalization Layer
model.add(layers.BatchNormalization())
### Max Pooling Layer
model.add(layers.MaxPool2D((2,2)))
"""
Dropout Layer
"""
model.add(layers.Dropout(0.25))
### Convolutional layer
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

# ANN Layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

model.summary()

# Compile and Train
model.compile(optimizer = 'adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics= ['accuracy'])

history3 = model.fit(train_images, train_labels, epochs=25, validation_data=(test_images, test_labels))
```
---
