# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

testing = False
epochs = 10
fashion_mnist = keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

model = keras.Sequential([
# TODO: add max pool and more layers to make model more accurate
  keras.layers.Conv2D(input_shape=(32,32,3), filters=8, kernel_size=3,
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Conv2D (kernel_size = 3, filters = 32, strides = 2,
                      activation='relu', name = 'Conv2'),
  keras.layers.Flatten(),
  keras.layers.Dense(1024, activation = 'relu', name= 'Dense1'),
  keras.layers.Dense(100, activation=tf.nn.softmax, name='Softmax')
])
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))

MODEL_DIR = "model"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save("model/mymodel.h5")
