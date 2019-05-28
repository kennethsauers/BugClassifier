import requests, json


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



def main():
    url = "http://127.0.0.1:5000/evaluate"
    data = test_images[4].reshape([28*28])
    data = data.tolist()
    body  ={'data' : data}
    headers = {'content-type': 'application/json'}

    r = requests.post(url, data=json.dumps(body), headers=headers)
    print(r.text)


if __name__ == "__main__":
    main()
