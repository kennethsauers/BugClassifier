import requests, json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
def onehotToLabel(onehot):
    for i in range(len(onehot[0])):
        if onehot[0][i] >= 0.5:
            return CIFAR100_LABELS_LIST[i]
    return -1

fashion_mnist = keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

def main():
    url = "http://192.168.0.252:5000/evaluate"
    data = test_images[4].reshape([32*32*3])
    data = data.tolist()
    body  ={'data' : data}
    headers = {'content-type': 'application/json'}

    r = requests.post(url, data=json.dumps(body), headers=headers)
    r = json.loads(r.text)
    print(onehotToLabel(r['data']))


if __name__ == "__main__":
    main()
