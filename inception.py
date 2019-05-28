import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

print(x_train[4])
