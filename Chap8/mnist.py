import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_original = input_data.read_data_sets("MNIST_data/", one_hot=False)
x_vals, y_vals = mnist_original.train.images, mnist_original.train.labels
_x_vals, _y_vals = mnist_original.validation.images, mnist_original.validation.labels