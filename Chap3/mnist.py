import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
import numpy as np
mnist = fetch_mldata('MNIST original')

x = mnist.data
y = np.asarray([[1 if i == ii else 0 for i in range(10)] for ii in mnist.target])

_x = tf.placeholder(tf.float32, shape=[None, x.shape[1]+1])
_y = tf.placeholder(tf.float32, shape=[None, y.shape[1]])
w = tf.Variable(tf.random_uniform([x.shape[1]+1, y.shape[1]]))
before_activation = tf.matmul(_x, w)
after_softmax = tf.nn.softmax(before_activation)
#after_activation = tf.select(tf.greater(before_activation, 0), before_activation / before_activation, before_activation * 0)
error = after_softmax - _y
update_w = tf.assign(w, w - 0.025 * tf.matmul(tf.transpose(_x), error))

kf = KFold(n_splits=5)

def equal_array(output, true_y):
    return np.argmax(output) == np.argmax(true_y)

for train_index, test_index in kf.split(x):
    print('NEW FOLD')
    train_bias = [[-1] for _ in range(len(train_index))]
    test_bias = [[-1] for _ in range(len(test_index))]
    x_train, x_test = np.append(x[train_index], train_bias, axis=1), np.append(x[test_index], test_bias, axis=1)
    y_train, y_test = y[train_index], y[test_index]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(10):
            sess.run(update_w, feed_dict = {_x: x_train, _y: y_train})
            o = sess.run(after_softmax, feed_dict={_x: x_test})
            tf_list = [1 if equal_array(output, true_y) else 0 for output, true_y in zip(o, y_test)]
            acc = sum(tf_list) / len(tf_list)
            print(acc)