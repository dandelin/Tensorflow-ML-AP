# 패리티 비트에 대한 단순 퍼셉트론
import tensorflow as tf

T, F, bias = 1, 0, -1

pinput = [
    [F, F, F, bias],
    [F, F, T, bias],
    [F, T, F, bias],
    [F, T, T, bias],
    [T, F, F, bias],
    [T, F, T, bias],
    [T, T, F, bias],
    [T, T, T, bias]
]

poutput = [
    [T],
    [F],
    [F],
    [T],
    [F],
    [T],
    [T],
    [F]
]

x = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.Variable(tf.random_uniform([4, 1]))
before_activation = tf.matmul(x, w)
after_activation = tf.select(tf.greater(before_activation, 0), before_activation / before_activation, before_activation * 0)
error = after_activation - Y
update_w = tf.assign(w, w - 0.25 * tf.matmul(tf.transpose(x), error))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10):
        print('{i}\'s iteration, weights : \n'.format(i=i), sess.run(w))
        print('final outputs are : \n', sess.run(after_activation, feed_dict={x: pinput}))
        sess.run(update_w, feed_dict={x: pinput, Y: poutput})