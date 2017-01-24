# THIS CODE IS MODIFICATION OF https://github.com/llSourcell/Tensorboard_demo/blob/master/simple.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def create_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def create_model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    with tf.name_scope('Input_layer'):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope('Hidden_layer'):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope('Output_layer'): 
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o)

# GET DATA
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# CREATE PLACEHOLDER
X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')

# INITIALIZE WEIGHTS
w_h = create_weights([784, 60], "w_h")
w_h2 = create_weights([60, 60], "w_h2")
w_o = create_weights([60, 10], "w_o")

# [TENSORBOARD] ADD HISTOGRAM SUMMARIES FOR WEIGHTS
tf.summary.histogram('w_h_summ', w_h)
tf.summary.histogram('w_h2_summ', w_h2)
tf.summary.histogram('w_o_summ', w_o)

# ADD DROPOUT
p_keep_input = tf.placeholder(dtype=tf.float32, name="p_keep_input")
p_keep_hidden = tf.placeholder(dtype=tf.float32, name="p_keep_hidden")

# THE MODEL
model = create_model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# COST FUNCTION
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(cost)
    # [TENSORBOARD] ADD SCALAR SUMMARY FOR COST
    tf.summary.scalar('cost', cost)

# ACCURACY
with tf.name_scope('accuracy'):
    corrects = tf.equal(tf.argmax(Y, axis=1), tf.argmax(model, axis=1))
    acc_op = tf.reduce_mean(tf.cast(corrects, tf.float32))
    # [TENSORBOARD] ADD SCALAR SUMMARY FOR ACCURACY
    tf.summary.scalar('accuracy', acc_op)

# SESSION
with tf.Session() as sess:
    # [TENSORBOARD] CREATE LOG WRITER
    writer = tf.summary.FileWriter('./logs', sess.graph)
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            train_feed = {
                X: trX[start:end],
                Y: trY[start:end],
                p_keep_input: 0.8,
                p_keep_hidden: 0.5
            }
            sess.run(train_op, feed_dict=train_feed)
            test_feed = {
                X: teX,
                Y: teY,
                p_keep_input: 1.0,
                p_keep_hidden: 1.0
            }
            summary, acc = sess.run([merged, acc_op], feed_dict=test_feed)
            writer.add_summary(summary, i)
        print(i, acc)