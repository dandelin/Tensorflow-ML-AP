import tensorflow as tf

def sigmoid(z):
    return 1 / (1 + tf.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def naive_mlp(NUM_HIDDEN, a_0, y):
    NUM_INPUT = 2
    NUM_OUTPUT = 1

    # MODEL
    with tf.name_scope('naive_{nh}'.format(nh=NUM_HIDDEN)):
        w_1 = tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN], seed=1111))
        b_1 = tf.Variable(tf.random_normal([1, NUM_HIDDEN], seed=1111))
        # [_, NUM_HIDDEN], broadcasting b_1
        z_1 = tf.matmul(a_0, w_1) + b_1
        # [_, NUM_HIDDEN]
        a_1 = sigmoid(z_1)
        w_2 = tf.Variable(tf.random_normal([NUM_HIDDEN, NUM_OUTPUT], seed=1111))
        b_2 = tf.Variable(tf.random_normal([1, NUM_OUTPUT], seed=1111))
        # [_, NUM_OUTPUT], broadcasting b_2
        z_2 = tf.matmul(a_1, w_2) + b_2
        # [_, NUM_OUTPUT]
        a_2 = sigmoid(z_2)
        error = a_2 - y

        # COST
        cost = tf.mul(0.5, error * error, name='cost')

        # BACK PROPAGATION
        d_error = error
        # [_, NUM_OUTPUT]
        d_a_2 = error
        d_z_2 = d_a_2 * sigmoid_prime(z_2)
        d_b_2 = d_z_2
        # [NUM_HIDDEN, NUM_OUTPUT]
        d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)
        # [_, NUM_HIDDEN]
        d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
        d_z_1 = d_a_1 * sigmoid_prime(z_1)
        d_b_1 = d_z_1
        # [NUM_INPUT, NUM_HIDDEN]
        d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)
        # [_, NUM_INPUT]
        d_a_0 = tf.matmul(d_z_1, tf.transpose(w_1))

        # UPDATES WEIGHTS AND BIASES
        eta = 0.5
        step = [
            tf.assign(w_1, w_1 - eta * d_w_1),
            tf.assign(b_1, b_1 - eta * tf.reduce_mean(d_b_1, 0)),
            tf.assign(w_2, w_2 - eta * d_w_2),
            tf.assign(b_2, b_2 - eta * tf.reduce_mean(d_b_2, 0))
        ]

    return [step, cost]

def built_in_mlp(NUM_HIDDEN, a_0, y):
    NUM_INPUT = 2
    NUM_OUTPUT = 1

    # MODEL
    with tf.name_scope('built_in_{nh}'.format(nh=NUM_HIDDEN)):
        w_1 = tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN], seed=1111))
        b_1 = tf.Variable(tf.random_normal([1, NUM_HIDDEN], seed=1111))
        # [_, NUM_HIDDEN], broadcasting b_1
        z_1 = tf.matmul(a_0, w_1) + b_1
        # [_, NUM_HIDDEN]
        a_1 = sigmoid(z_1)
        w_2 = tf.Variable(tf.random_normal([NUM_HIDDEN, NUM_OUTPUT], seed=1111))
        b_2 = tf.Variable(tf.random_normal([1, NUM_OUTPUT], seed=1111))
        # [_, NUM_OUTPUT], broadcasting b_2
        z_2 = tf.matmul(a_1, w_2) + b_2
        # [_, NUM_OUTPUT]
        a_2 = sigmoid(z_2)
        error = a_2 - y

        # COST
        cost = tf.mul(0.5, error * error, name='cost')

        # BACK PROPAGATION AND UPDATE WEIGHTS AND BIASES
        step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    
    return [step, cost]

# DATA

_in = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]


_out = [
    [0],
    [1],
    [1],
    [0]
]

a_0 = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

MAX_NUM = 11

# TRAIN

for nh in range(1, MAX_NUM):
    with tf.Session() as sess:
        naive_step, naive_cost = naive_mlp(nh, a_0, y)
        built_in_step, built_in_cost = built_in_mlp(nh, a_0, y)
        
        with tf.name_scope('summary_{nh}'.format(nh=nh)):
            ns = tf.summary.scalar('naive', tf.reduce_mean(naive_cost), collections=[str(nh)])
            bs = tf.summary.scalar('built_in', tf.reduce_mean(built_in_cost), collections=[str(nh)])
        
        merged = tf.summary.merge_all(key=str(nh))
        writer = tf.summary.FileWriter("./log/{nh}".format(nh=nh), sess.graph)

        sess.run(tf.global_variables_initializer())
        
        for epoch in range(1000):
            sess.run(naive_step, feed_dict={a_0: _in, y: _out})
            sess.run(built_in_step, feed_dict={a_0: _in, y: _out})
            summary = sess.run(merged, feed_dict={a_0: _in, y: _out})
            writer.add_summary(summary, epoch)