import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def mnist_mlp(NUM_INPUT, NUM_OUTPUT, a_0, y, mb_size):
    # 경험에 근거한 법칙은 0.1 * (DATA_SIZE) > (L + 1) * M + (M + 1) * N 을 만족하도록 M을 정하는 것
    # 즉 M < (0.1 * (DATA_SIZE) - N) / (L + 1 + N)
    # L = Input, M = Hidden, N = Output
    DATA_SIZE = mnist_original.train.images.shape[0]
    NUM_HIDDEN = int((0.1 * (DATA_SIZE) - NUM_OUTPUT) / (NUM_INPUT + 1 + NUM_OUTPUT))

    # MODEL
    with tf.name_scope('mlp_{minibatch_size}'.format(minibatch_size=mb_size)):
        w_1 = tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN], seed=1111))
        b_1 = tf.Variable(tf.random_normal([1, NUM_HIDDEN], seed=1111))
        # [_, NUM_HIDDEN], broadcasting b_1
        z_1 = tf.matmul(a_0, w_1) + b_1
        # [_, NUM_HIDDEN]
        # 절대 ReLU을 사용하지 말 것, unscaled log probability가 굉장히 커져서 softmax시 cost의 backpropagation이 제대로 되지 않는다.
        a_1 = tf.nn.sigmoid(z_1)
        w_2 = tf.Variable(tf.random_normal([NUM_HIDDEN, NUM_OUTPUT], seed=1111))
        b_2 = tf.Variable(tf.random_normal([1, NUM_OUTPUT], seed=1111))
        # [_, NUM_OUTPUT], broadcasting b_2
        z_2 = tf.matmul(a_1, w_2) + b_2
        # [_, NUM_OUTPUT]
        a_2 = tf.nn.softmax(z_2)

        cost = tf.reduce_sum(-y * tf.log(tf.clip_by_value(a_2, 1e-10, 1.0)) - (1 - y) * tf.log(tf.clip_by_value(1 - a_2, 1e-10, 1.0)), 1, name='cost')

         # BACK PROPAGATION AND UPDATE WEIGHTS AND BIASES
        step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(a_2, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    return [step, cost, a_2, accuracy]

# TRAIN
for mb_size in [1, 5, 10, 20, 50]:
    a_0 = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    mnist_original = input_data.read_data_sets("MNIST_data/", one_hot=True)

    step, cost, prediction, accuracy = mnist_mlp(784, 10, a_0, y, mb_size)

    with tf.name_scope('summary_{mb_size}'.format(mb_size=mb_size)):
        ns = tf.summary.scalar('cost', tf.reduce_mean(cost), collections=[str(mb_size)])
        acc = tf.summary.scalar('accuracy', accuracy, collections=[str(mb_size)])

    merged = tf.summary.merge_all(key=str(mb_size))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./log/{mb_size}".format(mb_size=mb_size), sess.graph)
        sess.run(tf.global_variables_initializer())
        validation_images, validation_labels = mnist_original.validation.images, mnist_original.validation.labels

        for epoch in range(int(mnist_original.train.images.shape[0] / mb_size) + 1):
            train_batch_images, train_batch_labels = mnist_original.train.next_batch(mb_size)

            sess.run(step, feed_dict={a_0: train_batch_images, y: train_batch_labels})
            if epoch % mb_size == 0:
                summary = sess.run(merged, feed_dict={a_0: validation_images, y: validation_labels})
                writer.add_summary(summary, epoch)