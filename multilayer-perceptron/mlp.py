import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


class Mlp:
    def __init__(self, n, hidden_layer, learning_rate):
        self.n_iterations = n
        self.hidden_neurons = hidden_layer
        self.learning_rate = learning_rate

    @staticmethod
    def init_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def init_bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    @staticmethod
    def model(x, weight_input, weight_output, bias_input, bias_output):
        activation = tf.nn.sigmoid(tf.matmul(x, weight_input) + bias_input)
        # softmax applied later (in main thread)
        return tf.matmul(activation, weight_output) + bias_output

    def train(self):
        print('Downloading MNIST dataset...')
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        print('MNIST dataset downloaded')

        x = tf.placeholder("float", [None, 784])
        w_i = Mlp.init_weights([784, self.hidden_neurons])
        w_o = Mlp.init_weights([self.hidden_neurons, 10])
        b_i = Mlp.init_bias([self.hidden_neurons])
        b_o = Mlp.init_bias([10])

        y = Mlp.model(x, w_i, w_o, b_i, b_o)
        y_ = tf.placeholder("float", [None, 10])

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print("step,training accuracy,test accuracy")
        for i in range(self.n_iterations):
            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

            if i % 100 == 0:
                test_accuracy = sess.run(accuracy,
                    feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                train_accuracy = sess.run(accuracy,
                    feed_dict={x: mnist.train.images, y_: mnist.train.labels})
                print('{:d},{:1.5f},{:1.5f}'.format(i, train_accuracy, test_accuracy))
