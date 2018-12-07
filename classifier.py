import time

import fire
import numpy as np
import tensorflow as tf

from base_model import Model
from math_tools import softmax
from math_tools import cross_entropy_loss
from utils import Batch_Utils

class Config(object):
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4

class softmax_ce_classifer(Model):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, (self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, (self.config.batch_size, self.config.n_classes))

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.labels_placeholder: labels_batch
        }
        return feed_dict

    def add_prediction_op(self):
        with tf.variable_scope('transformation'):
            bias = tf.Variable(tf.random_uniform([self.config.n_classes]))
            W = tf.Variable(tf.random_uniform([self.config.n_features, self.config.n_classes]))
            z = tf.matmul(self.input_placeholder, W) + bias
            pred = softmax(z)
        return pred

    def add_loss_op(self, pred):
        loss = cross_entropy_loss(self.labels_placeholder, pred)
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
        return train_op
    
    def run_epoch(self, sess, inputs, labels):
        n_minibatches, total_loss = 0, 0
        for input_batch, label_batch in Batch_Utils.get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, label_batch)
        return total_loss / n_minibatches
    def fit(self, sess, inputs, labels):
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            end_time = time.time()
            duration = start_time - end_time
            print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        return losses

    def __init__(self, config):
        self.config = config
        self.build()

def test_softmax_ce_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 0] = 1

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default() as graph:
        # Build the model and add the variable initializer op
        model = softmax_ce_classifer(config)
        init_op = tf.global_variables_initializer()
    # Finalizing the graph causes tensorflow to raise an exception if you try to modify the graph
    # further. This is good practice because it makes explicit the distinction between building and
    # running the graph.·
    graph.finalize()

    # Create a session for running ops in the graph
    with tf.Session(graph=graph) as sess:
        # Run the op to initialize the variables.
        sess.run(init_op)
        # Fit the model
        losses = model.fit(sess, inputs, labels)

    # If ops are implemented correctly, the average loss should fall close to zero
    # rapidly.
    assert losses[-1] < .5
    print ("Basic (non-exhaustive) classifier tests pass")

if __name__ == '__main__':
    fire.Fire()
