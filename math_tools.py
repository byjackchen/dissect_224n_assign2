import fire

import numpy as np
import tensorflow as tf

def softmax(x):
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    x_exp = tf.exp(tf.subtract(x,x_max))
    sfmx = tf.div(x_exp, tf.reduce_sum(x_exp, axis=1, keepdims=True))
    return sfmx

def corss_entropy_loss(y, y_hat):
    y_hat_l = tf.log(y_hat)
    ce = tf.negative(tf.reduce_sum(tf.multiply(tf.to_float(y),y_hat_l)))
    return ce

def test_softmax():
    a = tf.constant(np.array([[1001, 1002],[3,4]]), dtype=tf.float32)
    sfmx_a = softmax(a)
    with tf.Session() as sess:
        sfmx_a = sess.run(sfmx_a)
        print(80*'=')
        print(sfmx_a)
    utils_all_close('softmax test a', sfmx_a, np.array([[0.26894142, 0.73105858],
                                                      [0.26894142, 0.73105858]]))

def test_cross_entropy():
    b = tf.constant(np.array([[0, 1], [1, 0], [1, 0]]), dtype=tf.float32)
    b_hat = tf.constant(np.array([[.5, .5], [.5, .5], [.5, .5]]),
                                dtype=tf.float32)

    ce_b = corss_entropy_loss(b,b_hat)
    with tf.Session() as sess:
        ce_b = sess.run(ce_b)
        print(80*'=')
        print(ce_b)
    ce_b_expected = -3*np.log(.5)
    utils_all_close('corss entropy testing', ce_b_expected, ce_b)


def utils_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} failed, expected output to have shape {:} but has shape {:}"
                         .format(name, expected.shape, actual.shape))
    if np.amax(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} failed, expected {:} but value is {:}".format(name, expected, actual))
    else:
        print (name, "passed!")


if __name__ == '__main__':
    fire.Fire()

