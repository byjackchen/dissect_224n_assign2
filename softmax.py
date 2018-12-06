import fire

import numpy as np
import tensorflow as tf

def softmax(x):
    x_max = tf.reduce_max(x, axis=1, keep_dim=True)

