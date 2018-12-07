import sys
import time
import numpy as np

class Batch_Utils(object):
    @staticmethod
    def get_minibatches(data, minibatch_size, shuffle=True):
        flag_data_is_list = type(data) is list and \
                            (type(data[0]) is list or type(data[0]) is np.ndarray)
        data_size = len(data[0]) if flag_data_is_list else len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, data_size, minibatch_size):
            minibatch_indices = indices[minibatch_start: minibatch_start+minibatch_size]
            yield [Batch_Utils._minibatch(d, minibatch_indices) for d in data] if flag_data_is_list \
                else Batch_Utils._minibatch(data, minibatch_indices)
    @staticmethod
    def _minibatch(data, minibatch_idx):
        return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

