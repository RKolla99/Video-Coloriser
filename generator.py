import tensorflow as tf
import ops
import numpy as np


class Generator(object):
    def __init__(self, name, is_train, norm='instance', activation='relu',
                 image_size=256, output_channels=3):
        print('Init Generator '+name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self.output_channels = output_channels
        self._num_res_block = 9
        self._reuse = False

    def __call__(self, input):
        with tf.compat.v1.variable_scope(self.name, reuse=self._reuse):
            G = ops.conv3_block(input, 64, 'c7s1-32', 7, 1, self._is_train,
                                self._reuse, self._norm, self._activation, pad='REFLECT')
            G = ops.conv3_block(G, 128, 'd64', 3, 2, self._is_train,
                                self._reuse, self._norm, self._activation)
            G = ops.conv3_block(G, 256, 'd128', 3, 2, self._is_train,
                                self._reuse, self._norm, self._activation)
            for i in range(self._num_res_block):
                G = ops.residual3(G, 256, 'R128_{}'.format(i), self._is_train,
                                  self._reuse, self._norm)
            G = ops.deconv3_block(G, 128, 'u64', 3, 2, self._is_train,
                                  self._reuse, self._norm, self._activation)
            G = ops.deconv3_block(G, 64, 'u32', 3, 2, self._is_train,
                                  self._reuse, self._norm, self._activation)
            G = tf.pad(G, [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            G = ops.conv3_block(G, self.output_channels, 'c7s1-3', 7, 1, self._is_train,
                                self._reuse, norm=None, activation='tanh', pad='VALID')

            self._reuse = True
            self.var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return G
