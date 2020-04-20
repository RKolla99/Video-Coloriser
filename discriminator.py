import tensorflow as tf
import ops


class Discriminator(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky'):
        print('Init Discriminator '+name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, input):
        with tf.compat.v1.variable_scope(self.name, reuse=self._reuse):
            D = ops.conv3_block(input, 64, 'CD1', 4, 2, self._is_train,
                                self._reuse, norm=None, activation=self._activation)
            D = ops.conv3_block(D, 128, 'CD2', 4, 2, self._is_train,
                                self._reuse, self._norm, self._activation)
            D = ops.conv3_block(D, 256, 'CD3', 4, 2, self._is_train,
                                self._reuse, self._norm, self._activation)
            D = ops.conv3_block(D, 512, 'CD4', 4, 2, self._is_train,
                                self._reuse, self._norm, self._activation)
            D = ops.conv3_block(D, 1, 'CD5', 4, 1, self._is_train,
                                self._reuse, norm=None, activation=None, bias=True)
            D = tf.reduce_mean(D, axis=[1, 2, 3])

            self._reuse = True
            self.var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return D
