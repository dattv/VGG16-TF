import numpy as np
import tensorflow as tf
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    """
    A trainable version of VGG16
    """

    def __init__(self, vgg16_npy_path=None, trainable=True, dropout=0.5):
        if vgg16_npy_path is None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = []
        self.trainable = trainable
        self.dropout = dropout

    def build(self, rgb, train_mode=False):
        """
        load variable from npy to build VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode:  a bool tensor, unusually a placeholder: if True, dropout will be turned on
        :return:
        """
        rgb_scaled = rgb * 255.e0
        # convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                        green - VGG_MEAN[1],
                                        red - VGG_MEAN[2],])

        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_chanels, out_chanels, name):
        with tf.variable_scope(name):


    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], mean=0.e0,
                                            stddev=1.e-2)


    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name, trainable=self.trainable)
        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()
        return var

    def save_npy(self, sess, npy_path="./vgg16_saved.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}
        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count