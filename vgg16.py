import inspect
import os
import numpy as np
import tensorflow as tf

import time
import urllib.request

VGG_MEAN = [103.939, 116.779, 123.68]

class vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            if not os.path.isfile(path):
                print("Error in loading vgg16.npy file")
                return

            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        print(self.data_dict.keys())

    def build_graph(self, rgb):
        """
        Load variable from npy to build the VGG16
        :param rgb:
        :return:
        """
        start_time = time.time()
        print("Build model is started")
        rgb_scaled = rgb * 255.e0

        # convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                        green - VGG_MEAN[1],
                                        red - VGG_MEAN[2],
                                        ])

        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, "pool3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, "pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")



    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME", name=name + "_weight")

            conv_bias = self.get_bias_(name)
            bias = tf.nn.bias_add(conv, conv_bias, name=name + "_bias")

            relu = tf.nn.relu(bias, name=name + "_relu")
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1

            for d in shape[1:]:
                dim *= d

            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weights(name)
            biases = self.get_bias_(name)

            # fully connected layer
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases, name=name)
            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name=name)

    def get_bias_(self, name):
        return tf.constant(self.data_dict[name][1], name=name)

    def get_fc_weights(self, name):
        return tf.constant(self.data_dict[name][0], name=name)

