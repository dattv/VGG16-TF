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
        print("npy file loadded")
        print(self.data_dict["conv1_1"][0])

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

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)



