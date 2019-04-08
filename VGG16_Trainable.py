import tensorflow as tf
import tensorflow.contrib.tensorrt
import numpy as np

import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    """
    A trainable version of VGG16
    """
