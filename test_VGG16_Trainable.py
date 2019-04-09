import tensorflow as tf
import numpy as np
from VGG16_Trainable import *
import os
import skimage.io
import matplotlib.pyplot as plt

TEST_DATA_DIR = "./test_data"
if not os.path.isdir(TEST_DATA_DIR):
    print("{} is not folder".format(TEST_DATA_DIR))

else:
    RESULT_DIR = "./result"
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    files = os.listdir(TEST_DATA_DIR)
    img = []
    for f in files:
        file = os.path.join(os.getcwd(), TEST_DATA_DIR[2:])
        file = os.path.join(file, f)

        img.append(skimage.io.imread(file))

    plt.subplot(1, 2, 1).set_title("first")
    plt.imshow(img[0])
    plt.subplot(1, 2, 2).set_title("second")
    plt.imshow(img[1])

