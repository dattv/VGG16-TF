import tensorflow as tf
import numpy as np
from VGG16_Trainable import *
import os


TEST_DATA_DIR = "./test_data"
if os.path.isdir(TEST_DATA_DIR):
    print("{} is not folder".format(TEST_DATA_DIR))

else:
    RESULT_DIR = "./result"
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    files = os.listdir(TEST_DATA_DIR)
    print(files)
