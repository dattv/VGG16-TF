import numpy as np
import tensorflow as tf

import os
import time

from vgg16 import *

model = vgg16("./WEIGHTS/vgg16.npy")
