# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from PIL import Image
import caffe2
import os, sys
assert("3." in sys.version)
# 1) for each data input (image), load the pixels into np array
# 2) determine how to deal w/ diffferent size inputs?? (zero padding...)
# 3) train it into a caffe2 RNN [train/CV or just train]
# 4) test it using [CV/test], or just test

#determine the max width and max height of training data
def max_width_max_height():
	pass

def zero_padding(max_width,max_height):
	pass
	#pad all images with zeroes until they resize