# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
assert("3.7" in sys.version)
from CASIA import CASIA
from PIL import Image 
from mini_lambs import JOIN
import os
import logging
from logging import debug,info,warning
logging.basicConfig(level=logging.INFO)


def load_and_partition_images(basepath):
	cas = CASIA()
	path = JOIN(os.getcwd(),basepath)
	char_dict = {}
	for filename in os.listdir(path):
		info("Current filename: {}".format(filename))
		img_lab_list = cas.load_gnt_file(JOIN(path,filename))
		chin_char_path = JOIN(os.getcwd(),"chin_char_{}".format(basepath[7:-4])) # extract out ['trn'/'tst'/'cv'] respectively
		try: os.mkdir(chin_char_path)
		except: warning("Directory {} already made - moving on...".format(chin_char_path))
		for img,lab in img_lab_list:
		 	chin_char_path_char = JOIN(chin_char_path,"{}".format(lab))
		 	if(lab not in os.listdir(chin_char_path)): os.mkdir(chin_char_path_char)
		 	num_elems = len(os.listdir(chin_char_path_char))
		 	img.save(JOIN(chin_char_path_char, '{}_{}.jpg'.format(lab,num_elems)))
	return
	

load_and_partition_images('HWDB1.1trn_gnt')
load_and_partition_images('HWDB1.1cv_gnt')
load_and_partition_images('HWDB1.1tst_gnt')

