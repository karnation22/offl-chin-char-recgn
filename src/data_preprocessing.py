# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from PIL import Image
from resizeimage import resizeimage
import caffe2 #handle caffe2_pb2?
import os, sys
assert("3." in sys.version)
# 1) for each data input (image), load the pixels into np array
# 2) determine how to deal w/ diffferent size inputs?? (zero padding...)
# 3) train it into a caffe2 RNN [train/CV or just train]
# 4) test it using [CV/test], or just test

#determine the max width and max height of training data
def find_max_width_max_height(filepath):
	chin_char_cwd = os.getcwd()+'\\'+filepath
	print(os.listdir(chin_char_cwd))
	max_width,max_height = -1,-1
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%100==0): print("Image index is "+str(index))
		jpg_path = chin_char_cwd +'\\'+chin_char
		for char in os.listdir(jpg_path):
			char_path = jpg_path + "\\"+char
			im = Image.open(char_path)
			width,height = im.size
			if(width>max_width): max_width=width 
			if(height>max_height): max_height=height
	return max_width, max_height

def zero_padding(max_width,max_height, filename_train_old, filename_train_new):
	chin_char_cwd = os.getcwd()+'\\'+filename_train_old
	print(os.listdir(chin_char_cwd))
	try: new_chin_char_cwd = os.mkdir(os.getcwd()+'\\'+filename_train_new)
	except: 
		new_chin_char_cwd = os.getcwd()+'\\'+filename_train_new
		print("new_chin_char_cwd directory exists; moving on...",end='\n\n') 
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%100==0): print("Index currently is "+str(index))
		jpg_path = chin_char_cwd+'\\'+chin_char
		for char in os.listdir(jpg_path):
				#print(char,type(char))
			if(chin_char not in os.listdir(new_chin_char_cwd)): os.mkdir(new_chin_char_cwd + "\\"+chin_char)
			new_chin_char_char_cwd = new_chin_char_cwd+'\\'+chin_char
				#print(new_chin_char_char_cwd)
				#print(max_width,max_height)
			im_char = resizeimage.resize_contain(Image.open(jpg_path+'\\'+char), [max_width,max_height])
				#print(im_char.size)
			im_char.save(new_chin_char_char_cwd+"\\"+char)
	return 


max_width, max_height = find_max_width_max_height('chin_char_trn')
print(max_width, max_height) # 215 253
mwmh = open('mwmh.txt',"rw")
mwmh.writelines(["Max Width: {}".format(max_width), "Max Height: {}".format(max_height)])
mwmh.close()


zero_padding(215,253,'chin_char_trn','chin_char_trn_preproc')
