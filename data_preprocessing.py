# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from PIL import Image
from resizeimage import resizeimage
from mini_lambs import JOIN
import numpy as np
import os, sys
assert("3.7" in sys.version)

# 1) for each data input (image), load the pixels into np array
# 2) determine how to deal w/ diffferent size inputs?? (zero padding...)
# 3) train it into a caffe2 RNN [train/CV or just train]
# 4) test it using [CV/test], or just test

#determine the max width and max height of training data
def find_max_width_max_height(filepath):
	chin_char_cwd = JOIN(os.getcwd(),filepath)
	print(os.listdir(chin_char_cwd))
	max_width,max_height = -1,-1
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%100==0): print("Image index is "+str(index))
		jpg_path = JOIN(chin_char_cwd,chin_char)
		for char in os.listdir(jpg_path):
			char_path = JOIN(jpg_path,char)
			im = Image.open(char_path)
			width,height = im.size
			if(width>max_width): max_width=width 
			if(height>max_height): max_height=height
	print(max_width, max_height) # 215 253
	mwmh = open('maxwmaxh.txt',"w")
	mwmh.writelines(["Max Width: {}\n".format(max_width), "Max Height: {}".format(max_height)])
	mwmh.close()
	return 

#write to file - don't return anything
def find_min_width_min_height(filepath):
	chin_char_cwd = JOIN(os.getcwd(),filepath) #chin_char_trn
	print(os.listdir(chin_char_cwd))
	width, height = sys.maxsize, sys.maxsize
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%100==0): print("Image index is %d"%(index),end="\n\n")
		jpg_path = JOIN(chin_char_cwd,chin_char)
		for char in os.listdir(jpg_path):
			char_path = jpg_path + '\\' +char 
			im = Image.open(char_path)
			cur_width,cur_height = im.size
			if(cur_width<width): width=cur_width
			if(cur_height<height): height=cur_height
	print(width,height)
	mwmh = open("minwminh.txt", "w")
	mwmh.writelines(["Min Width: {}\n".format(width), "Min Height: {}\n".format(height)])
	mwmh.close()
	return

def find_avg_width_avg_height(filepath):
	chin_char_cwd = JOIN(os.getcwd() ,filepath) #chin_char_trn
	list_chin_char_cwd = (os.listdir(chin_char_cwd))
	chin_char_list = open("chin_char_list.txt", "w",encoding='utf-8')
	chin_char_list.write('\n'.join(list_chin_char_cwd)+'\n')
	chin_char_list.close()
	cum_width,cum_height,img_count = 0,0,0
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%100==0): print("Image index is %d"%(index),end="\n\n")
		jpg_path = JOIN(chin_char_cwd,chin_char)
		for char in os.listdir(jpg_path):
			char_path = JOIN(jpg_path,char)
			im = Image.open(char_path)
			cur_width,cur_height = im.size
			cum_width+=cur_width
			cum_height+=cur_height
			img_count+=1
	awah = open("avgw_avgh_{}.txt".format(filepath), "w")
	awah.writelines(["Avg. Width: {}\n".format(round(cum_width/img_count)), 
		"Avg. Height: {}\n".format(round(cum_height/img_count))])
	awah.close()
	return round(cum_width/img_count),round(cum_height/img_count)

def zero_padding(max_width,max_height, filename_train_old, filename_train_new):
	chin_char_cwd = JOIN(os.getcwd(),filename_train_old)
	print(os.listdir(chin_char_cwd))
	try: new_chin_char_cwd = os.mkdir(JOIN(os.getcwd(),filename_train_new))
	except: 
		new_chin_char_cwd = JOIN(os.getcwd(),filename_train_new)
		print("new_chin_char_cwd directory exists; moving on...",end='\n\n') 
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%100==0): print("Index currently is "+str(index))
		jpg_path = JOIN(chin_char_cwd,chin_char)
		for char in os.listdir(jpg_path):
			if(chin_char not in os.listdir(new_chin_char_cwd)): os.mkdir(JOIN(new_chin_char_cwd,chin_char))
			new_chin_char_char_cwd = JOIN(new_chin_char_cwd,chin_char)
			im_char = resizeimage.resize_contain(Image.open(JOIN(jpg_path,char)), [max_width,max_height])
			im_char.save(JOIN(new_chin_char_char_cwd,char))
	return 


# resize image to new minimum...
def resize(width, height, filename_train_old, filename_train_new):
	chin_char_cwd = JOIN(os.getcwd(),filename_train_old)
	print(os.listdir(chin_char_cwd))
	try: 
		os.mkdir(JOIN(os.getcwd(),filename_train_new))
		new_chin_char_cwd = JOIN(os.getcwd(),filename_train_new)
		print("directory just created - moving on...")
	except:
		new_chin_char_cwd = JOIN(os.getcwd(),filename_train_new)
		print("directory already exists - moving on...")
	info("Current Chinese Character WD: {}".format(new_chin_char_cwd))
	for index,chin_char in enumerate(os.listdir(chin_char_cwd)):
		if(index%50==0): print("\tindex value is %d"%index)
		jpg_path = JOIN(chin_char_cwd,chin_char)
		for char in os.listdir(jpg_path):
			if(chin_char not in os.listdir(new_chin_char_cwd)): os.mkdir(JOIN(new_chin_char_cwd,chin_char))
			new_chin_char_char_cwd = JOIN(new_chin_char_cwd,chin_char)
			im_char = Image.open(JOIN(jpg_path,char))
			im_char_copy = im_char.copy()
			im_char_copy = im_char_copy.resize((width,height))
			im_char_copy.save(JOIN(new_chin_char_char_cwd,char))
	return

def main():
	AVG_WID_TRN,AVG_HEIGHT_TRN = find_avg_width_avg_height('chin_char_trn')
	print("done trn")
	AVG_WID_CV,AVG_HEIGHT_CV = find_avg_width_avg_height('chin_char_cv')
	print("done cv")
	AVG_WID_TST,AVG_HEIGHT_TST = find_avg_width_avg_height('chin_char_tst')
	print("done tst")
	#Resize all images to exact same width/height;
	AVG_WIDTH = round((AVG_WID_TRN+AVG_WID_CV+AVG_WID_TST)/3)
	AVG_HEIGHT = round((AVG_HEIGHT_TRN+AVG_HEIGHT_CV+AVG_HEIGHT_TST)/3)
	resize(AVG_WIDTH,AVG_HEIGHT,"chin_char_trn","chin_char_trn_preproc")
	resize(AVG_WIDTH,AVG_HEIGHT,"chin_char_cv","chin_char_cv_preproc")
	resize(AVG_WIDTH,AVG_HEIGHT,"chin_char_tst","chin_char_tst_preproc")
	return

main()

 

