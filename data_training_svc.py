# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pickle
import svmutil
import sys,os
from svm import *
from sklearn.svm import LinearSVC
from PIL import Image
import numpy as np



def join(base,extension):
	return base+'\\'+extension
def str_write(chin_index, NP2D):
	return str(chin_index)+" "+"".join([str(feat_index+1)+":"+str(round(float(val)/256, 4))+" " 
			for feat_index,val in enumerate(list(NP2D))])+"\n"

def run_loop(CLASSES,new_filename,class_path,NUM_CLASSES,NUM_PTS_PER_CLASS):
	f_data = open(new_filename, 'w', encoding='utf-8')
	for chin_index,chin_char in enumerate(CLASSES):
		if(chin_index%5==0): 
			print("chin_index: {}".format(chin_index))
			if(chin_index==NUM_CLASSES): break
		jpg_path = join(class_path, chin_char)
		for img_index,img in enumerate(os.listdir(jpg_path)):
			if(img_index%5==0):
				print("\timg_index: {}".format(img_index))
				if(img_index==NUM_PTS_PER_CLASS): break
			NP2D = np.asarray(Image.open(join(jpg_path,img)).getdata()).reshape(5670).astype(str)
			write_data = str_write(chin_index, NP2D)
			f_data.write(write_data)
	f_data.close()
	return

#preprocess the data given filepath (train and test);
# calculate accuracy percentage
def preprocess_data(filepath1, filepath2, NUM_CLASSES=150, NUM_PTS_PER_CLASS_1=100, NUM_PTS_PER_CLASS_2=20):
	class_path_1 = join(os.getcwd(), filepath1)
	class_path_2 = join(os.getcwd(), filepath2)
	CLASSES = os.listdir(class_path_1)
	new_filename_1 = os.getcwd()+"\\libsvm\\"+filepath1[:9]+"_2.tr"
	new_filename_2 = os.getcwd()+"\\libsvm\\"+filepath1[:9]+"_2.te"
	run_loop(CLASSES,new_filename_1,class_path_1,NUM_CLASSES,NUM_PTS_PER_CLASS_1) 
	run_loop(CLASSES,new_filename_2,class_path_2,NUM_CLASSES,NUM_PTS_PER_CLASS_2)
	# print("Len f_data object: {}".format(len(f_data.readlines())))
	return

# 100 samples/class train; 20 samples/class test
### preprocess_data("chin_char_trn_preproc2", "chin_char_tst_tst_preproc2")

# create distribution of accuracy per class (/20 per class)
def compare_accuracy_by_class(pred,y_test):
	return

def SKLEARN_SVM(penalty='l2'):
	def train_test_X_Y(filepath1):
		X_t,y_t = [],[]
		with open(filepath1, 'r', encoding='utf-8') as f:
			for index,line in enumerate(f.readlines()):
				if(index%100==0): print("index: {}".format(index))
				y_t.append(int(line[0]))
				X_t_new = list(filter(lambda x: ":" in x, line[1:].strip().split(" ")))
				assert(len(X_t_new)==5670)
				X_t_new_2 = [float(line[line.index("."):]) for line in X_t_new]
				X_t.append(X_t_new_2)
		return np.asarray(X_t),np.asarray(y_t)
	X_train,y_train = train_test_X_Y(join(join(os.getcwd(), "libsvm"), "chin_char_2.tr"))
	print("done loading train; X_shape: {}; Y_shape: {}".format(X_train.shape, y_train.shape))
	X_test, y_test  = train_test_X_Y(join(join(os.getcwd(), "libsvm"), "chin_char_2.te"))
	print("done loading test")
	clf = LinearSVC(penalty=penalty)
	clf.fit(X_train,y_train)
	pred = clf.predict(X_test)
	compare_accuracy_by_class(pred,y_test)
	score = clf.score(X_test, y_test)
	print("score: {}".format(score))

# determien how to work w/ the training data... (only 4 lines for training??)
SKLEARN_SVM()