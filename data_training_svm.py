# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
import svmutil
import sys,os
from svm import *
from sklearn.svm import LinearSVC
from PIL import Image
from mini_lambs import JOIN,STR_WRITE
import numpy as np
import pickle
from libsvm import *

def run_loop(CLASSES,new_filename,class_path,NUM_CLASSES,NUM_PTS_PER_CLASS):
	f_data = open(new_filename, 'w', encoding='utf-8')
	for chin_index,chin_char in enumerate(CLASSES):
		if(chin_index%5==0): 
			print("chin_index: {}".format(chin_index))
			if(chin_index==NUM_CLASSES): break
		jpg_path = JOIN(class_path, chin_char)
		for img_index,img in enumerate(os.listdir(jpg_path)):
			if(img_index%5==0):
				print("\timg_index: {}".format(img_index))
				if(img_index==NUM_PTS_PER_CLASS): break
			NP2D = np.asarray(Image.open(JOIN(jpg_path,img)).getdata()).reshape(5670).astype(str)
			write_data = STR_WRITE(chin_index, NP2D)
			f_data.write(write_data)
	f_data.close()
	return

#preprocess the data given filepath (train and test);
# calculate accuracy percentage
def preprocess_data_svm(filepath1, filepath2, NUM_CLASSES, NUM_PTS_PER_CLASS_1, NUM_PTS_PER_CLASS_2):
	class_path_1 = JOIN(os.getcwd(), filepath1)
	class_path_2 = JOIN(os.getcwd(), filepath2)
	CLASSES = os.listdir(class_path_1)
	try: os.mkdir("svm_data")	
	except: print("Directory present - moving on...")
	new_filename_1 = JOIN(os.getcwd(),JOIN("svm_data", filepath1[:9]+".tr"))
	new_filename_2 = JOIN(os.getcwd(),JOIN("svm_data", filepath2[:9]+".te"))
	run_loop(CLASSES,new_filename_1,class_path_1,NUM_CLASSES,NUM_PTS_PER_CLASS_1) 
	run_loop(CLASSES,new_filename_2,class_path_2,NUM_CLASSES,NUM_PTS_PER_CLASS_2)
	return

##loads relevant data from filepath provided...
def train_test_X_Y(filepath1): #NOTE: data MUST be in format provided given filepath
	X_t,y_t = [],[]
	with open(filepath1, 'r', encoding='utf-8') as f:
		for index,line in enumerate(f.readlines()):
			if(index%100==0): print("\tindex: {}".format(index))
			y_t.append(int(line[0]))
			X_t_new = list(filter(lambda x: ":" in x, line[1:].strip().split(" ")))
			X_t_new_2 = [float(line[line.index("."):]) for line in X_t_new] #(extract float - leave out index...)
			X_t.append(X_t_new_2)
	return np.asarray(X_t),np.asarray(y_t)

def sklearn_libsvm_wrapper(X_train,y_train,X_test,y_test):
	def SKLEARN_SVM(X_train,y_train,X_test,y_test,penalty='l2'):
		clf = LinearSVC(penalty=penalty)
		clf.fit(X_train,y_train)
		with open("sklearn_svm.pkl", "wb") as f_pkl:
			pickle.dump(clf, f_pkl)
		score = clf.score(X_test, y_test)
		print("score: {}".format(score))
		with open("sklearn_score.txt", "w") as f_score:
			f_score.write("score: {}".format(score))
	def LIBSVM_svm(X_train,y_train,X_test,y_test,C=10):
		prob = svm_problem(y_train,X_train)
		print("done prob")
		params = svm_parameter(kernel_type='LINEAR',C=C)
		print("done param")
		model = svm_train(prob, params)
		print("done train")
		_,(accr,MSE,SCC),_ = svm_predict(y_test,X_test,model)
		print("Accuracy: {}\nMSE: {}\nSCC: {}\n".format(accr,MSE,SCC))
		svm_save_model("chin_char.model",model)
	SKLEARN_SVM(X_train,y_train,X_test,y_test)
	LIBSVM_svm(X_train,y_train,X_test,y_test)

# determine how to work w/ the training data

# 100 samples/class train; 20 samples/class test
def main_shell():
	print("Two SVMs: one using Sklearn and another using LIBSVM")
	parser = argparse.ArgumentParser(description="""Argument parser for SVM:\n""")
	parser.add_argument('--NUM_CLASSES',type=int, default=200, help='input denoting number of classes to discern')
	parser.add_argument('--NUM_PTS_PER_CLASS_1',type=int, default=100, help='number of training pts per class [MAX=118]')
	parser.add_argument('--NUM_PTS_PER_CLASS_2',type=int,default=20, help='number of test pts per class [MAX=28]')
	args = parser.parse_args()
	preprocess_data_svm("chin_char_trn_preproc","chin_char_tst_preproc",args.NUM_CLASSES,args.NUM_PTS_PER_CLASS_1,args.NUM_PTS_PER_CLASS_2)
	#NOTE: data available in "chin_char.tr" and "chin_char.te" files respectively; 
		# You can navigate to "svm_data" from CMD and run via command line, provided you setup libsvm
	X_train,y_train = train_test_X_Y(JOIN(JOIN(os.getcwd(), "svm_data"), "chin_char.tr"))
	X_test,y_test  = train_test_X_Y(JOIN(JOIN(os.getcwd(), "svm_data"), "chin_char.te"))
	sklearn_libsvm_wrapper(X_train,y_train,X_test,y_test)
main_shell()
