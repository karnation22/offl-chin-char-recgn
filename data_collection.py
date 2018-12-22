# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
print(sys.version)
from CASIA import CASIA
from PIL import Image 
import os



def load_and_partition_images():
	cas = CASIA()
	basepath = '\HWDB1.1tst_gnt' #['\HWDB1.1trn_gnt', '\HWDB1.1tst.gnt']
	print(os.getcwd()+basepath)
	path = os.getcwd()+basepath
	char_dict = {}
	for filename in os.listdir(path)[1:]:
		print(filename)
		img_lab_list = cas.load_gnt_file(path+'\\' + filename)
		img,lab =(img_lab_list[0])
		print(lab.encode('utf-8'))
		chin_char_path = os.getcwd()+"\chin_char_tst_{}".format('trn' if 'trn' in basepath else 'tst')
		try: os.mkdir(chin_char_path)
		except: print("Handle issue later...")
		for index,(img,lab) in enumerate(img_lab_list[1:]):
		 	chin_char_path_char = chin_char_path+"\{}".format(lab)
		 	print(chin_char_path_char)
		 	if(lab not in os.listdir(chin_char_path)): os.mkdir(chin_char_path_char)
		 	num_elems = len(os.listdir(chin_char_path_char))
		 	img.save(chin_char_path_char+ '\\'+ '{}_{}.jpg'.format(lab,num_elems))
	return
	

#load_and_partition_images()




