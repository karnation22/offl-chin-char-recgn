# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from PIL import Image
import os, sys, argparse
assert("3.7" in sys.version)
from skimage import io
import torch
torch.set_default_tensor_type("torch.DoubleTensor")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import pandas as pd,numpy as np
from numpy import linalg as LA
np.set_printoptions(threshold=np.nan)
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from collections import Counter
from statistics import mean
from random import shuffle 
from mini_lambs import JOIN
from matplotlib import pyplot as plt
import logging
from logging import debug,info
logging.basicConfig(level=logging.INFO)

# HYPERPARAMETERS
NUM_EPOCH=20
NUM_CLASSES=200
L_RATE=0.01
DECAY=0.66
DECAY_FREQ=4
MOMEMTUM=0.9
KERNEL=2
STRIDE=1
INTER_FC=1024
INTER_FC1=512
INTER_FC2=324
INTER_FC3=256
INTER_FC4=240
BATCH_SIZE=200
LOG_INTERVAL=5

# OTHER MARCORS
TOT_CLASSES=3755
MAX_BRIGHT = 255
MIN_COUNT=1 
MEAN_COUNT=5.0 
classes = open('chin_char_list.txt', 'r',encoding='utf-8').readlines()
CLASSES = [clas.strip() for clas in classes] 
JOIN = lambda base,ext: base+"\\"+ext 

class NNC3FC2(nn.Module):
	def __init__(self,l_rate=L_RATE,batch=BATCH_SIZE,l_interval=LOG_INTERVAL,
		num_epoch=NUM_EPOCH, kernel=KERNEL,stride=STRIDE, momentum=MOMEMTUM,output_size=NUM_CLASSES):
		super(NNC3FC2, self).__init__()
		self.l_interval = l_interval
		self.l_rate, self.stride = l_rate,stride
		self.num_epoch, self.kernel = num_epoch,kernel
	
		self.conv1 = nn.Conv2d(1,4,kernel,stride)
		self.conv2 = nn.Conv2d(4,16,kernel,stride)
		self.conv3 = nn.Conv2d(16,256, kernel, stride)	
		self.fc1 = nn.Linear(256*9*7,INTER_FC4)
		self.fc2 = nn.Linear(INTER_FC4, NUM_CLASSES)
		self.batch1 = nn.BatchNorm2d(256*9*7)
		self.batch2 = nn.BatchNorm1d(output_size)

	def forward(self,x):
		# print("shape coming in is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv1(x)), self.kernel)
		# print("shape after round 1 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv2(x)), self.kernel)
		# print("shape after round 2 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv3(x)), self.kernel)
		# print("shape after round 3 is "+str(x.shape))
		x = x.view(-1, self.flatten_features(x))
		# print("shape after round 4 view is "+str(x.shape))
		x = F.relu(self.fc1(x))
		# print("shape after round 5 linear 1 is "+str(x.shape))
		x = self.batch2(self.fc2(x))
		# print("shape after round 6 linear 2 is "+str(x.shape))
		return F.log_softmax(x,dim=1)

	def flatten_features(self, x):
		size = x.size()[1:]  # all dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class NNC4FC2(nn.Module):
	def __init__(self,l_rate=L_RATE,batch=BATCH_SIZE,l_interval=LOG_INTERVAL,
		num_epoch=NUM_EPOCH, kernel=KERNEL,stride=STRIDE, momentum=MOMEMTUM,output_size=NUM_CLASSES):
		super(NNC4FC2, self).__init__()
		self.l_interval = l_interval
		self.l_rate, self.stride = l_rate,stride
		self.num_epoch, self.kernel = num_epoch,kernel
	
		self.conv1 = nn.Conv2d(1,4,kernel,stride)
		self.conv2 = nn.Conv2d(4,16,kernel,stride)
		self.conv3 = nn.Conv2d(16,64, kernel, stride)	
		self.conv4 = nn.Conv2d(64,256,kernel,stride)
		self.fc1 = nn.Linear(256*4*3,INTER_FC)
		self.fc2 = nn.Linear(INTER_FC, NUM_CLASSES)
		self.batch2 = nn.BatchNorm1d(output_size)

	def forward(self,x):
		# print("shape coming in is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv1(x)), self.kernel)
		# print("shape after round 1 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv2(x)), self.kernel)
		# print("shape after round 2 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv3(x)), self.kernel)
		# print("shape after round 3 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv4(x)), self.kernel)
		# print("shape after round 4 is "+str(x.shape))
		x = x.view(-1, self.flatten_features(x))
		# print("shape after round 5 view is "+str(x.shape))
		x = F.relu(self.fc1(x))
		# print("shape after round 6 linear 1 is "+str(x.shape))
		x = self.batch2(self.fc2(x))
		# print("shape after round 7 linear 2 is "+str(x.shape))
		return F.log_softmax(x,dim=1)

	def flatten_features(self, x):
		size = x.size()[1:]  # all dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class NNC5FC3(nn.Module):
	def __init__(self,l_rate=L_RATE,batch=BATCH_SIZE,l_interval=LOG_INTERVAL,
		num_epoch=NUM_EPOCH, kernel=KERNEL,stride=STRIDE, momentum=MOMEMTUM,output_size=NUM_CLASSES):
		super(NNC5FC3, self).__init__()
		self.l_interval = l_interval
		self.l_rate, self.stride = l_rate,stride
		self.num_epoch, self.kernel = num_epoch,kernel
	
		self.conv1 = nn.Conv2d(1,4,kernel,stride)
		self.conv2 = nn.Conv2d(4,16,kernel,stride)
		self.conv3 = nn.Conv2d(16,64, kernel, stride)
		self.conv4 = nn.Conv2d(64, 128, kernel, stride)
		self.conv5 = nn.Conv2d(128,256, kernel, stride)	
		self.fc1 = nn.Linear(256*8*7,INTER_FC)
		self.fc2 = nn.Linear(INTER_FC, INTER_FC1)
		self.fc3 = nn.Linear(INTER_FC1, NUM_CLASSES)
		self.batch2 = nn.BatchNorm1d(output_size)

	def forward(self,x):
		# print("shape coming in is "+str(x.shape))
		x = F.relu(self.conv1(x))
		# print("shape after round 1 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv2(x)), self.kernel)
		# print("shape after round 2 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv3(x)), self.kernel)
		# print("shape after round 3 is "+str(x.shape))
		x = F.relu(self.conv4(x))
		# print("shape after round 4 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv5(x)), self.kernel)
		# print("shape after round 5 is "+str(x.shape))
		x = x.view(-1, self.flatten_features(x))
		# print("shape after round 6 view is "+str(x.shape))
		x = F.relu(self.fc1(x))
		# print("shape after round 7 linear 1 is "+str(x.shape))
		x = F.relu(self.fc2(x))
		# print("shape after round 8 linear 2 is "+str(x.shape))
		x = self.batch2(self.fc3(x))
		# print("shape after round 9 linear 3 is "+str(x.shape))
		return F.log_softmax(x,dim=1)

	def flatten_features(self, x):
		size = x.size()[1:]  # all dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class NNC6FC3(nn.Module):
	def __init__(self,l_rate=L_RATE,batch=BATCH_SIZE,l_interval=LOG_INTERVAL,
		num_epoch=NUM_EPOCH, kernel=KERNEL,stride=STRIDE, momentum=MOMEMTUM,output_size=NUM_CLASSES):
		super(NNC6FC3, self).__init__()
		self.l_interval = l_interval
		self.l_rate, self.stride = l_rate,stride
		self.num_epoch, self.kernel = num_epoch,kernel
	
		self.conv1 = nn.Conv2d(1,2,kernel,stride)
		self.conv2 = nn.Conv2d(2,4,kernel,stride)
		self.conv3 = nn.Conv2d(4,8, kernel, stride)
		self.conv4 = nn.Conv2d(8,16,kernel,stride)
		self.conv5 = nn.Conv2d(16,64,kernel,stride)
		self.conv6 = nn.Conv2d(64,256,kernel,stride)
		self.fc1 = nn.Linear(256*3*2,INTER_FC2)
		self.fc2 = nn.Linear(INTER_FC2, INTER_FC4)
		self.fc3 = nn.Linear(INTER_FC4, NUM_CLASSES)
		self.batch2 = nn.BatchNorm1d(output_size)

	def forward(self,x):
		# print("shape coming in is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv1(x)), self.kernel)
		# print("shape after round 1 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv2(x)), self.kernel)
		# print("shape after round 2 is "+str(x.shape))
		x = F.relu(self.conv3(x))
		# print("shape after round 3 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv4(x)), self.kernel)
		# print("shape after round 4 is "+str(x.shape))
		x = F.relu(self.conv5(x))
		# print("shape after round 5 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv6(x)), self.kernel)
		# print("shape after round 6 is "+str(x.shape))
		x = x.view(-1, self.flatten_features(x))
		# print("shape after round 7 view is "+str(x.shape))
		x = F.relu(self.fc1(x))
		# print("shape after round 8 linear 1 is "+str(x.shape))
		x = F.relu(self.fc2(x))
		# print("shape after round 9 linear 2 is "+str(x.shape))
		x = self.batch2(self.fc3(x))
		# print("shape after round 10 linear 3 is "+str(x.shape))
		return F.log_softmax(x,dim=1)

	def flatten_features(self, x):
		size = x.size()[1:]  # all dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class NNC7FC2(nn.Module):
	def __init__(self,l_rate=L_RATE,batch=BATCH_SIZE,l_interval=LOG_INTERVAL,
		num_epoch=NUM_EPOCH, kernel=KERNEL,stride=STRIDE, momentum=MOMEMTUM,output_size=NUM_CLASSES):
		super(NNC7FC2, self).__init__()
		self.l_interval = l_interval
		self.l_rate, self.stride = l_rate,stride
		self.num_epoch, self.kernel = num_epoch,kernel
	
		self.conv1 = nn.Conv2d(1,2,kernel,stride)
		self.conv2 = nn.Conv2d(2,4,kernel,stride)
		self.conv3 = nn.Conv2d(4,8, kernel, stride)
		self.conv4 = nn.Conv2d(8,16,kernel,stride)
		self.conv5 = nn.Conv2d(16,32,kernel,stride)
		self.conv6 = nn.Conv2d(32,128,kernel,stride)
		self.conv7 = nn.Conv2d(128,256,kernel,stride)
		self.fc1 = nn.Linear(256*2*2,INTER_FC)
		self.fc2 = nn.Linear(INTER_FC, INTER_FC2)
		self.fc3 = nn.Linear(INTER_FC2, INTER_FC3)
		self.fc4 = nn.Linear(INTER_FC3, NUM_CLASSES)
		self.batch2 = nn.BatchNorm1d(output_size)

	def forward(self,x):
		# print("shape coming in is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv1(x)), self.kernel)
		# print("shape after round 1 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv2(x)), self.kernel)
		# print("shape after round 2 is "+str(x.shape))
		x = F.relu(self.conv3(x))
		# print("shape after round 3 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv4(x)), self.kernel)
		# print("shape after round 4 is "+str(x.shape))
		x = F.relu(self.conv5(x))
		# print("shape after round 5 is "+str(x.shape))
		x = F.relu(self.conv6(x))
		# print("shape after round 6 is "+str(x.shape))
		x = F.max_pool2d(F.relu(self.conv7(x)), self.kernel)
		# print("shape after round 7 is "+str(x.shape))
		x = x.view(-1, self.flatten_features(x))
		# print("shape after round 8 view is "+str(x.shape))
		x = F.relu(self.fc1(x))
		# print("shape after round 9 linear 1 is "+str(x.shape))
		x = F.relu(self.fc2(x))
		# print("shape after round 10 linear 2 is "+str(x.shape))
		x = F.relu(self.fc3(x))
		# print("shape after round 11 linear 3 is "+str(x.shape))
		x = self.batch2(self.fc4(x))
		# print("shape after round 12 linear 4 is "+str(x.shape))
		return F.log_softmax(x,dim=1)

	def flatten_features(self, x):
		size = x.size()[1:]  # all dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class NNC8FC4(nn.Module):
	def __init__(self,l_rate=L_RATE,batch=BATCH_SIZE,l_interval=LOG_INTERVAL,
		num_epoch=NUM_EPOCH, kernel=KERNEL,stride=STRIDE, momentum=MOMEMTUM,output_size=NUM_CLASSES):
		super(NNC8FC4, self).__init__()
		self.l_interval = l_interval
		self.l_rate, self.stride = l_rate,stride
		self.num_epoch, self.kernel = num_epoch,kernel
	
		self.conv1 = nn.Conv2d(1,2,kernel,stride)
		self.conv2 = nn.Conv2d(2,4,kernel,stride)
		self.conv3 = nn.Conv2d(4,8, kernel,stride)	
		self.conv4 = nn.Conv2d(8,16, kernel,stride)
		self.conv5 = nn.Conv2d(16,32,kernel,stride)
		self.conv6 = nn.Conv2d(32,64,kernel,stride)
		self.conv7 = nn.Conv2d(64,128,kernel,stride)
		self.conv8 = nn.Conv2d(128,256,kernel,stride)
		self.fc1 = nn.Linear(256*7*6,INTER_FC)
		self.fc2 = nn.Linear(INTER_FC, INTER_FC2)
		self.fc3 = nn.Linear(INTER_FC2, INTER_FC3)
		self.fc4 = nn.Linear(INTER_FC3, NUM_CLASSES)
		self.batch2 = nn.BatchNorm1d(output_size)

	def forward(self,x):
		# print("shape coming in is "+str(x.shape))
		x = F.relu(self.conv1(x))
		# print("shape after round 1 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv2(x)), self.kernel)
		# print("shape after round 2 is "+str(x.shape))
		x = F.relu(self.conv3(x))
		# print("shape after round 3 is "+str(x.shape))
		x = F.relu(self.conv4(x))
		# print("shape after round 4 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv5(x)), self.kernel)
		# print("shape after round 5 is "+str(x.shape))
		x = F.relu(self.conv6(x))
		# print("shape after round 6 is "+str(x.shape))
		x = F.relu(self.conv7(x))
		# print("shape after round 7 is "+ str(x.shape))
		x = F.max_pool2d(F.relu(self.conv8(x)), self.kernel)
		# print("shape after round 8 is "+str(x.shape))
		x = x.view(-1, self.flatten_features(x))
		# print("shape after round 9 view is "+str(x.shape))
		x = F.relu(self.fc1(x))
		# print("shape after round 10 linear 1 is "+str(x.shape))
		x = F.relu(self.fc2(x))
		# print("shape after round 11 linear 2 is "+str(x.shape))
		x = F.relu(self.fc3(x))
		# print("shape after round 12 linear 3 is "+str(x.shape))
		x = self.batch2(self.fc4(x))
		# print("shape after round 13 linear 4 is "+str(x.shape))
		return F.log_softmax(x,dim=1)

	def flatten_features(self, x):
		size = x.size()[1:]  # all dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


class Chin_Char_Dataset(Dataset):
	def __init__(self, data_dict, data_lab):
		self.data_dict = data_dict
		self.data_lab = data_lab
		# self.transforms = transforms.ToTensor()
		assert(type(data_dict)==dict and type(data_lab)==list)
	def __len__(self):
		assert(len(self.data_lab)==len(self.data_dict))
		return len(self.data_lab)
	def __getitem__(self,index):
		jpg_image = self.data_lab[index]
		jpg_label = self.data_dict[jpg_image]
		new_array = np.asarray(Image.open(jpg_image).getdata()).reshape(1,81,70)
		normed_array = (new_array-new_array.min())/(new_array.max()-new_array.min())
		return normed_array, jpg_label


# 1) CV to fine tune hyperparams? Test & document accuracy...
# 2) Determine why each output layer is the exact same!!

# [BATCH, NUM_INPUT_CHANNELS, HEIGHT, WIDTH] (4D Conv2D input)


def load_dict(chin_char_path, args):
	partition,jpgs = {},[] #dictionary / .jpg path as key, class as value...
	chin_classes = os.listdir(chin_char_path)
	print(chin_classes)
	for chin_index,chin_class in enumerate(chin_classes):
		if(chin_index%args.l_interval==0): 
			print("chin_index=={}".format(chin_index))
		if(chin_index==args.num_classes): break
		jpg_path = JOIN(chin_char_path,chin_class)
		for image in os.listdir(jpg_path):
			image_path = JOIN(jpg_path,image)
			partition[image_path]=chin_class
			jpgs.append(image_path)	
	return partition,jpgs


def parser_func():
	parser = argparse.ArgumentParser(description="""Argument parser Chinese Character classfiction:\n\n
		NOTE: You are welcome to change macro hyperparameters above.""")
	parser.add_argument('--batch_size',type=int, default=BATCH_SIZE, help='input denoting number of batches')
	parser.add_argument('--epochs', type=int, default=NUM_EPOCH, help='denotes number of overall rounds')
	parser.add_argument('--l_rate', type=float,default=L_RATE, help='determine GD learning rate')
	parser.add_argument('--l_interval',type=int, default=LOG_INTERVAL, help="determine batch frequency for logging (printing)")
	parser.add_argument('--cv_flag', type=bool, default=False, help="denotes if we are testing wrt cv (hyperparameter tuning) or test set")
	parser.add_argument('--decay', type=float, default=DECAY, help="denotes the decay of learning rate (type 1.0 if you want no decay)")
	parser.add_argument('--decay_freq', type=int,default=DECAY_FREQ,help="denotes frequency in epochs by which we multiply decay constant")
	parser.add_argument('--momentum', type=float,default=MOMEMTUM,help="denotes momentum of CNN classifier")
	parser.add_argument('--num_classes',type=int,default=NUM_CLASSES,help="denotes number of classes needed for examination")
	parser.add_argument('--kernel',type=int,default=KERNEL,help="denotes kernel size for maxpool")
	parser.add_argument('--stride',type=int,default=STRIDE,help="denote stride of maxpool")
	return parser.parse_args()

def index_encode(char):
	return CLASSES.index(char)

def train_batch(model,optimizer,device,train_loader,epoch,args):
	model.train()
	for p_group in optimizer.param_groups:
		p_group['lr'] = args.l_rate * (args.decay**(epoch//args.decay_freq))
	for batch_index, (image,chin_char) in enumerate(train_loader):
		print("Epoch: {}; Batch Index: {}".format(epoch+1, batch_index))
		chin_char = tensor([index_encode(char) for char in chin_char])
		image, chin_char = image.to(device), chin_char.to(device)
		output = model(image.type('torch.DoubleTensor'))
		optimizer.zero_grad()
		loss = F.nll_loss(output, chin_char)
		loss.backward()
		optimizer.step()
		if batch_index%args.l_interval==0:
			print("\tTrain Epoch: {}\n\tBatch Index: {}\n\tData Count: {} \n\tLoss Value: {:3f}\n ".
				format(epoch+1, batch_index, batch_index*len(image), loss.item()))
	return 
		
def cv_test_batch(model, epoch, device, cv_test_loader, args):
	model.eval()
	test_loss,correct,batch_total = 0,0,0
	with torch.no_grad():
		for batch_index, (image,chin_char) in enumerate(cv_test_loader):
			print("Epoch: {}; Batch Index: {}".format(epoch+1,batch_index))
			chin_char = tensor([index_encode(char) for char in chin_char])
			image, chin_char = image.to(device), chin_char.to(device)
			output = model(image.type('torch.DoubleTensor'))
			_, pred = torch.max(output, 1)
			test_loss += F.nll_loss(output, chin_char, reduction='sum').item()
			correct += pred.eq(chin_char.view_as(pred)).sum().item()
			batch_total+= args.batch_size
	print("Correct: {}".format(correct))
	print("\tAverage Loss: {}\nAccuracy:{}\n".format(test_loss/len(cv_test_loader), float(correct)/batch_total))
	return 100*(1.0-float(correct)/batch_total) # denotes the average error for a particular epoch

def _Data_Loader(batch_type, args):
	if(batch_type=="train"): chin_char_path = JOIN(os.getcwd(),'chin_char_trn_preproc')
	elif(batch_type=="cv"): chin_char_path = JOIN(os.getcwd(),'chin_char_tst_preproc')
	elif(batch_type=="test"): chin_char_path = JOIN(os.getcwd(),'chin_char_tst_preproc')
	else: 
		print("invalid batch_type")
		return None
	_dict,_labs = load_dict(chin_char_path, args)
	with open("{}_dict_mini.txt".format(batch_type), "w", encoding="utf-8") as fdict: fdict.write(str(_dict))
	with open("{}_labs_mini.txt".format(batch_type), "w", encoding="utf-8") as flabs: flabs.write(str(_labs))
	_dataset = Chin_Char_Dataset(_dict,_labs)
	return DataLoader(dataset=_dataset, batch_size=args.batch_size, shuffle=True)

def model_initializer(device):
	m_c3fc2 = NNC3FC2().to(device)
	m_c4fc2 = NNC4FC2().to(device)
	m_c5fc3 = NNC5FC3().to(device)
	m_c6fc3 = NNC6FC3().to(device)
	m_c7fc4 = NNC7FC2().to(device)
	m_c8fc4 = NNC8FC4().to(device)
	return ({0:'m_c3fc2',1:'m_c4fc2', 2:'m_c5fc3', 3:'m_c6fc3', 4:'m_c7fc2', 5:'m_c8fc4'},
			[m_c3fc2, m_c4fc2, m_c5fc3, m_c6fc3, m_c7fc4, m_c8fc4])

#do the actual plots save save them
def do_plots(error_list,m_index,ind_name):
	epoch = [epoch for epoch in range(1,NUM_EPOCH+1)]
	plt.plot(epoch,error_list)
	plt.title("Error Distribution for {}".format(ind_name[m_index][2:].upper()))
	plt.ylabel("Error percentage %")
	plt.xlabel("Epoch Number")
	plt.savefig(JOIN(os.getcwd(),JOIN("torch_cnn_data","{}_plot.png".format(ind_name[m_index]))))
	plt.figure()
	return

def main_shell():
	args = parser_func()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #lets see how cpu plays out...
	ind_name,models = model_initializer(device)
	train_loader = _Data_Loader("train", args)
	if(args.cv_flag): cv_loader  = _Data_Loader("cv", args)
	else: test_loader  = _Data_Loader("test", args)
	for m_index,model in enumerate(models):
		optimizer = optim.SGD(model.parameters(),lr=args.l_rate,momentum=MOMEMTUM)
		error_list = []
		print("working model {}".format(ind_name[m_index]),end='\n\n')
		for epoch in range(args.epochs):
			train_batch(model, optimizer, device, train_loader, epoch, args)
			if(args.cv_flag): incorrect = cv_test_batch(model, epoch, device, cv_loader, args)
			else: incorrect = cv_test_batch(model, epoch, device, test_loader, args)
			error_list.append(incorrect)
		try: os.mkdir("torch_cnn_data")
		except: print("directory present - moving on...") 
		do_plots(error_list,m_index,ind_name)
		torch.save(model.state_dict(), JOIN(os.getcwd(),JOIN("torch_cnn_data",'{}_mini.dat'.format(ind_name[m_index]))))

main_shell()
## NOTE, you can toggle with capital hyperparameters above
