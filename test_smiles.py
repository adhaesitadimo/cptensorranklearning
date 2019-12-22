import numpy as np
import tensorflow as tf 
import sklearn
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.linalg import norm
import os
import time

torch.manual_seed(69)
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

class RankLearnNet(nn.Module):
    def __init__(self, tensor, flat_shape):
        super(RankLearnNet, self).__init__()
        print(type(tensor))
        self.conv1 = nn.Conv3d(1, 32, (3, 3, 1), padding = (1,1,0))
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 1), padding = (1,1,0))
        self.pool1 = nn.AvgPool3d((2, 2, 1), padding=(1,1,0))
        self.conv3 = nn.Conv3d(32, 64, (3, 3, 1), padding = (1,1,0))
        self.conv4 = nn.Conv3d(64, 64, (3, 3, 1), padding = (1,1,0))
        self.pool2 = nn.AvgPool3d((2, 2, 3), padding=(1,1,0))
        self.fc1 = nn.Linear(flat_shape, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, inp):
        #print(inp.shape)
        inp = self.conv1(inp)
        inp = self.relu(inp)
        #print(inp.shape)
        inp = self.conv2(inp)
        #print(inp.shape)
        inp = self.relu(inp)
        inp = self.pool1(inp)
        #print(inp.shape)
        
        inp = self.conv3(inp)
        inp = self.relu(inp)
        #print(inp.shape)
        inp = self.conv4(inp)
        #print(inp.shape)
        inp = self.relu(inp)
        inp = self.pool1(inp)
        #print(inp.shape)

        inp = self.conv4(inp)
        inp = self.relu(inp)
        #print(inp.shape)
        inp = self.conv4(inp)
        inp = self.relu(inp)
        #print(inp.shape)
        inp = self.pool2(inp)
        #print(inp.shape)

        inp = torch.flatten(inp, start_dim = 1)
        #print(inp.shape)
        inp = self.fc1(inp)
        #print(inp.shape)
        inp = self.relu(inp)
        inp = self.dropout(inp)
        inp = self.fc2(inp)
        #print(inp.shape)
        inp = self.relu(inp)
        inp = self.dropout(inp)
        inp = self.fc3(inp)
        #print(inp.shape)
        return inp

model = torch.load('NN180noise.model')
def outer_product(x,y,z):
	I = [len(x),len(y),len(z)]
	res = np.zeros((I[0],I[1],I[2]))
	for i in range(I[0]):
		for j in range(I[1]):
			for k in range(I[2]):
				res[i,j,k] = x[i]*y[j]*z[k]
	return res

def KRON(A,B):
	I = A.shape[0]
	J = A.shape[1]
	K = B.shape[0]
	L = B.shape[1]
	C = np.zeros((I*K,J*L))
	for i in range(I):
		for j in range(J):
			C[i*K:(i+1)*K,j*L:(j+1)*L] = A[i][j]*B
	return C

def KRAO(A,B):
	I = A.shape[0]
	K = A.shape[1]
	J = B.shape[0]
	C = np.zeros((I*J,K))
	for i in range(I):
		for k in range(K):
			C[i*(J):(i+1)*(J),k] = A[i,k]*B[:,k]
	return C  

def HAD(A,B):
	return A*B

def mat_tens(X,n):
	x = X.shape[0]
	y = X.shape[1]
	z = X.shape[2]
	if n==0:
		res = np.zeros((x,y*z))
		for i in range(y):
			for j in range(z):
				res[:,j*y+i] = X[:,i,j]
	if n==1:
		res = np.zeros((y,x*z))
		for i in range(x):
			for j in range(z):
				res[:,j*x+i] = X[i,:,j]

	if n==2:
		res = np.zeros((z,x*y))
		for i in range(x):
			for j in range(y):
				res[:,j*x+i] = X[i,j,:]
	return res

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

# Plot the image here using matplotlib.
def plot_image(tensor, name='smile_orig.png'):
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.savefig(name)



def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def ALS_CP(X,r = 70,iter=100,eps=4e-3,pad=True):
	A = np.random.randn(X.shape[0],r)
	B = np.random.randn(X.shape[1],r)
	C = np.random.randn(X.shape[2],r)
	A_prev = sklearn.preprocessing.normalize(A,axis=0)
	B_prev = sklearn.preprocessing.normalize(B,axis=0)
	C_prev = sklearn.preprocessing.normalize(C,axis=0)
	err = 100
	X_0 = mat_tens(X,0)
	X_1 = mat_tens(X,1)
	X_2 = mat_tens(X,2)
	errs = []
	for i in tqdm(range(iter)):
		A = X_0.dot(KRAO(C,B)).dot(np.linalg.pinv(HAD(C.T.dot(C),B.T.dot(B))))
		B = X_1.dot(KRAO(C,A)).dot(np.linalg.pinv(HAD(C.T.dot(C),A.T.dot(A))))
		C = X_2.dot(KRAO(B,A)).dot(np.linalg.pinv(HAD(B.T.dot(B),A.T.dot(A))))  
		#err = torch.norm(torch.tensor(X) - torch.tensor(A), p='fro')
		err = np.linalg.norm(sklearn.preprocessing.normalize(A,axis=0)-A_prev,ord='fro')
		errs.append(err)
		#print(err)
		#err += np.linalg.norm(B-B_new,ord='fro')
		if err < eps:
			break
		A = sklearn.preprocessing.normalize(A,axis=0)
		B = sklearn.preprocessing.normalize(B,axis=0)
		C = sklearn.preprocessing.normalize(C,axis=0)
		A_prev = A
		B_prev = B
		C_prev = C
	np.save('errs.np', np.array(errs))
	#print(A.shape)
	#print(B.shape)
	#print(C.shape)
	#A = pad_along_axis(A, C.shape[0], axis=0)

	#res = np.stack([A,B,C], axis=2)
	if pad:
		C = pad_along_axis(C, A.shape[0], axis=0)
		res = np.stack([A,B,C], axis=2)
		return res
	else:
		return A, B, C

# Read the image from file. Assuming it is in the same directory.
errs = []
times = []
ranks = []
for item in os.listdir('/home/student/cpals/smiles'):
	print(item)
	pil_image = Image.open('/home/student/cpals/smiles/' + item)
	rgb_image = pil2tensor(pil_image)

	print(rgb_image.shape)

	plot_image(rgb_image, '/home/student/cpals/smiles_res/' + item[:-3] + '_orig.png')

	res = ALS_CP(rgb_image.view(60,60,3), r=180, iter=500)

	res = torch.FloatTensor(res).view(1, 1, 60, 180, 3)
	#print(res)
	rank = int(model(res.to(device)).item())
	print(rank)

	ranks.append(rank)

	img_1, img_2, img_3 = ALS_CP(rgb_image.view(60,60,3), r=rank, iter=500, pad=False)
	#print(img_1.shape)
	#print(img_2.shape)
	#print(img_3.shape)
	sum_ = 0
	start = time.time()
	for i in tqdm(range(rank)):
		sum_ += outer_product(img_1[:,i],img_2[:,i],img_3[:,i])

	#print(sum_.shape)
	end = time.time() - start
	err = torch.norm(rgb_image.view(60,60,3) - torch.FloatTensor(sum_), p='fro') 
	print(err)
	errs.append(err.item())
	times.append(end)

	plot_image(torch.FloatTensor(sum_).view(3,60,60), '/home/student/cpals/smiles_res/' + item[:-3] + '_pred.png')

print('Mean error by 10 smiles:', np.mean(errs))
print('Mean time by 10 smiles (in s):', np.mean(times))
print('Mean rank by 10 smiles:', np.mean(ranks))
