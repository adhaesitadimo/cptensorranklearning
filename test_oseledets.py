import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import tensorflow as tf 
import sklearn
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import time
from scipy.linalg import norm

# pytorch provides a function to convert PIL images to tensors.
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

# Read the image from file. Assuming it is in the same directory.
pil_image = Image.open('oseledets.png')
rgb_image = pil2tensor(pil_image)

print(rgb_image.shape)

# Plot the image here using matplotlib.
def plot_image(tensor, name='orig.png'):
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.savefig(name)

plot_image(rgb_image)

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


def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b


def ALS_CP(X,r = 70,iter=100,eps=1e-2):
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
	for i in tqdm(range(iter)):
		A = X_0.dot(KRAO(C,B)).dot(np.linalg.pinv(HAD(C.T.dot(C),B.T.dot(B))))
		B = X_1.dot(KRAO(C,A)).dot(np.linalg.pinv(HAD(C.T.dot(C),A.T.dot(A))))
		C = X_2.dot(KRAO(B,A)).dot(np.linalg.pinv(HAD(B.T.dot(B),A.T.dot(A))))  
		#err = torch.norm(torch.tensor(X) - torch.tensor(A), p='fro')
		err = np.linalg.norm(sklearn.preprocessing.normalize(A,axis=0)-A_prev,ord='fro')
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
	#np.save('errs.np', np.array(errs))
	print(A.shape)
	print(B.shape)
	print(C.shape)
	#A = pad_along_axis(A, C.shape[0], axis=0)

	#res = np.stack([A,B,C], axis=2)

	return A,B,C

errs = []
times = []
for rank in tqdm(range(1, 100)):
	start = time.time()
	res1, res2, res3 = ALS_CP(rgb_image, r=rank, iter=300)
	sum_ = 0
	for i in tqdm(range(rank)):
		sum_ += outer_product(res1[:,i],res2[:,i],res3[:,i])
	end = time.time() - start
	times.append(end)
	err = torch.norm(rgb_image - torch.tensor(sum_), p='fro') 
	print(err)
	errs.append(err.item())

	plot_image(torch.FloatTensor(sum_), 'tens' + str(rank) + '.png')

np.save('errs.np', np.array(errs))
np.save('times.np', np.array(times))
