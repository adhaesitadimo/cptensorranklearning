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
from scipy.linalg import norm


#np.random.seed(69)

#def generate_tensors(shape=(20,20,20)):
#    res = np.empty(shape=shape)
    
#    def generate_one_tensor():
#        for z in range(shape[2]):
#            v1, v2 = np.random.rand(shape[0]), np.random.rand(shape[1])
#            res[:,:,z] = np.outer(v1, v2)
#        return res
    
#    ranks = np.repeat(np.arange(20,51), 70)
#    np.random.shuffle(ranks)
#    print(ranks)
#    return np.array([sum([generate_one_tensor() for r in range(rank)]) for rank in ranks]), ranks

#tens, ranks = generate_tensors()

np.random.seed(69)

def generate_tensors(shape=(60,60,3), sigma=0.004):
    res = np.empty(shape=shape)
    ranks = list(map(int,np.random.normal(100, 10, 2500)))
    print(np.max(ranks), np.min(ranks))
    #print(ranks)
    #ranks = np.repeat(np.arange(20,51), 70)
    #np.random.shuffle(ranks)

    cov = np.diag(np.array([sigma] * 2500))
    gauss_noise = np.random.multivariate_normal(np.zeros(2500), cov, shape)
    print(gauss_noise.shape)

    def generate_one_tensor(index):
        res = np.empty(shape=shape)
        for z in range(shape[2]):
            v1, v2 = np.random.rand(shape[0]), np.random.rand(shape[1])
            res[:,:,z] = np.outer(v1, v2)
            #print(res.shape)
            #print(gauss_noise.shape)
            #print(gauss_noise[:,:,:,index])
            res += gauss_noise[:,:,:,index]
        #print(res)
        return res
    
    return np.array([sum([generate_one_tensor(i) for r in range(rank)]) 
    	for i, rank in tqdm(enumerate(ranks))]), ranks

tens, ranks = generate_tensors()

torch.save(torch.FloatTensor(tens), 'tens60603normnoisesigma01.pt')
torch.save(torch.FloatTensor(ranks), 'tens60603normranksnoisesigma01.pt')

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


def ALS_CP(X,r = 70,iter=100,eps=5e-3):
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
	for i in range(iter):
		A = X_0.dot(KRAO(C,B)).dot(np.linalg.pinv(HAD(C.T.dot(C),B.T.dot(B))))
		B = X_1.dot(KRAO(C,A)).dot(np.linalg.pinv(HAD(C.T.dot(C),A.T.dot(A))))
		C = X_2.dot(KRAO(B,A)).dot(np.linalg.pinv(HAD(B.T.dot(B),A.T.dot(A))))  
		err = np.linalg.norm(sklearn.preprocessing.normalize(A,axis=0)-A_prev,ord='fro')
		#err += np.linalg.norm(B-B_new,ord='fro')
		#print(errs)
		if err<eps:
			break
		A = sklearn.preprocessing.normalize(A,axis=0)
		B = sklearn.preprocessing.normalize(B,axis=0)
		C = sklearn.preprocessing.normalize(C,axis=0)
		A_prev = A
		B_prev = B
		C_prev = C


	C = pad_along_axis(C, A.shape[0], axis=0)

	res = np.stack([A,B,C], axis=2)

	return res

print(tens.shape)

#predecomposed1 = []

#for ten in tqdm(tens):
#	res = ALS_CP(ten, r=60, iter=20)
#	predecomposed1.append(res)

#torch.save(torch.FloatTensor(predecomposed1), 'predecomposed60normnoisesigma01.pt')

'''
predecomposed1 = []

errs = []
for iteration 
for ten in tqdm(tens[:20]):
	res = ALS_CP(ten, r=60, iter=1)
	sum_ = 0
	for i in range(60):
		sum_ += outer_product(res[:,i,0],res[:,i,1],res[:,i,2]) 
	#err = torch.norm(torch.tensor(ten) - torch.tensor(sum_[:,:,:3]), p='fro')
	err = torch.norm(torch.tensor(ten) - torch.tensor(sum_), p='fro')
	errs.append(err)
	predecomposed1.append(res)
print(np.mean(errs))
'''


#torch.save(torch.FloatTensor(predecomposed1), 'predecomposed6060380normnoisesigma01.pt')

predecomposed1 = []

for ten in tqdm(tens):
	res = ALS_CP(ten, r=180, iter=200)
	predecomposed1.append(res)

torch.save(torch.FloatTensor(predecomposed1), 'predecomposed60603180normnoisesigma01.pt')