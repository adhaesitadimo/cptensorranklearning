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

torch.manual_seed(69)
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target)) 

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


def train(tensors_train, model, target):
    epochs = 25
    learning_rate = 1e-4
    batch_size = 25

    criterion = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    model.train()
    samples = tensors_train.shape[0]

    mae_history = []
    mse_history = []
    mape_history = []


    for epoch in range(epochs):
        batch_ids = batch_size
        print('EPOCH {} STARTED'.format(epoch + 1))
        losses = []
        mse_losses = []
        mape_losses = []
        for batch in tqdm(range(int(samples / batch_size))):
            train_batch = tensors_train[batch_ids - batch_size:batch_ids].to(device)
            target_batch = target[batch_ids - batch_size:batch_ids].to(device)
            
            optimizer.zero_grad()
            output = model(train_batch)
            #print(output)
            #print(target_batch)
            loss = criterion(output, target_batch)
            mse_loss = mse(output, target_batch)
            mape_loss = MAPELoss(output, target_batch)
            #print(mape_loss.item())
            mse_losses.append(np.sqrt(mse_loss.item()))
            mape_losses.append(mape_loss.item()) 
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            batch_ids += batch_size

        print('MAE loss in epoch {}: {}'.format(epoch + 1, np.mean(losses)))
        print('MAPE loss in epoch {}: {}'.format(epoch + 1, np.mean(mape_losses)))
        print('MSE loss in epoch {}: {}'.format(epoch + 1, np.mean(mse_losses)))
        mae_history.append(np.mean(losses))
        mape_history.append(np.mean(mape_losses))
        mse_history.append(np.mean(mse_losses))

    np.save('NN150noiseesigma01_maeloss.npy', mae_history)
    np.save('NN150noiseesigma01_mapeloss.npy', mape_history)
    np.save('NN150noiseesigma01_mseloss.npy', mse_history)


def test(tensors_test, model, targets):
    with torch.no_grad():
        output = model(tensors_test.to(device))
    print(output)
    print(targets)
    print('MAE on test: {}'.format(mean_absolute_error(output, targets)))
    print('MAE on test: {}'.format(mean_squared_error(output, targets)))


predecomposed1 = torch.load('predecomposed60603150normnoisesigma01.pt')
ranks = torch.load('tens60603normranksnoisesigma01.pt')
print(predecomposed1.shape)
tens = torch.FloatTensor(predecomposed1).view(2500, 1, 60, 150, 3)
tensors_train = tens
ranks_train = ranks
print(tensors_train.shape)
tensors_test = tens[2400:]
ranks_test = ranks[2400:]

model = RankLearnNet(tensors_train, 11520)
model.to(device)
train(tensors_train, model, torch.FloatTensor(ranks_train))

torch.save(model, 'NN150noiseesigma01.model')
test(tensors_test, model, torch.FloatTensor(ranks_test))