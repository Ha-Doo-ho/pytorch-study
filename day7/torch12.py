import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import torch
from torch import tensor
from torch.nn import Sequential, Dropout, Linear, Conv2d, MaxPool1d, ReLU
from torch.utils.data import TensorDataset , DataLoader
from torchsummary import summary 
from torch.cuda import is_available
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


if is_available():
    device = "cuda"
else:
    device = "cpu"

def makeDataSet(x_train, x_val, y_train, y_val, batch_size=32):
    x_train_tensor = tensor(data=x_train, dtype=torch.float)
    x_val_tensor = tensor(data=x_val, dtype=torch.float)     
    
    y_train_tensor = tensor(data=y_train, dtype=torch.long) # 다중분류이므로 dtype은 무조건 torch.long이어야 한다.
    y_val_tensor = tensor(data=y_val, dtype=torch.long)
    
    train_Dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(train_Dataset)
    
    return train_DataLoader, x_val_tensor, y_val_tensor

# 미니배치 단위로 학습함.
def train(device, model, train_DataLoader, loss_fn, optimizer):
    num_batches = len(DataLoader)
    train_loss = 0
    
    for batch_idx, (x,y) in enumerate(DataLoader):
        x = x.to(device)
        y = y.to(device)
        
        # 순전파 FeedForward
        model.train()
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss 
        
        # 역전파 Backpropagation
        loss.backward() # loss 계산한 것을 바탕으로 기울기 계산 시작
        optimizer.step() # 계산된 기울기를 바탕으로 가중치 변경
        optimizer.zero_grad()
        
    train_loss /= num_batches
    return train_loss.item()

def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    with torch.no_grad():
        x = x_val_tensor.to()
        y = y_val_tensor.to()
        
        pred = model(x)
        eval_loss = loss_fn(pred, y)
        
        return pred, eval_loss.item()
    
def dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list):

    epochs = list(range(1, len(tr_loss_list)+1))
    plt.plot(epochs, tr_loss_list, label='train_err', marker = '.')
    plt.plot(epochs, val_loss_list, label='val_err', marker = '.')
    plt.plot(epochs, val_acc_list, label='val_acc', marker = '.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

# 1 데이터    
train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

# 1-2 가변수 처리 pass

# 1-3 데이터셋 분리
x_val, x_test = test_dataset.data[:5000], test_dataset.data[5000:]
y_val, y_test = test_dataset.targets[:5000], test_dataset.targets[5000:]

# 1-4 스케일링
x_val = x_val / 255
x_test = x_test / 255

print(x_val.shape, type(x_val))
exit()
x_val = x_val.reshape(-1, )

# 
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)