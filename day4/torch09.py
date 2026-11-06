# 다중 분류 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch import tensor
from torch.nn import Sequential, Linear, ReLU, Sigmoid, CrossEntropyLoss, Softmax

def make_DataSet(x_train, x_val, y_train, y_val, batch_size = 32):
    x_train_tensor = tensor(data=x_train, dtype=torch.float)
    y_train_tensor = tensor(data=y_train, dtype=torch.long).reshape(-1, 1)
    x_val_tensor = tensor(data=x_val, dtype=torch.float)
    y_val_tensor = tensor(data=y_val, dtype=torch.long).reshape(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_DataLoader, x_val_tensor, y_val_tensor

# 핵심: 미니배치 단위로 학습
def train(device, model, train_DataLoader, loss_fn, optimizer):
    num_batches = len(train_DataLoader)
    train_loss = 0
    
    model.train()
    for batch_idx, (x, y) in enumerate(train_DataLoader):
        x = x.to(device)
        y = y.to(device)
        
        #순전파
        pred = model(x)
        loss = loss_fn(pred, y)    
        train_loss += loss
        print(type(train_loss))
        
        #역전파 
        loss.backward() # loss를 통한 기울기 계산
        optimizer.step() # 해당 기울기를 가지고 가중치 조정
        optimizer.zero_grad() # 가중치 초기화. 초기화 하지 않으면, 계속 쌓임.
    
    train_loss /= num_batches
    return train_loss.item()

def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    model.eval()
    
    x = x_val_tensor.to(device)
    y = y_val_tensor.to(device)
    
    with torch.no_grad(): # 기울기 계산 종료
        pred = model(x)
        eval_loss = loss_fn(pred, y)
        
        print("pred 의 타입: ", type(pred))
        print("eval_loss 의 타입: ", type(eval_loss))
        
    return pred, eval_loss.item()

