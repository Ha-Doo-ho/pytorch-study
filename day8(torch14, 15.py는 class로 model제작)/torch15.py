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
from torch.nn import Module, Sequential, Dropout, Linear, Conv2d, MaxPool2d, ReLU, Flatten, BCELoss, BatchNorm2d, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.utils.data import TensorDataset , DataLoader
from torchsummary import summary 
from torch.cuda import is_available
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor # 이미지 → 파이토치 텐서 로 변경할 때 사용한다. 

# 0 디바이스 결정
if is_available():
    device = "cuda"
else:
    device = "cpu"

# 1 데이터 
# 데이터들을 텐서로 만들고 훈련 데이터는 미니배치로 제작하기 위해 데이터로더로 만든다.
def makeDataSet(x_train, y_train, x_val, y_val, batch_size=32):
    x_train_tensor = tensor(data=x_train, dtype=torch.float).permute(0, 3, 1 ,2) / 255 
    x_val_tensor = tensor(data=x_val, dtype=torch.float).permute(0, 3, 1, 2) / 255 
    
    y_train_tensor = tensor(data=y_train, dtype=torch.long)
    y_val_tensor = tensor(data=y_val, dtype=torch.long)
    
    train_tensor = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    
    return train_DataLoader, x_val_tensor, y_val_tensor
    
# 2 모델 생성
# 클래스 방식으로는 다음과 같이 작성하며 보통 클래스 방식으로 많이들 작성한다.
""" 
현재 방식이 가장 많이 보는 방식으로 진행된다. 
"""

class CustomCNN(Module): # JAVA의 class CustomCNN extends Module { }와 동일함. 
    def __init__(self, n_class): # JAVA에서의 public customCNN(n_class) { this.n_class = n_class } 와 동일함.
                                 # 마지막에 출력층이 몇개의 노드로 구성되어 있는지, 그거 확인하기 위한 용도이다. 
        super(CustomCNN, self).__init__() # 파이썬은 이걸 명시적으로 작성해야 한다. 자바는 그럴 필요 없음.
        
        """
        __init__()에서는 계층을 만들고, forward()에서 실제로 사용하는 방식
        
        """
        
        # 특징 추출 층
        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1,padding="same"),
            ReLU(),
            BatchNorm2d(num_features=8),
            MaxPool2d(kernel_size=2),
            
            Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=16),
            MaxPool2d(kernel_size=2),
            
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2)
        )
        
        # 판단 분석 및 최종 판단(분류)층 ---> 핵심은 가로/세로 크기 무조건 알아야 함. 그 크기를 바탕으로 ÷2 씩 쪼개서 내려오기 때문이다.
        self.classifier = Sequential(
            Flatten(),
            Linear(in_features=32 * 4 * 4, out_features=32),
            ReLU(),
            Dropout(p=0.3),
            Linear(in_features=32, out_features=16),
            ReLU(),
            
        #최종 판단(분류층)
            Linear(in_features=16, out_features=n_class)
        )
        
    def forward(self, x): # 모델에 데이터 넣을 때 그 데이터
        # 모델 설계
        x = self.features(x)
        x = self.classifier(x)
        return x 
        
        
# 3 컴파일 및 훈련
# 핵심: 미니배치 단위로 훈련함.
def train(device, model, train_DataLoader, loss_fn, optimizer):
    
    train_loss = 0
    num_batches = len(train_DataLoader)
    
    model.train()
    for batch_idx, (x,y) in enumerate(train_DataLoader):
        x = x.to(device)
        y = y.to(device)
        
        # FeedForward
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss 
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    train_loss /= num_batches
    train_loss = train_loss.item()
    print(f"{batch_idx+1}번 배치의 train_loss: {train_loss}")
    print("train_loss.item()의 타입", train_loss)
    return train_loss

# 4 평가 및 예측
def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    with torch.no_grad():
        model.eval()
        
        x = x_val_tensor.to(device)
        y = y_val_tensor.to(device)
        
        pred = model(x)
        val_loss = loss_fn(pred, y)
        
        val_loss = val_loss.item()
        print(f"pred의 타입: {type(pred)}, val_loss.item걸었을 때 타입: {type(val_loss)}")
        
        
        return pred, val_loss
    
# 1-1 데이터
"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in "10 classes" <-- 10개임 , with 6000 images per class. 
There are 50000 training images and 10000 test images. 
"""
train_dataset = datasets.CIFAR10(root="data", train=True, download=True) # 50000
test_dataset= datasets.CIFAR10(root="data", train=False, download=True)  # 10000

print("train_dataset.classes: ",train_dataset.classes) #['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 클래스 수가 10개임.

train_data = train_dataset.data
train_label = np.asarray(train_dataset.targets) # 파이썬의 리스트, 튜플 등을 numpy.ndarray로 만들어 줌. pandas에서 NumPy로 갈 때는 to_numpy또는 .values, TensorFlow / PyTorch의 형태에서 NumPy로 갈 때는 .numpy()

x_test = test_dataset.data
y_test = np.asarray(test_dataset.targets)

print(f"train_dataset.shape: {train_data.shape}, test_dataset.shape: {train_label.shape}")

# 1-2 가변수 처리 패스

# 1-3 데이터 분할 
x_train, x_val, y_train, y_val = train_test_split(train_data, train_label, train_size=0.8, random_state=11, shuffle=True, stratify=train_label)

# 1-4 스케일링
print(x_train.shape, x_test.shape) # (40000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) # (40000,) (10000,)

train_DataLoader, x_val_tensor, y_val_tensor = makeDataSet(x_train=x_train, y_train=y_train, x_val=x_val, y_val= y_val, batch_size=64)
x_test_tensor = tensor(data=x_test, dtype=torch.float).permute(0, 3, 1, 2) / 255 
y_test_tensor = tensor(data=y_test, dtype=torch.long)

print(x_val_tensor.shape)  # torch.Size([10000, 3, 32, 32])
print(x_test_tensor.shape) # torch.Size([10000, 3, 32, 32])

# 2 모델 구성
model = CustomCNN(n_class=10).to(device=device)

# 3 컴파일 및 훈련
loss_fn = CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=0.002)

epoch = 100
train_loss_list = []
val_loss_list = []
val_acc_list = []

counter = 0
patience = 5

for i in range(epoch):
    train_loss = train(device=device, model=model, train_DataLoader=train_DataLoader, loss_fn=loss_fn, optimizer=optimizer)
    pred, eval_loss = evaluate(device=device, model=model, x_val_tensor=x_val_tensor, y_val_tensor=y_val_tensor, loss_fn=loss_fn)

    print(type(pred))
    
    train_loss_list.append(train_loss)
    val_loss_list.append(eval_loss)
    
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_true=y_test, y_pred=pred)
    
    val_acc_list.append(acc)
    
    
   