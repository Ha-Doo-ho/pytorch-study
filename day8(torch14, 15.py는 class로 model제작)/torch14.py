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
from torchvision.transforms import ToTensor

# 2 모델 생성
# 클래스 방식으로는 다음과 같이 작성하며 보통 클래스 방식으로 많이들 작성한다.
class CustomCNN(Module): #모든 모델은 Module을 상속받는다. PyTorch 신경망의 기본 클래스가 Module이기 때문이다.
    def __init__(self, n_node): # kwargs: Keyword Arguments: 키=value 형태로 함수에 전달하는 인자. 이렇게 쓰려면 {}(딕셔너리) 내부에 넣어야 하고 {}안에 들어있는 것 넣으려면 ** 필요함.
        super(CustomCNN, self).__init__() # 기준 클래스, 그에 대한 객체(this임)순서. # 부모인 Module의 "초기화 메서드 호출함"으로써 계속 다른 형태로 제작이 가능하다.
                                          # 바어있어도 된다. super().__init__ 처럼 비워도 된다.
        
        #Conv층
        self.conv = Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding="same") #컬러면 in_channels = 3일것이다.
        self.relu1 = ReLU()
        self.pool = MaxPool2d(kernel_size=2)
        
        # 판단 종합층
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=8 * 14 * 14, out_features=32) # 처음 원본 이미지의 크기가 28 * 28이라고 가정했을 때의 예시이다. 
        self.relu2 = ReLU()
        
        # 판단층
        self.linear2 = Linear(in_features=32, out_features=10)
    
    # 순전파 - FeedForward 라서 이름이 forward임. 그래서 model을 호출하면 pytorch는 알아서 forword를 찾는 기능이 내장되어 있다. 
    # forward라는 메서드가 실제로 존재해서 오버라이딩 하는 것이다.  
    def forward(self, x):
        # 모델 설계
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x