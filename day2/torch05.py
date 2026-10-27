import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# 모델링을 위한 데이터로더 만들기
# 이유는 학습 시, 배치 단위로 데이터를 처리하기 위함이다.
def make_DataSet(x_train, x_val, y_train, y_val, batch_size = 32) :

    # 데이터 텐서로 변환
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # TensorDataset 생성 : 텐서 데이터셋으로 합치기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, x_val_tensor, y_val_tensor

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    tr_loss = 0                  # 전체 데이터셋의 크기
    model.train()                                   # 훈련 모드로 설정
    for batch, (X, y) in enumerate(dataloader):     # batch : 현재 배치 번호, (X, y) : 입력 데이터와 레이블
        X, y = X.to(device), y.to(device)           # 입력 데이터와 레이블을 지정된 장치(device, CPU 또는 GPU)로 연결

        # Feed Forward
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss

        # Backpropagation
        loss.backward()             # 역전파를 통해 모델의 각 파라미터에 대한 손실의 기울기를 계산
        optimizer.step()            # 옵티마이저가 계산된 기울기를 사용하여 모델의 파라미터를 업데이트
        optimizer.zero_grad()       # 옵티마이저의 기울기 값 초기화. 기울기가 누적되는 것 방지

    tr_loss /= num_batches          # 모든 배치에서의 loss 평균

    return tr_loss.item()

def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    model.eval()                        # 모델을 평가 모드로 설정

    with torch.no_grad():               # 평가 과정에서 기울기를 계산하지 않도록 설정(메모리 사용을 줄이고 평가 속도를 높입니다.)
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        eval_loss = loss_fn(pred, y).item()    # 예측 값 pred와 실제 값 y 사이의 손실 계산

    # print(f"evaluete_err: {eval_loss:>7f}")

    return eval_loss

# cpu 혹은 gpu 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 1 데이터
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/advertising.csv"
adv = pd.read_csv(path)
print(adv.head(), adv.shape) #(200, 4)

# 1-2 데이터 전처리 (x / y가 무엇인지 구분. 무조건 x에서만 작업이 이루어져야 한다.)
# x: TV / Radio / Newspaper 
# y: Sales
x = adv.drop("Sales", axis=1) 
y= adv["Sales"]

print(x.shape, y.shape) #(200, 3) (200,)

# 가변수화 할 것은 없으므로 패스

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=11, shuffle=True)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)

# 여기서부터 기존 머신러닝과 다른 점이 나온다. --> 텐서와 데이터로더 
# 1. 데이터셋을 데이터 텐서로 변환
# 2. 텐서 데이터셋으로 합치기: x, y
# 3. DataLoader 생성
# ★★★★★텐서 / 데이터 로더를 만드는 과정이 필요하다.★★★★★
# -> 그게 make_DataLoader이다.

train_loader, x_val_ts, y_val_ts = make_DataSet(x_train, x_val, y_train, y_val, batch_size = 4)
for x, y in train_loader:
    print(f"Shape of x [rows, columns]: {x.shape}") # torch.Size([4, 3]) 배치 사이즈를 4로 두었기 때문에 4, 가 나오게 됨
    print(f"Shape of y: {y.shape} {y.dtype}")       # torch.Size([4, 1]) 배치 사이즈를 4로 두었기 때문에 마찬가지임
    break

""" 
이것도 됨.
x, y = next(iter(train_loader)) # iter(train_loader): train_loader에서 배치를 차례대로 꺼낼 반복자 제작
                                # next(train_iterator): 반복자에서 다음 배치 하나를 가져옴. 
print(x.shape, y.shape) #torch.Size([4, 3]) torch.Size([4, 1])
"""


# 2 모델 구성

# 학습해야 하는 데이터셋의 열이 3이므로 무조건 기억할 것.  --> 무조건 행무시 열우선 
# 열과 같은 말: feature, attribute, dimension, column, 특징, 속성, 특성, 차원 . . . 외울 것.
# nn.Linear() 에 들어가는 인수가 in_feature, out_feature이다. 열 숫자 넣는 것이다.

# nn.Seqential 혹은 nn.모델 형태 에 ㅈㄴ 많이 적는다 정도로만 이해하면 된다.
model = nn.Sequential(
    nn.Linear(3, 3), # input, output 순서이다. 여기가 model.add(Dense(10,input_shape=(4,), activation='relu')) 과 같다.
    nn.ReLU(),
    nn.Linear(3, 1) # 여기가 ouput_layer 이다. keras로 치면 쭉 Dense층 작성하다가, model.add(Dense(3, activation='softmax')) 이렇게 끝내는 것과 같다. 
).to(device) 
print(model)

# 3 컴파일 및 훈련. pytorch에서는 약간 다르다.
# 우선 이렇게 한다 정도만 알아놓자.
loss_fn = nn.MSELoss() #loss 만드는 거 loss는 비용, 손실, 에러 등이 있다. 그래서 cost, Error 도 다 loss이다.
optimizer = Adam(model.parameters(), lr=0.01) # model.parameters() : 모델의 가중치와 편향

# keras에서는 model.fit 걸면 되는데, Pytorch는 조금 더 복잡하다. 
# 20번 학습을 시키되 학습이 끝날 때마다 validation 즉, 모의고사 한번씩 보라는 것이니까 val도 20번 하는 것.
epochs = 20
for t in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer, device) 
    val_loss = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
    print(f"Epoch:{t + 1}, train_loss: {train_loss: 4f}, val_loss: {val_loss: 4f}")