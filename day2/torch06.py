import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import torch
from torch import tensor 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset # Pytorch를 이용한 딥러닝에서는 DatatLoader 와 TensorDataset이 필수적이다.
from torch.optim import Adam


# 모델링을 위한 데이터로더 만들기 --> 이 데이터 로더는 Tensor Dataset으로 구성되어 있다.
# 이유는 학습 시, 배치 단위로 데이터를 처리하기 위함이다. 
# ★★★★★가장 중요한 과정이며 이것이 딥러닝과 머신러닝의 차이★★★★★
def make_DataSet(x_train, x_val, y_train, y_val, batch_size = 32):
    
    # 데이터셋을 텐서 데이터셋으로 변환
    # tensor는 파이토치(PyTorch)나 텐서플로우(TensorFlow)에서 연산을 수행하는 기본 단위이다. 
    x_train_tensor = tensor(data=x_train, dtype=torch.float32)
    x_val_tensor = tensor(data=x_val, dtype=torch.float32)
    y_train_tensor = tensor(data=y_train.values, dtype=torch.float32).view(-1, 1)
    y_val_tensor = tensor(data=y_val.values, dtype=torch.float32).view(-1, 1)
    
    # TensorDataset 생성 : 텐서 데이터셋으로 합치기
    # TensorDataset: 파이토치 등에서 여러 개의 텐서를 하나의 세트로 묶어 관리하기 위한 컨테이너(Dataset 클래스)
    # 이때 한 묶음이 문제-답 쌍(Pair)이 되는 것이다. 
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    
    # DataLoader 생성
    # 이제 묶는 것까지 끝났으니, 어떻게 가져오냐가 중요하다. 그 방법은 배치 단위로 가져오는 것이다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/boston.csv'
data = pd.read_csv(path)
print(data.shape, data.head()) #(506, 13) 

"""
0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3   4.98  24.0
1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8   9.14  21.6
2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8   4.03  34.7
3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7   2.94  33.4
4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7   5.33  36.2
"""

x = data.drop(labels="medv", axis=1)
y = data["medv"]

# 1-2 가변수화 --> 범주형 변수로 잡을만 한 것은 보이지 않는다. 패스

# 1-3 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=11)

# 1-4 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)

# 1-5 데이터로더 적용 및 확인
train_loader, x_val_ts, y_val_ts = make_DataSet(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, batch_size=4)

x,y = next(iter(train_loader))
print(x.shape, y.shape) #torch.Size([4, 12]) torch.Size([4, 1]) 입력: 4행 12열 / 4행 1열 
print(f"x:{x} | y: {y}")

# 2 모델 구성
model = nn.Sequential(
    nn.Linear(12, 24),
    nn.ReLU(),
    nn.Linear(24,1)
).to(device)
print(model)

# 3 컴파일 및 훈련. 
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

epochs = 20
for t in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
    print(f"Epoch:{t + 1}, train_loss: {train_loss: .4f}, val_loss: {val_loss: .4f} Device: {device}")


# 4 평가 및 예측
evaluate(x_val_ts, y_val_ts, model, loss_fn, device)