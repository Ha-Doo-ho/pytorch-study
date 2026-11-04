import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch import tensor
from torch.cuda import is_available

def make_DataSet(x_train, x_val, y_train, y_val, batch_size=32): #batch_size를 미리 정할 수 있는데, 디폴트가 32이다.
    # 텐서 데이터로 변환. (텐서 데이터셋을 만드려면 우선 데이터를 텐서로 변환해야 한다.)
    x_train_tensor = tensor(data=x_train, dtype=torch.float32)
    y_train_tensor = tensor(data=y_train.values, dtype=torch.float32).view(-1,1) # numpy의 reshape와 똑같은 기능을 한다.
    x_val_tensor = tensor(data=x_val, dtype=torch.float32)
    y_val_tensor = tensor(data=y_val.values, dtype=torch.float32).view(-1,1)    
    
    # 텐서 데이터셋 합치기 --> 이때 train데이터만 TensorDatataSet이 된다. 미니배치 단위로 학습시키기 위함이다.
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)    

    #DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, x_val_tensor, y_val_tensor # validation은 굳이 데이터로더로 만들지 않는다. 마치 시험문제를 주면 문제 3개씩 주고 3개씩 풀어봐가 아닌, 문제 전부 다 주고 다 풀어봐 이렇게 하는 것처럼.

def train(dataloader, model, loss_fn, optimizer, device): #핵심은 미니배치 단위이다.
    size = len(dataloader.dataset) # 전체 데이터셋의 크기
    num_batches = len(dataloader) # 배치 크기
    tr_loss = 0
    model.train()
    for batch, (x,y) in enumerate(dataloader): # ※ 미니배치 단위로 다음과 같은 작업이 진행된다.
        x,y = x.to(device), y.to(device)
        
         # Feed Forward --> 순전파.
        pred = model(x)
        loss = loss_fn(pred, y)
        tr_loss += loss

        # ★★Backpropagation → 가중치 업데이트 (반드시 미니배치 단위로 가중치가 업데이트 된다는 것을 알아야 함)
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

    return eval_loss, pred

def dl_learning_curve(tr_loss_list, val_loss_list):

    epochs = list(range(1, len(tr_loss_list)+1))
    plt.plot(epochs, tr_loss_list, label='train_err', marker = '.')
    plt.plot(epochs, val_loss_list, label='val_err', marker = '.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

# cpu 혹은 gpu 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 1 데이터
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/boston.csv'
data = pd.read_csv(path) #(506, 13)
print(data.shape,data.head())

# 1-1 데이터 전처리
x = data.drop(labels="medv", axis=1)
y = data["medv"]

# 1-2 가변수화 --> 숫자로 바꿀 것이 없으니 패스

# 1-3 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=11)

print(type(x_train)) # <class 'pandas.DataFrame'> --> 어떻게 보면 2차원 행렬이다. 행렬은 Pandas에서 DataFrame이라고 부른다.
print(type(y_train)) # <class 'pandas.Series'> --> 어떻게 보면 1차원 벡터이다. 벡터는 Pandas에서 Series라고 부른다.

# 1-4 스케일링 --> 넘파이로 바꾼다. 그래서 y_train_tensor나 y_val_tensor는 판다스의 Series라서 NumPy로 바꿔야 한다. 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
print(type(x_train), type(x_val)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

# 2 모델구성
train_loader, x_val_tensor, y_val_tensor = make_DataSet(x_train, x_val, y_train, y_val, 32)

x,y = next(iter(train_loader))
print(type(x), x.shape) # <class 'torch.Tensor'> torch.Size([32, 12])
print(type(y), y.shape) # <class 'torch.Tensor'> torch.Size([32, 1])

model = nn.Sequential(
    nn.Linear(12, 24),
    nn.ReLU(),
    nn.Linear(24, 1)   
).to(device)

print(model)

# 3. 컴파일 및 훈련
# 해당 부분을 많이 강조함. --> 학습 필요?
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.08)


epochs = 50
train_loss_list, val_loss_list = [], [] # loss 담기 위한 리스트

for t in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device)
    
    # 리스트에 loss 추가 --> learning curve 그리기 휘함이다.
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {t+1}, train loss:{np.round(train_loss,4)}, val_loss: {np.round(val_loss, 4)}")    
    
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, Value: {param.data}")


# 4 평가
loss, pred = evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device)
print(f'MSE : {loss}')

mae = mean_absolute_error(y_val_tensor.detach().cpu().numpy(), pred.detach().cpu().numpy()) # mean_absolute_error 는 numpy / DataFrame / list 3개 중 하나를 요구한다. 그래서 numpy로 바꾼다. 
mape = mean_absolute_percentage_error(y_val_tensor.detach().cpu().numpy(), pred.detach().cpu().numpy()) # 여기도 마찬가지
print(f'MAE : {mae}')
print(f'MAPE : {mape}')
