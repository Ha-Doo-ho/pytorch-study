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
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.nn import Sequential
from torch.nn import ReLU, Dropout, Linear, Sigmoid, Softmax

# 0 device 결정
if is_available():
  device = "cuda"
else:
  device = "cpu"
  
# 1 데이터
# tensor형태로 만들고, trainDataSet은 DataLoader로 제작해야 함.
def make_DataSet(x_train, x_val, y_train, y_val, batch_size=32):
  x_train_tensor = tensor(x_train, dtype=torch.float32)
  y_train_tensor = tensor(y_train, dtype=torch.long)
  # 정답값(Target): 클래스 인덱스 (예: tensor([0, 2, 1]), 데이터 타입은 반드시 torch.LongTensor형이어야 함)
  # 이때는 .values나 .to_numpy를 사용하면 안되는데, 그 이유는 y값에 보통 LabelEncoding을 걸어야 하는데,
  # 그걸 걸면 ndArray 즉 넘파이 배열로 바뀌기 때문에 여기서 걸면 2번 바꿔서 에러터진다.
  x_val_tensor = tensor(x_val, dtype=torch.float32)
  y_val_tensor = tensor(y_val, dtype=torch.long)

  train_Dataset = TensorDataset(x_train_tensor, y_train_tensor)
  train_Loader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

  return train_Loader, x_val_tensor, y_val_tensor

# 3 컴파일 및 훈련
# 핵심은 미니배치 단위로 학습한다는 것이 중요하다.
def train(model, device, DataLoader, lose_fn, optimizer):
    num_batches = len(DataLoader)
    train_loss = 0

    for batch_idx, (x, y) in enumerate(DataLoader):
      model.train()

      x = x.to(device)
      y = y.to(device)

      # 순전파 --> FeedForward
      pred = model(x)
      loss = loss_fn(pred, y) # 정답과 원본의 차이 즉 loss (손실, 에러 비용....)를 구한다.
      train_loss += loss
      print("train_loss의 타입",type(train_loss))

      # 역전파 --> Backpropagation
      loss.backward()  # loss를 가지고 얼마만큼 가야 하는지, 즉 기울기를 계산한다.
      optimizer.step() # 기울기를 움직인다. 즉, 가중치를 수정한다.
      optimizer.zero_grad() #가중치가 이게 없으면 계속 누적되기 때문에 초기화 하지 않으면, 가충치가 매우 크게 변경될 수 있다.

    train_loss /= num_batches
    return train_loss.item()

# 4 평가 및 예측
def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    model.eval()
    x = x_val_tensor.to(device)
    y = y_val_tensor.to(device)

    with torch.no_grad(): #기울기 계산 기능을 종료한다. 더 이상 기울기를 계산할 필요가 없다. 원본과 정답을 통해, loss만 계산하면 된다.
      pred = model(x)
      eval_loss = loss_fn(pred, y)
      print("pred 의 타입: ", type(pred))
      print("eval_loss 의 타입: ", type(eval_loss))

    return pred, eval_loss.item()


# 1 데이터
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv"
data = pd.read_csv(path)
print(data.shape) #[150 rows x 5 columns]> 미자막 열인 Species 를 구해야 함. 그게 y임
print(data.head)

# 1-2 가변수 처리
# 따로 가변수 처리 할 것이 없다. Pass

# 1-3 데이터셋 분리
x = data.drop("Species", axis=1)
y = data["Species"]
print(x.shape)
print(y.shape)
print(type(x), type(y))
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=11, shuffle=True,stratify=y)

# 1-4 x값에 스케일러 적용, y값이 현재 문자열로 되어 있으므로 LabelEncoding을 적용해야 한다.
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
print(type(x_train)) #<class 'numpy.ndarray'> 스케일러 거치면 <class 'numpy.ndarray'> 임.
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_val = labelencoder.transform(y_val)
print(type(y_train)) # LabelEncoder 거쳐도 <class 'numpy.ndarray'>

# 2 모델구성
# 다중분류에서는 마지막 열의 Activation_Function을 따로 주지 않는다.
# CrossEntropyLoss (Keras의 categorical_crossentropy)가 softmax기능을 포함하고 있기 때문이다.
# 대신 Softmax는 예측한 값을 0~1 사이(클래스 확률 전부 더하면 1)의 값으로 정규화 하는데 사용한다.

train_Loader, x_val_tensor, y_val_tensor = make_DataSet(x_train, x_val, y_train, y_val)

model = Sequential(
    Linear(in_features=4, out_features = 10),
    ReLU(),
    Linear(in_features=10, out_features=15),
    ReLU(),
    Linear(in_features=15, out_features=20),
    ReLU(),
    Linear(in_features=20, out_features=15),
    ReLU(),
    Linear(in_features=15, out_features=3),
).to(device)

# 3 컴파일 및 훈련
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.08)

epoch=100
train_loss_list = []
val_loss_list = []
for i in range(epoch):
    train_loss = train(model, device, train_Loader, loss_fn, optimizer)
    _,val_loss = evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn)

    # 학습 곡선을 그리기 위해 사용한다.
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch:{epoch+i} train_loss:{np.round(train_loss,4)} val_loss:{np.round(val_loss,4)} device:{device}")


# 4 평가 및 예측
pred, _ = evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn)
#print(pred)
print(type(pred)) #pred 의 타입:  <class 'torch.Tensor'>
pred = torch.softmax(pred, dim=1) #열에 대하여 Softmax함수를 걸어야 함.

pred = np.argmax(pred.cpu().numpy(), axis=1)
print(pred[:100])    
    