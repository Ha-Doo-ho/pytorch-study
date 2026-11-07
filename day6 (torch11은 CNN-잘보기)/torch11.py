import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
from torch import nn
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.cuda import is_available
from torch.nn import Sequential, ReLU, Sigmoid, BCELoss, CrossEntropyLoss, Conv2d, MaxPool2d, Dropout

# 모델 요약
from torchsummary import summary

# 0 디바이스 설정
if is_available():
    device = "cuda"
else:
    device = "cpu"

# 1 데이터
def makeDataSet(x_train, y_train, x_val, y_val, batch_size=32):
    x_train_tensor = tensor(data=x_train, dtype=torch.float)
    x_val_tensor = tensor(data=x_val, dtype=torch.float)
    y_train_tensor = tensor(data=y_train, dtype=torch.long)
    y_val_tensor = tensor(data= y_val, dtype=torch.long)
    
    train_Dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle = True)
    
    return train_DataLoader, x_val_tensor, y_val_tensor

# 3 컴파일 및 훈련
# 핵심은 미니배치 단위로 훈련을 진행한다는 것을 늘 염두에 둘 것. 
def train(model, device, DataLoader, loss_fn, optimizer):
    model.train()
    num_batches = len(DataLoader)
    train_loss = 0
    
    for batch_idx, (x, y) in enumerate(DataLoader):
        x = x.to(device)
        y = y.to(device)
        
        # 순전파 Feed-Forward
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss
        
        # 역전파 Backpropagation
        loss.backward() # loss_fn을 가지고 구한 loss를 가지고 기울기 계산을 시작
        optimizer.step() # 계산된 기울기를 가지고 가중치를 조정
        optimizer.zero_grad() # 가중치 초기화
    
    train_loss /= num_batches
    return train_loss.item()

# 4 컴파일 및 훈련
def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    model.eval()
    with torch.no_grad():
        
        x = x_val_tensor.to(device)
        y = y_val_tensor.to(device)

        pred = model(x)
        eval_loss = loss_fn(pred, y)
        print("eval_loss의 타입: ",type(eval_loss))
        
    return pred, eval_loss.item()

#학습곡선
def dl_learning_curve(train_loss_list, val_loss_list):

    epochs = list(range(1, len(train_loss_list)+1))
    plt.plot(epochs, train_loss_list, label='train_err', marker = '.')
    plt.plot(epochs, val_loss_list, label='val_err', marker = '.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()


# 1 데이터
# Download training data from open datasets.
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),   # 픽셀값을 [0,1] 사이로 정규화하고 텐서로 변환
)

# Download test data from open datasets.
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(train_dataset.data.shape, type(train_dataset))  # torch.Size([60000, 28, 28]) <class 'torchvision.datasets.mnist.MNIST'>
print(test_dataset.targets.shape, type(test_dataset)) # torch.Size([10000]) <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset.classes)
print(test_dataset[0])

# 1-2 가변수 처리 pass

# 1-3 데이터 분할 
x_val, x_test = test_dataset.data[:5000], test_dataset.data[5000:]
y_val, y_test = test_dataset.targets[:5000], test_dataset.targets[5000:]
print(x_val.shape, y_val.shape) # torch.Size([5000, 28, 28]) torch.Size([5000])

# 1-4 스케일링
x_val = x_val.reshape(-1, 1, 28, 28) / 255
x_test = x_test.reshape(-1, 1, 28, 28)/ 255
print(x_val.shape, x_test.shape) # torch.Size([5000, 1, 28, 28]) torch.Size([5000, 1, 28, 28])

# 2 모델 설계
# 핵심은 컨볼루션층 (Conv2d, Activation_fn, Pooling) 통과할 때마다, padding을 걸지 않으면, 절반으로 노드 수가 줄어든다는  것이다.
train_dataloader = DataLoader(train_dataset, batch_size = 64)
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding="same"),  # input = 1*28*28, output = 8*28*28
    nn.ReLU(),
    # Conv와 ReLU를 거친 데이터는 다음 형태이다. (배치, out_channels, 높이, 너비) 배치가 현재 32이므로 (32,8,28,28)이다.
    # 정규화한 결과의 크기와 위치를 조절하는 학습 파라미터 γ, β도 "채널마다" 하나씩 갖는다. 
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),     # input = 8*28*28, output = 8*14*14
    
    nn.Flatten(),
    nn.Linear(8*14*14, 64),    # 이전 레이어에서의 출력 수를 계산해서 입력으로 넣어야 함. 
    nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

summary(model, input_size = (1,28,28))

# 3 컴파일 및 훈련
loss_fn = nn.CrossEntropyLoss() # 정보량과 희소성의 관점에서 바라보면 됨.
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 10
tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(model, device, train_dataloader, loss_fn, optimizer)
    _, val_loss = evaluate(device, model, x_val, y_val, loss_fn)

    # 리스트에 loss 추가 --> learning curve 그리기 위해.
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {t+1}, train loss : {tr_loss:.4f}, val loss : {val_loss:.4f}")

# 학습곡선    
dl_learning_curve(tr_loss_list, val_loss_list)

# 모델 평가 및 예
pred,_ = evaluate(device, model, x_test, y_test, loss_fn) #device, model, x_val_tensor, y_val_tensor, loss_fn
pred[:5]

pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis = 1)

cm = confusion_matrix(y_test.numpy(), pred)
print(accuracy_score(y_test.numpy(), pred))
print('-'*100)
print(classification_report(y_test.numpy(), pred))