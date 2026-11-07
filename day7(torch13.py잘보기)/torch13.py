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
from torch.nn import Sequential, Dropout, Linear, Conv2d, MaxPool2d, ReLU, Flatten, BCELoss, BatchNorm2d, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.utils.data import TensorDataset , DataLoader
from torchsummary import summary 
from torch.cuda import is_available
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

def dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list):

    epochs = list(range(1, len(tr_loss_list)+1))
    plt.plot(epochs, tr_loss_list, label='train_err', marker = '.')
    plt.plot(epochs, val_loss_list, label='val_err', marker = '.')
    plt.plot(epochs, val_acc_list, label='val_acc', marker = '.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    
    plt.savefig(
    "/home/ma1/dooho/project/learning_curve.png",
    dpi=150,
    bbox_inches="tight"
)
    
    plt.show()

# 0 디바이스 설정
if is_available():
    device = "cuda"
else:
    device = "cpu"

# 1 데이터셋은 텐서로 바꾸고, 훈련 데이터셋은 DataLoader로 바꿔야 한다. 미니배치 단위로 만들어 줌.
def makeDataSet(x_train, x_val, y_train, y_val, batch_size=32):
    x_train_tensor = tensor(data=x_train, dtype=torch.float).reshape(-1,1,28,28)
    x_val_tensor = tensor(data=x_val, dtype=torch.float).reshape(-1,1,28,28)     
    
    y_train_tensor = tensor(data=y_train, dtype=torch.long) # 다중분류이므로 dtype은 무조건 torch.long이어야 한다.
    y_val_tensor = tensor(data=y_val, dtype=torch.long)
    
    train_Dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)
    
    return train_DataLoader, x_val_tensor, y_val_tensor 

# 3 모델 학습
# 핵심: 미니배치 단위로 학습 --> 미니배치 기준으로 단위 잡으면 됨.
def train(device, model, train_DataLoader, loss_fn, optimizer):
    train_loss = 0
    num_batches = len(train_DataLoader)
    model.train()
    
    for batch_idx, (x,y) in enumerate(train_DataLoader):
        x = x.to(device)
        y = y.to(device)
        
        # 순전파
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss
        
        #역전파
        loss.backward() # loss_fn으로 구한 loss값 가지고 기울기 계산
        optimizer.step() # 구한 기울기로 파라미터 조정
        optimizer.zero_grad()
        
    train_loss /= num_batches
    print(f"train_loss타입: {type(train_loss)}")
    return train_loss.item()

def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    with torch.no_grad():
        model.eval()
        
        x = x_val_tensor.to(device)
        y = y_val_tensor.to(device)
        
        pred = model(x) 
        eval_loss = loss_fn(pred, y)
        
        print("eval_loss타입: ",type(eval_loss))
    return pred, eval_loss.item()

# 1-1 데이터
train_dataset = datasets.FashionMNIST(root='data', train=True, transform=ToTensor())
test_dataset = datasets.FashionMNIST(root='data', train=False, transform=ToTensor())

print(type(train_dataset)) #<class 'torchvision.datasets.mnist.FashionMNIST'>
#exit()

# 1-2 가변수 처리. --> 할 거 없음 pass

# 스케일링 먼저 
train_data = train_dataset.data.numpy() / 255 # torchvision이니까 파이토치 데이터 타입이다. 그래서 .numpy()이다. 만약 판다스였다면 넘파이로 바꿀 때, values 혹은 to_numpy()사용했을 것이다.
test_data = test_dataset.data.numpy() / 255

train_labels = train_dataset.targets.numpy() 
test_labels = test_dataset.targets.numpy() 


# 1-3 데이터셋 분리
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, train_size= 50000, random_state = 10, stratify = train_labels)
_, x_test, _, y_test = train_test_split(test_data, test_labels, test_size = 9000, random_state = 10, stratify = test_labels)

print(x_train.shape, y_train.shape) #(20000,) --> 벡터임 답지라는 뜻.
print("y_test 타입:", type(y_test))
# 1-4 스케일링

# 2 모델 구성
train_DataLoader, x_val_tensor, y_val_tensor = makeDataSet(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, batch_size=64)
x_test_tensor = tensor(data=x_test, dtype=torch.float).reshape(-1,1,28,28)
y_test_tensor = tensor(data=y_test, dtype=torch.long)

model = Sequential(
    
    # 컨벌루션 연산 층
    Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding="same"),
    ReLU(),
    # Conv와 ReLU를 거친 데이터는 다음 형태이다. (배치, out_channels, 높이, 너비) 
    # 정규화한 결과의 크기와 위치를 조절하는 학습 파라미터 γ, β도 "채널마다" 하나씩 갖는다. 
    BatchNorm2d(num_features=8),
    MaxPool2d(kernel_size=2),
    
    
    
    Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding="same"),
    ReLU(),
    BatchNorm2d(num_features=16),
    MaxPool2d(kernel_size=2),
        
    Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
    ReLU(),
    BatchNorm2d(num_features=32),
    MaxPool2d(kernel_size=2),
    
    # 판단
    Flatten(),
    Linear(in_features=3*3*32, out_features= 128), #마지막 계산 잘해야 한다. 행/열크기가 28이었지만 14 -> 7 -> 3 이 된다. 
    ReLU(),
    
    Linear(in_features=128, out_features=64),
    ReLU(),
    
    # 출력층
    Linear(in_features=64, out_features=10)    
).to(device)

# 3 컴파일 및 훈련
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.002)

epoch = 40
train_loss_list = []
val_loss_list = []
val_acc_list = []

val_loss = 0
best_weight = float('inf')
counter = 0
patience = 5

start_time = time.time()
for i in range(epoch):
    train_loss = train(device=device, model=model, train_DataLoader=train_DataLoader, loss_fn=loss_fn, optimizer=optimizer)
    pred, val_loss = evaluate(device=device, model=model, x_val_tensor=x_val_tensor, y_val_tensor=y_val_tensor, loss_fn=loss_fn)
    
    pred = np.argmax(pred.cpu().numpy(), axis=1) # CrossEntropyLoss를 걸면 마지막 층에  Softmax 활성화함수가 자동으로 걸린다. 그래서 여기서 바로 구해도 됨.
    accuracy = accuracy_score(y_true=y_val_tensor.cpu().numpy(), y_pred=pred)
    
    val_acc_list.append(accuracy)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    
    print(f"Epoch:{i+1} train_loss:{round(train_loss,4)} val_loss:{round(val_loss, 4)} device:{device}")    
    
    #EarlyStopping
    if val_loss >= best_weight:
        counter += 1
        print(f"EarlyStopping Start! counter:{counter}")
    else:
        best_weight = val_loss
        counter = 0
    
    if counter > patience:
        print("EarlyStopping Triggered!")
        break
    
end_time = time.time()
print(f"소요시간: {round(end_time - start_time, 4)}")

dl_learning_curve(train_loss_list, val_loss_list, val_acc_list)
         
# 4 평가 및 예측
# 예측
pred, _ = evaluate(device=device, model=model, x_val_tensor=x_test_tensor, y_val_tensor=y_test_tensor, loss_fn=loss_fn)
pred = softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis = 1)

classes = train_dataset.classes

# confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.xticks(rotation=90)

plt.savefig(
    "/home/ma1/dooho/project/confusion_metrix.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

# classification report
print('='*80)
print(f'Accuracy : {accuracy_score(y_test, pred)}')
print('-'*80)
print(classification_report(y_test, pred, target_names=classes))