import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, root_mean_squared_error, root_mean_squared_log_error, mean_absolute_percentage_error

import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
from torch.cuda import is_available
from torch.nn import Sequential, Linear, ReLU, Dropout, Sigmoid, BCELoss

# 역전파 하는 부분을 잘 볼 것. 기초를 코드로 연결한 것이므로 잘 봐야 한다. 

# 디바이스 결정
if is_available():
    device = "cuda"
else:
    device = "cpu"
print("사용장치: ", device)

# 학습 곡선
def dl_learning_curve(tr_loss_list, val_loss_list):

    epochs = list(range(1, len(tr_loss_list)+1))
    plt.plot(epochs, tr_loss_list, label='train_err', marker = '.')
    plt.plot(epochs, val_loss_list, label='val_err', marker = '.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

# 1. 데이터 
def make_Dataset(x_train, x_val, y_train, y_val, batch_size=32):
    x_train_tesor = tensor(data=x_train, dtype=torch.float) # float == torch.float == torch.float32이다. 다만 torch.float이나 torch.float32를 사용하는 이유는 가독성 때문이다. 
    x_val_tensor = tensor(data=x_val, dtype=torch.float)
    y_trian_tensor = tensor(data=y_train.values, dtype=torch.float).view(-1,1)
    y_val_tensor = tensor(data=y_val.values, dtype=torch.float).view(-1,1)
    
    train_dataset = TensorDataset(x_train_tesor, y_trian_tensor) # 여기에는 입력 데이터 x와 정답 데이터 y가 쌍(Pair)으로 보관되어 있다. 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # DataLoadeer는 이 Pair들을 batch단위로 꺼낸다. 만약 배치가 32 배치면 입력32개, 정답32개씩 꺼낸다. 
    
    """ 
    DataLoader 에 들어있는 것
    
        입력 데이터           정답 데이터

    x_train_tensor[0]  ↔  y_train_tensor[0]
    x_train_tensor[1]  ↔  y_train_tensor[1]
    x_train_tensor[2]  ↔  y_train_tensor[2]
    ... 

    """
    return train_loader, x_val_tensor, y_val_tensor


# 3. 컴파일 및 훈련
# 데이터 --> 데이터 로더, 모델, 손실함수, 옵티마이저..., 디바이스
def train(dataloader, model, loss_fn, optimizer, device): 
    # 핵심: 미니배치 단위로 훈련이 진행된다.
    size = len(dataloader.dataset) # 데이터의 개수 (단위: 개)
    num_batches = len(dataloader)  # 배치의 수 (단위: batch)
    
    train_loss = 0
    model.train() #모델을 학습 모드로 설정한다. def train() 호출하는 것이 아니다. 
    
    for batch_idx, (x, y) in enumerate(dataloader): # 미니배치 하나를 직접 꺼냄. 그래서 25, 27Line에 설명되어 있듯 "x:이번 배치의 문제 데이터", "y: 이번 배치의 정답 데이터" 이다.
        x = x.to(device) # 입력과 정답 데이터를 학습을 수행할 장치로 이동시켜야 함. --> CPU에서 할건지 GPU에서 할 건지... -> 실무에서 Pytorch를 사용하는 이유이다.         
        y = y.to(device)

        # Feed Forward → 순전파
        pred = model(x)
        loss = loss_fn(pred, y) # loss, error, cost, 손실, 에러, 비용 전부 동의어. 예측(pred)과 정답(y)가 얼마나 다른지 확인 
        train_loss += loss  # 한 에폭 동안의 전체 학습 데이터에 대한 손실을 요약한 값
        
        # TODO: 여긴 핵심이니까 무조건 암기할 것!
        # Backpropagation → 가중치 업데이트 
        # TODO: 예측과 정답이 얼마나 다른지를 누적하여 확인하였다. 그 손실을 줄이기 위해 가중치를 어떻게 바꿔야 하는지 그 값을 정해야 한다.
        # 이전 배치에서 계산한 기울기(gradient)를 초기화
        # 기울기란? --> 현재 가중치를 어느 방향으로 얼마나 바꾸면 손실이 줄어드는가? 손실을 줄어들게 하는 값이 기울기.
        # 모델의 각 가중치에 대해 기울기를 계산하고, 옵티마이저가 그 정보를 이용하여 가중치를 변경한다. 
        optimizer.zero_grad() # 기울기 초기화
        loss.backward()  # ★★계산한 손실 loss를 기준★★으로 역전파를 수행하여 각 가중치의 기울기를 계산 (가중치 변경 안함!!!)
        optimizer.step() # 옵티마이저가 loss.backward()를 통해 계산된 기울기를 사용하여 모델의 파라미터를 업데이트(가중치를 변경)
    
    train_loss /= num_batches          # 모든 배치에서의 loss 평균손실 계산
    
    return train_loss.item() #텐서 안의 숫자를 Python 숫자로 꺼냄. 이건 그냥 외워라.
         
""" 
    
    Ex_ 데이터가 100개가 있다면?
    현재 배치는 32 배치이다. 단위가 batch임.
    최대 32개씩 나눈다는 뜻이므로, 다음과 같다.

    배치            배치에 포함된 데이터 수
    1번 배치:               32게 --> 가중치 갱신
    2번 배치:               32개 --> 가중치 갱신
    3번 배치:               32개 --> 가중치 갱신
    4번 배치:               4개 --> 가중치 갱신
    
    합계:                   100개
    
    1번 배치 ~ 4번 배치까지 전부 돌았을 때, 1 epoch이다.  
    1 epoch 당 가중치 갱신은 4번 일어난다.
    
    100 epochs하면 가중치 갱신은 400번 일어난다는 것이다.
-------------------------------------------------------------------    
학습 과정

첫 번째 배치
    ↓
기울기 초기화
    ↓
첫 번째 배치의 기울기 계산
    ↓
가중치 갱신

두 번째 배치
    ↓
기울기 초기화
    ↓
두 번째 배치의 기울기 계산
    ↓
가중치 갱신

기울기를 초기화 하는 이유: Pytorch의 .backward()는 호출하면 계산된 기울기가 기존 기울기에 "누적?" 된다. --> 해보면 진짜 그럼. 
그래서 미니배치 학습에서는 새로운 배치를 학습하기 전에 이전 기울기를 지운다.

"""
# 4. 평가 및 예측
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    with torch.no_grad():   # 기울기 계산에 필요한 연산 기록을 끔. loss.backward()로 기울기 계산을 켜는데, 이걸로 잠근다. 
                            # 없어도 되긴 하는데, 없으면 PyTorch는 나중에 backward()를 할 수 있도록 모델의 예측과 손실 계산 과정을 기록하게 되어서 시간이 소요됨.
        model.eval() # 모델을 평가 모드로 설정 --> 이게 있어야 Dropout 같은 것을 걸 때, 훈련때만 적용하고 평가에는 적용되지 않는 경우에 이게 있어야 결과가 올바르게 나온다.          
        x = x_val_tensor.to(device) #입력과 정답 데이터를 평가를 수행할 장치로 이동시켜야 함. --> CPU에서 할건지 GPU에서 할 건지... -> 실무에서 Pytorch를 사용하는 이유이다.         
        y = y_val_tensor.to(device)
        pred = model(x) 

        eval_loss = loss_fn(pred, y).item() # 예측값과 정답의 손실을 계산 후, item()으로 텐서의 숫자를 꺼낸다.
        
        return eval_loss, pred
    

# 1. 데이터
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/titanic.3.csv"
data = pd.read_csv(path)
print(data.shape) #(891, 11)
#print(data.head()) # Pclass(n등석)  Sex(성별)   Age(나이) Survied(1: 생존, 0: 사망)를 맞출 수 있도록 훈련시켜보자. 

x = data.drop(["Survived", "SibSp", "Parch", "Fare", "Embarked", "AgeGroup", "Family", "Age_scale1"], axis=1)
y = data["Survived"]
print(x.shape, y.shape) #(891, 3) (891,)
print(f"type_x {type(x)}, type_y: {type(y)}") #type_x <class 'pandas.DataFrame'>, type_y: <class 'pandas.Series'>
print(x.head())
print(y.head())

# 1-2 가변수화
# x의 Sex에 대해서 가변수화 해야 한다. 
x = pd.get_dummies(data=x, columns=["Sex"], drop_first=True) # 리스트 요구하기 때문에 하나라도 []를 넣는다.
print(x.head())

# 1-3 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=11, stratify=y)


# 1-4 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
print(type(x_train)) #<class 'numpy.ndarray'>

# 2 모델 구성
train_loader, x_val_tensor, y_val_tensor = make_Dataset(x_train, x_val, y_train, y_val, batch_size=32)

model = Sequential(
    Linear(3, 10),
    ReLU(),
    Linear(10, 15),
    ReLU(),
    Linear(15, 10),
    ReLU(),
    Linear(10, 1),
    Sigmoid()
).to(device)

# 3 컴파일 및 훈련
loss_fn = BCELoss()
optimizer = Adam(model.parameters(), lr=0.008)

epochs = 100
train_loss_list = []
val_loss_list = []
for i in range(epochs):
    train_loss = train(dataloader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
    val_loss, _ = evaluate(x_val_tensor=x_val_tensor, y_val_tensor=y_val_tensor, model = model, loss_fn=loss_fn, device=device)
    
    train_loss_list.append(train_loss) 
    val_loss_list.append(val_loss)
    
    print(f"Epoch {i+1}, train loss : {train_loss:4f}, val loss : {val_loss:4f}, device:{device}")

dl_learning_curve(train_loss_list, val_loss_list)


# 4 평가 및 예측
_, pred = evaluate(x_val_tensor=x_val_tensor, y_val_tensor=y_val_tensor, model=model, loss_fn=loss_fn, device=device)
print(type(pred)) # <class 'torch.Tensor'> --> Pandas의 형태인 시리즈, 데이터프레임 에서 NumPy 배열인 ndArray 로 가려면 to_numpy() 혹은 .values이다.
                  # 반면 이번에 해야 하는  torch.Tensor에서 NumPy 배열인 ndArray로 가려면 .numpy()를 사용해야 한다.
pred = np.where(pred.cpu().numpy() > 0.5, 1, 0) # numpy는 cpu 전용이다. GPU에서 동작 안함.
print(pred)

# 5 혼동 행렬
labels=["Dead", "Servived"]
cm = confusion_matrix(y_val_tensor.numpy(), pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(cmap="Blues_r", xticks_rotation=45)
plt.show()