import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

#Classification : 대학원 지원이 가능한 점수인가 아닌가를 분류하는 문제

# 1. 데이터
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/Graduate_apply.csv"
data = pd.read_csv(path)
print(data.shape) # (400, 4) 
print(data.head()) # --> admit은 0이 탈락 1은 합격이며, gre는 대학원 입학시험 점수이다.

# 여기서 x는 gre, gpa, rank이다. y가 합격여부(admit)이다.
x = data.drop("admit", axis=1)
y = data["admit"]

# 핵심은 가변수화. 형태는 숫자인데, 의미가 범주형인 경우가 있다. --> 대표적으로 월(1월/2월/3월 . . . ) 이런 것들은 겉으로 보기에는 숫자인데, 반드시 가변수화를 해주어야 한다.
# 가변수화가 필요하다.  필요한 부분은 rank이다. 자칫 가중치?, 숫자 값으로 오해할 수 있기 때문이다.
cat_cols = ["rank"]
x = pd.get_dummies(y, cat_cols, drop_first=True)

x_train, x_val, y_train, y_val = train_test_split(x,y, train_size=0.9, random_state=11, shuffle=True, stratify=y)

#스케일링
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# 2. 모델구성
model = RandomForestClassifier(n_estimators=100,)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 및 예측
y_predict = model.predict(x_val)
print(y_predict)
auc = accuracy_score(y_val, y_predict)
print(np.round(auc, 4))