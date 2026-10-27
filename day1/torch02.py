import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression


# 어떤 메체의 광고가 매출에 영향을 얼마나 주는지..

# 1 데이터 --> x는 TV, Radio, Newspaper 이며, y는 sales
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/advertising.csv"
data1 = pd.read_csv(path)
print(data1.head()) # 혹은 .columns 를 정말로 많이 사용한다.

# 1-1 데이터 전처리
# y는 이전에도 설명했지만, 답지이므로 따로 빼야 한다. drop을 사용하면 된다. 절대 axis를 빼지 않도록 실수에 유의하자.
x = data1.drop("Sales", axis=1) 
y = data1["Sales"] #혹은 data1.loc[:, "Sales"]

print(x, y)

# 범주형 변수가 있으면 가변수화 하는데, 그런거 없으니 패스

# train / val / test 분리
x_train, x_val, y_train, y_val = train_test_split(x,y, train_size=0.7, random_state=11, shuffle=True)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)

# 2 모델 구성
model = LinearRegression()

# 3 학습
model.fit(X=x_train, y=y_train)

# 4 평가 및 예측
y_predict = model.predict(x_val)
print(y_predict)
mae = mean_absolute_error(y_val, y_predict)
print(mae)