# In Colab
# Ref:  https://medium.com/@soubhikkhankary28/univariate-time-series-forecasting-using-rnn-lstm-32702bd5cf4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Reading train CSV file
df1 = pd.read_csv("train_data.csv")
print(df1.shape)
plt.figure(figsize=(10,8))
print(df1.plot(x="time", y='feed_pressure'))

# df1.head()
df1=df1[['feed_pressure']]
df1.head()

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# feature_range를 튜플로 지정
sc = MinMaxScaler(feature_range=(0, 1))
df1_scaled_train = sc.fit_transform(df1)
print(type(df1_scaled_train))
df1_scaled_train

df1_scaled_train.shape[0]

hops = 60  # 입력 데이터의 길이 설정 (과거 60개의 시점 데이터를 사용)

total_len = df1_scaled_train.shape[0]  # 데이터의 총 길이
X_train = []  # 입력 데이터 리스트
y_train = []  # 출력 데이터 리스트

for i in range(60, total_len):
    # 과거 60개의 데이터 포인트를 입력으로 사용
    X_train.append(df1_scaled_train[i - 60:i])  # i번째 데이터를 기준으로 이전 60개 데이터를 X에 추가
    y_train.append(df1_scaled_train[i])         # i번째 데이터를 y에 추가 (예측할 값)

# 리스트를 numpy 배열로 변환
X_train = np.array(X_train)
y_train = np.array(y_train)

print(len(X_train))
print(len(y_train))

m1_len=X_train.shape[0]
m2_len=X_train.shape[1]

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32)

# Reading train CSV file 
df1_test=pd.read_csv("test_data.csv")
print(df1_test.shape)
print(df1_test.describe())
print(df1_test['feed_pressure'].plot())

df1_total=pd.concat((df1['feed_pressure'],df1_test['feed_pressure']),axis=0)
df1_new=df1_total.values
test_arr=df1_new[len(df1_new)-len(df1_test)-60:]
# verify if 80 records are present or not
len(test_arr)
test_arr

test_arr_1=sc.transform(test_arr.reshape(-1,1))
test_arr_1.shape

n_hops=60
n_features=1
X_test=[]

y_test=[]
for i in range(n_hops,test_arr_1.shape[0]):
    X_test.append(test_arr_1[i-n_hops:i])
X_test=np.array(X_test)

y_test_pred=model.predict(X_test)
print(len(y_test_pred))
y_test_pred

y_test_pred_actual=sc.inverse_transform(y_test_pred)
y_test_pred_actual

test_pred_1=pd.DataFrame(y_test_pred_actual,columns=['actual'])
test_actual_1=df1_test[['time', 'feed_pressure']]
full_test_actual_1=pd.concat([test_pred_1, test_actual_1], axis=1)
full_test_actual_1

full_test_actual_1.index=pd.to_datetime(full_test_actual_1['time'])
plt.plot(full_test_actual_1['feed_pressure'], color='red', label='actual')
plt.plot(full_test_actual_1['actual'], color='blue', label='pred')
plt.legend()
plt.plot()

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

# RMSE (Root Mean Squared Error) 계산
rmse = np.sqrt(mean_squared_error(full_test_actual_1['feed_pressure'], full_test_actual_1['actual']))

# R^2 (Coefficient of Determination) 계산
r2 = r2_score(full_test_actual_1['feed_pressure'], full_test_actual_1['actual'])

# MAPE (Mean Absolute Percentage Error) 계산
mape = mean_absolute_percentage_error(full_test_actual_1['feed_pressure'], full_test_actual_1['actual']) * 100

# 결과 출력
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 (Coefficient of Determination):", r2)
print("Mean Absolute Percentage Error (MAPE):", mape, "%")
