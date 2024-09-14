import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# .xlsx 파일 경로
xlsx_path = r"pressure_data.csv"
_, fileExtension = os.path.splitext(xlsx_path)
if fileExtension == ".csv":
    data = pd.read_csv(xlsx_path)
elif fileExtension == ".xlsx":
    data = pd.read_excel(xlsx_path)
else:
    raise Exception("Not Supported File Format")
# .csv 파일로 저장할 경로
csv_path = r"pressure_data2.csv"
# .csv 파일로 저장
data.to_csv(csv_path, index=False)

def get_outlier(df=None, column=None, weight=1.5):
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight
  
    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight
  
    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    return outlier_idx

outlier_idx = get_outlier(df=data, column='유입 압력', weight=1.5)
data.drop(outlier_idx, axis=0, inplace=True)

X_feed = data["유입 압력"].values
y_feed = data["유입 압력"].values

# Min-Max 스케일링
scaler = MinMaxScaler()
X_feed = scaler.fit_transform(X_feed.reshape(-1, 1))
y_feed = scaler.fit_transform(y_feed.reshape(-1, 1))

# 데이터 시퀀스 생성 함수
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# 시퀀스 길이
sequence_length = 50

# Feed pressure 데이터 시퀀스 생성
X_feed_seq, y_feed_seq = create_sequences(X_feed, sequence_length)

# 데이터 분할
X_feed_train, X_feed_test, y_feed_train, y_feed_test = train_test_split(X_feed_seq, y_feed_seq, test_size=0.2, random_state=42)

# LSTM 모델 구성
model_feed = tf.keras.models.Sequential()
model_feed.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model_feed.add(tf.keras.layers.Dense(1))
model_feed.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model_feed.fit(X_feed_train, y_feed_train, epochs=50, batch_size=32)

# Feed pressure 예측
y_feed_pred = model_feed.predict(X_feed_test)

# 이동 평균을 적용할 윈도우 크기 설정
window_size = 5

# 이동 평균을 계산할 함수 정의
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# 패딩을 추가하기 위한 함수 정의
def add_padding(data, padding_size):
    start_padding = np.full(padding_size, data[0])
    end_padding = np.full(padding_size, data[-1])
    return np.concatenate([start_padding, data, end_padding])

# 데이터에 패딩 추가
y_feed_test_padded = add_padding(y_feed_test.flatten(), window_size)
y_feed_pred_padded = add_padding(y_feed_pred.flatten(), window_size)

# 이동 평균 적용
y_feed_test_rolled = moving_average(y_feed_test_padded, window_size)
y_feed_pred_rolled = moving_average(y_feed_pred_padded, window_size)

# 패딩
y_feed_test_rolled_trimmed = y_feed_test_rolled[window_size:-window_size]
y_feed_pred_rolled_trimmed = y_feed_pred_rolled[window_size:-window_size]

# 인터폴레이션을 위한 새로운 x축 포인트 생성
x_new = np.linspace(0, len(y_feed_test_rolled_trimmed) - 1, len(y_feed_test) * 2)

# 인터폴레이션 함수 생성
f_actual = interp1d(range(len(y_feed_test_rolled_trimmed)), y_feed_test_rolled_trimmed, kind='linear')
f_predicted = interp1d(range(len(y_feed_pred_rolled_trimmed)), y_feed_pred_rolled_trimmed, kind='linear')

# 새로운 x축 포인트에서의 인터폴레이션 값 계산
y_feed_test_interpolated = f_actual(x_new)
y_feed_pred_interpolated = f_predicted(x_new)

# RMSE 계산 (이동 평균 적용된 데이터에 대해서)
mse_processed = mean_squared_error(y_feed_test_rolled_trimmed, y_feed_pred_rolled_trimmed)
print(f"처리된 데이터에 대한 RMSE: {np.sqrt(mse_processed)}")

# 원본 데이터에 대한 RMSE 계산
mse_original = mean_squared_error(y_feed_test, y_feed_pred)
print(f"원본 데이터에 대한 RMSE: {np.sqrt(mse_original)}")

# 결과 시각화
plt.figure(figsize=(40, 20))

# 원본 데이터 그래프
plt.subplot(1, 2, 1)
plt.plot(y_feed_test, label='Actual Feed Pressure (Original)', color='violet', alpha=0.5)
plt.plot(y_feed_pred, label='Predicted Feed Pressure (Original)', color='dodgerblue', alpha=0.5)
plt.title('Feed Pressure Prediction (Original)')
plt.legend()

# 처리된 데이터 그래프
plt.subplot(1, 2, 2)
plt.plot(y_feed_test_rolled_trimmed, label='Actual Feed Pressure (Smoothed & Interpolated)', color='red', alpha=0.5)
plt.plot(y_feed_pred_rolled_trimmed, label='Predicted Feed Pressure (Smoothed & Interpolated)', color='green', alpha=0.5)
plt.title('Feed Pressure Prediction (Smoothed & Interpolated)')
plt.legend()

plt.savefig('feed_pressure_comparison.png')
plt.show()
