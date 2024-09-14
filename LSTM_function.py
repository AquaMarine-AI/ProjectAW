#LSTM
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from typing import List
import matplotlib.pyplot as plt


def LSTM_train_and_get_result(
    csv_path : str,             # 학습 데이터 csv 파일 경로
    image_dir : str,           # 학습 결과 그래프 이미지 저장 경로
    input : List[str],          # 학습 입력으로 쓰일 데이터의 컬럼명
    output : List[str],         # 학습 출력으로 쓰일 데이터의 컬럼명
    test_size : float = 0.2,    # 테스트 데이터 비율
    random_state : int = 42,    # 랜덤 시드
    sequence_length : int = 10, # LSTM 입력 시퀀스 길이
    units : int = 50,           # 뉴런 개수                     (많으면 더 복잡한 패턴 학습 가능하나 계산 시간이 길어지고 과적합 위험)
    epochs : int = 50,          # 학습 반복 횟수                 (많으면 정확성 및 과적합 증가)
    batch_size : int = 32,      # 한 번에 학습할 데이터 개수       (크면 GPU 계산 효율 증가, 불안정성 감소 그러나 메모리 사용량 증가하고 학습 속도 하락, 2^n으로 보통 잡음)
    min_max_scaler : bool = True,# Min-Max 스케일링 여부
    outlier_check : bool = False, # 이상치 검출 여부
    outlier_column : str = None, # 이상치 검출할 컬럼명
    outlier_weight : float = 1.5,# 이상치 검출 가중치             (크면 이상치로 절사될 데이터 개수 감소)
    verbose : int = 1,           # 학습 과정 출력 여부
):
    # csv 데이터 로드
    data : pd.DataFrame = pd.read_csv(csv_path)
   
    if outlier_check is True:
        quantile_25 = np.percentile(data[outlier_column].values, 25)
        quantile_75 = np.percentile(data[outlier_column].values, 75)

        IQR = quantile_75 - quantile_25
        IQR_weight = IQR * outlier_weight

        lowest = quantile_25 - IQR_weight
        highest = quantile_75 + IQR_weight

        outlier_idx = data[outlier_column][ (data[outlier_column] < lowest) | (data[outlier_column] > highest)].index
        data.drop(outlier_idx, axis=0, inplace=True)

    #학습 데이터의 입력/출력 생성
    input_feed = np.hstack([data[col].values for col in input])
    output_feed = np.hstack([data[col].values for col in output])

    if min_max_scaler is True:
        scaler = MinMaxScaler()
        input_feed = scaler.fit_transform(input_feed.reshape(-1, len(input)))
        output_feed = scaler.fit_transform(output_feed.reshape(-1, len(output)))

    # 데이터 시퀀스 생성 (입력 데이터, 출력 데이터, 시퀀스 길이 -> Ex) 30일간의 입력데이터로 31일째의 출력 데이터 예측하게 데이터 형태 만들기)
    input_feed_seq, output_feed_seq = [], []
    for i in range(len(input_feed) - sequence_length):
        input_feed_seq.append(input_feed[i:i+sequence_length])
        output_feed_seq.append(output_feed[i+sequence_length])

    input_feed_seq = np.array(input_feed_seq)
    output_feed_seq = np.array(output_feed_seq)

    # 학습/테스트 데이터 분할
    input_feed_train, input_feed_test, output_feed_train, output_feed_test = train_test_split(input_feed_seq, output_feed_seq, test_size=test_size, random_state=random_state)

    # LSTM 모델 구성
    model_feed = tf.keras.models.Sequential()
    model_feed.add(tf.keras.layers.LSTM(units, activation='relu', input_shape=(sequence_length, len(input))))
    model_feed.add(tf.keras.layers.Dense(1))
    model_feed.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 훈련
    model_feed.fit(input_feed_train, output_feed_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # 결과 예측
    output_feed_pred = model_feed.predict(input_feed_test)

    #성능 평가(mse)
    output_feed_test = scaler.inverse_transform(output_feed_test)  # 스케일링 해제
    output_feed_pred = scaler.inverse_transform(output_feed_pred)
    mse = mean_squared_error(output_feed_test, output_feed_pred)
    print(f"(RMSE): {np.sqrt(mse)}")
    mse_string = f"{mse:011.8f}"

    # 결과 시각화
    plt.figure(figsize=(40, 20))
    plt.plot(output_feed_test, label='Actual Feed Pressure', color = 'violet')
    plt.plot(output_feed_pred, label='Predicted Feed Pressure', color = 'dodgerblue')
    plt.legend()
    plt.title('input : {}, sequence_length : {}\n\
               units : {}, epochs : {}, batch_size : {}\n\
               outlier_check : {}, {}'
              .format(", ".join([str(item) for item in input]), sequence_length, units, epochs, batch_size, outlier_check, outlier_weight)
              , fontsize=20)
    
    idx = ["./AIDataSet/data1.csv", "./AIDataSet/data2.csv", "./AIDataSet/data3.csv", "./AIDataSet/data3Out0.csv"].index(csv_path)
    target = ['data1_', 'data2_', 'data3_', 'data4_']
    plt.savefig(f"{image_dir}/{target[idx]}{mse_string}.png")
    plt.close() # 그래프 초기화