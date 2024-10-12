import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
import torch

# 데이터 로드
file_path = '(0923-0930)시간-처리수전기전도도-유입압력.csv'
data = pd.read_csv(file_path, encoding='cp949')

# 데이터 전처리
data['Time'] = pd.to_datetime(data['Time'], format='%Y %m %d %H:%M:%S')
data = data.sort_values('Time')

# 분 단위의 시간 인덱스 생성
data['time_idx'] = (data['Time'] - data['Time'].min()).dt.total_seconds() // 60  # 분 단위 인덱스 생성
data['time_idx'] = data['time_idx'].astype(int)  # 정수형으로 변환

# NaN 값 제거 및 인덱스 재설정
data = data.dropna().reset_index(drop=True)

# 그룹 ID 추가 (모든 데이터를 하나의 그룹으로)
data['group_id'] = 0

print(data.head())

# 변수 이름 설정
target_variable = "Permeate Conductivity"  # 처리수 전기전도도
target_variable_2 = "Feed Pressure"  # 유입 압력

# TFT에 필요한 시계열 데이터셋 생성
max_encoder_length = 90  # 과거 데이터 길이 (90분)
max_prediction_length = 120  # 예측할 길이 (120분)
training_cutoff = data['time_idx'].max() - max_prediction_length * 10

# 학습 데이터셋 정의
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=[target_variable, target_variable_2],  # 다중 타겟 변수
    group_ids=["group_id"],
    min_encoder_length=30,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[target_variable, target_variable_2],
    target_normalizer=MultiNormalizer([GroupNormalizer(groups=["group_id"]), GroupNormalizer(groups=["group_id"])]),
    allow_missing_timesteps=True,  # 불규칙한 시간 간격 허용
)

# 검증 데이터셋 정의
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# 데이터 로더 생성
batch_size = 64 
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# 모델 정의
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=RMSE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# LightningModule로 감싸기
class TFTLightningModule(LightningModule):
    def __init__(self, tft):
        super().__init__()
        self.model = tft

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat["prediction"]
        loss = self.model.loss(y_hat, y)
        self.log("train_loss", loss, batch_size=len(x['encoder_cont'].squeeze()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat["prediction"]
        loss = self.model.loss(y_hat, y)
        self.log("val_loss", loss, batch_size=len(x['encoder_cont'].squeeze()))
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers()

# TFT를 감싼 LightningModule 생성
tft_module = TFTLightningModule(tft)

# 모델 학습
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

trainer = Trainer(
    max_epochs=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
)

trainer.fit(
    tft_module,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 예측 수행
predictions = tft_module.model.predict(val_dataloader)

# predictions는 리스트 형태이므로 NumPy 배열로 변환
predictions = np.array(predictions)
print("예측 결과 ", predictions)
print("예측한 개수 ", predictions.shape)
print("검증 데이터 크기 ", len(val_dataloader.dataset))

print("Training cutoff:", training_cutoff)
print("Max time_idx in training data:", data['time_idx'].max())
print("Training data size:", len(data[data['time_idx'] <= training_cutoff]))
print("Validation data size:", len(data[data['time_idx'] > training_cutoff]))


# 예측 결과가 3차원인 경우, 첫 번째 축을 제거하여 2차원으로 변환
# (batch_size, prediction_length, target_dim)을 (batch_size * prediction_length, target_dim)으로 변환
if predictions.ndim == 3:
    predictions = predictions.reshape(-1, predictions.shape[-1])

# 이제 predictions는 2차원 배열입니다.
# 예측 결과를 각각 'Predicted Permeate Conductivity'와 'Predicted Feed Pressure'로 분리
predicted_permeate_conductivity = predictions[0]  # 첫 번째 목표 변수 (예측된 모든 값)
predicted_feed_pressure = predictions[1]  # 두 번째 목표 변수 (예측된 모든 값)

print(predicted_permeate_conductivity)
print(predicted_feed_pressure)

# 기존 데이터에서 예측 시점 이후의 'Time', 'time_idx', 'group_id' 생성
new_time_idx = np.arange(data['time_idx'].max() + 1, data['time_idx'].max() + 1 + len(predicted_permeate_conductivity))
new_times = pd.date_range(start=data['Time'].max() + pd.Timedelta(minutes=1), periods=len(predicted_permeate_conductivity), freq='min')

# 새로운 예측 데이터를 DataFrame으로 생성
predicted_data = pd.DataFrame({
    'Time': new_times,
    'time_idx': new_time_idx,
    'group_id': 0,
    'Permeate Conductivity': predicted_permeate_conductivity,
    'Feed Pressure': predicted_feed_pressure
})

# 기존 데이터와 새로운 예측 데이터를 결합
combined_data = pd.concat([data, predicted_data], ignore_index=True)

# 예측 결과를 엑셀 파일로 저장
output_file_path = "combined_predicted_results.xlsx"
combined_data.to_excel(output_file_path, index=False)
print(f"Combined data with predictions saved to {output_file_path}")

# combined_data의 상위 5개 행 출력
print(combined_data.head())

# -------- 그래프 그리기 -------- #
plt.figure(figsize=(12, 6))

# 원래 데이터의 처리수 전기전도도 그래프 (파란색)
plt.plot(data['Time'], data['Permeate Conductivity'], label="Original Permeate Conductivity", color='blue')

# 예측 데이터의 처리수 전기전도도 그래프 (빨간색)
plt.plot(predicted_data['Time'], predicted_data['Permeate Conductivity'], label="Predicted Permeate Conductivity", color='red')

# 원래 데이터의 유입 압력 그래프 (초록색)
plt.plot(data['Time'], data['Feed Pressure'], label="Original Feed Pressure", color='green')

# 예측 데이터의 유입 압력 그래프 (주황색)
plt.plot(predicted_data['Time'], predicted_data['Feed Pressure'], label="Predicted Feed Pressure", color='orange')

# 그래프 제목 및 라벨 설정
plt.title("Permeate Conductivity and Feed Pressure: Original vs Predicted")
plt.xlabel("Time")
plt.ylabel("Value")

# 범례 추가
plt.legend()

# 그래프 출력
plt.show()