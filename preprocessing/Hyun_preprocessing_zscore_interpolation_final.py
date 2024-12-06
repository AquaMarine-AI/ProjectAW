# 앞, 뒤를 제외한 띄엄띄엄 데이터 있는 부분 모두 제거 후,
# (선형) 보간된 작업 및 작업된 부분에만 노이즈를 추가
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 새로 업로드된 파일 경로
file_path = 'preprocessing_final_zscore_replaced.csv'

# 파일 로드
data = pd.read_csv(file_path, header=None)

# 1행을 X축과 Y축의 이름으로 설정
x_axis_label = data.iloc[0, 0]
y_axis_label = data.iloc[0, 1]

# 2행부터 데이터를 사용
data = data[1:]
data.columns = [x_axis_label, y_axis_label]

# X축을 datetime 형식으로 변환
data[x_axis_label] = pd.to_datetime(data[x_axis_label], errors='coerce')
data[y_axis_label] = pd.to_numeric(data[y_axis_label], errors='coerce')

# 보간 전 원본 데이터를 복사
original_data = data.copy()

# 선형 보간법을 사용하여 누락된 값을 대체
data[y_axis_label] = data[y_axis_label].interpolate(method='linear')

# 보간된 값 식별
is_interpolated = original_data[y_axis_label].isna()

# 보간된 값에만 노이즈 추가
noise_lower = np.random.uniform(-0.5, 0, size=len(data[y_axis_label]))
noise_upper = np.random.uniform(0, 1, size=len(data[y_axis_label]))

for i in range(len(data[y_axis_label])):
    if is_interpolated.iloc[i]:  # 보간된 값에만 처리
        if np.random.rand() < 0.5:  # 50% 확률로 아래로 노이즈
            data.iloc[i, 1] += noise_lower[i]
        else:  # 50% 확률로 위로 노이즈
            data.iloc[i, 1] += noise_upper[i]

# 전처리된 데이터를 CSV로 저장
output_file_path = 'final_final_preprocessed_with_noise.csv'
data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"전처리된 데이터가 다음 경로에 저장되었습니다: {output_file_path}")


# 보간 및 노이즈 추가 결과를 시각화
plt.figure(figsize=(12, 6))
plt.plot(data[x_axis_label], data[y_axis_label], label=f'With Noise on Interpolated {y_axis_label}', linewidth=1, alpha=0.7)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(f'{y_axis_label} with Directional Noise on Interpolated Values vs {x_axis_label}')
plt.xticks(rotation=45)  # X축 레이블 회전
plt.legend()
plt.tight_layout()
plt.show()
