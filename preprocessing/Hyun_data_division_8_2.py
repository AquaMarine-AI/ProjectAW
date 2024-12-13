# 다시 코드 실행 - 필요한 라이브러리 임포트
import pandas as pd

# 파일 로드
file_path = 'final_final_preprocessed_with_updated_time_format.csv'
data = pd.read_csv(file_path)

# 데이터 분할: 8:2 비율로 분할
train_size = int(len(data) * 0.8)

train_data = data.iloc[:train_size]  # 앞부분 80%
test_data = data.iloc[train_size:]  # 뒷부분 20%

# 파일 저장
train_data_path = 'train_data.csv'
test_data_path = 'test_data.csv'

train_data.to_csv(train_data_path, index=False, encoding='utf-8')
test_data.to_csv(test_data_path, index=False, encoding='utf-8')

train_data_path, test_data_path
