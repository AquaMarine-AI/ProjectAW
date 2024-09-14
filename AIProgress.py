from AIProgress_ui import Ui_Dialog
from PySide6.QtWidgets import QDialog, QWidget, QVBoxLayout
from PySide6.QtCore import QThread, QSettings, Signal, Qt
import pandas as pd
from tensorflow import keras

class AIProgress(QDialog):
    def __init__(self, AIData : pd.DataFrame):
        super().__init__()

        """Fixed works of Runtime"""
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)       # self = parent of Ui_AIProgress

        """Dynamic works of Runtime"""
        self.ui.cancelButton.clicked.connect(self.cancel_button_function)
        self.AIThread = AIWorkingThread(AIData)
        # call dialog accept when AIThread is nomally finished
        self.AIThread.finished.connect(self.accept)
        # connect AIThread to progressBar
        self.AIThread.progress.connect(self.update_progress_ui)
        
        self.AIThread.start()
        
        
    def cancel_button_function(self):
        print("click!")
        self.reject()

    #predefined(override) function in Ui_AIProgress
    def reject(self) -> None:
        print("AI Progress Rejected")
        if self.AIThread.isRunning():
            self.AIThread.terminate()
        return super().reject()
    
    #predefined(override) function in Ui_AIProgress
    def accept(self) -> None:
        print("AI Progress Finished normally")
        return super().accept()

    def update_progress_ui(self, value, message):
        self.ui.progressBar.setValue(value)
        self.ui.progressLabel.setText(message)





    
class AIWorkingThread(QThread):
    progress = Signal(int, str)

    def __init__(self, data):
        super().__init__()
        self.data : pd.DataFrame = data
    
    def run(self):
        from ast import literal_eval
        
        self.progress.emit(0, "AI 인풋값 로딩중")

        aiSettings = QSettings("JS", "WaterAnalysis")
        imageDir = aiSettings.value("imageDirectory")
        input : list = literal_eval(aiSettings.value("input"))
        output : list = literal_eval(aiSettings.value("output"))
        sequence_length = int(aiSettings.value("sequenceLength"))
        units = int(aiSettings.value("units"))
        epochs = int(aiSettings.value("epochs"))
        batchSize = int(aiSettings.value("batchSize"))
        outlierCheck = (aiSettings.value("outlierCheck") == "True")
        #aiSettings.value("outlierColumn", "feed pressure")
        outlierWeight = float(aiSettings.value("outlierWeight"))
        verbose = int(aiSettings.value("verbose"))

        class FitCallback(keras.callbacks.Callback):
            def __init__(self, thread, epochs):
                super().__init__()
                self.thread : AIWorkingThread = thread
                self.epochs = epochs

            def on_epoch_begin(self, epoch, logs=None):
                self.thread.progress.emit(int(epoch/100 * 65) + 25, f"AI 학습 중... {epoch} / {self.epochs}")


        """AI CODE START"""
        self.progress.emit(2, "AI 라이브러리 로딩중...")
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        from scipy.interpolate import interp1d
        
        self.progress.emit(10, "데이터 전처리 중...")

        # .csv 파일로 저장할 경로
        csv_path = r"pressure_data.csv"
        # .csv 파일로 저장
        self.data.to_csv(csv_path, index=False)

        def get_outlier(df=None, column=None, weight=outlierWeight):
            quantile_25 = np.percentile(df[column].values, 25)
            quantile_75 = np.percentile(df[column].values, 75)

            IQR = quantile_75 - quantile_25
            IQR_weight = IQR * weight
        
            lowest = quantile_25 - IQR_weight
            highest = quantile_75 + IQR_weight
        
            outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
            return outlier_idx

        if outlierCheck == True:
            outlier_idx = get_outlier(df=self.data, column=input[0], weight=outlierWeight)
            self.data.drop(outlier_idx, axis=0, inplace=True)
        
### NOT designed for multiple input/output ###
        X_feed = self.data[input[0]].values
        y_feed = self.data[output[0]].values

        # Min-Max 스케일링
        scaler = MinMaxScaler()
        X_feed = scaler.fit_transform(X_feed.reshape(-1, 1))
        y_feed = scaler.fit_transform(y_feed.reshape(-1, 1))

        self.progress.emit(15, "데이터 전처리 중...")

        # 데이터 시퀀스 생성 함수
        def create_sequences(data, sequence_length):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:i + sequence_length])
                y.append(data[i + sequence_length])
            return np.array(X), np.array(y)

        # Feed pressure 데이터 시퀀스 생성
        X_feed_seq, y_feed_seq = create_sequences(X_feed, sequence_length)

        # 데이터 분할
        X_feed_train, X_feed_test, y_feed_train, y_feed_test = train_test_split(X_feed_seq, y_feed_seq, test_size=0.2, random_state=42)
        
        self.progress.emit(20, "AI 모델 구성 중...")

        # LSTM 모델 구성
        model_feed = keras.models.Sequential()
        model_feed.add(keras.layers.LSTM(units=units, activation='relu', input_shape=(sequence_length, 1)))
        model_feed.add(keras.layers.Dense(1))
        model_feed.compile(optimizer='adam', loss='mean_squared_error')

        self.progress.emit(25, "AI 학습 중...")

        # 모델 훈련 -> 극히 오래 걸리는 부분
        model_feed.fit(X_feed_train, y_feed_train, epochs=epochs, batch_size=batchSize, verbose=verbose, callbacks=[FitCallback(self, epochs)])
        
        self.progress.emit(90, "AI 모델 예측 중...")
        
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

        self.progress.emit(95, "AI 통계 작성 중...")

        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure

        # 결과 시각화
        self.graph_widget = QWidget()
        self.graph_widget.resize(1280, 720)
        self.graph_widget.setWindowTitle('Result Graph')

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        vertical_layout = QVBoxLayout(self.graph_widget)
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(NavigationToolbar(self.canvas, self.graph_widget))
        vertical_layout.setStretch(1, 0)

        # 원본 데이터 그래프
        ax = self.canvas.figure.add_subplot(1, 2, 1)
        ax.plot(y_feed_test, label='Actual Feed Pressure (Original)', color='violet', alpha=0.5)
        ax.plot(y_feed_pred, label='Predicted Feed Pressure (Original)', color='dodgerblue', alpha=0.5)
        ax.set_title('Feed Pressure Prediction (Original)')
        ax.legend()

        ax = self.canvas.figure.add_subplot(1, 2, 2)
        ax.plot(y_feed_test_rolled_trimmed, label='Actual Feed Pressure (Smoothed & Interpolated)', color='red', alpha=0.5)
        ax.plot(y_feed_pred_rolled_trimmed, label='Predicted Feed Pressure (Smoothed & Interpolated)', color='green', alpha=0.5)
        ax.set_title('Feed Pressure Prediction (Smoothed & Interpolated)')
        ax.legend()

        self.canvas.figure.savefig(imageDir + '/feed_pressure_comparison.png') 
        self.progress.emit(100, "완료")
        
        """AI CODE END"""
        
        self.finished.emit()

    