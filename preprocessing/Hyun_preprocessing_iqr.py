import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 슬라이딩 윈도우 크기 10 설정 후, 평균 아래 값들을 직전 5개 데이터 중 최대값으로 대체
# 2. iqr 이상치 탐지 기법 적용
#   이상치 식별 시 직전 3개 데이터 중 최대값으로 대체
# 3. data smoothing 적용


def replace_outliers_iqr(df, column):
    """Replace outliers based on the IQR method with the maximum of the last 3 values."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Iterate through the dataframe and replace outliers with the maximum of the last 3 values
    for i in range(len(df)):
        if df.loc[i, column] < lower_bound or df.loc[i, column] > upper_bound:
            # Find the maximum of the last 3 values, ensuring not to go out of bounds
            if i >= 3:
                df.loc[i, column] = df.loc[i-3:i-1, column].max()
            else:
                df.loc[i, column] = df.loc[0:i, column].max()
    return df

def main():
    # Load the CSV file
    file_path = 'Hyun_augumented_data_4cycles_16months_feed_pressure_10min_final.csv'
    data = pd.read_csv(file_path)
    
    # Convert 'Time' column to datetime for easier manipulation
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Create a rolling window of size 10 and calculate the average
    data['rolling_mean'] = data['feed_pressure'].rolling(window=10).mean()

    # Replace values below the rolling mean with the maximum of the last 5 values
    for i in range(len(data)):
        if data.loc[i, 'feed_pressure'] < data.loc[i, 'rolling_mean']:
            data.loc[i, 'feed_pressure'] = data.loc[max(0, i-5):i, 'feed_pressure'].max()

    # Replace IQR-based outliers with the maximum of the last 3 values
    data_cleaned = replace_outliers_iqr(data, 'feed_pressure')
    
    # Apply smoothing using a rolling mean with a window size of 5 for further smoothing
    data_cleaned['smoothed_feed_pressure'] = data_cleaned['feed_pressure'].rolling(window=5, min_periods=1).mean()

    # Save the preprocessed data to a new CSV file
    data_cleaned.to_csv('Hyun_preprocessed_data_iqr_replaced.csv', index=False)

    # Plot the original, modified, and smoothed data
    plt.figure(figsize=(14, 7))
    plt.plot(data_cleaned['Time'], data_cleaned['feed_pressure'], label='Modified Feed Pressure', color='orange', alpha=0.6)
    plt.plot(data_cleaned['Time'], data_cleaned['smoothed_feed_pressure'], label='Smoothed Feed Pressure', color='blue', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Feed Pressure')
    plt.title('Feed Pressure with IQR Outlier Replacement and Smoothing')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
