import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_outliers_iqr(df, column):
    """Remove outliers based on the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def main():
    # Load the CSV file
    file_path = 'four-cycle-16months-feed-pressure-10min-v2.csv'
    data = pd.read_csv(file_path)
    
    # Convert 'Time' column to datetime for easier manipulation
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Create a rolling window of size 10 and calculate the average
    data['rolling_mean'] = data['feed_pressure'].rolling(window=10).mean()

    # Replace values below the rolling mean with the maximum of the last 5 values
    for i in range(len(data)):
        if data.loc[i, 'feed_pressure'] < data.loc[i, 'rolling_mean']:
            data.loc[i, 'feed_pressure'] = data.loc[max(0, i-5):i, 'feed_pressure'].max()

    # Apply IQR-based outlier removal
    data_cleaned = remove_outliers_iqr(data, 'feed_pressure')
    
    # Apply smoothing using a rolling mean with a window size of 5 for further smoothing
    data_cleaned['smoothed_feed_pressure'] = data_cleaned['feed_pressure'].rolling(window=5, min_periods=1).mean()

    # Save the preprocessed data to a new CSV file
    data_cleaned.to_csv('preprocessing_final.csv', index=False)

    # Plot the original, modified, and smoothed data
    plt.figure(figsize=(14, 7))
    plt.plot(data_cleaned['Time'], data_cleaned['feed_pressure'], label='Modified Feed Pressure', color='orange', alpha=0.6)
    plt.plot(data_cleaned['Time'], data_cleaned['smoothed_feed_pressure'], label='Smoothed Feed Pressure', color='blue', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Feed Pressure')
    plt.title('Feed Pressure with IQR Outlier Removal and Smoothing')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
