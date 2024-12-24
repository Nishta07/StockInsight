import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from datetime import datetime, timedelta

# Define the stock symbol and adjust the date to the last 60 days
symbol = 'AAPL'  # Major tech stock (Apple)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)  # 60 days ago

# Convert dates to string format for yfinance
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Fetch 1-minute intraday data (within the last 60 days)
try:
    data = yf.download(symbol, interval='1m', start=start_date_str, end=end_date_str)

    # Check if data was downloaded successfully
    if data.empty:
        raise ValueError("No data was fetched for the given date range.")

    print(f"Data successfully fetched for {symbol}.")
except Exception as e:
    print(f"Error fetching data: {e}")
    data = pd.DataFrame()  # Set data as an empty DataFrame in case of error

# Proceed only if data is fetched successfully
if not data.empty:
    # Clean and preprocess the raw data
    # 1. Handle missing values
    data.ffill(inplace=True)  # Forward fill missing data


    # 2. Handle outliers (removing extreme values based on z-score)
    def remove_outliers(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]


    data = remove_outliers(data, 'Close')

    # 3. Calculate the rolling volatility (using a 20-period rolling window)
    data['rolling_volatility'] = data['Close'].rolling(window=20).std()

    # 4. Calculate the Volume-Weighted Average Price (VWAP)
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # 5. Calculate Moving Averages (20-period and 50-period)
    data['ma_20'] = data['Close'].rolling(window=20).mean()
    data['ma_50'] = data['Close'].rolling(window=50).mean()

    # Visualize the data: Closing Price, VWAP, and Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['vwap'], label='VWAP', color='green', linestyle='--')
    plt.plot(data['ma_20'], label='MA 20', color='red', linestyle='--')
    plt.plot(data['ma_50'], label='MA 50', color='orange', linestyle='--')
    plt.title(f'{symbol} - Price, VWAP, and Moving Averages')
    plt.legend(loc='upper left')

    # Save the plot as a PNG file
    plt.savefig('price_vwap_ma_plot.png')

    # Visualize the Rolling Volatility
    plt.figure(figsize=(12, 6))
    plt.plot(data['rolling_volatility'], label='Rolling Volatility (20 periods)', color='purple')
    plt.title(f'{symbol} - Rolling Volatility')
    plt.legend(loc='upper left')

    # Save the rolling volatility plot as a PNG file
    plt.savefig('rolling_volatility_plot.png')

    # 6. Identify Unusual Trading Patterns:
    # Define unusual trading behavior as when the price deviates significantly from the rolling mean
    data['price_deviation'] = abs(data['Close'] - data['Close'].rolling(window=20).mean())
    unusual_pattern = data[data['price_deviation'] > data['price_deviation'].quantile(0.95)]  # 95th percentile

    # Visualize unusual patterns
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.scatter(unusual_pattern.index, unusual_pattern['Close'], color='red', label='Unusual Patterns')
    plt.title(f'{symbol} - Unusual Trading Patterns')
    plt.legend(loc='upper left')

    # Save the unusual patterns plot as a PNG file
    plt.savefig('unusual_patterns_plot.png')

    # Statistical Analysis
    # 1. Calculate key statistical measures for the 'Close' price
    mean_price = data['Close'].mean()
    std_price = data['Close'].std()

    # Ensure skewness and kurtosis are calculated only with valid data
    skewness_price = skew(data['Close'].dropna()) if len(data['Close'].dropna()) > 2 else np.nan
    kurtosis_price = kurtosis(data['Close'].dropna()) if len(data['Close'].dropna()) > 2 else np.nan

    # Now print the values correctly
    print(f'Mean Price: {mean_price:}')
    print(f'Standard Deviation: {std_price:}')
    print(f'Skewness: {skewness_price:}' if not np.isnan(skewness_price) else "Skewness: Data insufficient")
    print(f'Kurtosis: {kurtosis_price:}' if not np.isnan(kurtosis_price) else "Kurtosis: Data insufficient")

    # 2. Correlation between Volume and Price Change (interesting pattern)
    data['price_change'] = data['Close'].diff()
    plt.figure(figsize=(12, 6))
    sns.heatmap(data[['Volume', 'price_change']].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Volume and Price Change')

    # Save the correlation plot as a PNG file
    plt.savefig('volume_price_correlation_plot.png')

    # Investigating one interesting pattern (e.g., Volume spikes and price moves)
    # Identify periods with unusually high volume
    high_volume_periods = data[data['Volume'] > data['Volume'].quantile(0.95)]

    # Visualize high volume periods and price changes
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.scatter(high_volume_periods.index, high_volume_periods['Close'], color='red', label='High Volume Periods')
    plt.title(f'{symbol} - High Volume Periods with Price Changes')
    plt.legend(loc='upper left')

    # Save the high volume periods plot as a PNG file
    plt.savefig('high_volume_price_change_plot.png')

else:
    print("No data available for analysis.")
