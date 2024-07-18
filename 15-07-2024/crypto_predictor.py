import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import schedule
import time
from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = '06QFYV9IAUPKCDSU'

# Initialize Alpha Vantage TimeSeries
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

def fetch_ohlcv(symbol, interval='1min'):
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
    data.reset_index(inplace=True)
    data.rename(columns={'date': 'timestamp', '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}, inplace=True)
    return data

def plot_chart(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data['timestamp'], data['close'], label='Close Price', color='b', marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def fetch_and_plot(symbol):
    data = fetch_ohlcv(symbol)
    directory = os.path.join('D:\\Internship', symbol.replace('/', '_'))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f"{symbol.replace('/', '_')}_1d.png")
    plot_chart(data, f"{symbol} - Last 1 Day", filename)
    csv_filename = os.path.join(directory, f"{symbol.replace('/', '_')}_1d.csv")
    data.to_csv(csv_filename, index=False)
    print(f"Chart and data for {symbol} saved as {filename} and {csv_filename}")

def train_model(data):
    X = data[['open', 'high', 'low', 'close', 'volume']]
    y = data['close'].shift(-1).dropna()
    X = X.iloc[:-1, :]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_next_day(symbol):
    data = fetch_ohlcv(symbol, interval='1min')
    model = train_model(data)
    last_row = data.iloc[-1][['open', 'high', 'low', 'close', 'volume']].values.reshape(1, -1)
    prediction = model.predict(last_row)
    
    # Adding the prediction to the data for plotting
    next_day = data['timestamp'].iloc[-1] + timedelta(minutes=1)
    data = data.append({'timestamp': next_day, 'close': prediction[0]}, ignore_index=True)

    filename = f'D:\\Internship\\{symbol.replace("/", "_")}_prediction.png'
    plot_chart(data, f"{symbol} - Prediction for Next Day", filename)
    print(f"Predicted chart for {symbol} saved as {filename}")

def job(symbols):
    for stock in symbols:
        fetch_and_plot(stock)
        predict_next_day(stock)

# Schedule the job to run at 4 AM EST
def schedule_jobs(stocks):
    schedule.every().day.at("04:00").do(job, stocks)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    input_stocks = input("Enter the stock symbols separated by commas: ").split(',')
    input_stocks = [stock.strip() for stock in input_stocks]
    schedule_jobs(input_stocks)
    job(input_stocks)  # Run the job initially
