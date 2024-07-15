import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import schedule
import time
import os

def simulate_data():
    dates = pd.date_range(end=datetime.now(), periods=10).tolist()
    return pd.DataFrame({
        'timestamp': dates,
        'open': [30000 + i * 50 for i in range(10)],
        'high': [31000 + i * 50 for i in range(10)],
        'low': [29500 + i * 50 for i in range(10)],
        'close': [30500 + i * 50 for i in range(10)],
        'volume': [1000 + i * 10 for i in range(10)]
    })

def preprocess_data(data):
    data['target'] = data['close'].shift(-1)
    return data.dropna()

def train_model(data):
    X = data[['open', 'high', 'low', 'close', 'volume']]
    y = data['target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def make_prediction(model, data):
    X = data[['open', 'high', 'low', 'close', 'volume']].iloc[-1].values.reshape(1, -1)
    return model.predict(pd.DataFrame(X, columns=['open', 'high', 'low', 'close', 'volume']))[0]

def plot_prediction(data, prediction, symbol, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data['timestamp'], data['close'], label='Actual Close Price', color='b', marker='o')
    plt.axvline(x=data['timestamp'].iloc[-1], color='r', linestyle='--', label='Prediction Point')
    plt.scatter(data['timestamp'].iloc[-1] + timedelta(days=1), prediction, color='r', label='Predicted Close Price', marker='x')
    plt.title(f"{symbol} - Close Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def fetch_and_predict(symbol):
    data = simulate_data()
    data = preprocess_data(data)
    
    if len(data) < 9:
        print(f"Not enough data to train model for {symbol}")
        return
    
    model = train_model(data)
    prediction = make_prediction(model, data)
    
    output_dir = os.path.join('D:\\Internship', symbol.replace('/', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"{symbol.replace('/', '_')}_prediction.png")
    plot_prediction(data, prediction, symbol, filename)
    
    print(f"Predicted closing price for {symbol} on {datetime.now() + timedelta(days=1)}: {prediction}")
    print(f"Prediction chart saved as {filename}")

def schedule_prediction():
    for asset in ['BTC/USDT']:
        fetch_and_predict(asset)

schedule.every().day.at("09:00").do(schedule_prediction)

while True:
    schedule.run_pending()
    time.sleep(1)
