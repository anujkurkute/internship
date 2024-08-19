import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time
import logging
import schedule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a list of 15 crypto assets
crypto_assets = [
    'bitcoin', 'ethereum', 'cardano', 'solana', 'ripple',
    'polkadot', 'dogecoin', 'litecoin', 'chainlink', 'uniswap',
    'binancecoin', 'stellar', 'vechain', 'tron', 'avalanche'
]

# Function to fetch daily price charts for the last 365 days with retry logic
def fetch_daily_chart(asset, retries=5):
    url = f'https://api.coingecko.com/api/v3/coins/{asset}/market_chart?vs_currency=usd&days=365&interval=daily'
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'prices' in data:
                    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price']).set_index('timestamp')
                    return df
                else:
                    logging.warning(f"Unexpected data format for {asset}. Data received: {data}")
                    return pd.DataFrame(columns=['timestamp', 'price'])
            elif response.status_code == 429:
                logging.warning(f"Rate limit exceeded for {asset}. Retrying {attempt + 1}/{retries}...")
                time.sleep(2 ** attempt + 5)
            else:
                logging.error(f"Failed to fetch data for {asset}. HTTP Status Code: {response.status_code}. Response: {response.text}")
                return pd.DataFrame(columns=['timestamp', 'price'])
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {asset}. Error: {e}")
            time.sleep(2 ** attempt + 5)
    logging.error(f"Failed to fetch data for {asset} after {retries} retries.")
    return pd.DataFrame(columns=['timestamp', 'price'])

# Function to fetch current price with retry logic
def fetch_current_price(asset, retries=5):
    url = f'https://api.coingecko.com/api/v3/simple/price?ids={asset}&vs_currencies=usd'
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get(asset, {}).get('usd', None)
            elif response.status_code == 429:
                logging.warning(f"Rate limit exceeded for {asset}. Retrying {attempt + 1}/{retries}...")
                time.sleep(2 ** attempt + 5)
            else:
                logging.error(f"Failed to fetch current price for {asset}. HTTP Status Code: {response.status_code}. Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {asset}. Error: {e}")
            time.sleep(2 ** attempt + 5)
    logging.error(f"Failed to fetch current price for {asset} after {retries} retries.")
    return None

# Function to calculate moving average
def calculate_moving_average(data, window=5):
    return data['price'].rolling(window=window).mean()

# Basic reinforcement learning agent for adjusting parameters
class SimpleReinforcementLearning:
    def __init__(self, learning_rate=0.1):
        self.accuracy_history = []
        self.learning_rate = learning_rate

    def update(self, accuracy):
        self.accuracy_history.append(accuracy)
        self.adjust_parameters()

    def adjust_parameters(self):
        if len(self.accuracy_history) > 10:
            recent_accuracy = np.mean(self.accuracy_history[-10:])
            if recent_accuracy < 0.5:
                self.learning_rate += 0.01
            else:
                self.learning_rate -= 0.01

rl_agent = SimpleReinforcementLearning()

# Match today's pattern with closest historical pattern using weighted Euclidean distance
def match_pattern(current_pattern, pattern_db, rl_agent, asset, current_prices):
    patterns, next_prices, accuracies = zip(*pattern_db)
    distances = euclidean_distances([current_pattern], patterns)
    weighted_distances = distances / np.array(accuracies)
    closest_pattern_idx = weighted_distances.argmin()
    predicted_price = next_prices[closest_pattern_idx]

    # Calculate accuracy for this prediction
    actual_price = current_prices.get(asset, None)
    if actual_price is not None:
        accuracy = 1 / abs(predicted_price - actual_price)
        rl_agent.update(accuracy)
    
    return predicted_price

def main():
    # Fetch and store data for all assets
    crypto_data = {}
    for asset in crypto_assets:
        logging.info(f"Fetching data for {asset}...")
        crypto_data[asset] = fetch_daily_chart(asset)
        time.sleep(3)

    current_prices = {}
    for asset in crypto_assets:
        logging.info(f"Fetching current price for {asset}...")
        current_prices[asset] = fetch_current_price(asset)
        time.sleep(3)

    # Proceed only if data is successfully fetched
    if any(not data.empty for data in crypto_data.values()):
        # Detect patterns and connect them to subsequent patterns
        pattern_db = []
        for asset, data in crypto_data.items():
            if not data.empty:
                data['moving_avg'] = calculate_moving_average(data)
                for i in range(len(data) - 6):
                    current_pattern = data['price'].iloc[i:i + 5].values
                    next_day_price = data['price'].iloc[i + 5]
                    actual_price = data['price'].iloc[i + 6] if i + 6 < len(data) else None
                    accuracy = 1 / abs(next_day_price - actual_price) if actual_price else 0
                    pattern_db.append((current_pattern, next_day_price, accuracy))

        # Predict the next day's price change
        predictions = []
        for asset, data in crypto_data.items():
            if not data.empty:
                current_pattern = data['price'].iloc[-5:].values
                predicted_next_day_price = match_pattern(current_pattern, pattern_db, rl_agent, asset, current_prices)
                current_price = current_prices.get(asset, None)
                predictions.append((asset, current_price, predicted_next_day_price))

        # Print sorted predictions
        predictions.sort(key=lambda x: x[2], reverse=True)
        print("Today's market prices and predictions for next day:")
        for asset, current_price, predicted_price in predictions:
            if current_price is not None:
                print(f"{asset.capitalize()}:")
                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Predicted Price for Next Day: ${predicted_price:.2f}")
            else:
                print(f"{asset.capitalize()}: Current price data is unavailable.")
    else:
        logging.warning("No valid data was fetched for any cryptocurrency.")

# Schedule the job every day at 6:00 AM
schedule.every().day.at("06:00").do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
