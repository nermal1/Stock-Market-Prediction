import os
import datetime
import numpy as np
import pandas as pd
import joblib

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

start = datetime.datetime(2020, 6, 1)
end = datetime.datetime(2025, 7, 1)
symbols = ["JPM", "KULR", "META", "MS", "MU", "NVDA", "OKLO", "AVGO"]

account = trading_client.get_account()
available_cash = float(account.cash)

positions = trading_client.get_all_positions()
held_symbols = {pos.symbol for pos in positions}

request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start,
    end=end
)
bars = data_client.get_stock_bars(request_params).df

symbol_dfs = {}
for symbol in symbols:
    symbol_df = bars.loc[bars.index.get_level_values('symbol') == symbol, ['close']]
    symbol_df.rename(columns={'close': 'Close'}, inplace=True)
    symbol_dfs[symbol] = symbol_df 

def preprocess_data(df, window_size=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y, scaler

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

models_dir = "trained_models"
os.makedirs(models_dir, exist_ok=True)

# Train and save model per stock
for symbol in symbols:
    df = symbol_dfs[symbol]
    if len(df) < 100:
        print(f"Skipping {symbol}, not enough data.")
        continue

    X, y, scaler = preprocess_data(df)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(f"Training model for {symbol}")
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    # Save model
    model.save(f"{models_dir}/{symbol}_lstm_model.h5")
    print(f"Saved model for {symbol}")

    # Save scaler too (optional but helpful for consistent predictions)
    import joblib
    joblib.dump(scaler, f"{models_dir}/{symbol}_scaler.save")

def predict_next_price(model, df, scaler, window_size=60):
    X, _, _ = preprocess_data(df, window_size)
    last_input = X[-1].reshape(1, window_size, 1)
    pred_scaled = model.predict(last_input)
    predicted_price = scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]
    last_price = df['Close'].iloc[-1]
    return predicted_price, last_price

def generate_signal(predicted, current, threshold=0.01):
    change = (predicted - current) / current
    if change > threshold:
        return "buy"
    elif change < -threshold:
        return "sell"
    return "hold"
def execute_trade(symbol, signal, current_price, held_symbols, available_cash):
    if signal == "buy":
        if symbol in held_symbols:
            print(f"Already holding {symbol}, skipping buy.")
            return
        if available_cash < current_price:
            print(f"Not enough cash to buy {symbol}. Needed: ${current_price:.2f}, Available: ${available_cash:.2f}")
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"BUY 1 share of {symbol} at ~${current_price:.2f}")

    elif signal == "sell":
        if symbol not in held_symbols:
            print(f"No holdings to sell in {symbol}, skipping.")
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"SELL 1 share of {symbol} at ~${current_price:.2f}")

    else:
        print(f"HOLD {symbol}")

account = trading_client.get_account()
available_cash = float(account.cash)
positions = trading_client.get_all_positions()
held_symbols = {pos.symbol for pos in positions}

window_size = 60

for symbol in symbols:
    df = symbol_dfs[symbol]
    if len(df) < window_size + 1:
        print(f"Not enough data for {symbol}")
        continue

    try:
        model = load_model(f"trained_models/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"trained_models/{symbol}_scaler.save")
    except:
        print(f"No model or scaler found for {symbol}, skipping.")
        continue

    predicted, current = predict_next_price(model, df, scaler)
    signal = generate_signal(predicted, current)
    execute_trade(symbol, signal, current, held_symbols, available_cash)


