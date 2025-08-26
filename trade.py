import os
import joblib
import numpy as np
import datetime
import pandas as pd
from tensorflow.keras.models import load_model

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

symbols = ["JPM", "KULR", "META", "MS", "MU", "NVDA", "OKLO", "AVGO"]
window_size = 60

account = trading_client.get_account()
available_cash = float(account.cash)
positions = trading_client.get_all_positions()
held_symbols = {pos.symbol for pos in positions}

# Fetch recent data
start = datetime.datetime.now() - datetime.timedelta(days=365)
end = datetime.datetime.now()
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed="iex"
)
bars = data_client.get_stock_bars(request_params).df

def preprocess_data(df, window_size=60):
    scaled = scaler.transform(df)
    X = []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
    return np.array(X).reshape(-1, window_size, 1)

def predict_next_price(model, df, scaler, window_size=60):
    scaled = scaler.transform(df)
    X = [scaled[-window_size:]]
    X = np.array(X).reshape(1, window_size, 1)
    pred_scaled = model.predict(X, verbose=0)
    return scaler.inverse_transform([[pred_scaled[0][0]]])[0][0], df[-1][0]

def generate_signal(predicted, current, threshold=0.01):
    change = (predicted - current) / current
    if change > threshold:
        return "buy"
    elif change < -threshold:
        return "sell"
    return "hold"

def execute_trade(symbol, signal, current_price):
    global available_cash
    if signal == "buy":
        if symbol in held_symbols:
            print(f"Holding {symbol}, skip buy.")
            return
        if available_cash < current_price:
            print(f"Not enough cash for {symbol}.")
            return
        order = MarketOrderRequest(symbol=symbol, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        trading_client.submit_order(order)
        print(f"BUY {symbol} at {current_price}")
    elif signal == "sell":
        if symbol not in held_symbols:
            print(f"No {symbol} to sell.")
            return
        order = MarketOrderRequest(symbol=symbol, qty=1, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        trading_client.submit_order(order)
        print(f"SELL {symbol} at {current_price}")
    else:
        print(f"HOLD {symbol}")

# Run predictions & trades
for symbol in symbols:
    try:
        model = load_model(f"trained_models/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"trained_models/{symbol}_scaler.save")
    except:
        print(f"No model for {symbol}, skipping.")
        continue

    df = bars.loc[bars.index.get_level_values("symbol") == symbol, ["close"]].values
    if len(df) < window_size + 1:
        continue

    predicted, current = predict_next_price(model, df, scaler)
    signal = generate_signal(predicted, current)
    execute_trade(symbol, signal, current)
