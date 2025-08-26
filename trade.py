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

from config import symbols, window_size  # ✅ shared config

API_KEY = os.getenv("ALPACA_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Account info
account = trading_client.get_account()
available_cash = float(account.cash)
positions = trading_client.get_all_positions()
held_symbols = {pos.symbol for pos in positions}

# Get the last year of daily bars
start = datetime.datetime.now() - datetime.timedelta(days=365)
end = datetime.datetime.now()
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed="iex"  # ✅ avoid SIP permission error
)
bars = data_client.get_stock_bars(request_params).df

def predict_next_price(model, df, scaler):
    X = scaler.transform(df)[-window_size:]  # last 60 days
    X = np.array([X]).reshape(1, window_size, 1)
    pred_scaled = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]
    current = df[-1][0]
    return predicted, current

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

# Loop over tickers
for symbol in symbols:
    try:
        model = load_model(f"trained_models/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"trained_models/{symbol}_scaler.save")
    except:
        print(f"No model for {symbol}, skipping.")
        continue

    df = bars.loc[bars.index.get_level_values("symbol") == symbol, ["close"]].values
    if len(df) < window_size + 1:
        print(f"Not enough data for {symbol}, skipping.")
        continue

    predicted, current = predict_next_price(model, df, scaler)
    signal = generate_signal(predicted, current)
    execute_trade(symbol, signal, current)
