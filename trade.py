import os
import datetime
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# --- CONFIG ---
API_KEY = os.getenv("ALPACA_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")

symbols = ["JPM", "KULR", "META", "MS", "MU", "NVDA", "OKLO", "AVGO"]
window_size = 60
risk_fraction = 0.03  # risk 3% of capital per buy
TRADE_LOG = "trade_log.csv"

# --- CLIENTS ---
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# --- HELPER FUNCTIONS ---
def log_trade(symbol, signal, price, action):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "symbol": symbol, "signal": signal, "price": price, "action": action}
    if os.path.exists(TRADE_LOG):
        pd.DataFrame([row]).to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(TRADE_LOG, index=False)

def preprocess_data(df, scaler):
    scaled = scaler.transform(df)
    X = [scaled[-window_size:]]
    return np.array(X).reshape(1, window_size, 1)

def predict_next_price(model, df, scaler):
    X = preprocess_data(df, scaler)
    pred_scaled = model.predict(X, verbose=0)
    predicted_price = scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]
    last_price = df[-1][0]
    return predicted_price, last_price

def generate_signal(predicted, current, threshold=0.01):
    change = (predicted - current) / current
    if change > threshold:
        return "buy"
    elif change < -threshold:
        return "sell"
    return "hold"

def execute_trade(symbol, signal, current_price, available_cash, held_qty):
    action_taken = "hold"
    if signal == "buy":
        max_qty = int((available_cash * risk_fraction) // current_price)
        if max_qty > 0:
            order = MarketOrderRequest(symbol=symbol, qty=max_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            action_taken = f"buy_{max_qty}"
            print(f"BUY {max_qty} shares of {symbol} at ${current_price:.2f}")
        else:
            action_taken = "skip_buy"
            print(f"Not enough cash to buy {symbol}")
    elif signal == "sell":
        if held_qty > 0:
            order = MarketOrderRequest(symbol=symbol, qty=held_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            action_taken = f"sell_{held_qty}"
            print(f"SELL {held_qty} shares of {symbol} at ${current_price:.2f}")
        else:
            action_taken = "skip_sell"
            print(f"No shares to sell for {symbol}")
    log_trade(symbol, signal, current_price, action_taken)

# --- MAIN ---
account = trading_client.get_account()
available_cash = float(account.cash)
positions = trading_client.get_all_positions()
held_symbols_qty = {pos.symbol: int(pos.qty) for pos in positions}

# Fetch historical data
start = datetime.datetime.now() - datetime.timedelta(days=365)
end = datetime.datetime.now()
request_params = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end)
bars = data_client.get_stock_bars(request_params).df

for symbol in symbols:
    try:
        model = load_model(f"trained_models/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"trained_models/{symbol}_scaler.save")
    except:
        print(f"No model/scaler found for {symbol}, skipping.")
        continue

    df = bars.loc[bars.index.get_level_values("symbol") == symbol, ["close"]].values
    if len(df) < window_size + 1:
        continue

    predicted, current = predict_next_price(model, df, scaler)
    signal = generate_signal(predicted, current)
    held_qty = held_symbols_qty.get(symbol, 0)
    execute_trade(symbol, signal, current, available_cash, held_qty)
