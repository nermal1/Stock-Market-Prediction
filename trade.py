import os
import joblib
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import symbols, window_size

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
models_dir = "trained_models"

account = trading_client.get_account()
available_cash = float(account.cash)
positions = trading_client.get_all_positions()
held_symbols = {pos.symbol for pos in positions}

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

def predict_next_price(model, df, scaler):
    X = scaler.transform(df)[-window_size:]
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
        if symbol in held_symbols or available_cash < current_price:
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"BUY {symbol} at {current_price}")
    elif signal == "sell":
        if symbol not in held_symbols:
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"SELL {symbol} at {current_price}")

for symbol in symbols:
    try:
        model = load_model(f"{models_dir}/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"{models_dir}/{symbol}_scaler.save")
    except:
        print(f"No model/scaler for {symbol}, skipping.")
        continue
    df = bars.loc[bars.index.get_level_values("symbol") == symbol, ["close"]].values
    if len(df) < window_size + 1:
        continue
    predicted, current = predict_next_price(model, df, scaler)
    signal = generate_signal(predicted, current)
    execute_trade(symbol, signal, current)

