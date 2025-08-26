import os
import joblib
import numpy as np
import datetime
import pandas as pd
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

# CSV log file
log_file = "trade_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        "timestamp", "symbol", "predicted_price", "current_price", "signal", "action_taken"
    ]).to_csv(log_file, index=False)

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
    action_taken = "none"
    if signal == "buy":
        if symbol in held_symbols or available_cash < current_price:
            action_taken = "skip_buy"
        else:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            action_taken = "buy"
    elif signal == "sell":
        if symbol not in held_symbols:
            action_taken = "skip_sell"
        else:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            action_taken = "sell"
    else:
        action_taken = "hold"
    return action_taken

# Run predictions and execute trades
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
    action_taken = execute_trade(symbol, signal, current)

    # Log trade
    log_entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "symbol": symbol,
        "predicted_price": round(predicted, 2),
        "current_price": round(current, 2),
        "signal": signal,
        "action_taken": action_taken
    }])
    log_entry.to_csv(log_file, mode="a", header=False, index=False)
