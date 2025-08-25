import os
import datetime
import joblib
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Load API keys
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

start = datetime.datetime(2020, 6, 1)
end = datetime.datetime.now()
symbols = ["JPM", "KULR", "META", "MS", "MU", "NVDA", "OKLO", "AVGO"]

models_dir = "trained_models"
os.makedirs(models_dir, exist_ok=True)

def preprocess_data(df, window_size=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
        y.append(scaled[i, 0])
    return np.array(X).reshape(-1, window_size, 1), np.array(y), scaler

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Fetch & train models
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed="iex"
)
bars = data_client.get_stock_bars(request_params).df

for symbol in symbols:
    df = bars.loc[bars.index.get_level_values("symbol") == symbol, ["close"]]
    df.rename(columns={"close": "Close"}, inplace=True)

    if len(df) < 100:
        print(f"Skipping {symbol}, not enough data.")
        continue

    X, y, scaler = preprocess_data(df.values)

    model = create_model((X.shape[1], 1))
    print(f"Training {symbol}...")
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    model.save(f"{models_dir}/{symbol}_lstm_model.h5")
    joblib.dump(scaler, f"{models_dir}/{symbol}_scaler.save")

print("✅ Training complete")

