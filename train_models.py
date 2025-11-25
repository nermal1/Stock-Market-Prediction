import os
import datetime
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from config import symbols, window_size

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
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

start = datetime.datetime(2020, 6, 1)
end = datetime.datetime.now()

request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed="iex"
)
bars = data_client.get_stock_bars(request_params).df

for symbol in symbols:
    print(f"\n--- Processing {symbol} ---")
    
    # 1. Prepare Data
    df = bars.loc[bars.index.get_level_values("symbol") == symbol, ["close"]]
    df.rename(columns={"close": "Close"}, inplace=True)
    
    if len(df) < 100:
        print(f"Skipping {symbol}, not enough data.")
        continue

    # 2. Preprocess
    # We need to inverse transform later, so we keep the scaler handy
    X, y, scaler = preprocess_data(df.values, window_size=window_size)

    # 3. Split into Train and Test (80% Train, 20% Test)
    # This is crucial to see how the model performs on data it hasn't seen
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # 4. Create and Train Model
    model = create_model((X_train.shape[1], 1))
    
    # verbose=1 shows the progress bar and loss
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1) 

    # 5. Evaluate (Get Accuracy Numbers)
    # Predict on the test set
    predictions = model.predict(X_test, verbose=0)

    # Inverse transform to get actual dollar values (not 0-1 scaled values)
    # We must reshape y_test to match the scaler's expected input
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    pred_inv = scaler.inverse_transform(predictions)

    # Calculate Error Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
    mae = mean_absolute_error(y_test_inv, pred_inv)

    # Calculate Directional Accuracy (Did it guess the direction right?)
    # We compare the direction of the prediction vs the direction of the actual
    results_df = pd.DataFrame({'Actual': y_test_inv.flatten(), 'Predicted': pred_inv.flatten()})
    results_df['Actual_Change'] = results_df['Actual'].diff()
    results_df['Pred_Change'] = results_df['Predicted'].diff()
    
    # logic: if both are positive or both are negative, sign is matching
    results_df['Correct_Direction'] = np.sign(results_df['Actual_Change']) == np.sign(results_df['Pred_Change'])
    dir_acc = results_df['Correct_Direction'].mean() * 100

    print(f"Results for {symbol}:")
    print(f"  RMSE: ${rmse:.2f} (Average error range)")
    print(f"  MAE:  ${mae:.2f} (Average dollar miss)")
    print(f"  Directional Accuracy: {dir_acc:.2f}%")

    # 6. Save
    model.save(f"{models_dir}/{symbol}_lstm_model.h5")
    joblib.dump(scaler, f"{models_dir}/{symbol}_scaler.save")

