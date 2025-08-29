import alpaca_trade_api as tradeapi
import csv
import os
from datetime import datetime

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

# Alpaca API setup
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"  # use paper trading

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# Path for log file
LOG_FILE = "trade_log.csv"

# Initialize CSV file with headers if not already present
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "symbol", "action", "qty", "price", "capital"])


def log_trade(symbol, action, qty, price, capital):
    """Log every trade action to CSV file immediately."""
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            action,
            qty,
            round(price, 2) if price else None,
            round(capital, 2)
        ])


def trade_stock(symbol, action, capital, price=None):
    """
    Execute a trade (buy/sell/hold) and log it.
    - symbol: stock ticker
    - action: "buy", "sell", or "hold"
    - capital: current account capital
    - price: optional stock price for logging
    """

    # If HOLD: log but don’t submit an order
    if action.lower() == "hold":
        log_trade(symbol, "HOLD", 0, price, capital)
        return

    try:
        # Get latest market price if not provided
        if price is None:
            barset = api.get_latest_trade(symbol)
            price = barset.price

        # Risk management: trade with 3% of capital
        trade_amount = capital * 0.03
        qty = int(trade_amount // price)

        if qty <= 0:
            print(f"Not enough capital to trade {symbol}")
            return

        if action.lower() == "buy":
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            log_trade(symbol, "BUY", qty, price, capital)

        elif action.lower() == "sell":
            # Sell ALL holdings for this stock
            position = api.get_position(symbol)
            qty = int(position.qty)

            if qty > 0:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                log_trade(symbol, "SELL", qty, price, capital)
            else:
                print(f"No holdings to sell for {symbol}")
                log_trade(symbol, "SELL_ATTEMPT_NO_HOLDINGS", 0, price, capital)

    except Exception as e:
        print(f"Error trading {symbol}: {e}")
        log_trade(symbol, "ERROR", 0, price, capital)
