import os
import json
import time
import pandas as pd
import numpy as np
import requests
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# ------------------------ HARD-CODED CREDENTIALS ------------------------
TELEGRAM_TOKEN = "8389895593:AAFCxqAT4nfqTX3uEBC17C-2SfNwyw27xsQ"
TELEGRAM_CHAT_ID = "1003057736884"
BINANCE_API_KEY = "jHGiXjUNarsWRvEZhG4G94ZXTftHo7Rc1FPZOvmNHttac0SvjnDDnSWOW8fCMO6i"
BINANCE_API_SECRET = "81mbLbBlC4gq9D2OEeL7d6WIpSKdVzvkesFNNBr3SMVL3KfZvaq9qSrlZq2GNIxZ"

# ------------------------ INITIAL SETUP ------------------------
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
DATA_DIR = "/opt/crypto-bot/data"
MODEL_DIR = "/opt/crypto-bot/models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TIMEFRAMES = ["5m", "15m", "1h"]
SYMBOL = "BTCUSDT"

# ------------------------ TELEGRAM ------------------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram send error:", e)

# ------------------------ DATA FETCHING ------------------------
def fetch_historical(symbol, interval, limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# ------------------------ INDICATORS ------------------------
def add_indicators(df):
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    df["ATR"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df.dropna(inplace=True)
    return df

# ------------------------ TARGET ------------------------
def add_target(df):
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    return df

# ------------------------ ML MODEL ------------------------
def train_ml(df, timeframe):
    features = ["MA20", "MA50", "RSI", "MACD", "MACD_signal", "ADX", "ATR"]
    X, y = df[features], df["target"]
    model_path = os.path.join(MODEL_DIR, f"model_{timeframe}.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model {timeframe} accuracy: {acc:.2f}")
    return model

# ------------------------ SIGNAL GENERATION ------------------------
def generate_signal(df, model):
    latest = df.iloc[-1:]
    features = ["MA20", "MA50", "RSI", "MACD", "MACD_signal", "ADX", "ATR"]
    prediction = model.predict(latest[features])[0]

    # Confluence example: simple strategy
    ma_signal = 1 if latest["MA20"].values[0] > latest["MA50"].values[0] else 0
    rsi_signal = 1 if latest["RSI"].values[0] < 30 else (0 if latest["RSI"].values[0] > 70 else ma_signal)

    # Combine ML + TA
    if prediction == 1 and ma_signal == 1 and rsi_signal == 1:
        side = "BUY"
    elif prediction == 0 and ma_signal == 0 and rsi_signal == 0:
        side = "SELL"
    else:
        side = "HOLD"

    # Calculate SL/TP
    close_price = latest["close"].values[0]
    if side == "BUY":
        sl = close_price * 0.995
        tp = close_price * 1.01
    elif side == "SELL":
        sl = close_price * 1.005
        tp = close_price * 0.99
    else:
        sl = tp = close_price

    return side, sl, tp

# ------------------------ MAIN LOOP ------------------------
def run_bot():
    for tf in TIMEFRAMES:
        df = fetch_historical(SYMBOL, tf)
        df = add_indicators(df)
        df = add_target(df)
        model = train_ml(df, tf)
        side, sl, tp = generate_signal(df, model)

        msg = f"Timeframe: {tf}\nSignal: {side}\nPrice: {df['close'].iloc[-1]:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}"
        print(msg)
        send_telegram(msg)
        time.sleep(1)  # Small delay to avoid rate limits

if _name_ == "_main_":
    while True:
        try:
            run_bot()
            time.sleep(60)  # run every 1 min
        except Exception as e:
            print("Error:", e)
            time.sleep(30)
