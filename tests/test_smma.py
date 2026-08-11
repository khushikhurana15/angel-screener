import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pyotp
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
from config import config

totp = pyotp.TOTP(config.ANGEL_TOTP_SECRET).now()
obj = SmartConnect(api_key=config.ANGEL_API_KEY)
session = obj.generateSession(config.ANGEL_CLIENT_ID, config.ANGEL_PASSWORD, totp)
print("✅ Login done")

time.sleep(2)

symbol_token = "10693"
exchange = "NSE"

to_date = datetime.now()
from_date = to_date - timedelta(days=3)

params = {
    "exchange": exchange,
    "symboltoken": symbol_token,
    "interval": "ONE_MINUTE",
    "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
    "todate": to_date.strftime("%Y-%m-%d %H:%M"),
}


def fetch_candles_with_retry(obj, params, max_retries=3, wait_seconds=15):
    for attempt in range(1, max_retries + 1):
        try:
            return obj.getCandleData(params)
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"⏳ {wait_seconds}s wait karke retry...")
                time.sleep(wait_seconds)
            else:
                raise


response = fetch_candles_with_retry(obj, params)
candles = response.get("data", [])
print(f"Total candles mile: {len(candles)}")

df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
print(df.tail(5))


def calculate_smma(prices, period):
    smma = [None] * len(prices)

    if len(prices) < period:
        return smma

    first_smma = sum(prices[:period]) / period
    smma[period - 1] = first_smma

    for i in range(period, len(prices)):
        prev = smma[i - 1]
        smma[i] = (prev * (period - 1) + prices[i]) / period

    return smma


close_prices = df["close"].tolist()

df["smma_20"] = calculate_smma(close_prices, config.SMMA_FAST)
df["smma_120"] = calculate_smma(close_prices, config.SMMA_SLOW)

print("\n📊 Last 5 rows with SMMA:")
print(df[["timestamp", "close", "smma_20", "smma_120"]].tail(5))


def detect_crossovers(df):
    signals = []

    for i in range(1, len(df)):
        prev_fast = df["smma_20"].iloc[i - 1]
        prev_slow = df["smma_120"].iloc[i - 1]
        curr_fast = df["smma_20"].iloc[i]
        curr_slow = df["smma_120"].iloc[i]

        if prev_fast is None or prev_slow is None or curr_fast is None or curr_slow is None:
            continue

        if prev_fast < prev_slow and curr_fast > curr_slow:
            signals.append({
                "timestamp": df["timestamp"].iloc[i],
                "signal": "BUY",
                "ltp": df["close"].iloc[i],
            })

        elif prev_fast > prev_slow and curr_fast < curr_slow:
            signals.append({
                "timestamp": df["timestamp"].iloc[i],
                "signal": "SELL",
                "ltp": df["close"].iloc[i],
            })

    return signals


crossover_signals = detect_crossovers(df)

print(f"\n🔔 Total crossovers found: {len(crossover_signals)}")
for sig in crossover_signals:
    print(sig)


def build_trades(signals):
    trades = []
    open_trade = None

    for sig in signals:
        if open_trade is not None:
            open_trade["exit_timestamp"] = sig["timestamp"]
            open_trade["exit_ltp"] = sig["ltp"]
            open_trade["pnl"] = open_trade["exit_ltp"] - open_trade["entry_ltp"]
            open_trade["status"] = "CLOSED"
            trades.append(open_trade)
            open_trade = None

        open_trade = {
            "trade_type": sig["signal"],
            "entry_timestamp": sig["timestamp"],
            "entry_ltp": sig["ltp"],
            "exit_timestamp": None,
            "exit_ltp": None,
            "pnl": None,
            "status": "OPEN",
        }

    if open_trade is not None:
        trades.append(open_trade)

    return trades


trades = build_trades(crossover_signals)

print(f"\n💰 Total trades: {len(trades)}")
for t in trades:
    print(t)