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


def fetch_candles_with_retry(obj, params, max_retries=3, wait_seconds=15):
    for attempt in range(1, max_retries + 1):
        try:
            return obj.getCandleData(params)
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                return None


def fetch_extended_candles(symbol_token, total_days=45, chunk_days=25):
    """
    Angel One ek request mein limited din ka data deta hai,
    isliye hum chunks mein maangte hain aur jodte hain.
    """
    all_candles = []
    to_date = datetime.now()

    days_covered = 0
    while days_covered < total_days:
        chunk_to = to_date - timedelta(days=days_covered)
        chunk_from = chunk_to - timedelta(days=min(chunk_days, total_days - days_covered))

        params = {
            "exchange": "NSE",
            "symboltoken": str(symbol_token),
            "interval": "ONE_MINUTE",
            "fromdate": chunk_from.strftime("%Y-%m-%d %H:%M"),
            "todate": chunk_to.strftime("%Y-%m-%d %H:%M"),
        }

        response = fetch_candles_with_retry(obj, params)
        if response is not None:
            chunk_candles = response.get("data", [])
            all_candles.extend(chunk_candles)

        days_covered += chunk_days
        time.sleep(1)

    return all_candles


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
            signals.append({"index": i, "timestamp": df["timestamp"].iloc[i], "signal": "BUY", "ltp": df["close"].iloc[i]})
        elif prev_fast > prev_slow and curr_fast < curr_slow:
            signals.append({"index": i, "timestamp": df["timestamp"].iloc[i], "signal": "SELL", "ltp": df["close"].iloc[i]})

    return signals

def extract_features(df, idx):
    if idx < 10:
        return None

    smma_fast = df["smma_20"].iloc[idx]
    smma_slow = df["smma_120"].iloc[idx]
    smma_gap_pct = ((smma_fast - smma_slow) / smma_slow) * 100

    recent_closes = df["close"].iloc[idx - 10:idx]
    volatility = recent_closes.std()

    timestamp_str = df["timestamp"].iloc[idx]
    hour = int(timestamp_str[11:13])

    vol_first_half = df["volume"].iloc[idx - 10:idx - 5].mean()
    vol_second_half = df["volume"].iloc[idx - 5:idx].mean()
    volume_trend = (vol_second_half - vol_first_half) / (vol_first_half + 1e-6)

    avg_vol_2min = df["volume"].iloc[idx - 2:idx].mean()
    avg_vol_5min = df["volume"].iloc[idx - 5:idx].mean()
    ltq_ratio = avg_vol_2min / avg_vol_5min if avg_vol_5min > 0 else None

    return {
        "smma_gap_pct": smma_gap_pct,
        "volatility": volatility,
        "hour": hour,
        "volume_trend": volume_trend,
        "ltq_ratio_2v5": ltq_ratio,
    }


def build_trades(signals, df):
    trades = []
    open_trade = None

    for sig in signals:
        if open_trade is not None:
            open_trade["exit_timestamp"] = sig["timestamp"]
            open_trade["exit_ltp"] = sig["ltp"]
            open_trade["pnl"] = open_trade["exit_ltp"] - open_trade["entry_ltp"]
            open_trade["status"] = "CLOSED"
            open_trade["profitable"] = 1 if open_trade["pnl"] > 0 else 0
            trades.append(open_trade)
            open_trade = None

        features = extract_features(df, sig["index"])

        open_trade = {
            "trade_type": sig["signal"],
            "entry_timestamp": sig["timestamp"],
            "entry_ltp": sig["ltp"],
            "smma_gap_pct": features["smma_gap_pct"] if features else None,
            "volatility": features["volatility"] if features else None,
            "hour": features["hour"] if features else None,
            "volume_trend": features["volume_trend"] if features else None,
            "ltq_ratio_2v5": features["ltq_ratio_2v5"] if features else None,
            "exit_timestamp": None,
            "exit_ltp": None,
            "pnl": None,
            "status": "OPEN",
            "profitable": None,
        }

    if open_trade is not None:
        trades.append(open_trade)

    return trades


def process_stock(symbol, token):
    candles = fetch_extended_candles(token, total_days=45, chunk_days=25)

    if len(candles) < config.SMMA_SLOW:
        return []

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    close_prices = df["close"].tolist()
    df["smma_20"] = calculate_smma(close_prices, config.SMMA_FAST)
    df["smma_120"] = calculate_smma(close_prices, config.SMMA_SLOW)

    signals = detect_crossovers(df)
    trades = build_trades(signals, df)

    for t in trades:
        t["symbol"] = symbol

    return trades


qualified_df = pd.read_csv("qualified_stocks.csv")

all_trades = []

for i, row in qualified_df.iterrows():
    symbol = row["symbol"]
    token = row["token"]
    print(f"[{i+1}/{len(qualified_df)}] Processing {symbol}...")

    trades = process_stock(symbol, token)
    all_trades.extend(trades)

    print(f"   → {len(trades)} trades found (total so far: {len(all_trades)})")
    time.sleep(1.5)

trades_df = pd.DataFrame(all_trades)
trades_df.to_csv("training_data.csv", index=False)

print(f"\n✅ Total trades collected: {len(all_trades)}")
closed = trades_df[trades_df["status"] == "CLOSED"]
print(f"   Closed trades (usable for ML): {len(closed)}")
print(f"   Open trades (excluded from training): {len(trades_df) - len(closed)}")
print("\n✅ Saved to training_data.csv")