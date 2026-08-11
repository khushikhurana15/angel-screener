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
symbol_name = "MOMENTUM-EQ"
exchange = "NSE"


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


to_date = datetime.now()
from_date = to_date - timedelta(hours=3)

params = {
    "exchange": exchange,
    "symboltoken": symbol_token,
    "interval": "ONE_MINUTE",
    "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
    "todate": to_date.strftime("%Y-%m-%d %H:%M"),
}

response = fetch_candles_with_retry(obj, params)
candles = response.get("data", [])
print(f"Total candles mile: {len(candles)}")

df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])


def calculate_etq(df, minutes):
    recent = df.tail(minutes)
    return recent["volume"].sum()


def calculate_avg_price(df, minutes):
    recent = df.tail(minutes)
    return recent["close"].mean()


etq_5min = calculate_etq(df, 5)
etq_20min = calculate_etq(df, 20)
etq_60min = calculate_etq(df, 60)

avg_price_20min = calculate_avg_price(df, 20)
avg_price_60min = calculate_avg_price(df, 60)

print(f"\n📦 ETQ (Exchange Traded Quantity):")
print(f"   Last 5 min:  {etq_5min}")
print(f"   Last 20 min: {etq_20min}")
print(f"   Last 60 min: {etq_60min}")

print(f"\n💰 Average Price:")
print(f"   Last 20 min: {avg_price_20min:.2f}")
print(f"   Last 60 min: {avg_price_60min:.2f}")


time.sleep(1)

quote_response = obj.getMarketData(mode="FULL", exchangeTokens={exchange: [symbol_token]})
stock_data = quote_response["data"]["fetched"][0]

best_bid = stock_data["depth"]["buy"][0]
best_ask = stock_data["depth"]["sell"][0]

print(f"\n📊 Market Depth:")
print(f"   Bid Price: {best_bid['price']}, Bid Qty: {best_bid['quantity']}")
print(f"   Ask Price: {best_ask['price']}, Ask Qty: {best_ask['quantity']}")