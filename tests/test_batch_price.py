import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pyotp
import pandas as pd
from SmartApi import SmartConnect
from config import config

totp = pyotp.TOTP(config.ANGEL_TOTP_SECRET).now()
obj = SmartConnect(api_key=config.ANGEL_API_KEY)
session = obj.generateSession(config.ANGEL_CLIENT_ID, config.ANGEL_PASSWORD, totp)
print("✅ Login done")


df = pd.read_csv("nse_stock_list.csv")
sample = df.head(10)


tokens_list = sample["token"].astype(str).tolist()


quote_params = {
    "mode": "FULL",
    "exchangeTokens": {
        "NSE": tokens_list
    }
}

response = obj.getMarketData(mode="FULL", exchangeTokens={"NSE": tokens_list})

print("📦 Raw response:")
print(response)