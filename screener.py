import time
import pyotp
import pandas as pd
from SmartApi import SmartConnect
from config import config

# Step 1: Login
totp = pyotp.TOTP(config.ANGEL_TOTP_SECRET).now()
obj = SmartConnect(api_key=config.ANGEL_API_KEY)
session = obj.generateSession(config.ANGEL_CLIENT_ID, config.ANGEL_PASSWORD, totp)
print("✅ Login done")

df = pd.read_csv("nse_stock_list.csv")
all_tokens = df["token"].astype(str).tolist()
print(f"Total stocks to scan: {len(all_tokens)}")

BATCH_SIZE = 50

def chunk_list(lst, size):
    """List ko chhote chhote groups mein todta hai"""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

qualified_stocks = []

for batch_num, batch_tokens in enumerate(chunk_list(all_tokens, BATCH_SIZE), start=1):
    try:
        response = obj.getMarketData(mode="FULL", exchangeTokens={"NSE": batch_tokens})
        fetched = response.get("data", {}).get("fetched", [])

        for stock in fetched:
            ltp = stock.get("ltp", 0)
            buy_qty = stock.get("totBuyQuan", 0)
            sell_qty = stock.get("totSellQuan", 0)

            # Filter 1: Price range
            price_ok = config.PRICE_MIN <= ltp <= config.PRICE_MAX

            # Filter 2: Liquidity
            liquidity_ok = buy_qty > config.MIN_BID_QTY and sell_qty > config.MIN_ASK_QTY

            if price_ok and liquidity_ok:
                qualified_stocks.append({
                    "symbol": stock.get("tradingSymbol"),
                    "token": stock.get("symbolToken"),
                    "ltp": ltp,
                    "bid_qty": buy_qty,
                    "ask_qty": sell_qty,
                })

        print(f"Batch {batch_num} done ({len(batch_tokens)} stocks checked, {len(qualified_stocks)} qualified so far)")

    except Exception as e:
        print(f"⚠️ Batch {batch_num} failed: {e}")

    time.sleep(0.5)

print(f"\n🎯 Total qualified stocks: {len(qualified_stocks)}")
result_df = pd.DataFrame(qualified_stocks)
print(result_df)

result_df.to_csv("qualified_stocks.csv", index=False)
print("✅ Saved to qualified_stocks.csv")