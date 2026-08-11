import requests
import json
import pandas as pd

MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

print("Downloading master list... (thoda time lagega, file bhaari hai)")
response = requests.get(MASTER_URL)
all_instruments = response.json()

print(f"Total entries mile: {len(all_instruments)}")

nse_stocks = [
    item for item in all_instruments
    if item.get("exch_seg") == "NSE" and item.get("symbol", "").endswith("-EQ")
]

print(f"NSE Equity stocks mile: {len(nse_stocks)}")

df = pd.DataFrame(nse_stocks)[["token", "symbol", "name"]]

df.to_csv("nse_stock_list.csv", index=False)
print("✅ Saved to nse_stock_list.csv")
print(df.head(10))