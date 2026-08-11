import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pyotp
from SmartApi import SmartConnect
from config import config

totp = pyotp.TOTP(config.ANGEL_TOTP_SECRET).now()
print(f"Generated TOTP: {totp}")


obj = SmartConnect(api_key=config.ANGEL_API_KEY)


try:
    session = obj.generateSession(
        config.ANGEL_CLIENT_ID,
        config.ANGEL_PASSWORD,
        totp
    )
    print("✅ Login Successful!")
    print(session)
except Exception as e:
    print("❌ Login Failed:")
    print(e)


ltp_data = obj.ltpData(exchange="NSE", tradingsymbol="RELIANCE-EQ", symboltoken="2885")
print("📈 Reliance Live Price Data:")
print(ltp_data)