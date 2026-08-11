"""
Central configuration loader.
Reads from a local .env file (never committed) so credentials stay out of source code.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into environment


class Config:
    # Angel One SmartAPI credentials
    ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
    ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
    ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD", "")
    ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

    # Groq (AI explanation layer)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    USE_MOCK_FEED = os.getenv("USE_MOCK_FEED", "true").lower() == "true"

    # Screening thresholds (from the assignment spec) 
    PRICE_MIN = 30
    PRICE_MAX = 500
    MIN_BID_QTY = 1_000_000
    MIN_ASK_QTY = 1_000_000

    # Indicator settings 
    SMMA_FAST = 20
    SMMA_SLOW = 120


config = Config()
