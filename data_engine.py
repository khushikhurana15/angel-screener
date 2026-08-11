import time
import pyotp
import pandas as pd
import joblib
import os
import pickle
import socket
from datetime import datetime, timedelta
from SmartApi import SmartConnect
from groq import Groq
from config import config

socket.setdefaulttimeout(20)

_obj = None
_model = None
_groq_client = None

_candle_cache = {}
_cache_dirty_count = 0
CACHE_FILE = "candle_cache.pkl"

def load_cache_from_disk():
    global _candle_cache
    try:
        with open(CACHE_FILE, "rb") as f:
            _candle_cache = pickle.load(f)
        print(f"CACHE: loaded {len(_candle_cache)} stocks from disk")
    except FileNotFoundError:
        _candle_cache = {}
        print("CACHE: no existing cache file, starting fresh")


def save_cache_to_disk():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(_candle_cache, f)


def login():
    global _obj
    totp = pyotp.TOTP(config.ANGEL_TOTP_SECRET).now()
    _obj = SmartConnect(api_key=config.ANGEL_API_KEY)
    _obj.generateSession(config.ANGEL_CLIENT_ID, config.ANGEL_PASSWORD, totp)
    time.sleep(2)
    return _obj


def load_model():
    global _model
    _model = joblib.load("models/crossover_model.joblib")
    return _model


def load_groq():
    global _groq_client
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def load_qualified_stocks(limit=None):
    df = pd.read_csv("qualified_stocks.csv")
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def _fetch_candles_raw(token, from_date, to_date, max_retries=3, wait_seconds=25):
    params = {
        "exchange": "NSE",
        "symboltoken": str(token),
        "interval": "ONE_MINUTE",
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M"),
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = _obj.getCandleData(params)
            candles = response.get("data")
            if candles:
                return candles
            return []
        except Exception:
            if attempt < max_retries:
                time.sleep(wait_seconds)
    return []

def get_candles_cached(token):
    """
    First call for a token: fetches 5 days of history and caches it
    (both in memory and, periodically, to disk). Every call after that:
    only fetches candles since the last cached timestamp and appends them,
    instead of re-fetching everything. Saving to disk means that if the
    process crashes and restarts, already-warmed stocks don't need to be
    re-fetched from scratch. Disk writes are throttled (every 10 updates)
    since pickling the full cache on every single stock update would slow
    the whole refresh cycle down as the cache grows.
    """
    global _cache_dirty_count
    now = datetime.now()

    if token not in _candle_cache:
        from_date = now - timedelta(days=5)
        raw = _fetch_candles_raw(token, from_date, now)
        if not raw:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        _candle_cache[token] = df
        _cache_dirty_count += 1
        if _cache_dirty_count % 10 == 0:
            save_cache_to_disk()
        return df.copy()

    cached_df = _candle_cache[token]
    last_timestamp = cached_df["timestamp"].iloc[-1] if not cached_df.empty else None

    if last_timestamp:
        last_dt = pd.to_datetime(last_timestamp).tz_localize(None)
        from_date = last_dt - timedelta(minutes=2)
    else:
        from_date = now - timedelta(days=5)

    new_raw = _fetch_candles_raw(token, from_date, now)

    if new_raw:
        new_df = pd.DataFrame(new_raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

        cutoff = now - timedelta(days=6)
        combined["_ts_parsed"] = pd.to_datetime(combined["timestamp"]).dt.tz_localize(None)
        combined = combined[combined["_ts_parsed"] >= cutoff].drop(columns=["_ts_parsed"]).reset_index(drop=True)

        _candle_cache[token] = combined
        _cache_dirty_count += 1
        if _cache_dirty_count % 10 == 0:
            save_cache_to_disk()
        return combined.copy()

    return cached_df.copy()

def calculate_smma(prices, period):
    smma = [None] * len(prices)
    if len(prices) < period:
        return smma
    smma[period - 1] = sum(prices[:period]) / period
    for i in range(period, len(prices)):
        smma[i] = (smma[i - 1] * (period - 1) + prices[i]) / period
    return smma


def find_latest_crossover(df):
    for i in range(len(df) - 1, 0, -1):
        pf, ps = df["smma_20"].iloc[i - 1], df["smma_120"].iloc[i - 1]
        cf, cs = df["smma_20"].iloc[i], df["smma_120"].iloc[i]
        if None in (pf, ps, cf, cs):
            continue
        if pf < ps and cf > cs:
            return {"index": i, "signal": "BUY"}
        if pf > ps and cf < cs:
            return {"index": i, "signal": "SELL"}
    return None


def extract_features(df, idx):
    if idx < 10:
        return None
    smma_fast, smma_slow = df["smma_20"].iloc[idx], df["smma_120"].iloc[idx]
    smma_gap_pct = ((smma_fast - smma_slow) / smma_slow) * 100
    volatility = df["close"].iloc[idx - 10:idx].std()
    hour = pd.to_datetime(df["timestamp"].iloc[idx]).hour
    vol_first = df["volume"].iloc[idx - 10:idx - 5].mean()
    vol_second = df["volume"].iloc[idx - 5:idx].mean()
    volume_trend = (vol_second - vol_first) / (vol_first + 1e-6)
    avg_2 = df["volume"].iloc[idx - 2:idx].mean()
    avg_5 = df["volume"].iloc[idx - 5:idx].mean()
    ltq_ratio = avg_2 / avg_5 if avg_5 > 0 else 1.0
    return {
        "smma_gap_pct": smma_gap_pct,
        "volatility": volatility if volatility is not None else 0,
        "hour": hour,
        "volume_trend": volume_trend,
        "ltq_ratio_2v5": ltq_ratio,
    }


def calculate_etq_and_avg(df):
    def etq(minutes):
        return int(df["volume"].tail(minutes).sum()) if not df.empty else 0

    def avg(minutes):
        return round(df["close"].tail(minutes).mean(), 2) if not df.empty else None

    return {
        "etq_5": etq(5), "etq_20": etq(20), "etq_60": etq(60),
        "avg_20": avg(20), "avg_60": avg(60),
    }


def predict_and_explain(features, symbol, signal):
    feature_order = ["smma_gap_pct", "volatility", "hour", "volume_trend", "ltq_ratio_2v5"]
    X = [[features[f] for f in feature_order]]

    pred = _model.predict(X)[0]
    proba = _model.predict_proba(X)[0]
    confidence = round(max(proba) * 100, 1)
    label = "Profitable" if pred == 1 else "Avoid"

    trade = {
        "symbol": symbol, "signal": signal,
        "smma_gap_pct": round(features["smma_gap_pct"], 3),
        "ltq_ratio_2v5": round(features["ltq_ratio_2v5"], 3),
        "volatility": round(features["volatility"], 3),
        "ml_prediction": label, "confidence": confidence,
    }

    explanation = generate_explanation(trade)
    return label, confidence, explanation


def generate_rule_based_explanation(trade):
    reasons = []
    if trade["smma_gap_pct"] > 0.5 or trade["smma_gap_pct"] < -0.5:
        reasons.append("the SMMA gap between the fast and slow line is strong, indicating a clear trend")
    else:
        reasons.append("the SMMA gap is small, indicating a weak or uncertain trend")
    if trade["ltq_ratio_2v5"] > 1.2:
        reasons.append("recent volume (last 2 min) is notably higher than the last 5 min, suggesting a pickup in activity")
    else:
        reasons.append("recent volume is not meaningfully higher than the last 5 min")
    verdict = "accepted" if trade["ml_prediction"] == "Profitable" else "avoided"
    return f"This signal should be {verdict} because {', and '.join(reasons)}."


def generate_explanation(trade):
    prompt = f"""The ML model has ALREADY decided: {trade['ml_prediction']} (confidence: {trade['confidence']}%).
Your only job is to explain WHY in exactly 1 short sentence, in clear English.
Do NOT disagree with or contradict the ML Prediction of "{trade['ml_prediction']}" - just explain it.

Use these EXACT definitions - do not invent your own meaning for these terms:
- ltq_ratio_2v5 = ratio of average traded volume in the last 2 minutes vs the last 5 minutes. 
  A value ABOVE 1 means recent trading activity is picking up (bullish signal for momentum). 
  A value BELOW 1 means recent activity has slowed down compared to the last 5 minutes.
- smma_gap_pct = percentage gap between the fast SMMA(20) and slow SMMA(120) at the moment of 
  crossover. A LARGER magnitude (positive or negative) means a stronger, more confident trend signal. 
  A value close to 0 means a weak, borderline signal.

Data: Symbol: {trade['symbol']}, Signal: {trade['signal']}, SMMA Gap%: {trade['smma_gap_pct']}, 
LTQ Ratio (2min vs 5min): {trade['ltq_ratio_2v5']}

Start your sentence with "This signal should be {'accepted' if trade['ml_prediction'] == 'Profitable' else 'avoided'}" 
and reference the actual numeric value(s) above using the definitions given - do not make up alternate names 
for these features."""
    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=70,
            timeout=6,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return generate_rule_based_explanation(trade)


def get_market_depth_batch(tokens):
    depth_map = {}
    batch_size = 40

    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i + batch_size]
        try:
            response = _obj.getMarketData(mode="FULL", exchangeTokens={"NSE": batch})
            fetched = response.get("data", {}).get("fetched", [])
            for stock in fetched:
                token = str(stock.get("symbolToken"))
                buy = stock.get("depth", {}).get("buy", [{}])
                sell = stock.get("depth", {}).get("sell", [{}])
                depth_map[token] = {
                    "ltp": stock.get("ltp", 0),
                    "bid_price": buy[0].get("price", 0) if buy else 0,
                    "bid_qty": buy[0].get("quantity", 0) if buy else 0,
                    "ask_price": sell[0].get("price", 0) if sell else 0,
                    "ask_qty": sell[0].get("quantity", 0) if sell else 0,
                    "tot_buy_qty": stock.get("totBuyQuan", 0),
                    "tot_sell_qty": stock.get("totSellQuan", 0),
                }
            time.sleep(0.5)
        except Exception:
            continue

    return depth_map


def still_qualifies(depth_info):
    """
    Re-checks a stock against the original screening criteria using the
    latest live depth data, so the dashboard reflects continuous
    screening rather than trusting a one-time CSV snapshot forever.

    Outside market hours, Angel One's frozen snapshot is often asymmetric -
    one side of the book reads 0 while the other holds a stale nonzero value.
    If EITHER side is missing, we don't have a reliable live reading to
    disqualify on, so we fall back to trusting the original screening.
    """
    ltp = depth_info.get("ltp", 0)
    tot_buy = depth_info.get("tot_buy_qty", 0)
    tot_sell = depth_info.get("tot_sell_qty", 0)

    if ltp == 0 or tot_buy == 0 or tot_sell == 0:
        return True

    price_ok = config.PRICE_MIN <= ltp <= config.PRICE_MAX
    liquidity_ok = tot_buy > config.MIN_BID_QTY and tot_sell > config.MIN_ASK_QTY

    if not (price_ok and liquidity_ok):
        print(f"DISQUALIFIED: ltp={ltp} price_ok={price_ok} tot_buy={tot_buy} tot_sell={tot_sell} liquidity_ok={liquidity_ok}")

    return price_ok and liquidity_ok


def build_stock_snapshot(symbol, token, depth_info):
    df = get_candles_cached(token)
    if len(df) < config.SMMA_SLOW:
        return None

    close_prices = df["close"].tolist()
    df = df.copy()
    df["smma_20"] = calculate_smma(close_prices, config.SMMA_FAST)
    df["smma_120"] = calculate_smma(close_prices, config.SMMA_SLOW)

    etq_avg = calculate_etq_and_avg(df)

    crossover = find_latest_crossover(df)
    signal, ml_pred, confidence, explanation = "-", "-", "-", "-"

    if crossover:
        features = extract_features(df, crossover["index"])
        if features:
            signal = crossover["signal"]
            ml_pred, confidence, explanation = predict_and_explain(features, symbol, signal)

    return {
        "Symbol": symbol,
        "LTP": depth_info.get("ltp", df["close"].iloc[-1]),
        "SMMA(20)": round(df["smma_20"].iloc[-1], 2) if df["smma_20"].iloc[-1] else "-",
        "SMMA(120)": round(df["smma_120"].iloc[-1], 2) if df["smma_120"].iloc[-1] else "-",
        "Signal": signal,
        "ETQ(5m)": etq_avg["etq_5"], "ETQ(20m)": etq_avg["etq_20"], "ETQ(60m)": etq_avg["etq_60"],
        "Avg(20m)": etq_avg["avg_20"], "Avg(60m)": etq_avg["avg_60"],
        "Bid Price": depth_info.get("bid_price", "-"), "Bid Qty": depth_info.get("bid_qty", "-"),
        "Ask Price": depth_info.get("ask_price", "-"), "Ask Qty": depth_info.get("ask_qty", "-"),
        "ML Pred": ml_pred, "Confidence": f"{confidence}%" if confidence != "-" else "-",
        "Explanation": explanation,
    }