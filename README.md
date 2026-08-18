# AI/ML-Based Stock Market Screening and Analysis System

An end-to-end Python application that screens NSE stocks in real time, calculates technical indicators, detects SMMA crossover signals, and uses a trained ML model to predict whether each signal is likely to be profitable — with human-readable explanations for every prediction.

Built for the AI/ML Engineer Quantitative Programming technical assessment.

---

## 🎯 Features

- **Live Stock Screening** — Scans all NSE-listed equity stocks and filters by price range (₹30–₹500)
- **Liquidity Filter** — Keeps only stocks with strong order-book depth (Bid & Ask Quantity > 10,00,000)
- **Continuous Re-Screening** — The live dashboard re-validates each stock's price and liquidity against fresh market depth data on every refresh cycle, rather than trusting the initial screening snapshot forever; stocks that drift outside the criteria are dropped from the current view
- **Technical Indicators** — SMMA(20) and SMMA(120), calculated from live 1-minute candle data
- **Exchange Traded Quantity (ETQ)** — Rolling traded volume over last 5 / 20 / 60 minutes
- **Average Price** — Rolling average LTP over last 20 / 60 minutes
- **Market Depth** — Real-time Bid Price, Bid Qty, Ask Price, Ask Qty
- **SMMA Crossover Detection** — Identifies Buy (fast SMMA crosses above slow SMMA) and Sell (fast crosses below slow) signals
- **ML-Based Trade Prediction** — An XGBoost model trained on ~4,800 historical crossover trades predicts whether each new signal is likely to be `Profitable` or should be `Avoided`, with a confidence score
- **AI-Generated Explanations** — Every prediction is paired with a plain-English reason (via Groq), grounded with explicit feature definitions to prevent hallucinated reasoning, with a rule-based fallback if the API is unavailable
- **Live Auto-Refreshing Dashboard** — A PySide6 desktop UI showing all of the above, one row per stock, updating continuously in the background without blocking the UI
- **Live Next-Day Validation Logging** — Every live crossover prediction is logged with a timestamp; a separate resolution script checks real subsequent outcomes to compute genuine out-of-sample accuracy over time (see Validation Analysis Report)

---

## 🏗️ Architecture

```
angel_screener/
├── dashboard.py                     # Main entry point — PySide6 UI + background data thread
├── data_engine.py                   # Core logic: Angel One login, candle fetching/caching, SMMA,
│                                     # ETQ/avg price, market depth, live re-screening, ML prediction,
│                                     # explanations, live prediction logging
├── config.py                        # Loads credentials/settings from .env
├── screener.py                      # Standalone script: initial price + liquidity screening
├── get_stock_list.py                # Downloads & filters NSE instrument master list
├── build_training_data.py           # Builds the historical trade dataset for ML training
├── train_model.py                   # Trains and saves the XGBoost model (stock-grouped split)
├── analyze_validation.py            # Held-out test set validation analysis (Section 3 of report)
├── resolve_predictions.py           # Resolves logged live predictions against real outcomes (Section 4)
├── models/
│   └── crossover_model.joblib       # Trained ML model
├── qualified_stocks.csv             # Initial screening universe (76 stocks) — re-validated live at runtime
├── training_data.csv                # ~4,800 historical labeled trades
├── live_predictions_log.csv         # Live logged predictions + resolved outcomes (next-day validation)
├── validation_analysis_summary.csv  # Output of analyze_validation.py
├── VALIDATION_REPORT.md             # Full validation analysis report (both methods)
├── tests/                           # Manual verification scripts used during development
│   └── README.md                    # Explains what each test script checks
├── requirements.txt
└── .env.example                      # Template for required API keys (no real credentials)
```

The dashboard runs the data pipeline in a **background thread**, separate from the UI thread. Candle history is cached per stock in memory and on disk — only the first fetch for a symbol pulls a full 5-day history; every subsequent refresh fetches only the new candles since the last update and appends them. This is what keeps the dashboard within Angel One's API rate limits while still refreshing continuously.

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.11+
- An active Angel One trading account with SmartAPI access
- A Groq API key (free tier) for AI explanations

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure credentials
```bash
cp .env.example .env
```
Fill in `.env` with your Angel One API key, client ID, password, TOTP secret, and Groq API key. **This file is git-ignored and must never be shared or committed.**

### 4. Run the full pipeline (first-time setup)
```bash
python get_stock_list.py          # Downloads NSE instrument list
python screener.py                # Applies price + liquidity filters → qualified_stocks.csv
python build_training_data.py     # Builds historical trade dataset (~15-20 min)
python train_model.py             # Trains the ML model → models/crossover_model.joblib
```

### 5. Launch the dashboard
```bash
python dashboard.py
```

### 6. Run validation analysis (optional, reproduces the Validation Report)
```bash
python analyze_validation.py      # Held-out test set analysis
python resolve_predictions.py     # Resolves live-logged predictions (run on a later trading day)
```

---

## 🧠 Machine Learning Approach

**Data**: ~4,800 historical SMMA crossover trades collected across 75 qualified stocks over a 45-day lookback window, each labeled `Profitable` or `Loss` based on the assignment's P&L formula.

**Features used**:

| Feature | Description |
|---|---|
| `smma_gap_pct` | % gap between SMMA(20) and SMMA(120) at the moment of crossover — a proxy for signal strength |
| `volatility` | Std. deviation of close price over the preceding 10 candles |
| `hour` | Hour of day the crossover occurred |
| `volume_trend` | Change in average volume between the preceding two 5-candle windows |
| `ltq_ratio_2v5` | Ratio of average traded quantity over the last 2 minutes vs. last 5 minutes — the assignment's suggested LTQ-spike signal |

**Model**: XGBoost classifier.

**Train/test methodology**: The dataset uses a **stock-grouped split** (`GroupShuffleSplit` on the `symbol` column) — every trade from a given stock goes entirely into either the training set or the test set, never split across both. This was a deliberate correction: an earlier version used a random row-level split, which let trades from the same stock appear in both sets, letting the model partially memorize a stock's typical price level or behavior instead of learning a generalizable crossover pattern. With the corrected split (60 training symbols, 15 test symbols, verified zero symbol overlap), the model achieves:

- **75.67% accuracy** on the held-out test set (stocks the model has never seen)
- Balanced precision/recall across both classes (Loss: 0.77 precision / 0.75 recall; Profit: 0.74 precision / 0.77 recall)
- Feature importance dominated by `smma_gap_pct` (66%), with `hour`, `volatility`, `ltq_ratio_2v5`, and `volume_trend` each contributing meaningfully (7–9% each)

**A key design decision worth noting**: an even earlier version of the model included `trade_type` (Buy/Sell) as a feature and reached a similar accuracy — but inspection of feature importances showed it was relying almost entirely (95%) on `trade_type` alone, rather than genuine market signals. Further analysis showed this was a statistical artifact of the P&L formula (see note below), not a real pattern, so `trade_type` was removed. The current model's accuracy is coincidentally similar in magnitude, but is now built on genuinely predictive, price/volume-based features rather than a shortcut.

**LTQ approximation**: The assignment distinguishes tick-level LTQ ("the quantity executed in the most recent trade, changing with every tick") from cumulative daily Volume. Angel One's historical REST API does not expose tick-level LTQ data, only OHLCV candles — so `ltq_ratio_2v5` is computed as a **1-minute candle volume proxy** (average volume over the last 2 one-minute candles vs. the last 5), rather than literal tick-by-tick LTQ. This is a reasonable engineering approximation given the available historical data, and captures the same underlying idea (a spike in recent trading activity) that the assignment describes. A closer match to the literal spec would require subscribing to Angel One's WebSocket tick feed for live LTQ, which was out of scope for the historical training pipeline.

---

## 📊 Validation Analysis — Does the ML filter actually improve profitability?

In response to feedback requesting explicit validation results, the model was evaluated two ways. Full detail, methodology, and honest limitations of both are in **`VALIDATION_REPORT.md`** — summary below.

**Note on methodology**: The assignment's "next-day validation" language specifically describes *temporal* generalization (does the model still work on data from after training). The held-out stock-grouped test set below tests a related but distinct property — *entity* generalization (does it work on stocks it never saw). Both are reported separately and not treated as equivalent; see `VALIDATION_REPORT.md` for the full distinction and the live pipeline built to address temporal validation directly.

**Held-out test set (1,011 signals, 15 stocks never seen during training):**

| Question | Result |
|---|---|
| % of crossovers the model recommended avoiding | 50.0% |
| % of avoided signals that were genuine losses (correctly avoided) | 77.5% |
| % of accepted signals that were actually profitable | 73.9% |
| % of accepted signals that were actually losses | 26.1% |
| Profitable rate with NO ML filter (blind acceptance) | 48.2% |
| Profitable rate WITH ML filter | 73.9% |
| **Improvement from the ML filter** | **+25.7 percentage points** |

This directly demonstrates the assignment's core hypothesis: incorporating LTQ-based and SMMA-gap features into a filtering model measurably improves the profitability of the crossover strategy — roughly **1.5x-ing** the proportion of profitable trades taken versus accepting every signal blindly.

A live logging-and-resolution pipeline (`log_live_prediction()` in `data_engine.py`, `resolve_predictions.py`) is also implemented and actively running during live sessions, producing genuine next-day-style validation as more trading days elapse — see `VALIDATION_REPORT.md` Section 4 for current (early-stage, small-sample) results and honest limitations.

---

## 📝 Important Notes & Assumptions

- **Candle timeframe**: The assignment does not specify a candlestick interval for SMMA calculation. We used **1-minute candles** as a reasonable, commonly-used default for intraday screening.
- **Liquidity definition**: "Bid Quantity" and "Ask Quantity" were interpreted as the **total order-book depth** (`totBuyQuan` / `totSellQuan` from Angel One's quote API), rather than only the best/top price level, since this better reflects overall market liquidity relative to the 10,00,000 threshold.
- **P&L formula**: We implemented the assignment's formula exactly as specified — `Profit/Loss = Exit LTP − Entry LTP`, applied identically to both Buy and Sell trades. One consequence of this literal formula: a Sell trade can be labeled "Profitable" even if price rose after entry (since exit − entry is still positive), which differs from traditional short-selling logic where a Sell profits from a price *decline*. We identified this early during model development — it initially caused the model to rely almost entirely on trade direction as a shortcut feature (see ML section above) — verified it was a mathematical property of the given formula rather than a bug, and explicitly excluded trade direction from the final feature set so the model learns from genuine market signals instead. We chose to preserve the assignment's literal P&L formula for consistency with the stated requirements, rather than substitute our own trading convention.
- **LTQ proxy**: As detailed in the ML section above, `ltq_ratio_2v5` uses 1-minute candle volume as a proxy for tick-level LTQ, since historical tick data isn't available via Angel One's REST API.
- **Live re-screening**: The dashboard re-checks each stock's live price and liquidity on every refresh cycle (using the same market depth data already being fetched for the Market Depth columns) and drops any stock that no longer qualifies from that cycle's view — the initial `qualified_stocks.csv` snapshot is only used as the starting universe, not as permanent ground truth. Outside of live market hours, or when order-book data is thin/asymmetric (one side reads zero), the system conservatively keeps the stock rather than disqualifying it on unreliable data.
- **Rate limiting**: Angel One's historical data and quote APIs enforce strict rate limits. The pipeline caches candle history per stock (in memory and on disk) and only fetches incremental updates after the first load, with retry logic and backoff delays as an additional safeguard.
- **Market hours**: ETQ, average price, and market depth calculations depend on live trading activity and will show reduced or stale values outside NSE trading hours (9:15 AM–3:30 PM IST, weekdays).
- **Next-day validation timeline**: True two-session (train on Day N, validate live on Day N+1) validation could not be completed within the resubmission timeline. A held-out stock-grouped test set was used as a methodologically comparable substitute for the primary reported numbers, and a live logging pipeline was built and is actively running toward producing genuine next-day results — see `VALIDATION_REPORT.md`.

---

## 🔐 Security

All credentials (Angel One API key, client ID, password, TOTP secret, Groq API key) are loaded from a local `.env` file, which is excluded from version control via `.gitignore`. No credentials are hardcoded or included in this submission.

---

## 🚀 Possible Future Improvements

- Let the live logging/resolution pipeline (`live_predictions_log.csv` + `resolve_predictions.py`) accumulate more trading days to produce a statistically robust temporal (next-day) validation figure, directly comparable to the held-out test set result
- WebSocket-based live tick streaming for true tick-level LTQ and lower-latency LTP updates, instead of periodic REST polling
- Additional engineered features (order book imbalance, sector-level trends)
- Larger historical lookback window for training data
- Parallelized candle fetching to further reduce full-cycle refresh time
- Automated retraining pipeline that periodically incorporates newly resolved live predictions back into the training set