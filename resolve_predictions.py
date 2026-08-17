"""
Run this on a SUBSEQUENT trading day (after live_predictions_log.csv has
entries from a previous live session) to resolve each logged prediction
against what actually happened - i.e. did the opposite crossover confirm
the predicted outcome, was it profitable or a loss.

This requires data_engine.py's log_live_prediction() (called from inside
predict_and_explain()) to have already run during at least one live
dashboard session before this script can produce results.

Key correctness point: this only looks at candles STRICTLY AFTER the
logged crossover's own timestamp (crossover_ts, from the candle data
itself), not the whole cached history - otherwise it could match a
crossover that happened before or at the same time as the logged signal,
or pick up the wrong entry price entirely.

Usage: python resolve_predictions.py
"""
import time
import pandas as pd
import data_engine as de

LOG_FILE = "live_predictions_log.csv"


def resolve_all():
    df = pd.read_csv(LOG_FILE)

    # pandas may read the boolean column back as string "False"/"True" after
    # a round-trip through CSV - normalize before filtering
    df["resolved"] = df["resolved"].astype(str).str.strip().str.lower() == "true"

    unresolved = df[df["resolved"] == False]

    print(f"Total logged predictions: {len(df)}")
    print(f"Unresolved (checking now): {len(unresolved)}")

    if len(unresolved) == 0:
        print("Nothing to resolve.")
        return

    de.login()
    de.load_cache_from_disk()

    qualified = {s["symbol"]: str(s["token"]) for s in de.load_qualified_stocks()}

    for idx, row in unresolved.iterrows():
        symbol = row["symbol"]
        token = qualified.get(symbol)
        if not token:
            print(f"SKIP: {symbol} not in current qualified_stocks.csv")
            continue

        entry_signal = row["signal"]
        entry_ltp = row["entry_ltp"]

        candles = de.get_candles_cached(token)
        if candles.empty:
            print(f"SKIP: {symbol} - no candle data available")
            continue

        candles = candles.copy()
        candles["_ts"] = pd.to_datetime(candles["timestamp"]).dt.tz_localize(None)

        row_ts = pd.to_datetime(row["crossover_ts"])
        if row_ts.tzinfo is not None:
            row_ts = row_ts.tz_localize(None)

        # Only consider candles strictly AFTER the logged signal's own
        # timestamp - this is what prevents matching a crossover that
        # happened before or during the logged signal itself.
        future_candles = candles[candles["_ts"] > row_ts].drop(columns=["_ts"]).reset_index(drop=True)

        if len(future_candles) < 10:
            print(f"WAIT: {symbol} - not enough new candles since signal yet ({len(future_candles)})")
            continue

        close_prices = future_candles["close"].tolist()
        future_candles["smma_20"] = de.calculate_smma(close_prices, de.config.SMMA_FAST)
        future_candles["smma_120"] = de.calculate_smma(close_prices, de.config.SMMA_SLOW)

        exit_crossover = de.find_latest_crossover(future_candles)

        if exit_crossover and exit_crossover["signal"] != entry_signal:
            exit_ltp = future_candles["close"].iloc[exit_crossover["index"]]
            pnl = exit_ltp - entry_ltp
            actual = "Profitable" if pnl > 0 else "Loss"

            df.at[idx, "actual_outcome"] = actual
            df.at[idx, "resolved"] = True
            print(f"RESOLVED: {symbol} ({entry_signal}) entry={entry_ltp} exit={exit_ltp:.2f} "
                  f"pnl={pnl:+.2f} predicted={row['ml_prediction']} actual={actual}")
        else:
            print(f"WAIT: {symbol} - no opposite crossover yet since signal")

        time.sleep(1)

    df.to_csv(LOG_FILE, index=False)
    print(f"\n✅ Updated {LOG_FILE}")

    resolved = df[df["resolved"] == True]
    if len(resolved) > 0:
        correctly_accepted = resolved[
            (resolved["ml_prediction"] == "Profitable") & (resolved["actual_outcome"] == "Profitable")
        ]
        correctly_avoided = resolved[
            (resolved["ml_prediction"] == "Avoid") & (resolved["actual_outcome"] == "Loss")
        ]
        correct = len(correctly_accepted) + len(correctly_avoided)
        print(f"\n--- Next-day live validation results ---")
        print(f"Total resolved signals: {len(resolved)}")
        print(f"Next-day validation accuracy: {correct}/{len(resolved)} ({correct/len(resolved)*100:.1f}%)")
    else:
        print("\nNo signals resolved yet - run again later once more opposite crossovers occur.")


if __name__ == "__main__":
    resolve_all()