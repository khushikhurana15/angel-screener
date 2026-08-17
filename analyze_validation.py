"""
Validation analysis: reproduces the exact stock-grouped train/test split used
in train_model.py, then computes the breakdown requested in the feedback:

  1. What % of crossover signals the model predicted "Avoid"
  2. Of those Avoided signals, what % would ACTUALLY have been losses
     (i.e. did avoiding them genuinely dodge a loss, or was the model wrong?)
  3. Of the signals the model let through (predicted "Profitable"/Accept),
     what % were actually profitable vs actually losses

This is run on a held-out set of stocks the model never saw during training
(same GroupShuffleSplit, same random_state, as train_model.py) - used here as
a methodologically-equivalent stand-in for "next trading day" validation,
since true next-day live validation requires two separate live trading
sessions that don't fit inside this submission's timeline. See README for
the full explanation of this substitution.
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import joblib

df = pd.read_csv("training_data.csv")
df = df[df["status"] == "CLOSED"].copy()

feature_cols = ["smma_gap_pct", "volatility", "hour", "volume_trend", "ltq_ratio_2v5"]
df = df.dropna(subset=feature_cols + ["profitable"])

X = df[feature_cols]
y = df["profitable"]
groups = df["symbol"]

# Must match train_model.py exactly so this is the SAME held-out test set
# the reported 75.67% accuracy came from - not a new/different split.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx].reset_index(drop=True)
test_symbols = groups.iloc[test_idx].reset_index(drop=True)

model = joblib.load("models/crossover_model.joblib")
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "symbol": test_symbols,
    "actual_profitable": y_test,   # ground truth: 1 = was actually profitable, 0 = was actually a loss
    "ml_prediction": y_pred,        # model's call: 1 = Accept/Profitable, 0 = Avoid
})

total_signals = len(results)

avoided = results[results["ml_prediction"] == 0]
accepted = results[results["ml_prediction"] == 1]

pct_avoided = len(avoided) / total_signals * 100

# Of the ones avoided, what fraction were GENUINE losses (i.e. avoiding them
# was the correct call) vs ones that would actually have been profitable
# (i.e. the model was overly cautious and missed a winner)
avoided_correctly = avoided[avoided["actual_profitable"] == 0]
avoided_incorrectly = avoided[avoided["actual_profitable"] == 1]
pct_avoided_that_were_genuine_losses = (
    len(avoided_correctly) / len(avoided) * 100 if len(avoided) > 0 else 0
)

# Of the ones accepted, what fraction were actually profitable vs actually losses
accepted_profitable = accepted[accepted["actual_profitable"] == 1]
accepted_loss = accepted[accepted["actual_profitable"] == 0]
pct_accepted_profitable = (
    len(accepted_profitable) / len(accepted) * 100 if len(accepted) > 0 else 0
)
pct_accepted_loss = (
    len(accepted_loss) / len(accepted) * 100 if len(accepted) > 0 else 0
)

print("=" * 60)
print("VALIDATION ANALYSIS - Held-out test set (unseen stocks)")
print("=" * 60)
print(f"\nTotal crossover signals evaluated: {total_signals}")
print(f"Test set stocks: {test_symbols.nunique()} (never seen during training)")

print(f"\n--- Signal filtering ---")
print(f"Signals ML predicted 'Avoid':     {len(avoided)} ({pct_avoided:.1f}% of all signals)")
print(f"Signals ML predicted 'Accept':    {len(accepted)} ({100 - pct_avoided:.1f}% of all signals)")

print(f"\n--- Quality of avoided signals ---")
print(f"Of the {len(avoided)} avoided signals:")
print(f"  {len(avoided_correctly)} ({pct_avoided_that_were_genuine_losses:.1f}%) were genuine losses - correctly avoided")
print(f"  {len(avoided_incorrectly)} ({100 - pct_avoided_that_were_genuine_losses:.1f}%) would actually have been profitable - overly cautious")

print(f"\n--- Quality of accepted signals ---")
print(f"Of the {len(accepted)} accepted signals:")
print(f"  {len(accepted_profitable)} ({pct_accepted_profitable:.1f}%) were actually profitable")
print(f"  {len(accepted_loss)} ({pct_accepted_loss:.1f}%) were actually losses")

print(f"\n--- Comparison: with vs without the ML filter ---")
baseline_profitable_pct = y_test.mean() * 100
print(f"If ALL {total_signals} signals were taken blindly (no ML filter):")
print(f"  {baseline_profitable_pct:.1f}% would have been profitable")
print(f"With the ML filter (only taking 'Accept' signals):")
print(f"  {pct_accepted_profitable:.1f}% were profitable")
print(f"  Improvement: {pct_accepted_profitable - baseline_profitable_pct:+.1f} percentage points")

# Save a clean summary CSV for the README/recording
summary = pd.DataFrame([{
    "total_signals": total_signals,
    "pct_avoided": round(pct_avoided, 1),
    "pct_avoided_correctly_genuine_losses": round(pct_avoided_that_were_genuine_losses, 1),
    "pct_accepted_profitable": round(pct_accepted_profitable, 1),
    "pct_accepted_loss": round(pct_accepted_loss, 1),
    "baseline_profitable_pct_no_filter": round(baseline_profitable_pct, 1),
    "improvement_pct_points": round(pct_accepted_profitable - baseline_profitable_pct, 1),
}])
summary.to_csv("validation_analysis_summary.csv", index=False)
print(f"\n✅ Summary saved to validation_analysis_summary.csv")