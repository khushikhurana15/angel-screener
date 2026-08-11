import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

df = pd.read_csv("training_data.csv")

df = df[df["status"] == "CLOSED"].copy()
print(f"Training on available rows: {len(df)}")

feature_cols = ["smma_gap_pct", "volatility", "hour", "volume_trend", "ltq_ratio_2v5"]
df = df.dropna(subset=feature_cols + ["profitable"])
print(f"Clean rows (after dropping missing values): {len(df)}")
print(f"Unique symbols: {df['symbol'].nunique()}")

X = df[feature_cols]
y = df["profitable"]
groups = df["symbol"]

# GroupShuffleSplit ensures every trade from a given symbol goes entirely
# into either train or test, never both. This avoids the model partially
# memorizing a stock's typical price level / behavior instead of learning
# generalizable crossover patterns - a real risk with the previous random
# row-level split, since each symbol had ~19-151 trades in the dataset.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_symbols = set(groups.iloc[train_idx])
test_symbols = set(groups.iloc[test_idx])
overlap = train_symbols & test_symbols
print(f"\nTrain symbols: {len(train_symbols)}, Test symbols: {len(test_symbols)}")
print(f"Symbol overlap between train/test: {len(overlap)} (should be 0)")

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy (stock-grouped split): {accuracy:.2%}")
print("\n📊 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=["Loss", "Profit"]))

print("\n📉 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted Loss   Predicted Profit")
print(f"Actual Loss      {cm[0][0]:<16} {cm[0][1]}")
print(f"Actual Profit    {cm[1][0]:<16} {cm[1][1]}")

print("\n🔑 Feature Importance:")
for feat, imp in zip(feature_cols, model.feature_importances_):
    print(f"   {feat}: {imp:.3f}")

joblib.dump(model, "models/crossover_model.joblib")
print("\n✅ Model saved to models/crossover_model.joblib")