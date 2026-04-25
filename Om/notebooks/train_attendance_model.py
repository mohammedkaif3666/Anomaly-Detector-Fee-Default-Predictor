import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("../data/attendance_features.csv")

features = [
    "attendance_rate",
    "longest_absence_streak",
    "absence_in_last_30_days",
    "day_of_week_variance"
]

X = df[features]
y_true = df["is_anomalous"]  # Ground truth labels (for evaluation only)

# ── 2. Train Isolation Forest ─────────────────────────────────────────────────
# contamination = 0.14 because Kaif built ~14% anomalies into the data
model = IsolationForest(
    n_estimators=100,
    contamination=0.14,
    random_state=42
)
model.fit(X)

# ── 3. Predict ────────────────────────────────────────────────────────────────
# Isolation Forest returns: -1 = anomaly, 1 = normal
# We convert to:            1  = anomaly, 0 = normal  (matches our labels)
raw_predictions = model.predict(X)
y_pred = (raw_predictions == -1).astype(int)

df["predicted_anomaly"] = y_pred

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("=" * 50)
print("ATTENDANCE ANOMALY DETECTION - RESULTS")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))

# Show flagged students
flagged = df[df["predicted_anomaly"] == 1][["student_id", "attendance_rate", "longest_absence_streak", "absence_in_last_30_days"]]
print(f"\nFlagged Students ({len(flagged)} total):")
print(flagged.to_string(index=False))

# ── 5. Save Model ─────────────────────────────────────────────────────────────
with open("models/attendance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as attendance_model.pkl")
print("   → Upload this file to the /models folder on GitHub")

# ── 6. Quick Sanity Check ─────────────────────────────────────────────────────
# Phanindra will verify this — make sure a 99% attendance student is NOT flagged
print("\n── Sanity Check ──")
test_cases = pd.DataFrame([
    {"attendance_rate": 0.99, "longest_absence_streak": 1, "absence_in_last_30_days": 0, "day_of_week_variance": 0.0001},  # Should be NORMAL
    {"attendance_rate": 0.36, "longest_absence_streak": 12, "absence_in_last_30_days": 21, "day_of_week_variance": 0.003},  # Should be ANOMALOUS
])

test_pred = model.predict(test_cases)
labels = ["✅ NORMAL" if p == 1 else "🚨 ANOMALOUS" for p in test_pred]
print(f"Student with 99% attendance → {labels[0]}  (should be NORMAL)")
print(f"Student with 36% attendance → {labels[1]}  (should be ANOMALOUS)")
