import pandas as pd
import numpy as np
import joblib
import os

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/qr_sessions.csv")

os.makedirs("models", exist_ok=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Select Features and Target
# ==========================================

X = df[
    [
        "scan_time_delta",
        "ip_request_velocity",
        "user_agent_score",
        "bank_webhook_received"
    ]
]

y = df["is_spoof_attempt"]


# ==========================================
# 3. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 4. Create LightGBM Model
# ==========================================

model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)


# ==========================================
# 5. Train Model
# ==========================================

print("\nTraining LightGBM model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. Predict Probabilities
# ==========================================

y_probs = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. Cost-Sensitive Threshold Optimization
# ==========================================

COST_FN = 500
COST_FP = 100

best_threshold = 0.5
min_total_cost = float("inf")

metrics_at_best = {}


for threshold in np.arange(0.05, 0.95, 0.01):

    y_pred = (y_probs >= threshold).astype(int)

    fn = np.sum(
        (y_test == 1) &
        (y_pred == 0)
    )

    fp = np.sum(
        (y_test == 0) &
        (y_pred == 1)
    )

    total_cost = (
        fn * COST_FN +
        fp * COST_FP
    )

    if total_cost < min_total_cost:

        min_total_cost = total_cost

        best_threshold = threshold

        metrics_at_best = {
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "fn": fn,
            "fp": fp,
            "threshold": threshold,
            "cost": total_cost
        }


# ==========================================
# 8. Performance Report
# ==========================================

print("\n")
print("==================================================")
print("       KAVACH-QR MODEL PERFORMANCE REPORT")
print("==================================================")

print(
    f"Optimal Decision Threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Precision @ Best Threshold : "
    f"{metrics_at_best['precision'] * 100:.2f}%"
)

print(
    f"Recall @ Best Threshold    : "
    f"{metrics_at_best['recall'] * 100:.2f}%"
)

print(
    f"False Positives (FP Count) : "
    f"{metrics_at_best['fp']}"
)

print(
    f"False Negatives (FN Count) : "
    f"{metrics_at_best['fn']}"
)

print(
    f"Total Financial Business Loss : "
    f"₹{metrics_at_best['cost']:,}"
)

print("==================================================")


# ==========================================
# 9. Compare With Default 0.5 Threshold
# ==========================================

default_pred = (
    y_probs >= 0.5
).astype(int)

default_fn = np.sum(
    (y_test == 1) &
    (default_pred == 0)
)

default_fp = np.sum(
    (y_test == 0) &
    (default_pred == 1)
)

default_cost = (
    default_fn * COST_FN +
    default_fp * COST_FP
)

savings = (
    default_cost -
    metrics_at_best["cost"]
)

print(
    f"Savings vs Standard 0.5 Threshold: "
    f"₹{savings:,}"
)


# ==========================================
# 10. Save Model
# ==========================================

joblib.dump(
    model,
    "models/lgbm_fraud_model.pkl"
)

joblib.dump(
    {"threshold": best_threshold},
    "models/model_config.pkl"
)


print("\n==========================================")
print("SUCCESS: Model training completed!")
print("==========================================")
print("Saved:")
print("1. models/lgbm_fraud_model.pkl")
print("2. models/model_config.pkl")
print("==========================================")