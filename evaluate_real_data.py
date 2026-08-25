import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# KAVACHQR - REAL DATA EVALUATION
# ============================================================

DATA_PATH = "data/fraud_dataset.csv"
MODEL_PATH = "models/lgbm_fraud_model.pkl"
CONFIG_PATH = "models/model_config.pkl"
OUTPUT_PATH = "data/real_data_evaluation_results.csv"


print("=" * 70)
print("KAVACHQR - REAL-WORLD-LIKE DATA EVALUATION")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1] Checking files...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Model configuration not found: {CONFIG_PATH}"
    )

print(f"Dataset : {DATA_PATH}")
print(f"Model   : {MODEL_PATH}")
print(f"Config  : {CONFIG_PATH}")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[2] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 3. SHOW ACTUAL DATASET COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL DATASET COLUMNS")
print("=" * 70)

for i, col in enumerate(df.columns):
    print(f"{i + 1:3}. {col}")


# ============================================================
# 4. SHOW SAMPLE DATA
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE DATA")
print("=" * 70)

print(df.head(5).to_string(index=False))


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\n[3] Loading trained LightGBM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model type: {type(model)}")


# ============================================================
# 6. LOAD MODEL CONFIG
# ============================================================

print("\n[4] Loading model configuration...")

config = joblib.load(CONFIG_PATH)

print("Configuration loaded.")

print("\nConfiguration:")
print(config)


# ============================================================
# 7. GET MODEL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("MODEL EXPECTED FEATURES")
print("=" * 70)

model_features = None

# LightGBM sklearn model
if hasattr(model, "feature_name_"):
    model_features = list(model.feature_name_)

# Some models expose feature names differently
elif hasattr(model, "booster_"):
    try:
        model_features = list(model.booster_.feature_name())
    except Exception:
        pass


if model_features is not None:

    for i, col in enumerate(model_features):
        print(f"{i + 1:3}. {col}")

else:
    print("Could not automatically determine model feature names.")


# ============================================================
# 8. EXPECTED KAVACH FEATURES
# ============================================================

required_features = [
    "scan_time_delta",
    "ip_request_velocity",
    "user_agent_score",
    "bank_webhook_received"
]

target_column = "is_spoof_attempt"


print("\n" + "=" * 70)
print("KAVACH EXPECTED FEATURES")
print("=" * 70)

for col in required_features:
    print(f" - {col}")

print(f"\nTarget:")
print(f" - {target_column}")


# ============================================================
# 9. CHECK FEATURE AVAILABILITY
# ============================================================

available_features = [
    col for col in required_features
    if col in df.columns
]

missing_features = [
    col for col in required_features
    if col not in df.columns
]


print("\n" + "=" * 70)
print("FEATURE AVAILABILITY")
print("=" * 70)

print("\nAvailable features:")

if available_features:
    for col in available_features:
        print(f"  + {col}")
else:
    print("  None")


print("\nMissing features:")

if missing_features:
    for col in missing_features:
        print(f"  - {col}")
else:
    print("  None")


# ============================================================
# 10. CHECK TARGET
# ============================================================

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

if target_column in df.columns:
    print(f"Target '{target_column}' found.")
else:
    print(f"Target '{target_column}' NOT found.")


# ============================================================
# 11. STOP IF DATASET IS NOT COMPATIBLE
# ============================================================

if missing_features:

    print("\n" + "=" * 70)
    print("DATASET / MODEL FEATURE MISMATCH")
    print("=" * 70)

    print(
        "\nThe current dataset cannot be evaluated directly by "
        "this trained model."
    )

    print("\nMissing features:")

    for col in missing_features:
        print(f"  - {col}")

    print("\nYour dataset contains:")

    for col in df.columns:
        print(f"  - {col}")

    print("\nIMPORTANT:")
    print(
        "Do NOT fill missing fraud/security features with random "
        "values or zeros just to make the model run."
    )

    print(
        "\nThe dataset needs to be mapped to the exact feature "
        "schema used during model training."
    )

    print("\nModel features:")

    if model_features:
        for col in model_features:
            print(f"  - {col}")

    print("\nEvaluation stopped safely.")

    raise SystemExit(0)


# ============================================================
# 12. CHECK TARGET
# ============================================================

if target_column not in df.columns:

    print("\nERROR:")
    print(
        f"Target column '{target_column}' is missing."
    )

    print(
        "\nWithout the target column, the model can make "
        "predictions but classification metrics cannot be calculated."
    )

    raise SystemExit(0)


# ============================================================
# 13. DETERMINE FINAL MODEL FEATURES
# ============================================================

print("\n[5] Preparing model features...")


if model_features:

    # Use exactly the features expected by the trained model
    feature_columns = model_features

else:

    # Fallback to manually specified features
    feature_columns = required_features


print("\nFinal feature order:")

for i, col in enumerate(feature_columns):
    print(f"{i + 1:3}. {col}")


# ============================================================
# 14. VERIFY FINAL FEATURES
# ============================================================

missing_model_features = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_model_features:

    print("\nERROR: Model features missing from dataset:")

    for col in missing_model_features:
        print(f"  - {col}")

    raise ValueError(
        "Dataset does not contain all features required by the model."
    )


# ============================================================
# 15. PREPARE X AND Y
# ============================================================

print("\n[6] Preparing X and y...")

X = df[feature_columns].copy()

y = df[target_column].astype(int)


print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ============================================================
# 16. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DATA TYPES")
print("=" * 70)

print(X.dtypes)


# ============================================================
# 17. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = X.isnull().sum()

print(missing_values)


if missing_values.sum() > 0:

    print("\nERROR: Missing feature values detected.")

    print(
        "\nMissing values must be handled using the same "
        "preprocessing strategy used during model training."
    )

    raise ValueError(
        "Missing feature values detected."
    )


# ============================================================
# 18. CHECK NUMERIC FEATURES
# ============================================================

print("\n[7] Checking feature compatibility...")

non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()


if non_numeric_columns:

    print("\nNon-numeric columns detected:")

    for col in non_numeric_columns:
        print(f"  - {col}")

    print(
        "\nYour LightGBM model may require the same encoding "
        "used during training."
    )

    raise ValueError(
        "Non-numeric features detected. "
        "Use the training preprocessing pipeline."
    )


print("All features are numeric.")


# ============================================================
# 19. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    y.value_counts().sort_index()
)


# ============================================================
# 20. DETERMINE THRESHOLD
# ============================================================

print("\n[8] Determining prediction threshold...")

if isinstance(config, dict) and "threshold" in config:

    threshold = float(config["threshold"])

else:

    threshold = 0.5


print(f"Prediction threshold: {threshold}")


# ============================================================
# 21. RUN MODEL
# ============================================================

print("\n[9] Running LightGBM predictions...")

if not hasattr(model, "predict_proba"):

    raise AttributeError(
        "Loaded model does not support predict_proba()."
    )


y_probs = model.predict_proba(X)[:, 1]

y_pred = (
    y_probs >= threshold
).astype(int)


print("Predictions completed.")


# ============================================================
# 22. ADD PREDICTIONS
# ============================================================

df["fraud_probability"] = y_probs

df["predicted_fraud"] = y_pred


# ============================================================
# 23. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)


print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y,
        y_pred,
        target_names=[
            "GENUINE",
            "SPOOF/FRAUD"
        ],
        zero_division=0
    )
)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y,
    y_pred
)

print(cm)


if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Breakdown:")

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")


# ============================================================
# 26. ACTUAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL CLASS DISTRIBUTION")
print("=" * 70)

actual_distribution = (
    df[target_column]
    .value_counts()
    .sort_index()
)

for label, count in actual_distribution.items():

    class_name = (
        "GENUINE"
        if label == 0
        else "SPOOF/FRAUD"
    )

    print(
        f"{class_name:15}: {count}"
    )


# ============================================================
# 27. PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)

predicted_distribution = (
    df["predicted_fraud"]
    .value_counts()
    .sort_index()
)

for label, count in predicted_distribution.items():

    class_name = (
        "GENUINE"
        if label == 0
        else "SPOOF/FRAUD"
    )

    print(
        f"{class_name:15}: {count}"
    )


# ============================================================
# 28. INCORRECT PREDICTIONS
# ============================================================

wrong = df[
    df[target_column] != df["predicted_fraud"]
].copy()


print("\n" + "=" * 70)
print("INCORRECT PREDICTIONS")
print("=" * 70)

print(
    f"\nIncorrect predictions: {len(wrong)}"
)


if len(wrong) > 0:

    columns_to_show = []

    # Add session ID if it exists
    if "session_id" in wrong.columns:
        columns_to_show.append("session_id")

    # Add available model features
    for col in feature_columns:

        if col in wrong.columns:
            columns_to_show.append(col)

    # Add target/predictions
    columns_to_show.extend([
        target_column,
        "fraud_probability",
        "predicted_fraud"
    ])

    print(
        wrong[
            columns_to_show
        ].head(20).to_string(index=False)
    )

else:

    print("\nNo incorrect predictions.")


# ============================================================
# 29. HIGH-RISK TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP HIGH-RISK TRANSACTIONS")
print("=" * 70)

high_risk = df.sort_values(
    "fraud_probability",
    ascending=False
)


columns_to_show = []

if "session_id" in df.columns:
    columns_to_show.append("session_id")

for col in feature_columns:

    if col in df.columns:
        columns_to_show.append(col)

columns_to_show.extend([
    target_column,
    "fraud_probability",
    "predicted_fraud"
])


print(
    high_risk[
        columns_to_show
    ].head(20).to_string(index=False)
)


# ============================================================
# 30. SAVE RESULTS
# ============================================================

print("\n[10] Saving evaluation results...")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nResults saved to:\n{OUTPUT_PATH}"
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"\nDataset rows : {len(df)}")
print(f"Features     : {len(feature_columns)}")

print(f"\nAccuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

print(
    f"\nResults file : {OUTPUT_PATH}"
)

print("\n" + "=" * 70)