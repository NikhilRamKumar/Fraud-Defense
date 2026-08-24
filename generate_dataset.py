import pandas as pd
import numpy as np
import os

# ==========================================
# KAVACH-QR Synthetic Dataset Generator
# ==========================================

np.random.seed(42)

n_samples = 10000

# Create data folder
os.makedirs("data", exist_ok=True)

# ==========================================
# 1. Real Customer Scans - 85%
# ==========================================

n_real = int(n_samples * 0.85)

real_data = {
    "session_id": [
        f"SESS_REAL_{i:05d}"
        for i in range(n_real)
    ],

    "scan_time_delta":
        np.random.exponential(
            scale=12,
            size=n_real
        ) + 1.5,

    "ip_request_velocity":
        np.random.poisson(
            lam=1.2,
            size=n_real
        ),

    "user_agent_score":
        np.random.normal(
            loc=0.95,
            scale=0.03,
            size=n_real
        ).clip(0, 1),

    "bank_webhook_received":
        np.random.choice(
            [1, 0],
            size=n_real,
            p=[0.98, 0.02]
        ),

    "is_spoof_attempt": 0
}


# ==========================================
# 2. Fake / Spoof Scans - 15%
# ==========================================

n_spoof = n_samples - n_real

spoof_data = {
    "session_id": [
        f"SESS_FAKE_{i:05d}"
        for i in range(n_spoof)
    ],

    "scan_time_delta":
        np.random.exponential(
            scale=45,
            size=n_spoof
        ) + 10,

    "ip_request_velocity":
        np.random.poisson(
            lam=8.5,
            size=n_spoof
        ),

    "user_agent_score":
        np.random.normal(
            loc=0.30,
            scale=0.15,
            size=n_spoof
        ).clip(0, 1),

    "bank_webhook_received":
        np.random.choice(
            [0, 1],
            size=n_spoof,
            p=[0.99, 0.01]
        ),

    "is_spoof_attempt": 1
}


# ==========================================
# 3. Create DataFrames
# ==========================================

real_df = pd.DataFrame(real_data)

spoof_df = pd.DataFrame(spoof_data)


# ==========================================
# 4. Combine and Shuffle
# ==========================================

df = pd.concat(
    [real_df, spoof_df],
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 5. Save Dataset
# ==========================================

output_path = "data/qr_sessions.csv"

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 6. Confirmation
# ==========================================

print("==========================================")
print("KAVACH-QR DATASET GENERATED")
print("==========================================")
print(f"Total samples : {len(df)}")
print(f"Real scans    : {sum(df['is_spoof_attempt'] == 0)}")
print(f"Spoof scans   : {sum(df['is_spoof_attempt'] == 1)}")
print(f"Saved to      : {output_path}")
print("==========================================")