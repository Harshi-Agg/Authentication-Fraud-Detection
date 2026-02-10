# ============================================================
# Adaptive Authentication Fraud Detection System
# Isolation Forest + Behavioral Risk Modeling
# (TensorFlow-FREE | IEEE-CIS Dataset)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# ------------------------------------------------------------
# 1. PATH SETUP
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"

TRAIN_TXN = DATA_DIR / "train_transaction.csv"
TRAIN_ID  = DATA_DIR / "train_identity.csv"

# ------------------------------------------------------------
# 2. LOAD & MERGE DATA
# ------------------------------------------------------------

print("Loading datasets...")

txn = pd.read_csv(TRAIN_TXN)
idt = pd.read_csv(TRAIN_ID)

df = txn.merge(idt, on="TransactionID", how="left")
print("Merged shape:", df.shape)

# ------------------------------------------------------------
# 3. FEATURE SELECTION
# ------------------------------------------------------------

cols = [
    "TransactionDT",
    "TransactionAmt",
    "card1", "card2",
    "addr1",
    "dist1",
    "C1", "C2", "C3",
    "D1", "D2",
    "DeviceType",
    "id_01", "id_02",
    "isFraud"
]

df = df[cols]

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

df["hour"] = (df["TransactionDT"] / 3600) % 24
df["day"]  = (df["TransactionDT"] / (3600 * 24)) % 7

df["DeviceType"] = df["DeviceType"].fillna("unknown")
df["device_change"] = (df["DeviceType"] == "unknown").astype(int)

df.drop(columns=["TransactionDT", "DeviceType"], inplace=True)

# ------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------------------------------------------------
# 6. ISOLATION FOREST (ANOMALY DETECTION)
# ------------------------------------------------------------

print("Training Isolation Forest...")

iso = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)

iso.fit(X[y == 0])

df["anomaly_raw"] = -iso.decision_function(X)

# Normalize anomaly score (VERY IMPORTANT)
anomaly_scaler = MinMaxScaler()
df["anomaly_score"] = anomaly_scaler.fit_transform(
    df[["anomaly_raw"]]
)

# ------------------------------------------------------------
# 7. BEHAVIORAL TEMPORAL MODELING
# ------------------------------------------------------------

WINDOW = 5
df["user_id"] = df["card1"]

df["behav_mean"] = 0.0
df["behav_std"]  = 0.0

for user in df["user_id"].unique():
    idx = df[df["user_id"] == user].index
    roll = df.loc[idx, "anomaly_score"].rolling(WINDOW)
    df.loc[idx, "behav_mean"] = roll.mean().fillna(0)
    df.loc[idx, "behav_std"]  = roll.std().fillna(0)

# ------------------------------------------------------------
# 8. BEHAVIORAL RISK CLASSIFIER
# ------------------------------------------------------------

risk_features = [
    "anomaly_score",
    "behav_mean",
    "behav_std"
]

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=6,
    random_state=42
)

rf.fit(df[risk_features], y)
df["behavior_risk"] = rf.predict_proba(df[risk_features])[:, 1]

# ------------------------------------------------------------
# 9. FINAL RISK FUSION
# ------------------------------------------------------------

df["final_risk"] = (
    0.6 * df["anomaly_score"] +
    0.4 * df["behavior_risk"]
)

# ------------------------------------------------------------
# 10. ADAPTIVE AUTHENTICATION DECISION
# ------------------------------------------------------------

def auth_decision(score):
    if score < 0.35:
        return "JWT Issued"
    elif score < 0.65:
        return "MFA Required"
    else:
        return "Access Blocked"

df["auth_decision"] = df["final_risk"].apply(auth_decision)

# ------------------------------------------------------------
# 11. DIAGNOSTICS
# ------------------------------------------------------------

print("\nFinal Risk Statistics:")
print(df["final_risk"].describe())

print("\nDecision Distribution:")
print(df["auth_decision"].value_counts())

# ------------------------------------------------------------
# 12. SAVE OUTPUT
# ------------------------------------------------------------

# OUTPUT_FILE = BASE_DIR / "authentication_results.csv"
# df.to_csv(OUTPUT_FILE, index=False)

# print(f"\nResults saved to: {OUTPUT_FILE}")
print("Pipeline executed successfully ✔")
