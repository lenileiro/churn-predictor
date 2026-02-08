"""Shared feature configuration for training and serving."""

from pathlib import Path

MODEL_PATH = Path("model.pkl")
MODEL_CHECKSUM_PATH = Path("model.pkl.sha256")
DATA_PATH = Path("data.csv")

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CHURN_THRESHOLD = 0.5
