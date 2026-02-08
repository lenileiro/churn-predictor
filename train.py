"""Train a churn prediction model on the Telco Customer Churn dataset."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_CHECKSUM_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
)


# ── Data loading & cleaning ──────────────────────────────────────────────────


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the CSV and apply basic cleaning.

    Cleaning decisions:
    - TotalCharges: the raw CSV stores this as a string column. 11 rows
      contain whitespace-only values (all customers with tenure=0 who
      haven't been billed yet). We coerce to numeric and fill with 0.
    - SeniorCitizen: stored as 0/1 integer unlike every other binary
      column. We convert to "Yes"/"No" strings so the encoder treats
      all categoricals uniformly.
    - Churn target: map to 0/1 for the classifier.
    - customerID: dropped — it's an identifier, not a feature.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


# ── Data exploration ─────────────────────────────────────────────────────────


def explore(df: pd.DataFrame) -> None:
    """Print dataset summary to understand what we're working with."""
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Missing values: none")
    else:
        print("Missing values:")
        for col, count in missing.items():
            print(f"  {col}: {count} ({count / len(df):.1%})")
    print()

    # Class balance
    churn_counts = df["Churn"].value_counts()
    print("Target distribution:")
    print(f"  No churn: {churn_counts[0]:>5d} ({churn_counts[0] / len(df):.1%})")
    print(f"  Churn:    {churn_counts[1]:>5d} ({churn_counts[1] / len(df):.1%})")
    print()

    # Numeric feature ranges
    print("Numeric features:")
    for col in NUMERIC_FEATURES:
        print(f"  {col:>20s}:  min={df[col].min():.1f}  median={df[col].median():.1f}  max={df[col].max():.1f}")
    print()

    # Churn rate by contract type — the single strongest predictor.
    print("Churn rate by Contract type:")
    for contract, group in df.groupby("Contract")["Churn"]:
        print(f"  {contract:>20s}: {group.mean():.1%}")
    print()


# ── Pipeline construction ────────────────────────────────────────────────────


def build_pipeline() -> Pipeline:
    """Build a sklearn pipeline: preprocessing → model.

    The ColumnTransformer processes:
    - Numeric features → StandardScaler
    - Categorical features → OneHotEncoder (unknown categories ignored
      at predict time so the API doesn't crash on unseen values)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    # scale_pos_weight ~2.7 compensates for class imbalance (73.5% no-churn vs
    # 26.5% churn). This pushes the model toward higher recall on the minority
    # class at a small precision cost — a reasonable tradeoff when the business
    # cost of missing a churner exceeds the cost of a false alarm.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=2.7,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


# ── Feature importance ───────────────────────────────────────────────────────


def print_feature_importance(pipeline: Pipeline, top_n: int = 10) -> None:
    """Print the top features by XGBoost importance (gain)."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    classifier: XGBClassifier = pipeline.named_steps["classifier"]

    # Reconstruct feature names from the ColumnTransformer.
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    all_names = num_names + cat_names

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"Top {top_n} features (by importance):")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:>2d}. {all_names[idx]:<40s} {importances[idx]:.4f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    df = load_and_clean(DATA_PATH)
    explore(df)

    X = df[ALL_FEATURES]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/test split: {len(X_train)} / {len(X_test)} (stratified)\n")

    print("=" * 60)
    print("Training XGBoost pipeline...")
    print("=" * 60)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print()

    print_feature_importance(pipeline)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("Evaluation (hold-out test set)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    joblib.dump(pipeline, MODEL_PATH)
    checksum = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    MODEL_CHECKSUM_PATH.write_text(checksum)
    print(f"Model saved to {MODEL_PATH} (sha256: {checksum})")


if __name__ == "__main__":
    main()
