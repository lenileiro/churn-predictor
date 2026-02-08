"""FastAPI application for serving churn predictions."""

from contextlib import asynccontextmanager
from typing import Literal

import hashlib
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from sklearn.pipeline import Pipeline

from config import ALL_FEATURES, CHURN_THRESHOLD, MODEL_CHECKSUM_PATH, MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, verifying its checksum first."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    if not MODEL_CHECKSUM_PATH.exists():
        raise RuntimeError(f"Checksum not found at {MODEL_CHECKSUM_PATH}. Run train.py first.")

    expected = MODEL_CHECKSUM_PATH.read_text().strip()
    actual = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Model checksum mismatch: expected {expected}, got {actual}. "
            "The model file may have been tampered with."
        )

    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Churn Predictor",
    description="Predict customer churn from telco customer data.",
    version="0.1.0",
    lifespan=lifespan,
)


Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
PhoneLineType = Literal["Yes", "No", "No phone service"]
InternetDependentType = Literal["Yes", "No", "No internet service"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerInput(BaseModel):
    """Input features for a single customer prediction."""

    gender: Gender = Field(..., examples=["Female"])
    SeniorCitizen: YesNo = Field(default="No", examples=["No"])
    Partner: YesNo = Field(default="No", examples=["Yes"])
    Dependents: YesNo = Field(default="No", examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: YesNo = Field(default="Yes", examples=["Yes"])
    MultipleLines: PhoneLineType = Field(default="No", examples=["No"])
    InternetService: InternetServiceType = Field(default="Fiber optic", examples=["DSL"])
    OnlineSecurity: InternetDependentType = Field(default="No", examples=["No"])
    OnlineBackup: InternetDependentType = Field(default="No", examples=["No"])
    DeviceProtection: InternetDependentType = Field(default="No", examples=["No"])
    TechSupport: InternetDependentType = Field(default="No", examples=["No"])
    StreamingTV: InternetDependentType = Field(default="No", examples=["No"])
    StreamingMovies: InternetDependentType = Field(default="No", examples=["No"])
    Contract: ContractType = Field(..., examples=["Month-to-month"])
    PaperlessBilling: YesNo = Field(default="Yes", examples=["Yes"])
    PaymentMethod: PaymentMethodType = Field(
        default="Electronic check", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(..., ge=0, examples=[45.3])
    TotalCharges: float = Field(default=0.0, ge=0, examples=[543.6])

    @model_validator(mode="after")
    def check_service_consistency(self):
        internet_dependent = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        if self.InternetService == "No":
            for field in internet_dependent:
                if getattr(self, field) == "Yes":
                    raise ValueError(
                        f"{field} cannot be 'Yes' when InternetService is 'No'"
                    )
        if self.PhoneService == "No" and self.MultipleLines == "Yes":
            raise ValueError(
                "MultipleLines cannot be 'Yes' when PhoneService is 'No'"
            )
        return self


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str


@app.get("/health")
def health():
    model: Pipeline | None = getattr(app.state, "model", None)
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    model: Pipeline | None = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    row = pd.DataFrame([{f: getattr(customer, f) for f in ALL_FEATURES}])

    probability = float(model.predict_proba(row)[0, 1])
    label = "Yes" if probability >= CHURN_THRESHOLD else "No"

    return PredictionResponse(churn_probability=round(probability, 4), prediction=label)
