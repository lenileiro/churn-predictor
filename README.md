# Churn Predictor

Binary classifier that predicts whether a telecom customer will churn, served via a FastAPI REST API.

Uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (7,043 customers, 19 features).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

For development (pytest):

```bash
pip install ".[dev]"
```

### Download the dataset

```bash
curl -o data.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Train the model

```bash
python train.py
```

This will:
- Explore the dataset (shape, missing values, class balance, feature ranges)
- Train an XGBoost classifier inside a sklearn Pipeline
- Print feature importances and evaluation metrics
- Save `model.pkl` to disk

Sample output:

```
Accuracy: 0.7530
F1 Score: 0.6217
AUC-ROC:  0.8357
```

## Run the API

```bash
uvicorn app:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "tenure": 12, "MonthlyCharges": 45.3, "Contract": "Month-to-month"}'
```

```json
{"churn_probability": 0.7756, "prediction": "Yes"}
```

Only `gender`, `tenure`, `MonthlyCharges`, and `Contract` are required. All other fields have sensible defaults. See `/docs` for the full schema with allowed values.

## Run tests

```bash
pytest -v
```

## Docker

```bash
python train.py
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

## Production considerations

- **Model versioning** — Track artifacts, training data, and hyperparameters with MLflow or DVC so runs are reproducible and comparable.
- **Monitoring & drift detection** — Log predictions and input distributions. Alert on data drift (e.g. EvidentlyAI) or performance degradation against a labeled holdout.
- **CI/CD pipeline** — Automated training, evaluation against a baseline, and gated deployment (canary or shadow mode).
- **Feature store** — Centralize feature definitions and serving so training and inference use identical transformations, eliminating train/serve skew.
- **Scalability** — Serve behind a load balancer, consider async prediction queues for batch scoring, and cache frequent predictions.
- **Security** — Rate limiting, authentication (API keys / OAuth), and HTTPS termination.
- **Observability** — Structured logging, request tracing, latency/error metrics (Prometheus + Grafana).
