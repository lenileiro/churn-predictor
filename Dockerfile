FROM python:3.12-slim AS build

WORKDIR /app

COPY pyproject.toml config.py train.py app.py ./
RUN pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY config.py app.py model.pkl model.pkl.sha256 ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
