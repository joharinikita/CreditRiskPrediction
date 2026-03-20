# Credit Risk Prediction

A data pipeline and FastAPI service that predicts whether a customer will default on a loan within 90 days, based on their transaction history.

---

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Step 1: Generate the training set from raw CSVs
python pipeline.py

# Step 2: Copy the provided model artifact
cp Atto/model.joblib artifacts/model.joblib

# Step 3: Start the prediction API
python -m uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

## API usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_0001",
    "num_transactions": 3,
    "total_debit": -65.98,
    "total_credit": 2500.0,
    "has_rent": false,
    "has_tesco": true,
    "has_restaurant": false,
    "has_salary": false,
    "has_utility": false,
    "has_transport": false
  }'
```

Response:
```json
{
  "customer_id": "CUST_0001",
  "probability": 0.12,
  "prediction": 0
}
```
Here's how to test it in the Swagger UI:

1)Open http://localhost:8000/docs in your browser
2)Click on the GET /health endpoint
3)Click "Try it out"
4) Click "Execute"
	You should see a response like:
	{
		"status": "ok"
	}

For the POST /predict endpoint:

1)Click on POST /predict
2)Click "Try it out"
3)Edit the request body with your values
4)Click "Execute"
5) You should see a response like:
{
  "customer_id": "CUST_0001",
  "probability": 0.123,
  "prediction": 0
}
---

## Documentation

### 1. What part of the exercise did you find most challenging, and why?

The data pipeline had the most subtle challenges. 
The original prototype was built in a Google Colab notebook using which  a clean, standalone script was created 
The API development was also non-trivial — 
specifically thinking about how the model artifact gets loaded at startup, ensuring Pydantic validation aligns exactly with the feature schema the model was trained on.

### 2. What tradeoffs did you make?

- **Simplicity over completeness**: Given the sample dataset has only 5 customers, model evaluation metrics are illustrative rather than meaningful. In production you'd want cross-validation and proper hold-out sets.
- **Single-process API**: The FastAPI service loads the model in-process. This is fine for the stated traffic (1000 predictions/hour) but would need revisiting at higher scale.
- **No input validation beyond types**: Added Pydantic type checking but skipped domain-level validation (e.g., `total_debit` should be ≤ 0). This keeps the code simple and avoids hard-coding assumptions about the data distribution.

### 3. Production on Azure with £500/month, <100ms latency, 1000 predictions/hour

1000 predictions/hour is ~0.28 req/sec — very low traffic. The entire stack would comfortably fit on **Azure Container Apps** (consumption plan, ~£20-50/month), which scales to zero when idle.

Changes I'd make first:
- **Model artifact in Azure Blob Storage** (not baked into the container image) so it can be updated without a redeployment.
- **Application Insights** for latency tracking and alerting — critical for catching p99 regressions before they become user-facing.
- **Health check + readiness probe** check on `/health` so the container orchestrator can restart unhealthy instances automatically.
- **Environment-based config** (model path, log level) via Azure Key Vault / env vars rather than hardcoded paths.

The <100ms latency target is easily achievable on this feature set ; the dominant cost will be network + serialisation, which Container Apps handles well at this traffic level.

### 4. How would you deploy the FastAPI service and make the model artifact available?

1. **Containerise**: `Dockerfile` → `uvicorn api:app --host 0.0.0.0 --port 8000`
2. **Push image** to Azure Container Registry (ACR)
3. **Deploy** to Azure Container Apps (or AKS for more control)
4. **Model artifact**: store `model.joblib` in Azure Blob Storage; the container downloads it on startup using the Azure SDK and the storage connection string passed as an environment variable. This decouples model versioning from image versioning.
5. **CI/CD**: GitHub Actions pipeline — on merge to `main`, rebuild image, push to ACR, trigger a new Container Apps revision.

### 5. If transaction volume jumped from thousands to millions per day, how would you rethink Part 1?

The current approach reads everything into pandas in memory — that breaks at millions of rows.

Rethink:
- **Batch pipeline**: Replace pandas with **Azure Databricks + PySpark** (or Synapse Analytics). Partition data by `customer_id` for parallelism.
- **Incremental processing**: Only process new transactions since the last run using watermarking (Delta Lake change data feed or Azure Event Hubs for streaming).
- **Feature store**: Materialise per-customer features into an **Azure ML Feature Store** or a dedicated table in Azure SQL / Cosmos DB. The API then reads pre-computed features rather than recomputing on the fly.
- **Orchestration**: **Azure Data Factory** or **Prefect/Airflow** to schedule and monitor pipeline runs, with alerting on failures.

### 6. What metrics would you track in production, and what could go wrong?

**Model metrics (tracked on a sample of decisions with delayed labels):**
- Precision, Recall,F1 — These measure how good the model is at detecting defaulters.
    Precision measures 
	Of all customers predicted to default, how many actually did? 
	Recall measures 
	Of all customers who actually defaulted, how many did we catch?
	F1
	Balance between Precision and Recall (2 × (Precision × Recall) / (Precision + Recall))
	
- AUC-ROC over rolling 30-day windows to catch model drift early
	measures how well the model ranks defaulters above non-defaulters

**Operational metrics:**
- Prediction latency (p50, p95, p99) — alerting threshold at 80ms to give headroom
- Request volume and error rate — sudden spikes or drops indicate upstream issues
- Prediction distribution (% predicted as default) — a shift here is often the first signal of data drift before labels arrive

**What could go wrong:**
- **Data drift**: Transaction patterns change seasonally (e.g., Christmas spending) or due to macroeconomic shocks. The model was trained on a small sample and may not generalise.
- **Label leakage**: If features are computed using data that wasn't available at the time of the original decision, the model will look better in training than it performs live.
- **Class imbalance**: Default rates are typically low (2-5%). Without careful handling, the model may optimise for accuracy by rarely predicting defaults — exactly the wrong behaviour.
- **Feedback loops**: If the model denies credit to certain customer profiles, those customers never default, which reinforces the model's prior — a well-known fairness and calibration risk in credit scoring.

### 7. AI tools used

Google COLAB was used for creating pipeline.py  
Claude .py sonnet-4-6 was used for the API development and for security analysis of the project.

These are the high Risk elements that came as a result of security analysis:
1. No Authentication on the API

/predict and /health are completely open — anyone who can reach the server can call them
In production, this should have API key authentication or OAuth2
Fix: Add an Authorization header check or use Azure API Management in front
2. Model loaded via joblib without integrity check

joblib.load() uses pickle under the hood — a maliciously crafted .joblib file can execute arbitrary code on load
If an attacker replaces artifacts/model.joblib, they own the server
Fix: Verify a SHA-256 checksum of the model file before loading it
3. No HTTPS
Running on plain http://localhost:8000 — in production, customer financial data would be sent unencrypted
Fix: Put behind Azure API Management or a reverse proxy (nginx) with TLS
All generated code was reviewed, tested, and adapted to match the actual data schema and project requirements. The core logic (feature engineering decisions, model choice, architecture recommendations) reflects independent judgement.
