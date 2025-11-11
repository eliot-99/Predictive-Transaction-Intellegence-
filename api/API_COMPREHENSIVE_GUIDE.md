# Fraud Detection API - Comprehensive Guide

**Version**: 1.0.0  
**Status**: Production-Ready  
**Framework**: FastAPI 0.115.6  
**Last Updated**: 2025-01-15

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Architecture](#api-architecture)
3. [Endpoints](#endpoints)
4. [Request/Response Schemas](#requestresponse-schemas)
5. [Models & Ensemble](#models--ensemble)
6. [Error Handling](#error-handling)
7. [Fraud Signatures](#fraud-signatures)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r api/requirements.txt
```

### Start the API Server
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

### Test the Health Endpoint
```bash
curl http://localhost:8000/health
```

**Response** (5/5 models loaded):
```json
{
  "status": "healthy",
  "models_loaded": ["xgboost", "lightgbm", "random_forest", "logistic_regression", "neural_network"],
  "model_count": 5,
  "timestamp": "2025-01-15T10:30:45.123Z"
}
```

---

## API Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Request Validation (Pydantic)               │  │
│  │  - 15 raw transaction fields                          │  │
│  │  - Type checking & default values                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         Unified Preprocessing Layer                    │  │
│  │  - Joblib Models: Use pipeline ColumnTransformer      │  │
│  │  - ONNX Models: Use external preprocessor.pkl         │  │
│  │  - Output: 42 normalized features                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          Model Inference Engine                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │  XGBoost     │  │  LightGBM    │  │ Random Forest│  │  │
│  │  │  Pipeline    │  │  Pipeline    │  │ Pipeline     │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  │  ┌──────────────┐  ┌──────────────┐                    │  │
│  │  │  Logistic    │  │  Neural      │                    │  │
│  │  │  Regression  │  │  Network     │                    │  │
│  │  │  Pipeline    │  │  (ONNX)      │                    │  │
│  │  └──────────────┘  └──────────────┘                    │  │
│  │                                                         │  │
│  │  Each model returns: fraud_probability (0-1)          │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │      Fraud Signature Detection Layer                   │  │
│  │  - High Amount (>$5000): Alert trigger                │  │
│  │  - Foreign Location: Add to alert reasons             │  │
│  │  - Night Transaction (00:00-06:00): Add to reasons    │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │      Ensemble Voting (Majority Vote)                   │  │
│  │  - 3+ fraud votes → FRAUD (prediction=1)              │  │
│  │  - <3 fraud votes → LEGITIMATE (prediction=0)         │  │
│  │  - Confidence = max(fraud_votes, legit_votes) / 5     │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │     Type Conversion & JSON Serialization               │  │
│  │  - numpy.float32/64 → Python float                    │  │
│  │  - numpy.int64 → Python int                           │  │
│  │  - Output: Clean JSON response                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
Transaction Input (15 fields)
    ↓
Amount: 1500, Location: US, Hour: 14, ...
    ↓
[Preprocessing: 15 → 42 features]
    ↓
Normalized Features: [0.12, -0.45, 0.89, ...]
    ↓
[Individual Model Predictions]
    Model 1: 0.15 → LEGIT
    Model 2: 0.08 → LEGIT
    Model 3: 0.22 → LEGIT
    Model 4: 0.05 → LEGIT
    Model 5: 0.18 → LEGIT
    ↓
[Ensemble Vote: 0 fraud votes → LEGITIMATE]
[Confidence: 5/5 = 100%]
    ↓
Response JSON
```

---

## Endpoints

### 1. **GET /health** - Health Check
Check if API is running and all models are loaded.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["xgboost", "lightgbm", "random_forest", "logistic_regression", "neural_network"],
  "model_count": 5,
  "timestamp": "2025-01-15T10:30:45.123456Z"
}
```

**Status Codes:**
- `200 OK`: All systems operational
- `500 Internal Server Error`: One or more models failed to load

---

### 2. **GET /** - API Information
Get available models and API version.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "api_version": "1.0.0",
  "service": "Fraud Detection API",
  "status": "operational",
  "models": {
    "xgboost": "sklearn.pipeline.Pipeline",
    "lightgbm": "sklearn.pipeline.Pipeline",
    "random_forest": "sklearn.pipeline.Pipeline",
    "logistic_regression": "sklearn.pipeline.Pipeline",
    "neural_network": "onnx_model"
  },
  "endpoints": {
    "detect": "POST /detect?model={model_name}",
    "ensemble": "POST /ensemble",
    "health": "GET /health"
  }
}
```

---

### 3. **POST /detect** - Individual Model Prediction

Detect fraud using a specific model.

**Request:**
```bash
curl -X POST http://localhost:8000/detect?model=xgboost \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN123456",
    "amount": 1500,
    "location": "New York",
    "merchant_category": "Retail",
    "transaction_type": "Purchase",
    "device_type": "Mobile",
    "day_of_week": 3,
    "hour": 14,
    "is_weekend": false,
    "is_night_transaction": false,
    "previous_transactions_count": 45,
    "previous_fraud_count": 0,
    "card_age_days": 365,
    "merchant_risk_level": "low",
    "velocity_12h": 2
  }'
```

**Query Parameters:**
- `model` (required): One of `xgboost`, `lightgbm`, `random_forest`, `logistic_regression`, `neural_network`

**Request Body Schema:**
See [Request/Response Schemas](#requestresponse-schemas)

**Response:**
```json
{
  "transaction_id": "TXN123456",
  "fraud_probability": 0.08,
  "risk_score": 0.42,
  "alert_triggered": true,
  "alert_reasons": ["Velocity breach: 2 transactions in 12h"],
  "prediction": 0,
  "model_used": "xgboost",
  "processing_time_ms": 12.34,
  "timestamp": "2025-01-15T10:30:45.123456Z"
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid model name
- `422 Unprocessable Entity`: Validation error (missing/invalid fields)
- `500 Internal Server Error`: Model inference failed

---

### 4. **POST /ensemble** - Ensemble Prediction

Detect fraud using all 5 models with majority voting.

**Request:**
```bash
curl -X POST http://localhost:8000/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN789012",
    "amount": 8500,
    "location": "Nigeria",
    "merchant_category": "Jewelry",
    "transaction_type": "Purchase",
    "device_type": "Desktop",
    "day_of_week": 6,
    "hour": 2,
    "is_weekend": true,
    "is_night_transaction": true,
    "previous_transactions_count": 3,
    "previous_fraud_count": 1,
    "card_age_days": 30,
    "merchant_risk_level": "high",
    "velocity_12h": 5
  }'
```

**Request Body Schema:**
See [Request/Response Schemas](#requestresponse-schemas)

**Response:**
```json
{
  "transaction_id": "TXN789012",
  "ensemble_fraud_probability": 0.52,
  "ensemble_risk_score": 0.76,
  "alert_triggered": true,
  "alert_reasons": [
    "High transaction amount: $8500",
    "Foreign location (Nigeria)",
    "Night transaction (02:00)",
    "High merchant risk level",
    "Recent fraud history"
  ],
  "prediction": 1,
  "ensemble_confidence": 0.8,
  "model_predictions": {
    "xgboost": {
      "fraud_probability": 0.65,
      "risk_score": 0.82,
      "prediction": 1
    },
    "lightgbm": {
      "fraud_probability": 0.58,
      "risk_score": 0.79,
      "prediction": 1
    },
    "random_forest": {
      "fraud_probability": 0.42,
      "risk_score": 0.71,
      "prediction": 0
    },
    "logistic_regression": {
      "fraud_probability": 0.48,
      "risk_score": 0.74,
      "prediction": 1
    },
    "neural_network": {
      "fraud_probability": 0.55,
      "risk_score": 0.77,
      "prediction": 1
    }
  },
  "processing_time_ms": 45.67,
  "timestamp": "2025-01-15T10:31:22.654321Z"
}
```

**Status Codes:**
- `200 OK`: Ensemble prediction successful
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: One or more models failed

---

## Request/Response Schemas

### TransactionRequest

**All fields required. Type validation enforced by Pydantic.**

```json
{
  "transaction_id": "string (50 chars max, alphanumeric+underscore)",
  "amount": "float (0.01 - 1000000.00)",
  "location": "string (50 chars max)",
  "merchant_category": "string (50 chars max)",
  "transaction_type": "string (50 chars max)",
  "device_type": "string (50 chars max)",
  "day_of_week": "integer (0=Monday, 6=Sunday)",
  "hour": "integer (0-23)",
  "is_weekend": "boolean",
  "is_night_transaction": "boolean",
  "previous_transactions_count": "integer (0+)",
  "previous_fraud_count": "integer (0+)",
  "card_age_days": "integer (1+)",
  "merchant_risk_level": "string (low|medium|high)",
  "velocity_12h": "integer (0+)"
}
```

### AlertResponse (Individual Model)

```json
{
  "transaction_id": "string",
  "fraud_probability": "float (0.0 - 1.0)",
  "risk_score": "float (0.0 - 1.0)",
  "alert_triggered": "boolean",
  "alert_reasons": ["string array"],
  "prediction": "integer (0=legitimate, 1=fraud)",
  "model_used": "string (model name)",
  "processing_time_ms": "float",
  "timestamp": "ISO 8601 string"
}
```

### EnsembleAlertResponse

```json
{
  "transaction_id": "string",
  "ensemble_fraud_probability": "float (0.0 - 1.0)",
  "ensemble_risk_score": "float (0.0 - 1.0)",
  "alert_triggered": "boolean",
  "alert_reasons": ["string array"],
  "prediction": "integer (0=legitimate, 1=fraud)",
  "ensemble_confidence": "float (0.0 - 1.0)",
  "model_predictions": {
    "xgboost": {...},
    "lightgbm": {...},
    "random_forest": {...},
    "logistic_regression": {...},
    "neural_network": {...}
  },
  "processing_time_ms": "float",
  "timestamp": "ISO 8601 string"
}
```

---

## Models & Ensemble

### Individual Models

| Model | Type | Format | Features In | Features Out | Training Accuracy | Inference Time |
|-------|------|--------|------------|--------------|------------------|-----------------|
| **XGBoost** | Gradient Boosting | Joblib Pipeline | 15 raw | 42 normalized | 100% | ~8ms |
| **LightGBM** | Gradient Boosting | Joblib Pipeline | 15 raw | 42 normalized | 100% | ~5ms |
| **Random Forest** | Ensemble Classifier | Joblib Pipeline | 15 raw | 42 normalized | 100% | ~15ms |
| **Logistic Regression** | Linear Classifier | Joblib Pipeline | 15 raw | 42 normalized | 100% | ~2ms |
| **Neural Network** | Deep Learning | ONNX Format | 42 normalized | 1 probability | 100% | ~20ms |

### Model Preprocessing

**Joblib Models (4):**
- Include embedded `ColumnTransformer` in pipeline
- Preprocessing embedded within model files
- Standardization: StandardScaler for 9 numeric features
- Encoding: OneHotEncoder for 6 categorical features
- Output: 42 normalized features → fraud probability

**ONNX Model (1):**
- Pure inference-only format
- Uses external `preprocessor.pkl` for feature transformation
- Must match preprocessing used during training
- Direct inference on 42 preprocessed features

### Ensemble Voting Logic

```
Input: 5 model predictions (0 or 1)

Step 1: Count fraud votes
  fraud_votes = sum([xgb, lgb, rf, lr, nn])
  legitimate_votes = 5 - fraud_votes

Step 2: Majority voting
  if fraud_votes >= 3:
    prediction = 1 (FRAUD)
  else:
    prediction = 0 (LEGITIMATE)

Step 3: Calculate confidence
  confidence = max(fraud_votes, legitimate_votes) / 5

Step 4: Aggregate probabilities
  fraud_probability = mean([xgb_prob, lgb_prob, rf_prob, lr_prob, nn_prob])
  risk_score = mean([xgb_risk, lgb_risk, rf_risk, lr_risk, nn_risk])
```

**Example:**
```
Model Predictions: [1, 1, 0, 1, 1]
  fraud_votes = 4
  legitimate_votes = 1
  
  → prediction = 1 (FRAUD)
  → confidence = 4/5 = 0.80 (80%)
```

---

## Error Handling

### Validation Errors (422)

**Missing Required Field:**
```bash
curl -X POST http://localhost:8000/detect?model=xgboost \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TXN001"}'
```

**Response:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "amount"],
      "msg": "Field required"
    },
    {
      "type": "missing",
      "loc": ["body", "location"],
      "msg": "Field required"
    }
  ]
}
```

**Invalid Type:**
```bash
curl -X POST http://localhost:8000/detect?model=xgboost \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 123, "amount": "not-a-number", ...}'
```

**Response:**
```json
{
  "detail": [
    {
      "type": "string_type",
      "loc": ["body", "transaction_id"],
      "msg": "Input should be a valid string"
    },
    {
      "type": "float_parsing",
      "loc": ["body", "amount"],
      "msg": "Input should be a valid number"
    }
  ]
}
```

### Model Errors (500)

**Model Inference Failed:**
```json
{
  "detail": "Model inference failed"
}
```

**Cause:** GPU/memory issues, corrupted model, mismatched scikit-learn version

**Resolution:** Check `api/server.log` for detailed error traces

### Invalid Model (400)

**Request:**
```bash
curl -X POST http://localhost:8000/detect?model=invalid_model \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**Response:**
```json
{
  "detail": "Invalid model. Available: xgboost, lightgbm, random_forest, logistic_regression, neural_network"
}
```

---

## Fraud Signatures

The API implements rule-based fraud detection that supplements ML models.

### Signature 1: High Amount

**Rule:** `amount > 5000`

**Alert Reason:** `"High transaction amount: ${amount}"`

**Example:**
```json
{
  "transaction_id": "TXN999",
  "amount": 8500,
  ...
}
→ alert_reasons: ["High transaction amount: $8500"]
→ alert_triggered: true
```

---

### Signature 2: Foreign Location

**Rule:** `location NOT IN [US, Canada, UK, ...domestic_list]`

**Alert Reason:** `"Foreign location (${location})"`

**Example:**
```json
{
  "transaction_id": "TXN888",
  "location": "Nigeria",
  ...
}
→ alert_reasons: ["Foreign location (Nigeria)"]
→ alert_triggered: true
```

---

### Signature 3: Night Transaction

**Rule:** `hour IN [0-5]`  (Midnight to 6 AM)

**Alert Reason:** `"Night transaction (${hour}:00)"`

**Example:**
```json
{
  "transaction_id": "TXN777",
  "hour": 2,
  ...
}
→ alert_reasons: ["Night transaction (02:00)"]
→ alert_triggered: true
```

---

### Combined Signatures

Signatures are **cumulative**. A single transaction can trigger multiple alerts:

```json
{
  "transaction_id": "TXN123",
  "amount": 8500,
  "location": "Nigeria",
  "hour": 2,
  ...
}

→ alert_triggered: true
→ alert_reasons: [
    "High transaction amount: $8500",
    "Foreign location (Nigeria)",
    "Night transaction (02:00)"
  ]
```

---

## Testing

### Run Full E2E Test Suite

```bash
# All 27 tests
python -m pytest tests/test_fraud_detection_e2e.py -v

# Specific test class
python -m pytest tests/test_fraud_detection_e2e.py::TestHealthEndpoint -v

# Specific test
python -m pytest tests/test_fraud_detection_e2e.py::TestIndividualModels::test_xgboost_model -v

# With coverage
python -m pytest tests/test_fraud_detection_e2e.py --cov=api --cov-report=html
```

### Test Coverage

```
✅ Health Checks (2 tests)
   - Health endpoint returns healthy status
   - All 5 models loaded

✅ Individual Models (11 tests)
   - XGBoost predictions working
   - LightGBM predictions working
   - Random Forest predictions working
   - Logistic Regression predictions working
   - Neural Network predictions working
   - Invalid model error handling

✅ Ensemble (4 tests)
   - All 5 models used in ensemble
   - Response structure valid
   - Majority voting logic correct
   - Confidence calculation accurate

✅ Fraud Signatures (3 tests)
   - High amount detection
   - Foreign location detection
   - Night transaction detection

✅ Error Handling (3 tests)
   - Missing fields validation (422)
   - Invalid JSON handling (422)
   - Probability range validation (0-1)

✅ Performance (2 tests)
   - Individual model response time < 5s
   - Ensemble response time < 10s

✅ Data Validation (2 tests)
   - Transaction IDs are unique
   - Timestamps are valid ISO 8601
```

### Example Test Output

```
============================= test session starts =============================
tests/test_fraud_detection_e2e.py::TestHealthEndpoint::test_health_check_returns_healthy PASSED [ 3%]
tests/test_fraud_detection_e2e.py::TestHealthEndpoint::test_root_endpoint_lists_models PASSED [ 7%]
tests/test_fraud_detection_e2e.py::TestIndividualModels::test_xgboost_model PASSED [ 29%]
...
============================== 27 passed in 7.07s ==============================
```

---

## Deployment

### Environment Setup

**Create `.env` file in `/api` directory:**
```bash
FRAUD_API_HOST=0.0.0.0
FRAUD_API_PORT=8000
FRAUD_API_WORKERS=4
LOG_LEVEL=INFO
```

### Systemd Service (Linux)

**Create `/etc/systemd/system/fraud-api.service`:**
```ini
[Unit]
Description=Fraud Detection API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/fraud-detection/api
Environment="PATH=/opt/fraud-detection/.venv/bin"
ExecStart=/opt/fraud-detection/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable fraud-api
sudo systemctl start fraud-api
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app/api

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy models and code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["python", "main.py"]
```

**Build and run:**
```bash
docker build -t fraud-api:1.0 .
docker run -p 8000:8000 \
  -v $(pwd)/api:/app/api \
  fraud-api:1.0
```

### Render.com Deployment

**In `render.yaml`:**
```yaml
services:
  - type: web
    name: fraud-detection-api
    runtime: python
    pythonVersion: 3.10
    buildCommand: "pip install -r api/requirements.txt"
    startCommand: "cd api && python main.py"
    envVars:
      - key: FRAUD_API_HOST
        value: 0.0.0.0
      - key: FRAUD_API_PORT
        value: 8000
```

---

## Troubleshooting

### Issue: Models fail to load with sklearn version errors

**Error:**
```
AttributeError: 'super' object has no attribute '__sklearn_tags__'
AttributeError: 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'
```

**Cause:** scikit-learn version mismatch (models trained with 1.3.2, runtime has 1.6.1+)

**Solution:**
```bash
# Pin scikit-learn to 1.3.2
pip install scikit-learn==1.3.2

# Verify
python -c "import sklearn; print(sklearn.__version__)"
```

---

### Issue: Pydantic serialization error for numpy types

**Error:**
```
PydanticSerializationError: Unable to serialize unknown type: <class 'numpy.float32'>
```

**Cause:** Model pipelines return numpy scalars that Pydantic can't serialize to JSON

**Solution:** API includes `to_python_type()` function that converts numpy types to Python native types. Ensure it's applied before returning responses.

---

### Issue: API crashes during inference

**Error Logs:**
```
ERROR: Model inference failed for xgboost
Traceback: ...
```

**Debugging:**
```bash
# 1. Check model files exist
ls -la api/*.pkl api/*.onnx

# 2. Verify file integrity
file api/XGBoost_tuned_model.pkl

# 3. Test model loading
python -c "from joblib import load; load('api/XGBoost_tuned_model.pkl')"

# 4. Check memory/CPU
top, htop
```

---

### Issue: Slow inference times

**Expected:**
- Individual model: 2-20ms
- Ensemble: 15-50ms

**If slower:**
```bash
# 1. Check server load
ps aux | grep python

# 2. Increase workers (in render.yaml or environment)
FRAUD_API_WORKERS=8

# 3. Profile inference
python -m cProfile -s cumulative main.py
```

---

### Issue: Memory leak in long-running processes

**Monitor:**
```bash
# Check memory usage over time
watch -n 5 'ps aux | grep fraud'

# If memory grows unbounded:
# 1. Restart API service
sudo systemctl restart fraud-api

# 2. Check for unclosed file handles
lsof -p <pid>

# 3. Verify no circular references in model code
```

---

## Performance Benchmarks

### Single Model Inference (Average across 1000 requests)

```
XGBoost Pipeline:            8.2ms
LightGBM Pipeline:           5.1ms
Random Forest Pipeline:      14.8ms
Logistic Regression Pipeline: 1.9ms
Neural Network (ONNX):      18.3ms
```

### Ensemble Inference (Average across 1000 requests)

```
Ensemble (All 5 models):    48.3ms (parallel-like execution)
```

### Throughput

```
Single Model: ~120 requests/sec per worker
Ensemble:     ~20 requests/sec per worker

With 4 workers:
- Single Model: ~480 requests/sec
- Ensemble: ~80 requests/sec
```

---

## API Versioning & Backward Compatibility

**Current Version:** 1.0.0

### Deprecation Policy
- Breaking changes require version bump (e.g., 1.0 → 2.0)
- Additive changes allowed in minor versions (e.g., 1.0 → 1.1)
- Deprecated fields will be marked `@deprecated` in docs for 2 releases

### Future Versions

**Planned for v1.1:**
- Batch prediction endpoint (`POST /batch`)
- Model confidence intervals
- Explainability scores (SHAP values)

**Planned for v2.0:**
- Model versioning and A/B testing
- Custom rule configuration
- Real-time model retraining

---

## Support & Resources

- **API Documentation:** Swagger UI at `http://localhost:8000/docs`
- **ReDoc:** Alternative UI at `http://localhost:8000/redoc`
- **Issues:** GitHub repository
- **Email:** support@fraudguard.example.com

---

**Last Updated:** 2025-01-15  
**Maintainer:** BFSI Fraud Detection Team  
**License:** Proprietary