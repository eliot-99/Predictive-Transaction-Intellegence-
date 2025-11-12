# Deployment Form Values

## Render Web Service — Backend API (FastAPI)

### Core Form Inputs
| Field | Value | Notes |
| --- | --- | --- |
| **Service Type** | Web Service | Required for HTTP-serving apps |
| **Name** | `fraudguard-api` | Matches `render.yaml` for auto-deploy compatibility |
| **Language** | Python 3 (Runtime: Python 3.13) | Align with `render.yaml` runtime |
| **Branch** | `main` | Use default production branch |
| **Region** | Oregon (US West) | Keeps parity with `render.yaml` (`region: oregon`) |
| **Root Directory** | `api` | Restricts build context to FastAPI service |
| **Build Command** | `pip install -r requirements.txt` | Installs FastAPI + ML dependencies |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` | Matches local `render.yaml` |
| **Instance Type** | Free (upgrade to Starter if cold-starts unacceptable) | Paid tiers give persistent uptime |
| **Health Check Path** | `/health` | Ensures Render monitors readiness |

### Required Environment Variables
| Key | Value | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Match default exposed by FastAPI app |
| `PYTHONUNBUFFERED` | `true` | Forces real-time stdout logging |
| `MODEL_PATH` | `fraud_model.onnx` (inside `api/`) | Override only if custom model filename |
| `PREPROC_PATH` | `preprocessor.pkl` (inside `api/`) | Override if preprocessing artifact renamed |

### Required Model Files
| Model | ONNX File | Joblib File | Purpose |
| --- | --- | --- | --- |
| XGBoost | `XGBoost_tuned_model.onnx` | `XGBoost_tuned_model.pkl` | Gradient boosting model for fraud detection |
| LightGBM | `LightGBM_tuned_model.onnx` | `LightGBM_tuned_model.pkl` | Gradient boosting model for fraud detection |
| Random Forest | `Random_Forest_tuned_model.onnx` | `Random_Forest_tuned_model.pkl` | Tree ensemble model for fraud detection |
| Logistic Regression | `Logistic_Regression_tuned_model.onnx` | `Logistic_Regression_tuned_model.pkl` | Linear model for fraud detection |
| Neural Network | `neural_network_model.onnx` | N/A | Deep learning model for fraud detection |
| Preprocessor | N/A | `preprocessor.pkl` | Data preprocessing pipeline |

### Deployment Checklist
1. Upload all required model files listed above (ONNX and joblib files) to the `api/` directory in your repository.
2. Confirm `render.yaml` is committed so future pushes auto-sync form settings.
3. After first deploy, copy the Render service URL (e.g., `https://fraudguard-api.onrender.com`).

## Vercel Project — Frontend WebApp (Flask)

### Core Project Settings
| Setting | Value | Notes |
| --- | --- | --- |
| **Project Name** | `fraudguard-webapp` | Matches `vercel.json` name |
| **Framework Preset** | Other | Uses custom WSGI entrypoint |
| **Root Directory** | Repository root (`.`) | Allows Vercel to read `vercel.json` |
| **Install Command** | `pip install -r WebApp/requirements.txt` | Ensures lightweight dependency set |
| **Build Command** | Leave empty (use Vercel default) | Python builder reads `vercel.json` |
| **Output Directory** | Leave empty | Serverless deployment, no static build step |
| **Python Version** | `3.11` (set `VERCEL_PYTHON_VERSION`) | Matches local dev |

### Required Environment Variables (Production)
| Key | Sample Value | Description |
| --- | --- | --- |
| `SECRET_KEY` | output from `generate_credentials.py` | Flask session secret |
| `DATA_ENCRYPTION_KEY` | Fernet key from `generate_credentials.py` | Encrypts MongoDB fields |
| `MONGODB_URI` | `mongodb+srv://USER:PASS@cluster.mongodb.net/fraudguard` | Atlas connection string |
| `MONGODB_DB_NAME` | `fraudguard` | Target database |
| `MONGODB_CONNECT_TIMEOUT_MS` | `5000` | Optional; keeps Atlas handshake fast |
| `MAIL_SERVER` | `smtp.gmail.com` | Outbound email host |
| `MAIL_PORT` | `587` | TLS submission port |
| `MAIL_USERNAME` | Gmail/SMTP username | Sender identity |
| `MAIL_PASSWORD` | App password (never raw Gmail password) | SMTP auth |
| `MAIL_USE_TLS` | `true` | Enables STARTTLS |
| `MAIL_USE_SSL` | `false` | Must stay false when TLS enabled |
| `MAIL_FROM_ADDRESS` | `alerts@fraudguard.com` | Display sender |
| `GEMINI_API_KEY` | Google AI Studio API key | Powers chatbot fallback |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Optional override |
| `FRAUD_API_URL` | Render URL from backend deploy (e.g., `https://fraudguard-api.onrender.com`) | Routes predictions through API |

### Post-Deploy Actions
1. Add the Render backend URL to `FRAUD_API_URL` and redeploy if it changes.
2. Configure custom domain (optional) → point DNS to Vercel project.
3. Verify protected routes by logging in and ensuring MongoDB data loads.

## Cross-Service Integration Values

| Consumer | Env Key | Value Source |
| --- | --- | --- |
| Vercel Frontend | `FRAUD_API_URL` | Render backend service base URL |
| Render Backend | N/A | Exposes REST endpoints at `/detect`, `/ensemble`, `/chat` |
| MongoDB Atlas | Whitelist | `0.0.0.0/0` (or specific Render/Vercel egress ranges) |

1. Deploy Render backend first to capture the live API URL.
2. Update Vercel environment variable `FRAUD_API_URL` with the Render URL before triggering Vercel redeploy.
3. Re-run Vercel redeploy (`Deploy` → `Redeploy with existing env`) so the frontend fetches the updated API base.