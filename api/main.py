import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, field_validator

from chatbot_handler import chatbot_handler

MODEL_PATH = Path(__file__).parent / os.getenv("MODEL_PATH", "fraud_model.onnx")
PREPROC_PATH = Path(__file__).parent / os.getenv("PREPROC_PATH", "preprocessor.pkl")
PORT = int(os.getenv("PORT", 8000))

# Available models - support both ONNX and joblib formats
AVAILABLE_MODELS = {
    "xgboost": {
        "onnx": str(Path(__file__).parent / "XGBoost_tuned_model.onnx"),
        "joblib": str(Path(__file__).parent / "XGBoost_tuned_model.pkl"),
        "type": "gradient_boosting"
    },
    "lightgbm": {
        "onnx": str(Path(__file__).parent / "LightGBM_tuned_model.onnx"),
        "joblib": str(Path(__file__).parent / "LightGBM_tuned_model.pkl"),
        "type": "gradient_boosting"
    },
    "random_forest": {
        "onnx": str(Path(__file__).parent / "Random_Forest_tuned_model.onnx"),
        "joblib": str(Path(__file__).parent / "Random_Forest_tuned_model.pkl"),
        "type": "tree_ensemble"
    },
    "logistic_regression": {
        "onnx": str(Path(__file__).parent / "Logistic_Regression_tuned_model.onnx"),
        "joblib": str(Path(__file__).parent / "Logistic_Regression_tuned_model.pkl"),
        "type": "linear"
    },
    "neural_network": {
        "onnx": str(Path(__file__).parent / "neural_network_model.onnx"),
        "joblib": None,
        "type": "neural_network"
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

FRAUD_SIGNATURES = {
    "high_amount": 50_000_000,
    "foreign_country": ["Russia", "Turkey", "USA", "China", "UAE"],
    "velocity_threshold": 10,
    "night_transaction": (0, 5),
}

USER_BEHAVIOR: Dict[int, Dict[str, Any]] = {}
BEHAVIOR_LOCK = Lock()


class Transaction(BaseModel):
    User_ID: int
    Transaction_Amount: float
    Transaction_Location: str
    Merchant_ID: int
    Device_ID: int
    Card_Type: str
    Transaction_Currency: str
    Transaction_Status: str
    Previous_Transaction_Count: int
    Distance_Between_Transactions_km: float
    Time_Since_Last_Transaction_min: int
    Authentication_Method: str
    Transaction_Velocity: int
    Transaction_Category: str
    Transaction_Hour: int
    Transaction_Day: int
    Transaction_Month: int
    Transaction_Weekday: int
    Log_Transaction_Amount: float
    Velocity_Distance_Interact: float
    Amount_Velocity_Interact: float
    Time_Distance_Interact: float
    Hour_sin: float
    Hour_cos: float
    Weekday_sin: float
    Weekday_cos: float
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_empty_and_convert(cls, v):
        """Convert empty strings to appropriate defaults and validate types."""
        if v == '' or v is None:
            return v  # Let Pydantic handle the default
        if isinstance(v, str):
            try:
                # Try to convert string numbers to appropriate types
                if '.' in v:
                    return float(v)
                return int(v)
            except (ValueError, AttributeError):
                return v  # Keep as string if conversion fails
        return v


class AlertResponse(BaseModel):
    model_config = {'protected_namespaces': ()}

    Transaction_ID: int
    User_ID: int
    Fraud_Probability: float
    Final_Risk_Score: float
    isFraud_pred: int
    alert_triggered: bool
    alert_reasons: List[str]
    timestamp: str
    model_used: str


class EnsembleAlertResponse(BaseModel):
    model_config = {'protected_namespaces': ()}

    Transaction_ID: int
    User_ID: int
    Fraud_Probability: float
    Final_Risk_Score: float
    isFraud_pred: int
    alert_triggered: bool
    alert_reasons: List[str]
    timestamp: str
    ensemble_prediction: int
    ensemble_confidence: float
    model_predictions: Dict[str, Dict[str, Any]]
    models_used: List[str]


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    bot_message: str
    type: str
    model: str = "phi3-bfsi"


ModelUnion = Union[Tuple[ort.InferenceSession, str, str], Tuple[Any, None, str]]  # (session/pipeline, input_name, model_type)


def load_artifacts() -> Tuple[Any, Dict[str, ModelUnion]]:
    """Load preprocessor and all models (ONNX or joblib)."""
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROC_PATH}")
    
    preprocessor = joblib.load(PREPROC_PATH)
    models_dict: Dict[str, ModelUnion] = {}
    
    for model_key, model_config in AVAILABLE_MODELS.items():
        loaded = False

        # Try ONNX first
        onnx_file = model_config.get("onnx")
        if onnx_file and Path(onnx_file).exists():
            try:
                session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
                input_name = session.get_inputs()[0].name
                models_dict[model_key] = (session, input_name, "onnx")
                logger.info(f"Loaded ONNX model: {model_key} from {onnx_file}")
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load ONNX model {model_key}: {e}")

        # Fallback to joblib
        if not loaded:
            joblib_file = model_config.get("joblib")
            if joblib_file and Path(joblib_file).exists():
                try:
                    model = joblib.load(joblib_file)
                    models_dict[model_key] = (model, None, "joblib")
                    logger.info(f"Loaded joblib model: {model_key} from {joblib_file}")
                    loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load joblib model {model_key}: {e}")

        if not loaded:
            logger.warning(f"Model not found for {model_key}")
    
    if not models_dict:
        raise RuntimeError("No models could be loaded")
    
    return preprocessor, models_dict


def to_python_type(val: Any) -> Any:
    """Convert numpy types to Python native types."""
    if isinstance(val, np.ndarray):
        return float(val)
    elif isinstance(val, (np.floating, np.integer)):
        return float(val)
    elif isinstance(val, (list, tuple)):
        return [to_python_type(v) for v in val]
    elif isinstance(val, dict):
        return {k: to_python_type(v) for k, v in val.items()}
    return val


def evaluate_ensemble(
    payload: Dict[str, Any],
    preprocessor: Any,
    models_dict: Dict[str, ModelUnion],
) -> Dict[str, Any]:
    """Evaluate fraud risk using ensemble of all models."""
    frame = pd.DataFrame([payload])
    tx_id = int(datetime.utcnow().timestamp() * 1000)
    frame.insert(0, "Transaction_ID", tx_id)

    model_predictions = {}
    probabilities = []

    for model_name, model_tuple in models_dict.items():
        try:
            result = evaluate_risk(payload, preprocessor, model_tuple, model_name)
            model_predictions[model_name] = {
                "probability": result["Fraud_Probability"],
                "prediction": result["isFraud_pred"],
                "confidence": result["Fraud_Probability"]
            }
            probabilities.append(result["Fraud_Probability"])
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue

    if not probabilities:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="All models failed")

    # Ensemble prediction: average of probabilities
    ensemble_probability = sum(probabilities) / len(probabilities)
    ensemble_prediction = int(ensemble_probability > 0.5)
    ensemble_confidence = max(probabilities) if ensemble_prediction else min(probabilities)

    reasons: List[str] = []
    boost = 0.0

    if payload["Transaction_Amount"] > FRAUD_SIGNATURES["high_amount"]:
        reasons.append("High Amount")
        boost += 0.35
    if FRAUD_SIGNATURES["night_transaction"][0] <= payload["Transaction_Hour"] < FRAUD_SIGNATURES["night_transaction"][1]:
        reasons.append("Night")
        boost += 0.25
    if payload["Transaction_Velocity"] > FRAUD_SIGNATURES["velocity_threshold"]:
        reasons.append("High Velocity")
        boost += 0.20
    if payload["Transaction_Location"] in FRAUD_SIGNATURES["foreign_country"]:
        reasons.append("Foreign")
        boost += 0.25

    now = datetime.utcnow()
    user_id = payload["User_ID"]

    with BEHAVIOR_LOCK:
        profile = USER_BEHAVIOR.setdefault(user_id, {"devices": set(), "tx_count_1h": 0, "last_hour": now.hour})
        if payload["Device_ID"] not in profile["devices"]:
            reasons.append("New Device")
            boost += 0.30
            profile["devices"].add(payload["Device_ID"])
        if now.hour != profile["last_hour"]:
            profile["tx_count_1h"] = 0
            profile["last_hour"] = now.hour
        profile["tx_count_1h"] += 1
        if profile["tx_count_1h"] > 8:
            reasons.append("Burst")
            boost += 0.20

    final_score = min(ensemble_probability + boost, 1.0)
    alert = final_score > 0.7 or ensemble_prediction == 1

    return {
        "Transaction_ID": tx_id,
        "User_ID": user_id,
        "Fraud_Probability": round(ensemble_probability, 4),
        "Final_Risk_Score": round(final_score, 4),
        "isFraud_pred": ensemble_prediction,
        "alert_triggered": alert,
        "alert_reasons": reasons,
        "timestamp": now.isoformat(),
        "ensemble_prediction": ensemble_prediction,
        "ensemble_confidence": round(ensemble_confidence, 4),
        "model_predictions": model_predictions,
        "models_used": list(model_predictions.keys()),
    }


def evaluate_risk(
    payload: Dict[str, Any],
    preprocessor: Any,
    model_tuple: ModelUnion,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Evaluate fraud risk using a model (ONNX or joblib)."""
    frame = pd.DataFrame([payload])
    tx_id = int(datetime.utcnow().timestamp() * 1000)
    frame.insert(0, "Transaction_ID", tx_id)
    
    try:
        # Get model type
        if len(model_tuple) == 3:
            model_obj, input_name, model_type = model_tuple
        else:
            model_obj, input_name = model_tuple
            model_type = "onnx" if input_name else "joblib"
        
        # Apply preprocessing based on model type
        if model_type == "onnx":
            # ONNX models: use full preprocessing (42 features after one-hot encoding)
            features = preprocessor.transform(frame.drop(columns=["Transaction_ID", "User_ID"], errors="ignore"))
            if hasattr(features, "toarray"):
                features = features.toarray()
            features = np.asarray(features, dtype=np.float32)
            
            # Run ONNX inference
            result = model_obj.run(None, {input_name: features})
            if isinstance(result[0], np.ndarray):
                if result[0].ndim == 0:
                    probability = float(result[0])
                elif result[0].ndim == 1:
                    probability = float(result[0][0])
                else:
                    probability = float(result[0][0][0])
            else:
                probability = float(result[0][0][0])
        else:
            # joblib pipeline inference (includes preprocessing internally)
            try:
                probability = float(model_obj.predict_proba(frame.drop(columns=["Transaction_ID"], errors="ignore"))[0][1])
            except AttributeError as attr_err:
                # Handle sklearn version compatibility issues (e.g., monotonic_cst attribute missing)
                logger.warning(f"predict_proba failed for {model_name} due to sklearn compatibility: {attr_err}. Using predict fallback.")
                prediction_fallback = model_obj.predict(frame.drop(columns=["Transaction_ID"], errors="ignore"))[0]
                probability = 1.0 if prediction_fallback == 1 else 0.0
            
    except Exception as exc:
        logger.exception(f"Model inference failed for {model_name}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed") from exc

    prediction = int(probability > 0.5)
    reasons: List[str] = []
    boost = 0.0

    if payload["Transaction_Amount"] > FRAUD_SIGNATURES["high_amount"]:
        reasons.append("High Amount")
        boost += 0.35
    if FRAUD_SIGNATURES["night_transaction"][0] <= payload["Transaction_Hour"] < FRAUD_SIGNATURES["night_transaction"][1]:
        reasons.append("Night")
        boost += 0.25
    if payload["Transaction_Velocity"] > FRAUD_SIGNATURES["velocity_threshold"]:
        reasons.append("High Velocity")
        boost += 0.20
    if payload["Transaction_Location"] in FRAUD_SIGNATURES["foreign_country"]:
        reasons.append("Foreign")
        boost += 0.25

    now = datetime.utcnow()
    user_id = payload["User_ID"]

    with BEHAVIOR_LOCK:
        profile = USER_BEHAVIOR.setdefault(user_id, {"devices": set(), "tx_count_1h": 0, "last_hour": now.hour})
        if payload["Device_ID"] not in profile["devices"]:
            reasons.append("New Device")
            boost += 0.30
            profile["devices"].add(payload["Device_ID"])
        if now.hour != profile["last_hour"]:
            profile["tx_count_1h"] = 0
            profile["last_hour"] = now.hour
        profile["tx_count_1h"] += 1
        if profile["tx_count_1h"] > 8:
            reasons.append("Burst")
            boost += 0.20

    final_score = min(probability + boost, 1.0)
    alert = final_score > 0.7 or prediction == 1

    return {
        "Transaction_ID": tx_id,
        "User_ID": user_id,
        "Fraud_Probability": round(probability, 4),
        "Final_Risk_Score": round(final_score, 4),
        "isFraud_pred": prediction,
        "alert_triggered": alert,
        "alert_reasons": reasons,
        "timestamp": now.isoformat(),
        "model_used": model_name,
    }


def evaluate_ensemble(
    payload: Dict[str, Any],
    preprocessor: Any,
    models_dict: Dict[str, ModelUnion],
) -> Dict[str, Any]:
    """Evaluate fraud using all available models with majority voting."""
    frame = pd.DataFrame([payload])
    tx_id = int(datetime.utcnow().timestamp() * 1000)
    frame.insert(0, "Transaction_ID", tx_id)
    
    model_predictions: Dict[str, Dict[str, Any]] = {}
    fraud_votes = 0
    legitimate_votes = 0
    probabilities = []
    
    # Preprocess features once for all models
    try:
        features = preprocessor.transform(frame.drop(columns=["Transaction_ID", "User_ID"], errors="ignore"))
        if hasattr(features, "toarray"):
            features = features.toarray()
        features = np.asarray(features, dtype=np.float32)
    except Exception as exc:
        logger.exception("Feature preprocessing failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Feature preprocessing failed") from exc
    
    # Run all models
    for model_name, model_tuple in models_dict.items():
        try:
            if len(model_tuple) == 3:
                model_obj, input_name, model_type = model_tuple
            else:
                model_obj, input_name = model_tuple
                model_type = "onnx" if input_name else "joblib"
            
            if model_type == "onnx":
                # ONNX inference
                result = model_obj.run(None, {input_name: features})
                if isinstance(result[0], np.ndarray):
                    if result[0].ndim == 0:
                        probability = float(result[0])
                    elif result[0].ndim == 1:
                        probability = float(result[0][0])
                    else:
                        probability = float(result[0][0][0])
                else:
                    probability = float(result[0][0][0])
            else:
                # joblib pipeline inference
                try:
                    probability = float(model_obj.predict_proba(frame.drop(columns=["Transaction_ID"], errors="ignore"))[0][1])
                except AttributeError as attr_err:
                    # Handle sklearn version compatibility issues (e.g., monotonic_cst attribute missing)
                    logger.warning(f"predict_proba failed for {model_name} due to sklearn compatibility: {attr_err}. Using predict fallback.")
                    prediction_fallback = model_obj.predict(frame.drop(columns=["Transaction_ID"], errors="ignore"))[0]
                    probability = 1.0 if prediction_fallback == 1 else 0.0
            
            probabilities.append(to_python_type(probability))
            prediction = int(probability > 0.5)
            
            if prediction == 1:
                fraud_votes += 1
            else:
                legitimate_votes += 1
            
            model_predictions[model_name] = {
                "probability": round(probability, 4),
                "prediction": prediction,
                "model_type": model_type,
            }
        except Exception as e:
            logger.error(f"Model {model_name} inference failed: {e}")
    
    # Ensemble decision
    total_votes = fraud_votes + legitimate_votes
    if total_votes == 0:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No models produced results")
    
    ensemble_prediction = 1 if fraud_votes > legitimate_votes else 0
    ensemble_confidence = float(max(fraud_votes, legitimate_votes) / total_votes)
    ensemble_probability = float(np.mean(probabilities))
    
    # Apply rule-based fraud signatures
    reasons: List[str] = []
    boost = 0.0

    if payload["Transaction_Amount"] > FRAUD_SIGNATURES["high_amount"]:
        reasons.append("High Amount")
        boost += 0.35
    if FRAUD_SIGNATURES["night_transaction"][0] <= payload["Transaction_Hour"] < FRAUD_SIGNATURES["night_transaction"][1]:
        reasons.append("Night")
        boost += 0.25
    if payload["Transaction_Velocity"] > FRAUD_SIGNATURES["velocity_threshold"]:
        reasons.append("High Velocity")
        boost += 0.20
    if payload["Transaction_Location"] in FRAUD_SIGNATURES["foreign_country"]:
        reasons.append("Foreign")
        boost += 0.25

    now = datetime.utcnow()
    user_id = payload["User_ID"]

    with BEHAVIOR_LOCK:
        profile = USER_BEHAVIOR.setdefault(user_id, {"devices": set(), "tx_count_1h": 0, "last_hour": now.hour})
        if payload["Device_ID"] not in profile["devices"]:
            reasons.append("New Device")
            boost += 0.30
            profile["devices"].add(payload["Device_ID"])
        if now.hour != profile["last_hour"]:
            profile["tx_count_1h"] = 0
            profile["last_hour"] = now.hour
        profile["tx_count_1h"] += 1
        if profile["tx_count_1h"] > 8:
            reasons.append("Burst")
            boost += 0.20

    final_score = min(ensemble_probability + boost, 1.0)
    alert = final_score > 0.7 or ensemble_prediction == 1

    return {
        "Transaction_ID": tx_id,
        "User_ID": user_id,
        "Fraud_Probability": round(ensemble_probability, 4),
        "Final_Risk_Score": round(final_score, 4),
        "isFraud_pred": ensemble_prediction,
        "alert_triggered": alert,
        "alert_reasons": reasons,
        "timestamp": now.isoformat(),
        "ensemble_prediction": ensemble_prediction,
        "ensemble_confidence": round(ensemble_confidence, 4),
        "model_predictions": model_predictions,
        "models_used": list(model_predictions.keys()),
    }


app = FastAPI(title="Fraud Detection API", version="2.0")

# Load models at startup
try:
    preprocessor, models_dict = load_artifacts()
    logger.info(f"Loaded {len(models_dict)} models")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    preprocessor = None
    models_dict = {}


@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API v2",
        "available_models": list(models_dict.keys()),
        "endpoints": {
            "/health": "Health check",
            "/detect": "Fraud detection",
            "/detect?model=xgboost": "Use specific model",
            "/ensemble": "Ensemble prediction",
            "/docs": "Swagger UI"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": len(models_dict)}


@app.post("/detect")
def detect_fraud(transaction: Transaction, model: Optional[str] = Query(None)):
    """Detect fraud in a transaction."""
    if not models_dict:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No models available")
    
    payload = transaction.dict()
    
    if model and model in models_dict:
        # Use specific model
        model_tuple = models_dict[model]
        result = evaluate_risk(payload, preprocessor, model_tuple, model)
        return AlertResponse(**result)
    elif model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model {model} not found")
    else:
        # Use first available model (usually random_forest if available)
        default_model = next(iter(models_dict.keys()))
        model_tuple = models_dict[default_model]
        result = evaluate_risk(payload, preprocessor, model_tuple, default_model)
        return AlertResponse(**result)


@app.post("/ensemble")
def detect_fraud_ensemble(transaction: Transaction):
    """Detect fraud using ensemble of all models."""
    if not models_dict or len(models_dict) < 2:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Ensemble requires at least 2 models")
    
    payload = transaction.dict()
    result = evaluate_ensemble(payload, preprocessor, models_dict)
    return EnsembleAlertResponse(**result)


@app.post("/chat")
def chat_with_chatbot(chat_request: ChatRequest):
    """Chat with Phi3 BFSI fine-tuned chatbot."""
    try:
        if not chat_request.message.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty message")
        
        if not chatbot_handler.loaded:
            logger.info("Loading chatbot model on first request...")
            if not chatbot_handler.load_model():
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chatbot model failed to load")
        
        response_text = chatbot_handler.generate_response(
            chat_request.message,
            system_prompt=chat_request.system_prompt
        )
        response_type = chatbot_handler.determine_response_type(response_text)
        
        return ChatResponse(
            bot_message=response_text,
            type=response_type,
            model="phi3-bfsi"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Chat processing failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)