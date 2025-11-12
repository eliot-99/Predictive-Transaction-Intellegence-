import pandas as pd
import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# API Configuration
API_BASE_URL = "https://fraudguard-api.onrender.com"
TIMEOUT = 30  # seconds

# Load test dataset
def load_test_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess test dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} test transactions")
    return df

def test_health_endpoint() -> Dict[str, Any]:
    """Test the /health endpoint."""
    print("\n=== Testing /health Endpoint ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        result = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "success": response.status_code == 200
        }
        print(f"Health check: {'PASS' if result['success'] else 'FAIL'}")
        print(f"Response: {result['response']}")
        return result
    except Exception as e:
        print(f"Health check failed with error: {e}")
        return {"success": False, "error": str(e)}

def prepare_transaction_data(row: pd.Series) -> Dict[str, Any]:
    """Prepare transaction data for API request."""
    # Map CSV columns to API expected fields
    transaction = {
        "User_ID": int(row["Transaction_ID"]),  # Using Transaction_ID as User_ID for testing
        "Transaction_Amount": float(row["Transaction_Amount"]),
        "Transaction_Location": str(row["Transaction_Location"]),
        "Merchant_ID": int(row["Merchant_ID"]),
        "Device_ID": int(row["Device_ID"]),
        "Card_Type": str(row["Card_Type"]),
        "Transaction_Currency": str(row["Transaction_Currency"]),
        "Transaction_Status": str(row["Transaction_Status"]),
        "Previous_Transaction_Count": int(row["Previous_Transaction_Count"]),
        "Distance_Between_Transactions_km": float(row["Distance_Between_Transactions_km"]),
        "Time_Since_Last_Transaction_min": int(row["Time_Since_Last_Transaction_min"]),
        "Authentication_Method": str(row["Authentication_Method"]),
        "Transaction_Velocity": int(row["Transaction_Velocity"]),
        "Transaction_Category": str(row["Transaction_Category"]),
        "Transaction_Hour": int(row["Transaction_Hour"]),
        "Transaction_Day": int(row["Transaction_Day"]),
        "Transaction_Month": int(row["Transaction_Month"]),
        "Transaction_Weekday": int(row["Transaction_Weekday"]),
        "Log_Transaction_Amount": float(row["Log_Transaction_Amount"]),
        "Velocity_Distance_Interact": float(row["Velocity_Distance_Interact"]),
        "Amount_Velocity_Interact": float(row["Amount_Velocity_Interact"]),
        "Time_Distance_Interact": float(row["Time_Distance_Interact"]),
        "Hour_sin": float(row["Hour_sin"]),
        "Hour_cos": float(row["Hour_cos"]),
        "Weekday_sin": float(row["Weekday_sin"]),
        "Weekday_cos": float(row["Weekday_cos"])
    }
    return transaction

def test_detect_endpoint(transaction: Dict[str, Any], model: str = None) -> Dict[str, Any]:
    """Test the /detect endpoint with optional model parameter."""
    try:
        url = f"{API_BASE_URL}/detect"
        if model:
            url += f"?model={model}"

        response = requests.post(url, json=transaction, timeout=TIMEOUT)
        result = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "success": response.status_code == 200,
            "model": model or "default"
        }
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "model": model or "default"}

def test_ensemble_endpoint(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Test the /ensemble endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}/ensemble", json=transaction, timeout=TIMEOUT)
        result = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "success": response.status_code == 200
        }
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def compare_predictions(api_response: Dict[str, Any], expected_fraud: int, expected_prob: float) -> Dict[str, Any]:
    """Compare API predictions with expected values."""
    if not api_response.get("success"):
        return {"comparison": "ERROR", "details": "API call failed"}

    response_data = api_response["response"]

    api_fraud = response_data.get("isFraud_pred")
    api_prob = response_data.get("Fraud_Probability")

    fraud_match = api_fraud == expected_fraud
    prob_diff = abs(api_prob - expected_prob) if api_prob is not None else None

    # Consider probability match if difference is less than 0.1
    prob_match = prob_diff is not None and prob_diff < 0.1

    overall_match = fraud_match and prob_match

    return {
        "comparison": "MATCH" if overall_match else "MISMATCH",
        "fraud_match": fraud_match,
        "prob_match": prob_match,
        "api_fraud": api_fraud,
        "expected_fraud": expected_fraud,
        "api_prob": round(api_prob, 4) if api_prob else None,
        "expected_prob": round(expected_prob, 4),
        "prob_diff": round(prob_diff, 4) if prob_diff else None
    }

def run_comprehensive_test(dataset_path: str, max_tests: int = 10):
    """Run comprehensive API testing."""
    print("Starting Comprehensive FraudGuard API Test")
    print("=" * 60)

    # Load test data
    df = load_test_data(dataset_path)

    # Test health endpoint
    health_result = test_health_endpoint()
    if not health_result["success"]:
        print("❌ Health check failed. Aborting tests.")
        return

    # Get available models from health response
    available_models = []
    if "models_loaded" in health_result["response"]:
        models_count = health_result["response"]["models_loaded"]
        print(f"API reports {models_count} models loaded")
        # Assume standard models if not specified
        available_models = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "neural_network"]

    # Test results storage
    test_results = {
        "health": health_result,
        "individual_tests": [],
        "ensemble_tests": [],
        "summary": {
            "total_transactions": min(len(df), max_tests),
            "models_tested": available_models,
            "success_rate": 0,
            "accuracy": 0
        }
    }

    # Test individual transactions
    print(f"\n=== Testing {min(len(df), max_tests)} Transactions ===")

    successful_tests = 0
    total_comparisons = 0
    correct_predictions = 0

    for idx, row in df.head(max_tests).iterrows():
        print(f"\n{'='*50}")
        print(f"TRANSACTION {idx + 1} (ID: {int(row['Transaction_ID'])})")
        print(f"Expected: Fraud={int(row['isFraud'])}, Probability={float(row['Fraud_Probability']):.4f}")
        print(f"{'='*50}")

        transaction = prepare_transaction_data(row)
        expected_fraud = int(row["isFraud"])
        expected_prob = float(row["Fraud_Probability"])

        transaction_results = {
            "transaction_id": idx + 1,
            "expected_fraud": expected_fraud,
            "expected_prob": expected_prob,
            "model_results": {}
        }

        # Test default model (no model parameter)
        print("\n--- Testing Default Model (/detect) ---")
        result = test_detect_endpoint(transaction)
        transaction_results["model_results"]["default"] = result

        if result["success"]:
            response_data = result["response"]
            print(f"✅ SUCCESS - Status: {result['status_code']}")
            print(f"   Model Used: {response_data.get('model_used', 'unknown')}")
            print(f"   Prediction: Fraud={response_data.get('isFraud_pred')}, Probability={response_data.get('Fraud_Probability', 0):.4f}")
            print(f"   Risk Score: {response_data.get('Final_Risk_Score', 0):.4f}")
            print(f"   Alert Triggered: {response_data.get('alert_triggered', False)}")
            if response_data.get('alert_reasons'):
                print(f"   Alert Reasons: {', '.join(response_data['alert_reasons'])}")

            comparison = compare_predictions(result, expected_fraud, expected_prob)
            transaction_results["model_results"]["default"]["comparison"] = comparison
            print(f"   Comparison: {comparison['comparison']}")

            if comparison["comparison"] == "MATCH":
                correct_predictions += 1
            total_comparisons += 1
        else:
            print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
            total_comparisons += 1

        # Test each specific model
        for model in available_models:
            print(f"\n--- Testing {model.upper()} Model (/detect?model={model}) ---")
            result = test_detect_endpoint(transaction, model)
            transaction_results["model_results"][model] = result

            if result["success"]:
                response_data = result["response"]
                print(f"✅ SUCCESS - Status: {result['status_code']}")
                print(f"   Prediction: Fraud={response_data.get('isFraud_pred')}, Probability={response_data.get('Fraud_Probability', 0):.4f}")
                print(f"   Risk Score: {response_data.get('Final_Risk_Score', 0):.4f}")
                print(f"   Alert Triggered: {response_data.get('alert_triggered', False)}")
                if response_data.get('alert_reasons'):
                    print(f"   Alert Reasons: {', '.join(response_data['alert_reasons'])}")

                comparison = compare_predictions(result, expected_fraud, expected_prob)
                transaction_results["model_results"][model]["comparison"] = comparison
                print(f"   Comparison: {comparison['comparison']}")

                if comparison["comparison"] == "MATCH":
                    correct_predictions += 1
                total_comparisons += 1
            else:
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                total_comparisons += 1

        # Test ensemble
        print(f"\n--- Testing ENSEMBLE Model (/ensemble) ---")
        ensemble_result = test_ensemble_endpoint(transaction)
        transaction_results["ensemble_result"] = ensemble_result

        if ensemble_result["success"]:
            response_data = ensemble_result["response"]
            print(f"✅ SUCCESS - Status: {ensemble_result['status_code']}")
            print(f"   Ensemble Prediction: Fraud={response_data.get('isFraud_pred')}, Probability={response_data.get('Fraud_Probability', 0):.4f}")
            print(f"   Risk Score: {response_data.get('Final_Risk_Score', 0):.4f}")
            print(f"   Alert Triggered: {response_data.get('alert_triggered', False)}")
            print(f"   Ensemble Confidence: {response_data.get('ensemble_confidence', 0):.4f}")
            print(f"   Models Used: {', '.join(response_data.get('models_used', []))}")
            if response_data.get('alert_reasons'):
                print(f"   Alert Reasons: {', '.join(response_data['alert_reasons'])}")

            # Show individual model predictions in ensemble
            if 'model_predictions' in response_data:
                print("   Individual Model Results:")
                for model_name, pred_data in response_data['model_predictions'].items():
                    print(f"     {model_name}: Fraud={pred_data.get('prediction')}, Prob={pred_data.get('probability', 0):.4f}")

            comparison = compare_predictions(ensemble_result, expected_fraud, expected_prob)
            transaction_results["ensemble_result"]["comparison"] = comparison
            print(f"   Comparison: {comparison['comparison']}")

            if comparison["comparison"] == "MATCH":
                correct_predictions += 1
            total_comparisons += 1
        else:
            print(f"❌ FAILED - {ensemble_result.get('error', 'Unknown error')}")
            total_comparisons += 1

        test_results["individual_tests"].append(transaction_results)

        # Count successful API calls for this transaction
        api_successes = sum(1 for r in transaction_results["model_results"].values() if r.get("success", False))
        if ensemble_result.get("success", False):
            api_successes += 1

        if api_successes > 0:
            successful_tests += 1

        print(f"\nTransaction {idx + 1} Summary: {api_successes}/{len(available_models) + 2} endpoints successful")

    # Calculate summary
    test_results["summary"]["success_rate"] = (successful_tests / min(len(df), max_tests)) * 100
    test_results["summary"]["accuracy"] = (correct_predictions / total_comparisons) * 100 if total_comparisons > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Transactions Tested: {test_results['summary']['total_transactions']}")
    print(f"API Success Rate: {test_results['summary']['success_rate']:.1f}%")
    print(f"Prediction Accuracy: {test_results['summary']['accuracy']:.1f}%")
    print(f"Models Tested: {', '.join(available_models)}")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"api_test_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    # Model-specific success rates
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 40)

    model_stats = {}
    for model in available_models + ["default", "ensemble"]:
        successes = 0
        total = 0
        for transaction in test_results["individual_tests"]:
            if model == "ensemble":
                result = transaction.get("ensemble_result", {})
            else:
                result = transaction["model_results"].get(model, {})

            if result.get("success", False):
                successes += 1
            total += 1

        success_rate = (successes / total * 100) if total > 0 else 0
        model_stats[model] = success_rate
        status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
        print(f"{status} {model.upper()}: {success_rate:.1f}% success rate ({successes}/{total})")

    # Print issues summary
    print("\nISSUES IDENTIFIED:")
    issues = []

    if health_result["response"].get("models_loaded", 0) < len(available_models):
        issues.append(f"Only {health_result['response'].get('models_loaded', 0)} models loaded instead of {len(available_models)}")

    failed_models = [model for model, rate in model_stats.items() if rate < 100]
    if failed_models:
        issues.append(f"Models with failures: {', '.join(failed_models)}")

    failed_tests = sum(1 for t in test_results["individual_tests"]
                      if not any(r.get("success", False) for r in t["model_results"].values())
                      and not t.get("ensemble_result", {}).get("success", False))
    if failed_tests > 0:
        issues.append(f"{failed_tests} transactions failed all API calls")

    if test_results["summary"]["accuracy"] < 80:
        issues.append(f"Low prediction accuracy: {test_results['summary']['accuracy']:.1f}%")

    if not issues:
        print("✅ No major issues detected")
    else:
        for issue in issues:
            print(f"⚠️  {issue}")

    return test_results

if __name__ == "__main__":
    # Path to the test dataset
    dataset_path = r"c:\Users\ghosh\Desktop\Predictive-Transaction-intelligence-for-bfsi\Dataset\test_dataset_100_mixed.csv"

    # Run tests (limit to 20 transactions for comprehensive testing, adjust as needed)
    run_comprehensive_test(dataset_path, max_tests=20)