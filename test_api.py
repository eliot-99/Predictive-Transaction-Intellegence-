#!/usr/bin/env python3
"""
Comprehensive API Test Script for Fraud Detection API
Tests all endpoints with adversarial test dataset
"""

import json
import time
from typing import Dict, List, Any
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

class APITester:
    def __init__(self, base_url: str = "https://fraudguard-api.onrender.com"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30

    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {
                "endpoint": "/health",
                "status": "success",
                "response": response.json(),
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "endpoint": "/health",
                "status": "failed",
                "error": str(e)
            }

    def test_root(self) -> Dict[str, Any]:
        """Test root endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return {
                "endpoint": "/",
                "status": "success",
                "response": response.json(),
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "endpoint": "/",
                "status": "failed",
                "error": str(e)
            }

    def test_detect_single(self, transaction_data: Dict[str, Any], model: str = None, expected_fraud: int = None) -> Dict[str, Any]:
        """Test /detect endpoint with single transaction"""
        try:
            url = f"{self.base_url}/detect"
            if model:
                url += f"?model={model}"

            response = self.session.post(url, json=transaction_data)
            response.raise_for_status()
            result = response.json()

            return {
                "endpoint": f"/detect{'?model=' + model if model else ''}",
                "status": "success",
                "transaction_id": transaction_data.get("Transaction_ID"),
                "user_id": transaction_data.get("User_ID"),
                "response": result,
                "response_time": response.elapsed.total_seconds(),
                "expected_fraud": expected_fraud,
                "predicted_fraud": result.get("isFraud_pred"),
                "match": expected_fraud == result.get("isFraud_pred") if expected_fraud is not None else None
            }
        except Exception as e:
            return {
                "endpoint": f"/detect{'?model=' + model if model else ''}",
                "status": "failed",
                "transaction_id": transaction_data.get("Transaction_ID"),
                "user_id": transaction_data.get("User_ID"),
                "error": str(e)
            }

    def test_detect_all_models(self, transaction_data: Dict[str, Any], expected_fraud: int = None) -> List[Dict[str, Any]]:
        """Test /detect with all available models"""
        models = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "neural_network"]
        results = []

        for model in models:
            result = self.test_detect_single(transaction_data, model, expected_fraud)
            results.append(result)
            time.sleep(0.1)  # Small delay to avoid overwhelming the API

        return results

    def test_ensemble(self, transaction_data: Dict[str, Any], expected_fraud: int = None) -> Dict[str, Any]:
        """Test /ensemble endpoint"""
        try:
            response = self.session.post(f"{self.base_url}/ensemble", json=transaction_data)
            response.raise_for_status()
            result = response.json()

            return {
                "endpoint": "/ensemble",
                "status": "success",
                "transaction_id": transaction_data.get("Transaction_ID"),
                "user_id": transaction_data.get("User_ID"),
                "response": result,
                "response_time": response.elapsed.total_seconds(),
                "expected_fraud": expected_fraud,
                "predicted_fraud": result.get("isFraud_pred"),
                "match": expected_fraud == result.get("isFraud_pred") if expected_fraud is not None else None,
                "models_used": result.get("models_used", [])
            }
        except Exception as e:
            return {
                "endpoint": "/ensemble",
                "status": "failed",
                "transaction_id": transaction_data.get("Transaction_ID"),
                "user_id": transaction_data.get("User_ID"),
                "error": str(e)
            }

    def test_chat(self, message: str = "What is fraud detection?") -> Dict[str, Any]:
        """Test /chat endpoint"""
        try:
            payload = {"message": message}
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            result = response.json()

            return {
                "endpoint": "/chat",
                "status": "success",
                "request": message,
                "response": result,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "endpoint": "/chat",
                "status": "failed",
                "request": message,
                "error": str(e)
            }

    def prepare_transaction_data(self, row: pd.Series) -> Dict[str, Any]:
        """Convert CSV row to API transaction format"""
        # Remove prediction columns and Transaction_ID (generated server-side)
        transaction = {}
        exclude_cols = ["isFraud", "Fraud_Probability", "isFraud_pred", "Transaction_ID"]

        for col in row.index:
            if col not in exclude_cols:
                value = row[col]
                # Convert numpy types to Python types
                if pd.isna(value):
                    continue
                if isinstance(value, (pd.Int64Dtype().type, pd.Float64Dtype().type)):
                    value = value.item() if hasattr(value, 'item') else float(value)
                # Ensure proper types for API
                if col in ["User_ID", "Merchant_ID", "Device_ID", "Previous_Transaction_Count", "Time_Since_Last_Transaction_min", "Transaction_Velocity", "Transaction_Hour", "Transaction_Day", "Transaction_Month", "Transaction_Weekday"]:
                    transaction[col] = int(float(value))
                elif col in ["Transaction_Amount", "Distance_Between_Transactions_km", "Log_Transaction_Amount", "Velocity_Distance_Interact", "Amount_Velocity_Interact", "Time_Distance_Interact", "Hour_sin", "Hour_cos", "Weekday_sin", "Weekday_cos"]:
                    transaction[col] = float(value)
                else:
                    transaction[col] = str(value)

        return transaction

def run_comprehensive_test(csv_path: str, base_url: str = "https://fraudguard-api.onrender.com",
                          max_samples: int = None, parallel: bool = False):
    """Run comprehensive API tests"""

    print(f"🚀 Starting API tests for {base_url}")
    print(f"📊 Loading test data from {csv_path}")

    # Load test data
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
        print(f"📋 Using first {max_samples} samples")

    print(f"📈 Total samples: {len(df)}")

    tester = APITester(base_url)

    # Test basic endpoints
    print("\n🏥 Testing basic endpoints...")
    health_result = tester.test_health()
    root_result = tester.test_root()

    print(f"Health check: {'✅' if health_result['status'] == 'success' else '❌'}")
    print(f"Root endpoint: {'✅' if root_result['status'] == 'success' else '❌'}")

    # Test chat endpoint
    print("\n💬 Testing chat endpoint...")
    chat_result = tester.test_chat("Explain what fraud detection is in banking")
    print(f"Chat endpoint: {'✅' if chat_result['status'] == 'success' else '❌'}")

    # Test fraud detection endpoints
    print("\n🔍 Testing fraud detection endpoints...")

    results = {
        "health": health_result,
        "root": root_result,
        "chat": chat_result,
        "detect_default": [],
        "detect_models": [],
        "ensemble": []
    }

    def process_sample(sample_data):
        """Process a single sample"""
        transaction = tester.prepare_transaction_data(sample_data)
        expected_fraud = int(sample_data.get("isFraud", 0))

        # Test default detect
        detect_result = tester.test_detect_single(transaction, expected_fraud=expected_fraud)
        results["detect_default"].append(detect_result)

        # Test all models
        model_results = tester.test_detect_all_models(transaction, expected_fraud=expected_fraud)
        results["detect_models"].extend(model_results)

        # Test ensemble
        ensemble_result = tester.test_ensemble(transaction, expected_fraud=expected_fraud)
        results["ensemble"].append(ensemble_result)

        return {
            "transaction_id": transaction.get("Transaction_ID"),
            "detect": detect_result,
            "ensemble": ensemble_result
        }

    if parallel and len(df) > 10:
        print("⚡ Running tests in parallel...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_sample, df.iloc[i]) for i in range(len(df))]
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(df)} samples")
    else:
        print("🔄 Running tests sequentially...")
        for i, (_, row) in enumerate(df.iterrows()):
            process_sample(row)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(df)} samples")

    # Generate summary
    print("\n📊 Generating test summary...")

    summary = generate_summary(results)

    # Save detailed results
    output_file = "api_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"💾 Detailed results saved to {output_file}")

    return summary, results

def generate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate test summary"""

    summary = {
        "total_samples": len(results.get("detect_default", [])),
        "endpoint_status": {},
        "accuracy_metrics": {},
        "performance_metrics": {},
        "error_analysis": {}
    }

    # Endpoint status
    for endpoint in ["health", "root", "chat"]:
        status = results[endpoint]["status"]
        summary["endpoint_status"][endpoint] = status

    # Detect default accuracy
    detect_results = results.get("detect_default", [])
    if detect_results:
        successful = [r for r in detect_results if r["status"] == "success"]
        matches = [r for r in successful if r.get("match", False)]

        summary["accuracy_metrics"]["detect_default"] = {
            "total_requests": len(detect_results),
            "successful_requests": len(successful),
            "accuracy": len(matches) / len(successful) if successful else 0,
            "avg_response_time": sum(r.get("response_time", 0) for r in successful) / len(successful) if successful else 0
        }

    # Ensemble accuracy
    ensemble_results = results.get("ensemble", [])
    if ensemble_results:
        successful = [r for r in ensemble_results if r["status"] == "success"]
        matches = [r for r in successful if r.get("match", False)]

        summary["accuracy_metrics"]["ensemble"] = {
            "total_requests": len(ensemble_results),
            "successful_requests": len(successful),
            "accuracy": len(matches) / len(successful) if successful else 0,
            "avg_response_time": sum(r.get("response_time", 0) for r in successful) / len(successful) if successful else 0
        }

    # Model-specific accuracy
    model_results = results.get("detect_models", [])
    if model_results:
        models = {}
        for result in model_results:
            if result["status"] == "success":
                model = result["endpoint"].split("=")[-1] if "=" in result["endpoint"] else "default"
                if model not in models:
                    models[model] = []
                models[model].append(result)

        for model, model_res in models.items():
            matches = [r for r in model_res if r.get("match", False)]
            summary["accuracy_metrics"][f"detect_{model}"] = {
                "total_requests": len(model_res),
                "accuracy": len(matches) / len(model_res) if model_res else 0,
                "avg_response_time": sum(r.get("response_time", 0) for r in model_res) / len(model_res) if model_res else 0
            }

    # Error analysis
    all_results = (detect_results + ensemble_results + model_results)
    failed_requests = [r for r in all_results if r["status"] == "failed"]
    summary["error_analysis"] = {
        "total_failed_requests": len(failed_requests),
        "error_types": {}
    }

    for failed in failed_requests:
        error = failed.get("error", "Unknown")
        summary["error_analysis"]["error_types"][error] = summary["error_analysis"]["error_types"].get(error, 0) + 1

    return summary

def print_summary(summary: Dict[str, Any]):
    """Print formatted summary"""

    print("\n" + "="*60)
    print("🎯 API TEST SUMMARY")
    print("="*60)

    print(f"\n📊 Test Overview:")
    print(f"  • Total samples tested: {summary['total_samples']}")

    print(f"\n🏥 Endpoint Status:")
    for endpoint, status in summary["endpoint_status"].items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {endpoint}: {status}")

    if summary["accuracy_metrics"]:
        print(f"\n🎯 Accuracy Metrics:")
        for test_type, metrics in summary["accuracy_metrics"].items():
            acc = metrics.get("accuracy", 0) * 100
            resp_time = metrics.get("avg_response_time", 0) * 1000
            success_rate = (metrics.get("successful_requests", 0) / metrics.get("total_requests", 1)) * 100

            print(f"  • {test_type}:")
            print(f"    - Success rate: {success_rate:.1f}%")
            print(f"    - Accuracy: {acc:.1f}%")
            print(f"    - Avg response time: {resp_time:.1f}ms")
    if summary["error_analysis"]["total_failed_requests"] > 0:
        print(f"\n❌ Error Analysis:")
        print(f"  • Total failed requests: {summary['error_analysis']['total_failed_requests']}")
        print("  • Error types:")
        for error, count in summary["error_analysis"]["error_types"].items():
            print(f"    - {error}: {count}")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Test Fraud Detection API")
    parser.add_argument("--url", default="https://fraudguard-api.onrender.com",
                       help="API base URL")
    parser.add_argument("--csv", default="Dataset/adversarial_test_100.csv",
                       help="Path to test CSV file")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to test")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel")

    args = parser.parse_args()

    try:
        summary, results = run_comprehensive_test(
            csv_path=args.csv,
            base_url=args.url,
            max_samples=args.max_samples,
            parallel=args.parallel
        )

        print_summary(summary)

        # Exit with error code if there were failures
        if summary["error_analysis"]["total_failed_requests"] > 0:
            exit(1)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        exit(1)

if __name__ == "__main__":
    main()