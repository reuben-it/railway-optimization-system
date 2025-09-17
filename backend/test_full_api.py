"""
Test script for the complete Railway Optimization API with both models.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Health check failed: {e}")
        return None

def test_delay_prediction():
    """Test delay prediction endpoint."""
    print("\nTesting delay prediction...")
    
    test_data = {
        "train_type": "express",
        "block_length_km": 150.0,
        "speed_limit_kmph": 110.0,
        "rake_length_m": 400.0,
        "priority_level": 1,
        "headway_seconds": 240,
        "tsr_active": "Y",
        "tsr_speed_kmph": 60.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_delay_simple", json=test_data)
        print(f"Delay Prediction Status: {response.status_code}")
        result = response.json()
        print(f"Predicted delay: {result['predicted_delay']} minutes")
        print(f"Model type: {result['model_type']}")
        return result
    except Exception as e:
        print(f"Delay prediction failed: {e}")
        return None

def test_conflict_detection():
    """Test conflict detection endpoint."""
    print("\nTesting conflict detection...")
    
    test_data = {
        "train_type": "express",
        "block_length_km": 150.0,
        "speed_limit_kmph": 110.0,
        "rake_length_m": 400.0,
        "priority_level": 1,
        "headway_seconds": 240,
        "tsr_active": "Y",
        "tsr_speed_kmph": 60.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/check_conflict_simple", json=test_data)
        print(f"Conflict Detection Status: {response.status_code}")
        result = response.json()
        print(f"Conflict likelihood: {result['conflict_likelihood']}")
        print(f"Model type: {result['model_type']}")
        return result
    except Exception as e:
        print(f"Conflict detection failed: {e}")
        return None

def test_optimization():
    """Test the integrated optimization endpoint."""
    print("\nTesting optimization endpoint...")
    
    test_data = {
        "train_type": "express",
        "block_length_km": 150.0,
        "speed_limit_kmph": 110.0,
        "rake_length_m": 400.0,
        "priority_level": 1,
        "headway_seconds": 240,
        "tsr_active": "Y",
        "tsr_speed_kmph": 60.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/optimize", json=test_data)
        print(f"Optimization Status: {response.status_code}")
        result = response.json()
        print("Optimization Results:")
        print(f"  - Predicted delay: {result['optimization_results']['predicted_delay_minutes']} minutes")
        print(f"  - Conflict likelihood: {result['optimization_results']['conflict_likelihood']}")
        print(f"  - Combined risk: {result['optimization_results']['combined_risk_score']}")
        print(f"  - Risk level: {result['optimization_results']['risk_level']}")
        print("  - Suggestions:")
        for suggestion in result['optimization_results']['optimization_suggestions']:
            print(f"    * {suggestion}")
        print(f"  - Model status: {result['optimization_results']['model_status']}")
        return result
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

def test_model_info():
    """Test model info endpoint."""
    print("\nTesting model info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Model Info Status: {response.status_code}")
        result = response.json()
        print("Model Information:")
        print(f"  Delay Model: {'Loaded' if result['delay_model']['loaded'] else 'Not Loaded'}")
        print(f"  Conflict Model: {'Loaded' if result['conflict_model']['loaded'] else 'Not Loaded'}")
        return result
    except Exception as e:
        print(f"Model info failed: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Railway Optimization API - Complete Test Suite")
    print("=" * 60)
    
    # Test all endpoints
    health = test_health()
    if health and health.get('status') == 'healthy':
        test_delay_prediction()
        test_conflict_detection()
        test_optimization()
        test_model_info()
    else:
        print("API is not healthy, skipping other tests")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")