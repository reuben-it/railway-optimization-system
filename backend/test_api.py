"""
Test script for the Railway Optimization API.
Tests all endpoints with sample data.
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_delay_prediction():
    """Test delay prediction endpoint."""
    sample_data = {
        "train_type": "express",
        "block_length_km": 120.5,
        "speed_limit_kmph": 110,
        "rake_length_m": 650,
        "priority_level": 1,
        "headway_seconds": 300,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    }
    
    try:
        response = requests.post(f"{API_BASE}/predict_delay", json=sample_data)
        print(f"\nDelay Prediction: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Delay prediction test failed: {e}")
        return False

def test_conflict_detection():
    """Test conflict detection endpoint."""
    sample_data = {
        "train_type": "passenger",
        "delay_minutes": 15,
        "block_length_km": 80,
        "speed_limit_kmph": 100,
        "priority_level": 2,
        "headway_seconds": 240,
        "propagated_delay_minutes": 5
    }
    
    try:
        response = requests.post(f"{API_BASE}/check_conflict", json=sample_data)
        print(f"\nConflict Detection: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Conflict detection test failed: {e}")
        return False

def test_action_suggestion():
    """Test action suggestion endpoint."""
    sample_data = {
        "train_type": "freight",
        "block_length_km": 150,
        "speed_limit_kmph": 80,
        "rake_length_m": 1200,
        "priority_level": 3,
        "headway_seconds": 600,
        "tsr_active": "Y",
        "tsr_speed_kmph": 45,
        "current_delay": 10
    }
    
    try:
        response = requests.post(f"{API_BASE}/suggest_action", json=sample_data)
        print(f"\nAction Suggestion: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Action suggestion test failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint."""
    try:
        response = requests.get(f"{API_BASE}/model_info")
        print(f"\nModel Info: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Railway Optimization API Tests ===")
    
    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Delay Prediction", test_delay_prediction),
        ("Conflict Detection", test_conflict_detection),
        ("Action Suggestion", test_action_suggestion),
        ("Model Info", test_model_info)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    print("\n=== Test Results ===")
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

if __name__ == "__main__":
    main()