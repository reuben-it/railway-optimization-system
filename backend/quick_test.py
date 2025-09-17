"""
Quick test script for the continuous prediction system
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Health check: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_train_registration():
    """Test train registration"""
    train_data = {
        "train_id": "TEST001",
        "train_type": "EXPRESS",
        "current_section": "SECTION_A1",
        "from_station": "STATION_A",
        "to_station": "STATION_B",
        "scheduled_arrival": "2025-09-17T15:30:00",
        "actual_delay": 0.0,
        "block_length_km": 25.5,
        "speed_limit_kmph": 120,
        "rake_length_m": 400,
        "priority_level": 1,
        "headway_seconds": 300,
        "tsr_active": "N",
        "tsr_speed_kmph": 0.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/trains/register", json=train_data)
        print(f"✓ Train registration: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Train registration failed: {e}")
        return False

def test_get_trains():
    """Test getting all trains"""
    try:
        response = requests.get(f"{BASE_URL}/trains")
        print(f"✓ Get trains: {response.status_code} - Found {len(response.json())} trains")
        return True
    except Exception as e:
        print(f"✗ Get trains failed: {e}")
        return False

def test_critical_situations():
    """Test getting critical situations"""
    try:
        response = requests.get(f"{BASE_URL}/critical-situations")
        data = response.json()
        print(f"✓ Critical situations: {response.status_code} - {len(data['critical_situations'])} critical, immediate attention: {data['immediate_attention_needed']}")
        return True
    except Exception as e:
        print(f"✗ Critical situations failed: {e}")
        return False

def test_predict_endpoint():
    """Test the delay prediction endpoint"""
    train_data = {
        "train_type": "EXPRESS",
        "block_length_km": 25.5,
        "speed_limit_kmph": 120,
        "rake_length_m": 400,
        "priority_level": 1,
        "headway_seconds": 300,
        "tsr_active": "N",
        "tsr_speed_kmph": 0.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_delay_simple", json=train_data)
        result = response.json()
        print(f"✓ Delay Prediction: {response.status_code} - Delay: {result['predicted_delay']:.2f} minutes")
        return True
    except Exception as e:
        print(f"✗ Delay prediction failed: {e}")
        return False

def test_conflict_prediction():
    """Test the conflict prediction endpoint"""
    train_data = {
        "train_type": "EXPRESS",
        "block_length_km": 25.5,
        "speed_limit_kmph": 120,
        "rake_length_m": 400,
        "priority_level": 1,
        "headway_seconds": 300,
        "tsr_active": "N",
        "tsr_speed_kmph": 0.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/check_conflict_simple", json=train_data)
        result = response.json()
        print(f"✓ Conflict Prediction: {response.status_code} - Likelihood: {result['conflict_likelihood']:.3f}")
        return True
    except Exception as e:
        print(f"✗ Conflict prediction failed: {e}")
        return False

def main():
    print("🚂 Testing Railway Optimization System with Continuous Prediction")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Train Registration", test_train_registration),
        ("Get Trains", test_get_trains),
        ("Critical Situations", test_critical_situations),
        ("Delay Prediction", test_predict_endpoint),
        ("Conflict Prediction", test_conflict_prediction)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The continuous prediction system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()