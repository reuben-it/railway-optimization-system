"""
Populate the railway system with default trains for testing and demonstration.
This script adds several trains to the continuous prediction engine so they appear
when operators log in to the system.
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Backend API URL
BASE_URL = "http://localhost:8001"

# Default trains data
DEFAULT_TRAINS = [
    {
        "train_id": "EXP001",
        "train_type": "express",
        "current_section": "SEC001",
        "from_station": "NDLS",
        "to_station": "BCT",
        "scheduled_arrival": "14:30:00",
        "actual_delay": 5.2,
        "block_length_km": 150.0,
        "speed_limit_kmph": 130,
        "rake_length_m": 750,
        "priority_level": 1,
        "headway_seconds": 600,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "PASS202",
        "train_type": "passenger",
        "current_section": "SEC015",
        "from_station": "HWH",
        "to_station": "CSMT",
        "scheduled_arrival": "16:45:00",
        "actual_delay": 12.8,
        "block_length_km": 85.0,
        "speed_limit_kmph": 110,
        "rake_length_m": 600,
        "priority_level": 2,
        "headway_seconds": 420,
        "tsr_active": "Y",
        "tsr_speed_kmph": 80
    },
    {
        "train_id": "FRT303",
        "train_type": "freight",
        "current_section": "SEC032",
        "from_station": "KYN",
        "to_station": "LKO",
        "scheduled_arrival": "20:15:00",
        "actual_delay": 25.3,
        "block_length_km": 200.0,
        "speed_limit_kmph": 75,
        "rake_length_m": 1200,
        "priority_level": 4,
        "headway_seconds": 900,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "SUB404",
        "train_type": "suburban",
        "current_section": "SEC008",
        "from_station": "CST",
        "to_station": "KYN",
        "scheduled_arrival": "11:20:00",
        "actual_delay": 2.1,
        "block_length_km": 45.0,
        "speed_limit_kmph": 90,
        "rake_length_m": 300,
        "priority_level": 3,
        "headway_seconds": 180,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "EXP505",
        "train_type": "express",
        "current_section": "SEC021",
        "from_station": "MAO",
        "to_station": "PUNE",
        "scheduled_arrival": "18:30:00",
        "actual_delay": 8.7,
        "block_length_km": 120.0,
        "speed_limit_kmph": 120,
        "rake_length_m": 650,
        "priority_level": 1,
        "headway_seconds": 480,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "PASS606",
        "train_type": "passenger",
        "current_section": "SEC044",
        "from_station": "BZA",
        "to_station": "HYB",
        "scheduled_arrival": "13:15:00",
        "actual_delay": 18.5,
        "block_length_km": 95.0,
        "speed_limit_kmph": 100,
        "rake_length_m": 550,
        "priority_level": 2,
        "headway_seconds": 360,
        "tsr_active": "Y",
        "tsr_speed_kmph": 60
    },
    {
        "train_id": "SUB707",
        "train_type": "suburban",
        "current_section": "SEC003",
        "from_station": "CCG",
        "to_station": "SEHI",
        "scheduled_arrival": "10:45:00",
        "actual_delay": 0.8,
        "block_length_km": 25.0,
        "speed_limit_kmph": 80,
        "rake_length_m": 250,
        "priority_level": 3,
        "headway_seconds": 120,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "FRT808",
        "train_type": "freight",
        "current_section": "SEC056",
        "from_station": "TPTY",
        "to_station": "MDU",
        "scheduled_arrival": "22:00:00",
        "actual_delay": 35.2,
        "block_length_km": 180.0,
        "speed_limit_kmph": 65,
        "rake_length_m": 1400,
        "priority_level": 4,
        "headway_seconds": 1200,
        "tsr_active": "Y",
        "tsr_speed_kmph": 45
    }
]

def check_backend_health():
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Backend is healthy - Version: {health_data.get('version', 'Unknown')}")
            print(f"   Models loaded: {health_data.get('models', {})}")
            print(f"   Prediction engine: {health_data.get('prediction_engine', {})}")
            return True
        else:
            print(f"❌ Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Make sure the backend is running on http://localhost:8001")
        return False

def register_train(train_data):
    """Register a single train with the backend."""
    try:
        response = requests.post(
            f"{BASE_URL}/trains/register",
            json=train_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Registered train {train_data['train_id']} ({train_data['train_type']})")
            return True
        else:
            print(f"❌ Failed to register train {train_data['train_id']}: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error registering train {train_data['train_id']}: {e}")
        return False

def populate_trains():
    """Populate the system with default trains."""
    print("🚂 Railway Optimization System - Default Train Population")
    print("=" * 60)
    
    # Check backend health first
    if not check_backend_health():
        print("\n❌ Cannot proceed without healthy backend connection.")
        return False
    
    print(f"\n📝 Registering {len(DEFAULT_TRAINS)} default trains...")
    print("-" * 40)
    
    successful_registrations = 0
    failed_registrations = 0
    
    for train in DEFAULT_TRAINS:
        if register_train(train):
            successful_registrations += 1
        else:
            failed_registrations += 1
        
        # Small delay between registrations to avoid overwhelming the server
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("📊 REGISTRATION SUMMARY")
    print(f"✅ Successfully registered: {successful_registrations} trains")
    print(f"❌ Failed registrations: {failed_registrations} trains")
    print(f"📈 Success rate: {(successful_registrations / len(DEFAULT_TRAINS)) * 100:.1f}%")
    
    if successful_registrations > 0:
        print(f"\n🎉 {successful_registrations} trains are now available in the operator dashboard!")
        print("   You can log in to the operator interface to see them.")
    
    return successful_registrations > 0

def list_current_trains():
    """List currently registered trains."""
    try:
        response = requests.get(f"{BASE_URL}/trains", timeout=5)
        if response.status_code == 200:
            data = response.json()
            trains = data.get('trains', []) if isinstance(data, dict) else data
            
            print(f"\n📋 Currently registered trains: {len(trains)}")
            if trains:
                print("-" * 80)
                print(f"{'Train ID':<10} {'Type':<10} {'Section':<10} {'Delay':<8} {'State':<12} {'From-To'}")
                print("-" * 80)
                for train in trains:
                    delay = f"{train.get('actual_delay', 0):.1f}m"
                    from_to = f"{train.get('from_station', 'N/A')}-{train.get('to_station', 'N/A')}"
                    print(f"{train.get('train_id', 'N/A'):<10} "
                          f"{train.get('train_type', 'N/A'):<10} "
                          f"{train.get('current_section', 'N/A'):<10} "
                          f"{delay:<8} "
                          f"{train.get('state', 'N/A'):<12} "
                          f"{from_to}")
            return trains
        else:
            print(f"❌ Failed to get trains: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting trains: {e}")
        return []

def main():
    """Main function with options."""
    print("🚂 Railway System Train Population Tool")
    print("=" * 50)
    print("1. Check current trains")
    print("2. Populate default trains")
    print("3. Check backend health")
    print("4. Do everything (check status + populate)")
    print("=" * 50)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        list_current_trains()
    elif choice == "2":
        populate_trains()
    elif choice == "3":
        check_backend_health()
    elif choice == "4":
        print("🔍 Checking current status...")
        list_current_trains()
        print("\n🚂 Starting population process...")
        populate_trains()
        print("\n🔍 Final status check...")
        list_current_trains()
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()