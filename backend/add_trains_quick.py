"""
Quick script to add default trains to the running railway system.
Run this script after starting the backend to populate trains immediately.
"""

import requests
import json

# Default trains - smaller set for immediate testing
QUICK_TRAINS = [
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
    }
]

def add_train(train_data):
    """Add a single train to the system."""
    try:
        response = requests.post(
            "http://localhost:8001/trains/register",
            json=train_data,
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ Added train {train_data['train_id']}")
            return True
        else:
            print(f"❌ Failed to add train {train_data['train_id']}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error adding train {train_data['train_id']}: {e}")
        return False

def main():
    print("🚂 Adding default trains to railway system...")
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not healthy")
            return
    except:
        print("❌ Cannot connect to backend. Make sure it's running on http://localhost:8001")
        return
    
    # Add trains
    success_count = 0
    for train in QUICK_TRAINS:
        if add_train(train):
            success_count += 1
    
    print(f"\n✅ Successfully added {success_count}/{len(QUICK_TRAINS)} trains")
    
    # Check final status
    try:
        response = requests.get("http://localhost:8001/trains", timeout=5)
        if response.status_code == 200:
            data = response.json()
            total_trains = len(data.get('trains', []))
            print(f"📊 Total trains now in system: {total_trains}")
        
        print("\n🎉 You can now log in to the operator dashboard to see the trains!")
        
    except Exception as e:
        print(f"⚠️ Could not verify final status: {e}")

if __name__ == "__main__":
    main()