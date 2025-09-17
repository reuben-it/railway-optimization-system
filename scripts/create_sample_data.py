#!/usr/bin/env python3
"""
Create sample data for testing the Railway Optimization System
This script will register multiple trains and create various scenarios for testing.
"""

import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8001"

def register_sample_trains():
    """Register sample trains for testing."""
    print("🚂 Registering Sample Trains...")
    
    trains = [
        {
            "train_id": "EXP001",
            "train_type": "EXPRESS",
            "current_section": "SECTION_A",
            "from_station": "Mumbai Central",
            "to_station": "New Delhi",
            "scheduled_arrival": (datetime.now() + timedelta(hours=2)).isoformat(),
            "block_length_km": 35.0,
            "speed_limit_kmph": 130,
            "rake_length_m": 450,
            "priority_level": 1,
            "headway_seconds": 240
        },
        {
            "train_id": "PASS002",
            "train_type": "PASSENGER",
            "current_section": "SECTION_B",
            "from_station": "Pune Junction",
            "to_station": "Mumbai CST",
            "scheduled_arrival": (datetime.now() + timedelta(hours=1)).isoformat(),
            "block_length_km": 20.0,
            "speed_limit_kmph": 100,
            "rake_length_m": 300,
            "priority_level": 3,
            "headway_seconds": 180
        },
        {
            "train_id": "FRGHT003",
            "train_type": "FREIGHT",
            "current_section": "SECTION_C",
            "from_station": "JNPT Port",
            "to_station": "Inland Container Depot",
            "scheduled_arrival": (datetime.now() + timedelta(hours=4)).isoformat(),
            "block_length_km": 15.0,
            "speed_limit_kmph": 75,
            "rake_length_m": 600,
            "priority_level": 4,
            "headway_seconds": 420
        },
        {
            "train_id": "SUB004",
            "train_type": "SUBURBAN",
            "current_section": "SECTION_D",
            "from_station": "Churchgate",
            "to_station": "Virar",
            "scheduled_arrival": (datetime.now() + timedelta(minutes=45)).isoformat(),
            "block_length_km": 10.0,
            "speed_limit_kmph": 90,
            "rake_length_m": 200,
            "priority_level": 2,
            "headway_seconds": 120
        },
        {
            "train_id": "EXP005",
            "train_type": "EXPRESS",
            "current_section": "SECTION_E",
            "from_station": "Chennai Central",
            "to_station": "Bangalore City",
            "scheduled_arrival": (datetime.now() + timedelta(hours=3)).isoformat(),
            "block_length_km": 40.0,
            "speed_limit_kmph": 120,
            "rake_length_m": 400,
            "priority_level": 1,
            "headway_seconds": 300
        }
    ]
    
    registered_trains = []
    for train in trains:
        try:
            response = requests.post(f"{BASE_URL}/trains/register", json=train)
            if response.status_code == 200:
                print(f"✅ Registered train {train['train_id']}: {train['from_station']} → {train['to_station']}")
                registered_trains.append(train['train_id'])
            else:
                print(f"❌ Failed to register {train['train_id']}: {response.text}")
        except Exception as e:
            print(f"❌ Error registering {train['train_id']}: {e}")
    
    return registered_trains

def create_operator_decisions(train_ids):
    """Create sample operator decisions for different scenarios."""
    print(f"\n⚡ Creating Sample Operator Decisions...")
    
    decisions = [
        {
            "train_id": train_ids[0] if len(train_ids) > 0 else "EXP001",
            "decision": "reduce_speed",
            "new_speed_limit": 80,
            "reason": "Heavy rainfall causing poor visibility",
            "operator_id": "OP123",
            "priority": "high"
        },
        {
            "train_id": train_ids[1] if len(train_ids) > 1 else "PASS002",
            "decision": "allow_delay",
            "delay_minutes": 15,
            "reason": "Waiting for connecting express train passengers",
            "operator_id": "OP456",
            "priority": "normal"
        },
        {
            "train_id": train_ids[2] if len(train_ids) > 2 else "FRGHT003",
            "decision": "hold_train",
            "reason": "Priority passenger train approaching - yield right of way",
            "operator_id": "OP123",
            "priority": "high"
        },
        {
            "train_id": train_ids[3] if len(train_ids) > 3 else "SUB004",
            "decision": "reduce_speed",
            "new_speed_limit": 60,
            "reason": "Track maintenance work ahead - proceed with caution",
            "operator_id": "OP456",
            "priority": "critical"
        }
    ]
    
    created_decisions = []
    for decision in decisions:
        try:
            response = requests.post(f"{BASE_URL}/operator/decisions", json=decision)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Decision for {decision['train_id']}: {decision['decision']} ({decision['priority']})")
                print(f"   Reason: {decision['reason']}")
                created_decisions.append(result)
            else:
                print(f"❌ Failed to create decision for {decision['train_id']}: {response.text}")
        except Exception as e:
            print(f"❌ Error creating decision for {decision['train_id']}: {e}")
        
        # Small delay between decisions
        time.sleep(0.5)
    
    return created_decisions

def show_test_credentials():
    """Display test credentials and usage instructions."""
    print(f"\n🔐 Test Credentials & Usage Instructions")
    print("=" * 50)
    
    print("\n👨‍💼 OPERATOR ACCOUNTS:")
    print("Login ID: OP123 | Password: op@123")
    print("Login ID: OP456 | Password: station@456")
    
    print("\n🚛 DRIVER ACCOUNTS:")
    print("Login ID: DR123 | Password: dr@123")
    print("Login ID: DR456 | Password: train@456")
    
    print("\n📱 TESTING SCENARIOS:")
    print("\n🔹 OPERATOR WORKFLOW:")
    print("1. Login with OP123/op@123")
    print("2. View real-time train status and critical situations")
    print("3. Click 'Decide' button on any train")
    print("4. Make decisions: reduce speed, allow delay, hold train, emergency stop")
    print("5. Enter reason and priority level")
    print("6. Send decision - driver will be automatically notified")
    
    print("\n🔹 DRIVER WORKFLOW:")
    print("1. Login with DR123/dr@123 (or DR456/train@456)")
    print("2. View pending notifications from operators")
    print("3. Acknowledge received instructions")
    print("4. Mark actions as completed")
    print("5. Check notification history")
    
    print("\n🔹 REAL-TIME FEATURES:")
    print("• Operator decisions instantly create driver notifications")
    print("• Auto-refresh every 30 seconds (operator) / 15 seconds (driver)")
    print("• Priority-based notification system (normal/high/critical)")
    print("• Comprehensive train state monitoring (SAFE/WATCHLIST/CRITICAL)")

def show_api_endpoints():
    """Display available API endpoints for testing."""
    print(f"\n🔗 Available API Endpoints (Base: {BASE_URL})")
    print("=" * 50)
    
    endpoints = [
        ("GET", "/", "System health and status"),
        ("GET", "/trains", "List all trains"),
        ("POST", "/trains/register", "Register new train"),
        ("GET", "/critical-situations", "Get critical situations"),
        ("POST", "/operator/decisions", "Make operator decision"),
        ("GET", "/operator/decisions", "Get all operator decisions"),
        ("GET", "/driver/notifications/{driver_id}", "Get driver notifications"),
        ("POST", "/driver/notifications/{id}/acknowledge", "Acknowledge notification"),
        ("POST", "/driver/notifications/{id}/complete", "Complete notification"),
    ]
    
    for method, endpoint, description in endpoints:
        print(f"{method:4} {endpoint:35} - {description}")

def main():
    """Main function to create all sample data."""
    print("🚂 Railway Optimization System - Sample Data Generator")
    print("=" * 60)
    
    # Register sample trains
    train_ids = register_sample_trains()
    
    if train_ids:
        # Create operator decisions
        decisions = create_operator_decisions(train_ids)
        print(f"\n✅ Created {len(decisions)} operator decisions")
    
    # Show test credentials and instructions
    show_test_credentials()
    
    # Show API endpoints
    show_api_endpoints()
    
    print(f"\n🏁 Sample data creation completed!")
    print(f"📱 You can now test the mobile app with the above credentials")
    print(f"🌐 Or test via API at {BASE_URL}/docs")

if __name__ == "__main__":
    main()