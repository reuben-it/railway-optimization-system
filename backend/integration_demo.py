"""
Railway Optimization System - Complete Integration Demo
This demonstrates both delay prediction and conflict detection working together.
"""

import sys
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
import random

def load_models():
    """Load both trained models."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load delay model
    delay_model_path = os.path.join(script_dir, "models", "delay_model.pkl")
    delay_model = joblib.load(delay_model_path)
    
    # Load conflict model
    conflict_model_path = os.path.join(script_dir, "models", "conflict_model.pkl")
    conflict_model = joblib.load(conflict_model_path)
    
    return delay_model, conflict_model

def prepare_features(train_type, block_length, speed_limit, rake_length, priority, headway, tsr_active, tsr_speed):
    """Prepare features for both models."""
    current_time = datetime.now()
    train_id = f"T{random.randint(10000, 99999)}"
    
    features = {
        'train_id': train_id,
        'train_type': train_type,
        'date': current_time.strftime('%Y-%m-%d'),
        'section_id': f"SEC{random.randint(1, 100):03d}",
        'from_station': 'NDLS',
        'to_station': 'HWH',
        'scheduled_departure': current_time.strftime('%H:%M:%S'),
        'scheduled_arrival': (current_time + timedelta(hours=2)).strftime('%H:%M:%S'),
        'actual_departure': current_time.strftime('%H:%M:%S'),
        'actual_arrival': (current_time + timedelta(hours=2)).strftime('%H:%M:%S'),
        'block_length_km': block_length,
        'track_type': 'double',
        'speed_limit_kmph': speed_limit,
        'rake_length_m': rake_length,
        'priority_level': priority,
        'headway_seconds': headway,
        'tsr_active': tsr_active,
        'tsr_speed_kmph': tsr_speed,
        'platform_assigned': f"P{random.randint(1, 8)}",
        'controller_action': 'none',
        'propagated_delay_minutes': 0.0
    }
    
    return pd.DataFrame([features])

def predict_delay(delay_model, features_df):
    """Predict delay using the trained model."""
    prediction = delay_model.predict(features_df)[0]
    return max(0, prediction)

def predict_conflict(conflict_model, features_df):
    """Predict conflict likelihood using the trained model."""
    if hasattr(conflict_model, 'predict_proba'):
        probability = conflict_model.predict_proba(features_df)[0][1]
        return probability
    else:
        prediction = conflict_model.predict(features_df)[0]
        return min(1.0, max(0.0, prediction))

def generate_recommendations(delay_minutes, conflict_probability):
    """Generate operational recommendations based on predictions."""
    recommendations = []
    
    # Delay-based recommendations
    if delay_minutes > 15:
        recommendations.append("🚨 CRITICAL: Consider immediate rerouting or cancellation")
    elif delay_minutes > 10:
        recommendations.append("⚠️  HIGH: Notify passengers and prepare contingency plans")
    elif delay_minutes > 5:
        recommendations.append("📢 MEDIUM: Inform passengers of potential delays")
    
    # Conflict-based recommendations
    if conflict_probability > 0.8:
        recommendations.append("🔴 CONFLICT ALERT: Implement immediate traffic control")
    elif conflict_probability > 0.6:
        recommendations.append("🟡 CONFLICT RISK: Monitor closely and increase headway")
    elif conflict_probability > 0.4:
        recommendations.append("🟢 CONFLICT WATCH: Standard monitoring protocols")
    
    # Combined risk assessment
    combined_risk = (delay_minutes / 60.0) * 0.6 + conflict_probability * 0.4
    
    if combined_risk > 0.8:
        risk_level = "CRITICAL"
        recommendations.append("🆘 OVERALL RISK: CRITICAL - Immediate intervention required")
    elif combined_risk > 0.6:
        risk_level = "HIGH"
        recommendations.append("⚠️  OVERALL RISK: HIGH - Close monitoring needed")
    elif combined_risk > 0.3:
        risk_level = "MEDIUM"
        recommendations.append("📊 OVERALL RISK: MEDIUM - Standard protocols")
    else:
        risk_level = "LOW"
        recommendations.append("✅ OVERALL RISK: LOW - Normal operations")
    
    return recommendations, risk_level, combined_risk

def demo_scenario(name, train_type, block_length, speed_limit, rake_length, priority, headway, tsr_active, tsr_speed):
    """Run a complete demo scenario."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    
    print(f"Train Type: {train_type}")
    print(f"Block Length: {block_length} km")
    print(f"Speed Limit: {speed_limit} km/h")
    print(f"Rake Length: {rake_length} m")
    print(f"Priority Level: {priority}")
    print(f"Headway: {headway} seconds")
    print(f"TSR Active: {tsr_active}")
    if tsr_active == 'Y':
        print(f"TSR Speed: {tsr_speed} km/h")
    
    # Prepare features
    features_df = prepare_features(train_type, block_length, speed_limit, rake_length, 
                                 priority, headway, tsr_active, tsr_speed)
    
    # Make predictions
    delay_prediction = predict_delay(delay_model, features_df)
    conflict_prediction = predict_conflict(conflict_model, features_df)
    
    print(f"\n📊 PREDICTIONS:")
    print(f"Predicted Delay: {delay_prediction:.2f} minutes")
    print(f"Conflict Probability: {conflict_prediction:.1%}")
    
    # Generate recommendations
    recommendations, risk_level, combined_risk = generate_recommendations(delay_prediction, conflict_prediction)
    
    print(f"\n🎯 RISK ASSESSMENT:")
    print(f"Risk Level: {risk_level}")
    print(f"Combined Risk Score: {combined_risk:.3f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

def main():
    """Run the complete integration demo."""
    print("🚂 Railway Optimization System - Complete Integration Demo")
    print("=" * 60)
    
    # Load models
    print("Loading trained models...")
    global delay_model, conflict_model
    delay_model, conflict_model = load_models()
    print("✅ Both models loaded successfully!")
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Express Train - Normal Conditions",
            "train_type": "express",
            "block_length": 80,
            "speed_limit": 120,
            "rake_length": 350,
            "priority": 1,
            "headway": 300,
            "tsr_active": "N",
            "tsr_speed": 0
        },
        {
            "name": "Freight Train - TSR Active",
            "train_type": "freight",
            "block_length": 200,
            "speed_limit": 80,
            "rake_length": 800,
            "priority": 3,
            "headway": 600,
            "tsr_active": "Y",
            "tsr_speed": 40
        },
        {
            "name": "Passenger Train - Short Headway",
            "train_type": "passenger",
            "block_length": 150,
            "speed_limit": 100,
            "rake_length": 600,
            "priority": 2,
            "headway": 180,
            "tsr_active": "N",
            "tsr_speed": 0
        },
        {
            "name": "Suburban Train - High Density",
            "train_type": "suburban",
            "block_length": 50,
            "speed_limit": 80,
            "rake_length": 300,
            "priority": 2,
            "headway": 120,
            "tsr_active": "N",
            "tsr_speed": 0
        }
    ]
    
    # Run all scenarios
    for scenario in scenarios:
        demo_scenario(**scenario)
    
    print(f"\n{'='*60}")
    print("🎉 DEMO COMPLETE!")
    print("✅ Delay Prediction Model: Working")
    print("✅ Conflict Detection Model: Working")
    print("✅ Integrated Optimization: Working")
    print("✅ Risk Assessment: Working")
    print("✅ Recommendations Engine: Working")
    print(f"{'='*60}")
    print("\n🚀 The Railway Optimization System is ready for production!")

if __name__ == "__main__":
    main()