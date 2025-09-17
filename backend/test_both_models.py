"""
Test both delay and conflict models to understand their feature requirements.
"""

import joblib
import pandas as pd
import os

def test_both_models():
    """Test both delay and conflict models."""
    script_dir = r"C:\Users\siriv\railway-optimization-system\backend"
    
    # Load delay model
    delay_model_path = os.path.join(script_dir, "models", "delay_model.pkl")
    delay_model = joblib.load(delay_model_path)
    print(f"Delay model: {type(delay_model)}")
    print(f"Delay model features: {delay_model.n_features_in_}")
    if hasattr(delay_model, 'feature_names_in_'):
        print(f"Delay model feature names: {list(delay_model.feature_names_in_)}")
    
    # Load conflict model
    conflict_model_path = os.path.join(script_dir, "models", "conflict_model.pkl")
    conflict_model = joblib.load(conflict_model_path)
    print(f"\nConflict model: {type(conflict_model)}")
    print(f"Conflict model features: {conflict_model.n_features_in_}")
    if hasattr(conflict_model, 'feature_names_in_'):
        print(f"Conflict model feature names: {list(conflict_model.feature_names_in_)}")
    
    # Create test data
    sample_data = {
        'train_id': ['T12345'],
        'train_type': ['express'],
        'date': ['2025-09-17'],
        'section_id': ['SEC001'],
        'from_station': ['NDLS'],
        'to_station': ['HWH'],
        'scheduled_departure': ['10:00:00'],
        'scheduled_arrival': ['12:00:00'],
        'actual_departure': ['10:05:00'],  # 5 min delay
        'actual_arrival': ['12:05:00'],    # 5 min delay
        'block_length_km': [120.5],
        'track_type': ['double'],
        'speed_limit_kmph': [110.0],
        'rake_length_m': [650.0],
        'priority_level': [1],
        'headway_seconds': [300],
        'tsr_active': ['N'],
        'tsr_speed_kmph': [0.0],
        'platform_assigned': ['P1'],
        'controller_action': ['none'],
        'propagated_delay_minutes': [0.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test delay prediction
    print("\n=== Testing Delay Model ===")
    try:
        delay_pred = delay_model.predict(df)
        print(f"Delay prediction: {delay_pred[0]:.2f} minutes")
    except Exception as e:
        print(f"Delay prediction error: {e}")
    
    # Test conflict prediction
    print("\n=== Testing Conflict Model ===")
    try:
        # Check if it's a classifier (has predict_proba) or regressor
        if hasattr(conflict_model, 'predict_proba'):
            conflict_prob = conflict_model.predict_proba(df)
            print(f"Conflict probability: {conflict_prob[0]}")
            print(f"Conflict likelihood: {conflict_prob[0][1]:.3f}")
        else:
            conflict_pred = conflict_model.predict(df)
            print(f"Conflict prediction: {conflict_pred[0]}")
    except Exception as e:
        print(f"Conflict prediction error: {e}")

if __name__ == "__main__":
    test_both_models()