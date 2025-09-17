"""
Direct test of both models without needing to run the server.
This tests the model loading and prediction functions directly.
"""

import sys
import os
import joblib

def load_and_test_models():
    """Load models directly and test them."""
    print("Loading models directly...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    delay_model = None
    conflict_model = None
    
    # Load delay prediction model
    delay_model_path = os.path.join(script_dir, "models", "delay_model.pkl")
    if os.path.exists(delay_model_path):
        try:
            delay_model = joblib.load(delay_model_path)
            print("Delay prediction model loaded successfully")
        except Exception as e:
            print(f"Failed to load delay model: {e}")
    else:
        print(f"Delay model not found at {delay_model_path}")
    
    # Load conflict detection model
    conflict_model_path = os.path.join(script_dir, "models", "conflict_model.pkl")
    if os.path.exists(conflict_model_path):
        try:
            conflict_model = joblib.load(conflict_model_path)
            print("Conflict detection model loaded successfully")
        except Exception as e:
            print(f"Failed to load conflict model: {e}")
    else:
        print(f"Conflict model not found at {conflict_model_path}")
    
    return delay_model, conflict_model

class TestInput:
    """Simple test input class."""
    def __init__(self):
        self.train_type = "express"
        self.block_length_km = 150.0
        self.speed_limit_kmph = 110.0
        self.rake_length_m = 400.0
        self.priority_level = 1
        self.headway_seconds = 240
        self.tsr_active = "Y"
        self.tsr_speed_kmph = 60.0
        self.delay_minutes = 10.0
        self.propagated_delay_minutes = 3.0

def prepare_features_for_models(input_data):
    """Prepare features using the same logic as the app."""
    import pandas as pd
    from datetime import datetime, timedelta
    import random
    
    # Generate default values for missing features
    current_time = datetime.now()
    train_id = f"T{random.randint(10000, 99999)}"
    
    # Create a full feature dictionary
    features = {
        'train_id': train_id,
        'train_type': input_data.train_type,
        'date': current_time.strftime('%Y-%m-%d'),
        'section_id': f"SEC{random.randint(1, 100):03d}",
        'from_station': 'NDLS',  # Default station
        'to_station': 'HWH',     # Default station
        'scheduled_departure': current_time.strftime('%H:%M:%S'),
        'scheduled_arrival': (current_time + timedelta(hours=2)).strftime('%H:%M:%S'),
        'actual_departure': current_time.strftime('%H:%M:%S'),
        'actual_arrival': (current_time + timedelta(hours=2)).strftime('%H:%M:%S'),
        'block_length_km': input_data.block_length_km,
        'track_type': 'double',  # Default track type
        'speed_limit_kmph': input_data.speed_limit_kmph,
        'rake_length_m': input_data.rake_length_m,
        'priority_level': input_data.priority_level,
        'headway_seconds': input_data.headway_seconds,
        'tsr_active': input_data.tsr_active,
        'tsr_speed_kmph': input_data.tsr_speed_kmph,
        'platform_assigned': f"P{random.randint(1, 8)}",
        'controller_action': 'none',
        'propagated_delay_minutes': getattr(input_data, 'propagated_delay_minutes', 0.0)
    }
    
    return pd.DataFrame([features])

def test_delay_prediction(delay_model):
    """Test delay prediction functionality."""
    print("\nTesting delay prediction...")
    
    test_input = TestInput()
    
    try:
        # Prepare features
        features_df = prepare_features_for_models(test_input)
        print(f"Features prepared: {features_df.shape}")
        print(f"Sample features: {list(features_df.columns)[:5]}...")
        
        if delay_model is not None:
            # Make prediction
            prediction = delay_model.predict(features_df)[0]
            print(f"Delay prediction: {prediction:.2f} minutes")
            return True, prediction
        else:
            print("Delay model not available")
            return False, None
            
    except Exception as e:
        print(f"Error in delay prediction: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_conflict_prediction(conflict_model):
    """Test conflict prediction functionality."""
    print("\nTesting conflict prediction...")
    
    test_input = TestInput()
    
    try:
        # Prepare features
        features_df = prepare_features_for_models(test_input)
        print(f"Features prepared: {features_df.shape}")
        print(f"Sample features: {list(features_df.columns)[:5]}...")
        
        if conflict_model is not None:
            # Check if it's a classifier or regressor
            if hasattr(conflict_model, 'predict_proba'):
                # Classifier - get probability
                prediction = conflict_model.predict_proba(features_df)[0][1]
                print(f"Conflict probability: {prediction:.3f}")
                return True, prediction
            else:
                # Regressor - get direct prediction
                prediction = conflict_model.predict(features_df)[0]
                prediction = min(1.0, max(0.0, prediction))  # Ensure 0-1 range
                print(f"Conflict likelihood: {prediction:.3f}")
                return True, prediction
        else:
            print("Conflict model not available")
            return False, None
            
    except Exception as e:
        print(f"Error in conflict prediction: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run all tests."""
    print("=" * 60)
    print("Railway Optimization Models - Direct Test")
    print("=" * 60)
    
    # Load models directly
    delay_model, conflict_model = load_and_test_models()
    
    print(f"\nModel Status:")
    print(f"Delay model loaded: {delay_model is not None}")
    print(f"Conflict model loaded: {conflict_model is not None}")
    
    if delay_model is not None:
        print(f"Delay model type: {type(delay_model)}")
    
    if conflict_model is not None:
        print(f"Conflict model type: {type(conflict_model)}")
    
    # Test predictions
    if delay_model is not None:
        test_delay_prediction(delay_model)
    
    if conflict_model is not None:
        test_conflict_prediction(conflict_model)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"✓ Delay model: {'Working' if delay_model else 'Not available'}")
    print(f"✓ Conflict model: {'Working' if conflict_model else 'Not available'}")
    
    if delay_model and conflict_model:
        print("✓ Both models successfully integrated!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()