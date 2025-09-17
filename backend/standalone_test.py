"""
Simple standalone test of the railway backend.
This will help us understand any issues.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test loading the model directly."""
    import joblib
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "delay_model.pkl")
    
    print(f"Looking for model at: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Model loaded successfully: {type(model)}")
            print(f"Model has {model.n_features_in_} features")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

def test_prediction_with_model(model):
    """Test making a prediction with the model."""
    if model is None:
        print("No model available for testing")
        return
    
    import pandas as pd
    
    # Create sample data with all required features
    sample_data = {
        'train_id': ['T12345'],
        'train_type': ['express'],
        'date': ['2025-09-17'],
        'section_id': ['SEC001'],
        'from_station': ['NDLS'],
        'to_station': ['HWH'],
        'scheduled_departure': ['10:00:00'],
        'scheduled_arrival': ['12:00:00'],
        'actual_departure': ['10:00:00'],
        'actual_arrival': ['12:00:00'],
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
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data columns: {list(df.columns)}")
    
    try:
        prediction = model.predict(df)
        print(f"Prediction successful: {prediction[0]}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== Railway Backend Standalone Test ===")
    
    # Test model loading
    print("\n1. Testing model loading...")
    model = test_model_loading()
    
    # Test prediction
    print("\n2. Testing prediction...")
    test_prediction_with_model(model)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()