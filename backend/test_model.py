"""
Model inspection and compatibility testing script.
Handles version compatibility issues and tests model loading.
"""

import joblib
import pickle
import pandas as pd
import numpy as np
import warnings
import os

# Suppress warnings for testing
warnings.filterwarnings('ignore')

def test_model_loading():
    """Test loading the XGBoost model with error handling."""
    model_path = "models/delay_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        # Try loading with joblib first
        print("Attempting to load model with joblib...")
        model = joblib.load(model_path)
        print(f"Model loaded successfully with joblib: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading with joblib: {e}")
        
        try:
            # Try loading with pickle as fallback
            print("Attempting to load model with pickle...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully with pickle: {type(model)}")
            return model
        except Exception as e2:
            print(f"Error loading with pickle: {e2}")
            return None

def test_model_prediction(model):
    """Test making predictions with the loaded model."""
    if model is None:
        print("No model available for testing")
        return False
    
    try:
        # Create sample input data
        sample_data = pd.DataFrame({
            'train_type': [1],  # express
            'block_length_km': [100.0],
            'speed_limit_kmph': [120.0],
            'rake_length_m': [600.0],
            'priority_level': [1],
            'headway_seconds': [300],
            'tsr_active': [0],
            'tsr_speed_kmph': [0.0]
        })
        
        print("Testing prediction with sample data...")
        prediction = model.predict(sample_data)
        print(f"Prediction successful: {prediction[0]}")
        return True
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        
        # Try to get model feature names if available
        try:
            if hasattr(model, 'feature_names_in_'):
                print(f"Model expects features: {model.feature_names_in_}")
            elif hasattr(model, 'get_booster'):
                print("XGBoost model detected")
                print(f"Model feature names: {model.get_booster().feature_names}")
        except:
            pass
        
        return False

def inspect_model_details(model):
    """Inspect model details and requirements."""
    if model is None:
        return
    
    print("\n=== Model Inspection ===")
    print(f"Model type: {type(model)}")
    
    # Check if it's an XGBoost model
    if str(type(model)).find('xgboost') != -1:
        print("XGBoost model detected")
        try:
            print(f"Number of features: {model.n_features_in_}")
            if hasattr(model, 'feature_names_in_'):
                print(f"Feature names: {list(model.feature_names_in_)}")
        except:
            pass
    
    # Check if it's a sklearn pipeline
    elif hasattr(model, 'steps'):
        print("Sklearn Pipeline detected")
        print(f"Pipeline steps: {[step[0] for step in model.steps]}")
    
    # Check other common attributes
    if hasattr(model, 'feature_names_in_'):
        print(f"Expected features: {list(model.feature_names_in_)}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"Number of input features: {model.n_features_in_}")

def create_compatible_test_data(model):
    """Create test data based on model requirements."""
    try:
        # Try to determine expected features
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            print(f"Creating test data for features: {feature_names}")
            
            # Create sample data based on feature names
            sample_data = {}
            for feature in feature_names:
                if 'train_type' in feature.lower():
                    sample_data[feature] = [1]
                elif 'length' in feature.lower():
                    sample_data[feature] = [100.0]
                elif 'speed' in feature.lower():
                    sample_data[feature] = [120.0]
                elif 'priority' in feature.lower():
                    sample_data[feature] = [1]
                elif 'headway' in feature.lower():
                    sample_data[feature] = [300]
                elif 'tsr' in feature.lower():
                    sample_data[feature] = [0]
                else:
                    sample_data[feature] = [0.0]
            
            test_df = pd.DataFrame(sample_data)
            print(f"Test data shape: {test_df.shape}")
            print(f"Test data columns: {list(test_df.columns)}")
            
            return test_df
            
    except Exception as e:
        print(f"Error creating compatible test data: {e}")
    
    return None

def main():
    """Main function to test model loading and compatibility."""
    print("=== Railway Optimization Model Testing ===")
    
    # Test model loading
    model = test_model_loading()
    
    if model is None:
        print("Failed to load model. Please check the model file.")
        return
    
    # Inspect model details
    inspect_model_details(model)
    
    # Test prediction with standard data
    success = test_model_prediction(model)
    
    if not success:
        print("\nStandard prediction failed. Trying with compatible data...")
        # Try with compatible test data
        compatible_data = create_compatible_test_data(model)
        if compatible_data is not None:
            try:
                prediction = model.predict(compatible_data)
                print(f"Compatible prediction successful: {prediction[0]}")
            except Exception as e:
                print(f"Compatible prediction also failed: {e}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()