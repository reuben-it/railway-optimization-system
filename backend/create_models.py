"""
Create sample ML models for the railway optimization system.
This script generates simple XGBoost models for delay prediction and conflict detection.
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import os

# Set up paths
DATA_PATH = "../data/synthetic_data.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Load the synthetic data for model training."""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} records from {DATA_PATH}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_delay_prediction_model(df):
    """Create and save a delay prediction model."""
    # Extract features for delay prediction
    feature_columns = [
        'train_type', 'block_length_km', 'speed_limit_kmph', 
        'rake_length_m', 'priority_level', 'headway_seconds', 
        'tsr_active', 'tsr_speed_kmph'
    ]
    
    # Convert categorical variables to numeric
    df_model = df.copy()
    df_model['train_type'] = df_model['train_type'].map({
        'express': 1, 'passenger': 2, 'freight': 3, 'suburban': 4
    })
    df_model['tsr_active'] = df_model['tsr_active'].map({'Y': 1, 'N': 0})
    
    # Define features and target
    X = df_model[feature_columns]
    y = df_model['delay_minutes']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Delay Prediction Model - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "delay_model.pkl")
    joblib.dump(model, model_path)
    print(f"Delay prediction model saved to {model_path}")
    
    # Save feature names for later use
    feature_names_path = os.path.join(MODEL_DIR, "delay_model_features.txt")
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_columns))
    
    return model

def create_conflict_detection_model(df):
    """Create and save a conflict detection model."""
    # Extract features for conflict detection
    feature_columns = [
        'train_type', 'delay_minutes', 'block_length_km', 
        'speed_limit_kmph', 'priority_level', 'headway_seconds',
        'propagated_delay_minutes'
    ]
    
    # Convert categorical variables to numeric
    df_model = df.copy()
    df_model['train_type'] = df_model['train_type'].map({
        'express': 1, 'passenger': 2, 'freight': 3, 'suburban': 4
    })
    
    # Create a binary target variable for conflicts
    # Assuming conflict_likelihood > 0.5 indicates a conflict
    df_model['is_conflict'] = (df_model['conflict_likelihood'] > 0.5).astype(int)
    
    # Define features and target
    X = df_model[feature_columns]
    y = df_model['is_conflict']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Conflict Detection Model - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "conflict_model.pkl")
    joblib.dump(model, model_path)
    print(f"Conflict detection model saved to {model_path}")
    
    # Save feature names for later use
    feature_names_path = os.path.join(MODEL_DIR, "conflict_model_features.txt")
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_columns))
    
    return model

def main():
    """Main function to create and save the models."""
    print("Loading synthetic data...")
    df = load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print("\nCreating delay prediction model...")
    delay_model = create_delay_prediction_model(df)
    
    print("\nCreating conflict detection model...")
    conflict_model = create_conflict_detection_model(df)
    
    print("\nModel creation completed successfully!")

if __name__ == "__main__":
    print("Starting model creation process...")
    main()
    print("Process completed!")