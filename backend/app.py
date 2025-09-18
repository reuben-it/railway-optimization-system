"""
FastAPI backend for Railway Optimization System.
Serves trained ML models for delay prediction and conflict detection.
Now includes continuous prediction engine for real-time monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import joblib
import re
import requests
import json
import numpy as np
import logging
import os
import time

# Import our new components
from train_state_manager import TrainStateManager, TrainState
from continuous_prediction_engine import ContinuousPredictionEngine, TrainData, PredictionResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Railway Optimization API - Continuous Prediction System",
    description="API for railway delay prediction, conflict detection, and continuous monitoring",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and engines
delay_model = None
conflict_model = None
state_manager = None
prediction_engine = None

# Global storage for operator decisions and driver notifications
operator_decisions: List[Dict] = []
driver_notifications: List[Dict] = []

# Load models from repository models/ folder
CONFLICT_MODEL_PATH = "./models/conflict_model.pkl"
DELAY_MODEL_PATH = "./models/delay_model.pkl"

try:
    conflict_model = joblib.load(CONFLICT_MODEL_PATH)
except Exception as e:
    conflict_model = None

try:
    delay_model = joblib.load(DELAY_MODEL_PATH)
except Exception as e:
    delay_model = None

class Scenario(BaseModel):
    data: Dict[str, Any]

# Define input schemas
class DelayPredictionInput(BaseModel):
    train_id: str
    train_type: str  # express, passenger, freight, suburban
    date: str  # YYYY-MM-DD format
    section_id: str
    from_station: str
    to_station: str
    scheduled_departure: str  # HH:MM:SS format
    scheduled_arrival: str    # HH:MM:SS format
    actual_departure: str     # HH:MM:SS format
    actual_arrival: str       # HH:MM:SS format
    block_length_km: float
    track_type: str          # single, double, electrified
    speed_limit_kmph: float
    rake_length_m: float
    priority_level: int
    headway_seconds: int
    tsr_active: str          # Y or N
    tsr_speed_kmph: Optional[float] = 0.0
    platform_assigned: str
    controller_action: str   # hold, divert, reschedule, none
    propagated_delay_minutes: Optional[float] = 0.0

# Simplified input for easier frontend integration
class SimpleDelayInput(BaseModel):
    train_type: str  # express, passenger, freight, suburban
    block_length_km: float
    speed_limit_kmph: float
    rake_length_m: float
    priority_level: int
    headway_seconds: int
    tsr_active: str  # Y or N
    tsr_speed_kmph: Optional[float] = 0.0

class ConflictDetectionInput(BaseModel):
    train_type: str
    delay_minutes: float
    block_length_km: float
    speed_limit_kmph: float
    priority_level: int
    headway_seconds: int
    propagated_delay_minutes: Optional[float] = 0.0

class OptimizationInput(BaseModel):
    train_type: str
    block_length_km: float
    speed_limit_kmph: float
    rake_length_m: float
    priority_level: int
    headway_seconds: int
    tsr_active: str
    tsr_speed_kmph: Optional[float] = 0.0
    current_delay: Optional[float] = 0.0

# New schemas for continuous prediction system
class TrainRegistrationInput(BaseModel):
    train_id: str
    train_type: str
    current_section: str
    from_station: str
    to_station: str
    scheduled_arrival: str
    actual_delay: float = 0.0
    block_length_km: float
    speed_limit_kmph: float
    rake_length_m: float
    priority_level: int
    headway_seconds: int
    tsr_active: str = "N"
    tsr_speed_kmph: float = 0.0

class TrainUpdateInput(BaseModel):
    train_id: str
    actual_delay: Optional[float] = None
    current_section: Optional[str] = None
    tsr_active: Optional[str] = None
    tsr_speed_kmph: Optional[float] = None
    priority_level: Optional[int] = None
    headway_seconds: Optional[int] = None

class OperatorActionInput(BaseModel):
    train_id: str
    action: str  # hold, pass, reroute, reschedule
    operator_id: str
    reason: Optional[str] = None

class OperatorDecisionInput(BaseModel):
    train_id: str
    decision: str  # allow_delay, reduce_speed, hold_train, emergency_stop
    delay_minutes: Optional[float] = None
    new_speed_limit: Optional[float] = None
    reason: str
    operator_id: str
    priority: Optional[str] = "normal"  # normal, high, critical
    priority: str = "normal"  # normal, high, critical

class DriverNotification(BaseModel):
    notification_id: str
    driver_id: str
    train_id: str
    message: str
    decision_type: str
    action_required: str
    timestamp: str
    status: str = "pending"  # pending, acknowledged, completed
    operator_id: str

def load_models():
    """Load the trained ML models."""
    global delay_model, conflict_model
    
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load delay prediction model with error handling
        delay_model_path = os.path.join(script_dir, "models", "delay_model.pkl")
        if os.path.exists(delay_model_path):
            try:
                delay_model = joblib.load(delay_model_path)
                logger.info("Delay prediction model loaded successfully with joblib")
            except Exception as e1:
                logger.warning(f"Failed to load with joblib: {e1}")
                try:
                    import pickle
                    with open(delay_model_path, 'rb') as f:
                        delay_model = pickle.load(f)
                    logger.info("Delay prediction model loaded successfully with pickle")
                except Exception as e2:
                    logger.error(f"Failed to load with pickle: {e2}")
                    logger.warning("Will use fallback prediction method")
                    delay_model = None
        else:
            logger.warning(f"Delay model not found at {delay_model_path}")
        
        # Load conflict detection model (if available)
        conflict_model_path = os.path.join(script_dir, "models", "conflict_model.pkl")
        if os.path.exists(conflict_model_path):
            try:
                conflict_model = joblib.load(conflict_model_path)
                logger.info("Conflict detection model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load conflict model: {e}")
                conflict_model = None
        else:
            logger.warning(f"Conflict model not found at {conflict_model_path}")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def initialize_prediction_system():
    """Initialize the continuous prediction system."""
    global state_manager, prediction_engine
    
    try:
        # Initialize state manager
        state_manager = TrainStateManager(
            base_cooldown=600,  # 10 minutes
            conflict_threshold=0.7,
            watchlist_threshold=0.4,
            critical_escalation_count=3
        )
        
        # Initialize prediction engine
        prediction_engine = ContinuousPredictionEngine(
            delay_model=delay_model,
            conflict_model=conflict_model,
            state_manager=state_manager,
            prediction_interval=30,  # 30 seconds
            max_concurrent_predictions=10
        )
        
        # Add callback for logging critical situations
        async def log_critical_callback(train_data, result, state):
            if state == TrainState.CRITICAL:
                logger.warning(f"CRITICAL: Train {train_data.train_id} - "
                             f"Conflict: {result.conflict_probability:.3f}, "
                             f"Delay: {result.delay_prediction:.1f}min")
        
        prediction_engine.add_prediction_callback(log_critical_callback)
        
        # Start the prediction engine
        prediction_engine.start()
        
        logger.info("Continuous prediction system initialized and started")
        
    except Exception as e:
        logger.error(f"Failed to initialize prediction system: {e}")
        raise

def fallback_delay_prediction(input_data):
    """Fallback delay prediction using heuristics when model is not available."""
    base_delay = 0
    
    # Get train type from input
    train_type = input_data.train_type if hasattr(input_data, 'train_type') else 'passenger'
    
    # Train type factor
    train_type_delays = {
        'express': 2,
        'passenger': 5,
        'freight': 10,
        'suburban': 3
    }
    base_delay += train_type_delays.get(train_type.lower(), 5)
    
    # Block length factor (longer blocks may have more delays)
    block_length = input_data.block_length_km if hasattr(input_data, 'block_length_km') else 100
    if block_length > 100:
        base_delay += 3
    elif block_length > 50:
        base_delay += 1
    
    # Speed limit factor (lower speeds may cause delays)
    speed_limit = input_data.speed_limit_kmph if hasattr(input_data, 'speed_limit_kmph') else 100
    if speed_limit < 80:
        base_delay += 2
    
    # Priority level factor (lower priority trains may face more delays)
    priority = input_data.priority_level if hasattr(input_data, 'priority_level') else 2
    base_delay += (priority - 1) * 2
    
    # Headway factor (shorter headways may cause more conflicts and delays)
    headway = input_data.headway_seconds if hasattr(input_data, 'headway_seconds') else 300
    if headway < 300:
        base_delay += 4
    elif headway < 600:
        base_delay += 2
    
    # TSR factor
    tsr_active = input_data.tsr_active if hasattr(input_data, 'tsr_active') else 'N'
    if tsr_active.upper() == 'Y':
        base_delay += 8
        tsr_speed = input_data.tsr_speed_kmph if hasattr(input_data, 'tsr_speed_kmph') else 60
        if tsr_speed < 50:
            base_delay += 5
    
    # Add some randomness but keep it deterministic for testing
    import hashlib
    input_str = f"{train_type}_{block_length}_{speed_limit}_{priority}_{headway}_{tsr_active}"
    seed = int(hashlib.md5(input_str.encode()).hexdigest()[:8], 16)
    np.random.seed(seed % 1000)
    variation = np.random.normal(0, 2)
    
    final_delay = max(0, base_delay + variation)
    return final_delay

def prepare_delay_features(input_data):
    """Prepare features for delay prediction with all required columns."""
    from datetime import datetime, timedelta
    import random
    
    # If it's a SimpleDelayInput, create full feature set with defaults
    if hasattr(input_data, 'train_type') and not hasattr(input_data, 'train_id'):
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
            'propagated_delay_minutes': 0.0
        }
    else:
        # Use provided full feature set
        features = {
            'train_id': input_data.train_id,
            'train_type': input_data.train_type,
            'date': input_data.date,
            'section_id': input_data.section_id,
            'from_station': input_data.from_station,
            'to_station': input_data.to_station,
            'scheduled_departure': input_data.scheduled_departure,
            'scheduled_arrival': input_data.scheduled_arrival,
            'actual_departure': input_data.actual_departure,
            'actual_arrival': input_data.actual_arrival,
            'block_length_km': input_data.block_length_km,
            'track_type': input_data.track_type,
            'speed_limit_kmph': input_data.speed_limit_kmph,
            'rake_length_m': input_data.rake_length_m,
            'priority_level': input_data.priority_level,
            'headway_seconds': input_data.headway_seconds,
            'tsr_active': input_data.tsr_active,
            'tsr_speed_kmph': input_data.tsr_speed_kmph,
            'platform_assigned': input_data.platform_assigned,
            'controller_action': input_data.controller_action,
            'propagated_delay_minutes': input_data.propagated_delay_minutes
        }
    
    return pd.DataFrame([features])

def prepare_conflict_features(input_data):
    """Prepare features for conflict detection using the same format as delay prediction."""
    from datetime import datetime, timedelta
    import random
    
    # If it's a simplified input, create full feature set with defaults
    if hasattr(input_data, 'train_type') and not hasattr(input_data, 'train_id'):
        # Generate default values for missing features
        current_time = datetime.now()
        train_id = f"T{random.randint(10000, 99999)}"
        
        # Use delay_minutes if provided, otherwise use current_delay or 0
        delay_minutes = getattr(input_data, 'delay_minutes', getattr(input_data, 'current_delay', 0))
        
        # Calculate actual times based on delay
        scheduled_dep = current_time.strftime('%H:%M:%S')
        scheduled_arr = (current_time + timedelta(hours=2)).strftime('%H:%M:%S')
        actual_dep = (current_time + timedelta(minutes=delay_minutes)).strftime('%H:%M:%S')
        actual_arr = (current_time + timedelta(hours=2, minutes=delay_minutes)).strftime('%H:%M:%S')
        
        # Create a full feature dictionary
        features = {
            'train_id': train_id,
            'train_type': input_data.train_type,
            'date': current_time.strftime('%Y-%m-%d'),
            'section_id': f"SEC{random.randint(1, 100):03d}",
            'from_station': 'NDLS',  # Default station
            'to_station': 'HWH',     # Default station
            'scheduled_departure': scheduled_dep,
            'scheduled_arrival': scheduled_arr,
            'actual_departure': actual_dep,
            'actual_arrival': actual_arr,
            'block_length_km': getattr(input_data, 'block_length_km', 100),
            'track_type': 'double',  # Default track type
            'speed_limit_kmph': getattr(input_data, 'speed_limit_kmph', 100),
            'rake_length_m': getattr(input_data, 'rake_length_m', 600),
            'priority_level': getattr(input_data, 'priority_level', 2),
            'headway_seconds': getattr(input_data, 'headway_seconds', 300),
            'tsr_active': getattr(input_data, 'tsr_active', 'N'),
            'tsr_speed_kmph': getattr(input_data, 'tsr_speed_kmph', 0),
            'platform_assigned': f"P{random.randint(1, 8)}",
            'controller_action': 'none',
            'propagated_delay_minutes': getattr(input_data, 'propagated_delay_minutes', 0)
        }
    else:
        # Use provided full feature set (same as delay prediction)
        features = {
            'train_id': input_data.train_id,
            'train_type': input_data.train_type,
            'date': input_data.date,
            'section_id': input_data.section_id,
            'from_station': input_data.from_station,
            'to_station': input_data.to_station,
            'scheduled_departure': input_data.scheduled_departure,
            'scheduled_arrival': input_data.scheduled_arrival,
            'actual_departure': input_data.actual_departure,
            'actual_arrival': input_data.actual_arrival,
            'block_length_km': input_data.block_length_km,
            'track_type': input_data.track_type,
            'speed_limit_kmph': input_data.speed_limit_kmph,
            'rake_length_m': input_data.rake_length_m,
            'priority_level': input_data.priority_level,
            'headway_seconds': input_data.headway_seconds,
            'tsr_active': input_data.tsr_active,
            'tsr_speed_kmph': input_data.tsr_speed_kmph,
            'platform_assigned': input_data.platform_assigned,
            'controller_action': input_data.controller_action,
            'propagated_delay_minutes': input_data.propagated_delay_minutes
        }
    
    return pd.DataFrame([features])

# Minimal LLM config kept optional; not required for local use
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_KEY = None
headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

def query_llm(prompt, max_tokens=300):
    if not HF_API_KEY:
        return None
    payload = {"inputs": prompt, "parameters": {"temperature": 0.0, "max_new_tokens": max_tokens}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    try:
        output = response.json()
        return output[0].get("generated_text", "")
    except Exception as e:
        return None


def rule_validator(parsed_json, recommendation):
    train_type = parsed_json.get("train_type", "")
    action = recommendation.get("action", "")
    section_type = parsed_json.get("track_type", "")
    delay_minutes = parsed_json.get("delay_minutes", 0)
    headway_seconds = parsed_json.get("headway_seconds", 0)

    if section_type == "single" and train_type == "freight" and action == "pass":
        recommendation["rule_validation"] = "Invalid - Freight cannot precede Express on single line."
        recommendation["action"] = "hold"

    if action == "reroute" and delay_minutes < 30:
        recommendation["rule_validation"] = "Invalid - Reroute not allowed for delay < 30 min."
        recommendation["action"] = "hold"

    if headway_seconds < 300 and action == "pass":
        recommendation["rule_validation"] = "Invalid - Headway violation."
        recommendation["action"] = "hold"

    if "rule_validation" not in recommendation:
        recommendation["rule_validation"] = "Valid"

    return recommendation


def extract_features_21(parsed_json):
    features = {
        "delay_minutes": parsed_json.get("delay_minutes", 0),
        "priority_level": parsed_json.get("priority_level", 1),
        "headway_seconds": parsed_json.get("headway_seconds", 300),
        "block_length_km": parsed_json.get("block_length_km", 0),
        "speed_limit_kmph": parsed_json.get("speed_limit_kmph", 0)
    }

    track_type = parsed_json.get("track_type", "")
    features["track_single"] = 1 if track_type == "single" else 0
    features["track_double"] = 1 if track_type == "double" else 0
    features["track_loop"] = 1 if track_type == "loop" else 0

    train_type = parsed_json.get("train_type", "").lower()
    features["train_express"] = 1 if train_type == "express" else 0
    features["train_freight"] = 1 if train_type == "freight" else 0
    features["train_passenger"] = 1 if train_type == "passenger" else 0
    features["train_suburban"] = 1 if train_type == "suburban" else 0
    features["train_maintenance"] = 1 if train_type == "maintenance" else 0

    features["tsr_active"] = 1 if parsed_json.get("tsr_active", "N") == "Y" else 0

    for i in range(15, 22):
        features[f"feature{i}"] = 0

    return features


def build_full_df(parsed_json):
    full_columns = [
        "train_id", "train_type", "from_station", "to_station",
        "scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival",
        "delay_minutes", "priority_level", "headway_seconds", "track_type", "block_length_km",
        "speed_limit_kmph", "tsr_active", "tsr_speed_kmph", "platform_assigned",
        "controller_action", "rake_length_m", "date", "propagated_delay_minutes", "section_id"
    ]

    string_columns = [
        "train_id", "train_type", "from_station", "to_station",
        "scheduled_departure", "scheduled_arrival", "actual_departure",
        "actual_arrival", "track_type", "section_id",
        "platform_assigned", "controller_action", "date"
    ]

    row = {}
    for col in full_columns:
        if col in parsed_json:
            row[col] = parsed_json[col]
        else:
            row[col] = "NA" if col in string_columns else 0

    features_21 = extract_features_21(parsed_json)
    row.update(features_21)

    df = pd.DataFrame([row])

    for col in string_columns:
        df[col] = df[col].astype(str)

    return df


def recommend_action(parsed_json: Dict[str, Any]):
    features_df = build_full_df(parsed_json)

    # === Input sanitization ===
    numeric_cols = [
        "delay_minutes", "priority_level", "headway_seconds", "block_length_km",
        "speed_limit_kmph", "tsr_active", "tsr_speed_kmph", "rake_length_m",
        "propagated_delay_minutes"
    ]
    for col in numeric_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    categorical_cols = [
        "train_type", "track_type", "platform_assigned", "controller_action", "section_id"
    ]
    for col in categorical_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str)

    # Try to align dtypes with what the model's ColumnTransformer expects
    def sanitize_for_model(model, df: pd.DataFrame):
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.pipeline import Pipeline
        except Exception:
            return df

        ct = None
        # search for ColumnTransformer inside pipeline or model
        if hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if isinstance(step, ColumnTransformer):
                    ct = step
                    break
                if isinstance(step, Pipeline):
                    for sub in step.steps:
                        if isinstance(sub[1], ColumnTransformer):
                            ct = sub[1]
                            break
                    if ct is not None:
                        break
        elif isinstance(model, ColumnTransformer):
            ct = model

        if ct is None:
            return df

        # ct.transformers_ is a list of (name, transformer, columns)
        for _, transformer, cols in ct.transformers_:
            # columns may be slice or list; handle only list/tuple
            if cols is None or cols == 'drop' or cols == 'passthrough':
                continue
            if isinstance(cols, (list, tuple)):
                col_list = list(cols)
            else:
                # if columns is a boolean mask or slice, skip
                col_list = []

            # If transformer is or contains OneHotEncoder -> stringify
            is_ohe = False
            if transformer is None:
                continue
            if isinstance(transformer, OneHotEncoder):
                is_ohe = True
            else:
                # check inside Pipeline
                try:
                    if hasattr(transformer, 'named_steps'):
                        for sub in transformer.named_steps.values():
                            if isinstance(sub, OneHotEncoder):
                                is_ohe = True
                                break
                except Exception:
                    pass

            for c in col_list:
                if c in df.columns:
                    if is_ohe:
                        df[c] = df[c].astype(str)
                    else:
                        # try to coerce numeric
                        df[c] = pd.to_numeric(df[c], errors='coerce')

        return df

    # sanitize based on model pipeline if available
    if conflict_model is not None:
        try:
            features_df = sanitize_for_model(conflict_model, features_df)
        except Exception:
            pass
    # If models not loaded, provide dummy outputs to allow integration testing
    if conflict_model is None or delay_model is None:
        conflict_prob = 0.0
        delay_pred = 0.0
    else:
        try:
            conflict_prob = float(conflict_model.predict_proba(features_df)[0][1])
        except Exception as e:
            # If we hit the isnan TypeError from sklearn encoders, attempt a broad sanitization
            msg = str(e)
            if 'isnan' in msg:
                # stringify all object columns and coerce numerics again
                for c in features_df.columns:
                    if features_df[c].dtype == 'object':
                        features_df[c] = features_df[c].astype(str)
                    else:
                        features_df[c] = pd.to_numeric(features_df[c], errors='coerce')
                # retry once
                conflict_prob = float(conflict_model.predict_proba(features_df)[0][1])
            else:
                raise
        delay_pred = float(delay_model.predict(features_df)[0])

    prompt = f"""
    You are a railway traffic advisor.
    Scenario: {json.dumps(parsed_json)}.
    ML Predictions: conflict={conflict_prob:.2f}, propagated_delay={delay_pred:.2f}.
    Suggest the best operational action (hold, pass, reroute, resequence).
    Return only valid JSON with keys: action, expected_delay_change, rationale.
    """

    llm_output = query_llm(prompt)

    recommendation = None
    if llm_output:
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if match:
            try:
                recommendation = json.loads(match.group())
            except json.JSONDecodeError:
                recommendation = None

    if recommendation is None:
        recommendation = {
            "action": "hold" if conflict_prob > 0.7 else "pass",
            "expected_delay_change": -delay_pred if conflict_prob > 0.7 else 0,
            "rationale": "Fallback rule-based recommendation.",
            "rule_validation": "Valid"
        }

    recommendation = rule_validator(parsed_json, recommendation)

    log_entry = {
        "timestamp": str(datetime.now()),
        "input": parsed_json,
        "conflict_prob": conflict_prob,
        "delay_pred": delay_pred,
        "llm_output": llm_output,
        "recommendation": recommendation
    }
    with open("dss_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "conflict_probability": conflict_prob,
        "delay_prediction": delay_pred,
        "recommendation": recommendation
    }


@app.on_event("startup")
async def startup_event():
    """Load models and initialize continuous prediction system on startup."""
    load_models()
    initialize_prediction_system()
    
    # Auto-populate default trains for demo/testing
    try:
        from auto_populate_trains import ensure_minimum_trains
        ensure_minimum_trains(prediction_engine, min_trains=5)
    except Exception as e:
        logger.warning(f"Failed to auto-populate trains: {e}")
        logger.info("System will start without default trains")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of prediction system."""
    global prediction_engine
    if prediction_engine:
        prediction_engine.stop()
        logger.info("Prediction engine stopped")

@app.middleware("http")
async def log_requests(request, call_next):
    """Log incoming requests for debugging."""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Railway Optimization API - Continuous Prediction System",
        "version": "2.0.0",
        "models_loaded": {
            "delay_model": delay_model is not None,
            "conflict_model": conflict_model is not None
        },
        "prediction_engine": {
            "running": prediction_engine.is_running if prediction_engine else False,
            "total_trains": len(prediction_engine.trains) if prediction_engine else 0
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    engine_stats = prediction_engine.get_engine_stats() if prediction_engine else {}
    
    return {
        "status": "healthy",
        "models": {
            "delay_prediction": "loaded" if delay_model else "not_found",
            "conflict_detection": "loaded" if conflict_model else "not_found"
        },
        "prediction_engine": engine_stats,
        "timestamp": time.time()
    }

# Continuous Prediction System Endpoints

@app.post("/trains/register")
async def register_train(train_input: TrainRegistrationInput):
    """Register a new train for continuous monitoring."""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    try:
        train_data = TrainData(
            train_id=train_input.train_id,
            train_type=train_input.train_type,
            current_section=train_input.current_section,
            from_station=train_input.from_station,
            to_station=train_input.to_station,
            scheduled_arrival=train_input.scheduled_arrival,
            actual_delay=train_input.actual_delay,
            block_length_km=train_input.block_length_km,
            speed_limit_kmph=train_input.speed_limit_kmph,
            rake_length_m=train_input.rake_length_m,
            priority_level=train_input.priority_level,
            headway_seconds=train_input.headway_seconds,
            tsr_active=train_input.tsr_active,
            tsr_speed_kmph=train_input.tsr_speed_kmph,
            last_updated=time.time()
        )
        
        prediction_engine.add_train(train_data)
        
        return {
            "message": f"Train {train_input.train_id} registered for continuous monitoring",
            "train_id": train_input.train_id,
            "status": "registered"
        }
        
    except Exception as e:
        logger.error(f"Error registering train {train_input.train_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/trains/{train_id}/update")
async def update_train(train_id: str, train_update: TrainUpdateInput):
    """Update train data for continuous monitoring."""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    if train_id not in prediction_engine.trains:
        raise HTTPException(status_code=404, detail=f"Train {train_id} not found")
    
    try:
        # Prepare update kwargs, filtering out None values
        update_kwargs = {k: v for k, v in train_update.dict().items() 
                        if v is not None and k != "train_id"}
        
        prediction_engine.update_train(train_id, **update_kwargs)
        
        return {
            "message": f"Train {train_id} updated successfully",
            "train_id": train_id,
            "updated_fields": list(update_kwargs.keys())
        }
        
    except Exception as e:
        logger.error(f"Error updating train {train_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/trains/{train_id}")
async def remove_train(train_id: str):
    """Remove a train from continuous monitoring."""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    if train_id not in prediction_engine.trains:
        raise HTTPException(status_code=404, detail=f"Train {train_id} not found")
    
    prediction_engine.remove_train(train_id)
    
    return {
        "message": f"Train {train_id} removed from monitoring",
        "train_id": train_id,
        "status": "removed"
    }

@app.post("/trains/{train_id}/action")
async def operator_action(train_id: str, action_input: OperatorActionInput):
    """Record an operator action on a train."""
    if not prediction_engine or not state_manager:
        raise HTTPException(status_code=503, detail="Prediction system not available")
    
    if train_id not in prediction_engine.trains:
        raise HTTPException(status_code=404, detail=f"Train {train_id} not found")
    
    try:
        state_manager.mark_train_handled(
            train_id=train_id,
            action=action_input.action,
            handled_by=action_input.operator_id
        )
        
        return {
            "message": f"Action '{action_input.action}' recorded for train {train_id}",
            "train_id": train_id,
            "action": action_input.action,
            "operator": action_input.operator_id,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error recording action for train {train_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trains")
async def list_trains():
    """Get list of all trains being monitored."""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    trains_info = []
    for train_id, train_data in prediction_engine.trains.items():
        train_status = state_manager.get_train_status(train_id)
        
        trains_info.append({
            "train_id": train_id,
            "train_type": train_data.train_type,
            "current_section": train_data.current_section,
            "actual_delay": train_data.actual_delay,
            "state": train_status.state.value if train_status else "UNKNOWN",
            "last_conflict_prob": train_status.last_conflict_prob if train_status else None,
            "last_delay_pred": train_status.last_delay_pred if train_status else None,
            "last_updated": train_data.last_updated
        })
    
    return {
        "trains": trains_info,
        "total_count": len(trains_info)
    }

@app.get("/trains/{train_id}")
async def get_train_details(train_id: str):
    """Get detailed information about a specific train."""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    if train_id not in prediction_engine.trains:
        raise HTTPException(status_code=404, detail=f"Train {train_id} not found")
    
    train_data = prediction_engine.trains[train_id]
    train_status = state_manager.get_train_status(train_id)
    recent_predictions = prediction_engine.get_train_predictions(train_id, limit=10)
    
    return {
        "train_data": {
            "train_id": train_data.train_id,
            "train_type": train_data.train_type,
            "current_section": train_data.current_section,
            "from_station": train_data.from_station,
            "to_station": train_data.to_station,
            "actual_delay": train_data.actual_delay,
            "priority_level": train_data.priority_level,
            "tsr_active": train_data.tsr_active,
            "last_updated": train_data.last_updated
        },
        "current_status": {
            "state": train_status.state.value if train_status else "UNKNOWN",
            "last_conflict_prob": train_status.last_conflict_prob if train_status else None,
            "last_delay_pred": train_status.last_delay_pred if train_status else None,
            "next_check_time": train_status.next_check_time if train_status else None,
            "handled_action": train_status.handled_action if train_status else None,
            "consecutive_critical": train_status.consecutive_critical_count if train_status else 0
        },
        "recent_predictions": [
            {
                "timestamp": pred.timestamp,
                "conflict_probability": pred.conflict_probability,
                "delay_prediction": pred.delay_prediction,
                "recommendations": pred.recommendations
            }
            for pred in recent_predictions
        ]
    }

@app.get("/critical-situations")
async def get_critical_situations():
    """Get current critical situations requiring operator attention."""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    return prediction_engine.get_critical_situations()

@app.get("/system-stats")
async def get_system_stats():
    """Get comprehensive system statistics."""
    if not prediction_engine or not state_manager:
        raise HTTPException(status_code=503, detail="Prediction system not available")
    
    engine_stats = prediction_engine.get_engine_stats()
    critical_situations = prediction_engine.get_critical_situations()
    
    return {
        "engine_stats": engine_stats,
        "critical_situations_summary": {
            "total_critical": critical_situations["total_critical"],
            "immediate_attention_needed": critical_situations["immediate_attention_needed"]
        },
        "models_status": {
            "delay_model": "loaded" if delay_model else "not_available",
            "conflict_model": "loaded" if conflict_model else "not_available"
        },
        "timestamp": time.time()
    }

@app.post("/predict_delay")
async def predict_delay(input_data: DelayPredictionInput):
    """Predict train delay based on complete input features."""
    try:
        if delay_model is not None:
            # Try to use the loaded model
            try:
                features_df = prepare_delay_features(input_data)
                predicted_delay = delay_model.predict(features_df)[0]
                predicted_delay = max(0, predicted_delay)
                
                return {
                    "predicted_delay": round(float(predicted_delay), 2),
                    "input_features": input_data.dict(),
                    "model_type": "sklearn_pipeline"
                }
            except Exception as model_error:
                logger.warning(f"Model prediction failed: {model_error}, using fallback")
                # Fall through to fallback method
        
        # Use fallback prediction method
        predicted_delay = fallback_delay_prediction(input_data)
        
        return {
            "predicted_delay": round(float(predicted_delay), 2),
            "input_features": input_data.dict(),
            "model_type": "heuristic_fallback"
        }
        
    except Exception as e:
        logger.error(f"Error in delay prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_delay_simple")
async def predict_delay_simple(input_data: SimpleDelayInput):
    """Predict train delay based on simplified input features."""
    try:
        if delay_model is not None:
            # Try to use the loaded model
            try:
                features_df = prepare_delay_features(input_data)
                predicted_delay = delay_model.predict(features_df)[0]
                predicted_delay = max(0, predicted_delay)
                
                return {
                    "predicted_delay": round(float(predicted_delay), 2),
                    "input_features": input_data.dict(),
                    "model_type": "sklearn_pipeline"
                }
            except Exception as model_error:
                logger.warning(f"Model prediction failed: {model_error}, using fallback")
                # Fall through to fallback method
        
        # Use fallback prediction method
        predicted_delay = fallback_delay_prediction(input_data)
        
        return {
            "predicted_delay": round(float(predicted_delay), 2),
            "input_features": input_data.dict(),
            "model_type": "heuristic_fallback"
        }
        
    except Exception as e:
        logger.error(f"Error in delay prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/check_conflict")
async def check_conflict(input_data: ConflictDetectionInput):
    """Check conflict likelihood based on input features."""
    try:
        if conflict_model is not None:
            # Try to use the loaded conflict model
            try:
                features_df = prepare_conflict_features(input_data)
                
                # Check if it's a classifier (has predict_proba) or regressor
                if hasattr(conflict_model, 'predict_proba'):
                    conflict_probability = conflict_model.predict_proba(features_df)[0][1]  # Get probability of positive class
                    conflict_likelihood = float(conflict_probability)
                else:
                    # If it's a regressor, get the prediction directly
                    conflict_prediction = conflict_model.predict(features_df)[0]
                    conflict_likelihood = min(1.0, max(0.0, float(conflict_prediction)))  # Ensure it's between 0 and 1
                
                return {
                    "conflict_likelihood": round(conflict_likelihood, 3),
                    "input_features": input_data.dict(),
                    "model_type": "sklearn_pipeline"
                }
                
            except Exception as model_error:
                logger.warning(f"Conflict model prediction failed: {model_error}, using fallback")
                # Fall through to fallback method
        
        # Fallback heuristic calculation
        conflict_likelihood = min(1.0, (input_data.delay_minutes / 30.0) + 
                                (input_data.propagated_delay_minutes / 20.0))
        
        return {
            "conflict_likelihood": round(conflict_likelihood, 3),
            "input_features": input_data.dict(),
            "model_type": "heuristic_fallback"
        }
        
    except Exception as e:
        logger.error(f"Error in conflict detection: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/check_conflict_simple")
async def check_conflict_simple(input_data: SimpleDelayInput):
    """Check conflict likelihood based on simplified input features."""
    # Create a ConflictDetectionInput-like object for processing
    class SimpleConflictInput:
        def __init__(self, simple_input, estimated_delay=5):
            self.train_type = simple_input.train_type
            self.delay_minutes = estimated_delay
            self.block_length_km = simple_input.block_length_km
            self.speed_limit_kmph = simple_input.speed_limit_kmph
            self.priority_level = simple_input.priority_level
            self.headway_seconds = simple_input.headway_seconds
            self.propagated_delay_minutes = estimated_delay * 0.3  # Estimate propagation
            
        def dict(self):
            return {
                'train_type': self.train_type,
                'delay_minutes': self.delay_minutes,
                'block_length_km': self.block_length_km,
                'speed_limit_kmph': self.speed_limit_kmph,
                'priority_level': self.priority_level,
                'headway_seconds': self.headway_seconds,
                'propagated_delay_minutes': self.propagated_delay_minutes
            }
    
    try:
        # Estimate delay based on input characteristics
        estimated_delay = 5  # Base estimate
        if input_data.tsr_active.upper() == 'Y':
            estimated_delay += 8
        if input_data.headway_seconds < 300:
            estimated_delay += 3
        
        conflict_input = SimpleConflictInput(input_data, estimated_delay)
        
        if conflict_model is not None:
            # Try to use the loaded conflict model
            try:
                features_df = prepare_conflict_features(conflict_input)
                
                if hasattr(conflict_model, 'predict_proba'):
                    conflict_probability = conflict_model.predict_proba(features_df)[0][1]
                    conflict_likelihood = float(conflict_probability)
                else:
                    conflict_prediction = conflict_model.predict(features_df)[0]
                    conflict_likelihood = min(1.0, max(0.0, float(conflict_prediction)))
                
                return {
                    "conflict_likelihood": round(conflict_likelihood, 3),
                    "estimated_delay": estimated_delay,
                    "input_features": input_data.dict(),
                    "model_type": "sklearn_pipeline"
                }
                
            except Exception as model_error:
                logger.warning(f"Conflict model prediction failed: {model_error}, using fallback")
        
        # Fallback heuristic calculation
        conflict_likelihood = min(1.0, (estimated_delay / 30.0))
        
        return {
            "conflict_likelihood": round(conflict_likelihood, 3),
            "estimated_delay": estimated_delay,
            "input_features": input_data.dict(),
            "model_type": "heuristic_fallback"
        }
        
    except Exception as e:
        logger.error(f"Error in simple conflict detection: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/suggest_action")
async def suggest_action(input_data: OptimizationInput):
    """Suggest operational action based on predictions."""
    try:
        if delay_model is None:
            raise HTTPException(status_code=503, detail="Delay prediction model not available")
        
        # Prepare delay prediction features
        delay_features = DelayPredictionInput(
            train_type=input_data.train_type,
            block_length_km=input_data.block_length_km,
            speed_limit_kmph=input_data.speed_limit_kmph,
            rake_length_m=input_data.rake_length_m,
            priority_level=input_data.priority_level,
            headway_seconds=input_data.headway_seconds,
            tsr_active=input_data.tsr_active,
            tsr_speed_kmph=input_data.tsr_speed_kmph
        )
        
        # Get delay prediction
        features_df = prepare_delay_features(delay_features)
        predicted_delay = max(0, delay_model.predict(features_df)[0])
        
        # Calculate total delay (current + predicted additional)
        total_delay = input_data.current_delay + predicted_delay
        
        # Prepare conflict detection features
        conflict_features = ConflictDetectionInput(
            train_type=input_data.train_type,
            delay_minutes=total_delay,
            block_length_km=input_data.block_length_km,
            speed_limit_kmph=input_data.speed_limit_kmph,
            priority_level=input_data.priority_level,
            headway_seconds=input_data.headway_seconds,
            propagated_delay_minutes=total_delay * 0.3  # Estimate propagation
        )
        
        # Get conflict likelihood
        if conflict_model is not None:
            conflict_df = prepare_conflict_features(conflict_features)
            conflict_prob = conflict_model.predict_proba(conflict_df)[0][1]
        else:
            # Heuristic calculation
            conflict_prob = min(1.0, (total_delay / 30.0))
        
        # Determine recommended action
        if total_delay > 20 or conflict_prob > 0.8:
            action = "Reschedule"
            priority = "High"
        elif total_delay > 10 or conflict_prob > 0.6:
            action = "Hold or Divert"
            priority = "Medium"
        elif total_delay > 5 or conflict_prob > 0.4:
            action = "Monitor Closely"
            priority = "Low"
        else:
            action = "Proceed as Scheduled"
            priority = "Normal"
        
        return {
            "predicted_delay": round(float(predicted_delay), 2),
            "current_delay": input_data.current_delay,
            "total_delay": round(float(total_delay), 2),
            "conflict_likelihood": round(float(conflict_prob), 2),
            "recommended_action": action,
            "priority": priority,
            "input_features": input_data.dict()
        }
        
    except Exception as e:
        logger.error(f"Error in action suggestion: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize")
async def optimize_railway_operations(input_data: SimpleDelayInput):
    """Comprehensive railway optimization combining delay prediction and conflict detection."""
    try:
        # First, predict delay
        delay_response = await predict_delay_simple(input_data)
        predicted_delay = delay_response["predicted_delay"]
        
        # Then, check conflict likelihood using the simple conflict endpoint
        conflict_response = await check_conflict_simple(input_data)
        conflict_likelihood = conflict_response["conflict_likelihood"]
        
        # Generate optimization suggestions based on both models
        suggestions = []
        
        # Delay-based suggestions
        if predicted_delay > 10:
            suggestions.append("Consider rerouting or increasing train priority")
        if predicted_delay > 5:
            suggestions.append("Notify passengers of potential delays")
        
        # Conflict-based suggestions
        if conflict_likelihood > 0.7:
            suggestions.append("HIGH RISK: Implement immediate traffic control measures")
            suggestions.append("Consider emergency slot reallocation")
        elif conflict_likelihood > 0.4:
            suggestions.append("Monitor closely and prepare contingency plans")
            suggestions.append("Increase headway if possible")
        elif conflict_likelihood > 0.2:
            suggestions.append("Standard monitoring protocols sufficient")
        
        # Combined risk assessment
        combined_risk = (predicted_delay / 60.0) * 0.6 + conflict_likelihood * 0.4  # Weight delay and conflict
        
        if combined_risk > 0.8:
            risk_level = "CRITICAL"
        elif combined_risk > 0.6:
            risk_level = "HIGH"
        elif combined_risk > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "optimization_results": {
                "predicted_delay_minutes": round(predicted_delay, 2),
                "conflict_likelihood": round(conflict_likelihood, 3),
                "combined_risk_score": round(combined_risk, 3),
                "risk_level": risk_level,
                "optimization_suggestions": suggestions,
                "model_status": {
                    "delay_model": delay_response.get("model_type", "unknown"),
                    "conflict_model": conflict_response.get("model_type", "unknown")
                }
            },
            "input_data": input_data.dict()
        }
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
async def model_info():
    """Get information about loaded models."""
    info = {
        "delay_model": {
            "loaded": delay_model is not None,
            "type": "XGBoost Regressor" if delay_model else None,
            "features": [
                "train_type", "block_length_km", "speed_limit_kmph", 
                "rake_length_m", "priority_level", "headway_seconds", 
                "tsr_active", "tsr_speed_kmph"
            ] if delay_model else None
        },
        "conflict_model": {
            "loaded": conflict_model is not None,
            "type": "XGBoost Classifier" if conflict_model else "Heuristic",
            "features": [
                "train_type", "delay_minutes", "block_length_km", 
                "speed_limit_kmph", "priority_level", "headway_seconds",
                "propagated_delay_minutes"
            ] if conflict_model else None
        }
    }
    
    return info

# Operator Decision Management Endpoints
@app.post("/operator/decisions")
async def make_operator_decision(decision: OperatorDecisionInput):
    """Record an operator decision and create driver notification."""
    global operator_decisions, driver_notifications
    
    try:
        # Create decision record
        decision_record = {
            "decision_id": f"DEC_{int(time.time())}_{decision.train_id}",
            "train_id": decision.train_id,
            "decision": decision.decision,
            "delay_minutes": decision.delay_minutes,
            "new_speed_limit": decision.new_speed_limit,
            "reason": decision.reason,
            "operator_id": decision.operator_id,
            "priority": decision.priority,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": "active"
        }
        
        operator_decisions.append(decision_record)
        
        # Create driver notification
        notification_id = f"NOT_{int(time.time())}_{decision.train_id}"
        
        # Generate appropriate message based on decision
        action_messages = {
            "allow_delay": f"Delay of {decision.delay_minutes} minutes approved for train {decision.train_id}. Proceed as scheduled.",
            "reduce_speed": f"Reduce speed to {decision.new_speed_limit} km/h for train {decision.train_id}. Reason: {decision.reason}",
            "hold_train": f"Hold train {decision.train_id} at current location. Await further instructions.",
            "emergency_stop": f"EMERGENCY: Stop train {decision.train_id} immediately. Contact operator."
        }
        
        driver_notification = {
            "notification_id": notification_id,
            "driver_id": f"DRIVER_{decision.train_id}",  # Assume driver ID based on train
            "train_id": decision.train_id,
            "message": action_messages.get(decision.decision, f"New instruction for train {decision.train_id}: {decision.reason}"),
            "decision_type": decision.decision,
            "action_required": decision.decision,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": "pending",
            "operator_id": decision.operator_id,
            "priority": decision.priority
        }
        
        driver_notifications.append(driver_notification)
        
        return {
            "message": "Decision recorded and driver notified",
            "decision_id": decision_record["decision_id"],
            "notification_id": notification_id
        }
        
    except Exception as e:
        logger.error(f"Error recording operator decision: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/operator/decisions")
async def get_operator_decisions():
    """Get all operator decisions."""
    return {
        "decisions": operator_decisions,
        "total_decisions": len(operator_decisions)
    }

@app.get("/operator/decisions/{train_id}")
async def get_train_decisions(train_id: str):
    """Get decisions for a specific train."""
    train_decisions = [d for d in operator_decisions if d["train_id"] == train_id]
    return {
        "train_id": train_id,
        "decisions": train_decisions,
        "total_decisions": len(train_decisions)
    }

# Driver Notification Endpoints
@app.get("/driver/notifications/{driver_id}")
async def get_driver_notifications(driver_id: str):
    """Get notifications for a specific driver."""
    driver_notifs = [n for n in driver_notifications if n["driver_id"] == driver_id]
    return {
        "driver_id": driver_id,
        "notifications": driver_notifs,
        "pending_notifications": len([n for n in driver_notifs if n["status"] == "pending"]),
        "total_notifications": len(driver_notifs)
    }

@app.get("/driver/notifications/train/{train_id}")
async def get_train_notifications(train_id: str):
    """Get notifications for a specific train."""
    train_notifs = [n for n in driver_notifications if n["train_id"] == train_id]
    return {
        "train_id": train_id,
        "notifications": train_notifs,
        "total_notifications": len(train_notifs)
    }

@app.post("/driver/notifications/{notification_id}/acknowledge")
async def acknowledge_notification(notification_id: str):
    """Mark a notification as acknowledged by driver."""
    global driver_notifications
    
    for notification in driver_notifications:
        if notification["notification_id"] == notification_id:
            notification["status"] = "acknowledged"
            notification["acknowledged_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
            return {"message": "Notification acknowledged", "notification_id": notification_id}
    
    raise HTTPException(status_code=404, detail="Notification not found")

@app.post("/driver/notifications/{notification_id}/complete")
async def complete_notification(notification_id: str):
    """Mark a notification as completed by driver."""
    global driver_notifications
    
    for notification in driver_notifications:
        if notification["notification_id"] == notification_id:
            notification["status"] = "completed"
            notification["completed_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
            return {"message": "Notification completed", "notification_id": notification_id}
    
    raise HTTPException(status_code=404, detail="Notification not found")

@app.get("/driver/notifications")
async def get_all_notifications():
    """Get all driver notifications."""
    return {
        "notifications": driver_notifications,
        "total_notifications": len(driver_notifications),
        "pending_notifications": len([n for n in driver_notifications if n["status"] == "pending"])
    }
@app.post("/recommend")
def recommend_endpoint(scenario: Scenario):
    try:
        out = recommend_action(scenario.data)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)