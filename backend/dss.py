import joblib
import requests
import json
import re
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

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


# FastAPI wrapper
app = FastAPI()


class Scenario(BaseModel):
    data: Dict[str, Any]


@app.post("/recommend")
def recommend_endpoint(scenario: Scenario):
    try:
        out = recommend_action(scenario.data)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
