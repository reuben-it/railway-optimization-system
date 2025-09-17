# Railway Optimization System - API Reference for LLM Integration

## 🤖 LLM Integration API Specification

This document provides detailed API specifications and examples for integrating Large Language Models into the Railway Optimization System.

## 🔄 Current API Endpoints (Production Ready)

### 1. System Health & Status

#### `GET /`
Returns system health and prediction engine status.

**Response:**
```json
{
    "status": "Railway Optimization System is running",
    "prediction_engine": {
        "running": true,
        "last_update": "2024-01-15T14:30:00Z",
        "total_predictions": 1234
    },
    "database": {
        "status": "connected",
        "total_trains": 15,
        "active_decisions": 8
    }
}
```

### 2. Train Management

#### `GET /trains`
Retrieves all registered trains with current status.

**Response:**
```json
{
    "trains": [
        {
            "train_id": "EXP001",
            "train_type": "EXPRESS",
            "current_section": "SECTION_A",
            "from_station": "Mumbai Central",
            "to_station": "New Delhi",
            "scheduled_arrival": "2024-01-15T16:30:00Z",
            "current_speed": 120,
            "speed_limit_kmph": 130,
            "status": "ON_TIME",
            "state": "SAFE",
            "block_length_km": 35.0,
            "rake_length_m": 450,
            "priority_level": 1,
            "headway_seconds": 240,
            "last_updated": "2024-01-15T14:30:00Z"
        }
    ]
}
```

#### `POST /trains/register`
Registers a new train in the system.

**Request:**
```json
{
    "train_id": "EXP002",
    "train_type": "EXPRESS",
    "current_section": "SECTION_B",
    "from_station": "Chennai Central",
    "to_station": "Bangalore City",
    "scheduled_arrival": "2024-01-15T18:00:00Z",
    "block_length_km": 40.0,
    "speed_limit_kmph": 120,
    "rake_length_m": 400,
    "priority_level": 1,
    "headway_seconds": 300
}
```

**Response:**
```json
{
    "message": "Train EXP002 registered successfully",
    "train_id": "EXP002"
}
```

### 3. Critical Situations

#### `GET /critical-situations`
Returns current critical situations requiring attention.

**Response:**
```json
{
    "critical_situations": [
        {
            "train_id": "EXP001",
            "situation_type": "SPEED_VIOLATION",
            "severity": "HIGH",
            "description": "Train exceeding speed limit in restricted zone",
            "current_speed": 95,
            "allowed_speed": 80,
            "location": "SECTION_A_KM_45",
            "detected_at": "2024-01-15T14:25:00Z"
        },
        {
            "train_id": "PASS002",
            "situation_type": "HEADWAY_VIOLATION",
            "severity": "MEDIUM",
            "description": "Insufficient headway distance maintained",
            "current_headway": 150,
            "required_headway": 180,
            "following_train": "EXP003",
            "detected_at": "2024-01-15T14:20:00Z"
        }
    ]
}
```

### 4. Operator Decisions

#### `POST /operator/decisions`
Creates an operator decision that generates driver notifications.

**Request:**
```json
{
    "train_id": "EXP001",
    "decision": "reduce_speed",
    "new_speed_limit": 80,
    "reason": "Heavy rainfall causing poor visibility conditions",
    "operator_id": "OP123",
    "priority": "high"
}
```

**Supported Decisions:**
- `reduce_speed` - Requires `new_speed_limit`
- `allow_delay` - Requires `delay_minutes`
- `hold_train` - Stops train at current location
- `emergency_stop` - Immediate emergency stop
- `reroute` - Requires `new_route_section`

**Priority Levels:**
- `normal` - Standard operational decision
- `high` - Important safety decision
- `critical` - Emergency situation

**Response:**
```json
{
    "message": "Decision recorded and driver notified successfully",
    "decision_id": 15,
    "notification_id": 23,
    "train_id": "EXP001"
}
```

#### `GET /operator/decisions`
Retrieves all operator decisions with pagination.

**Query Parameters:**
- `limit` (optional): Number of decisions to return (default: 50)
- `offset` (optional): Pagination offset (default: 0)
- `train_id` (optional): Filter by specific train

**Response:**
```json
{
    "decisions": [
        {
            "id": 15,
            "train_id": "EXP001",
            "decision": "reduce_speed",
            "new_speed_limit": 80,
            "reason": "Heavy rainfall causing poor visibility conditions",
            "operator_id": "OP123",
            "priority": "high",
            "created_at": "2024-01-15T14:30:00Z",
            "status": "executed"
        }
    ],
    "total": 1,
    "has_more": false
}
```

### 5. Driver Notifications

#### `GET /driver/notifications/{driver_id}`
Retrieves notifications for a specific driver.

**Response:**
```json
{
    "notifications": [
        {
            "id": 23,
            "decision_id": 15,
            "train_id": "EXP001",
            "message": "REDUCE SPEED to 80 km/h - Heavy rainfall causing poor visibility conditions",
            "priority": "high",
            "status": "pending",
            "created_at": "2024-01-15T14:30:00Z",
            "acknowledged_at": null,
            "completed_at": null
        }
    ]
}
```

#### `POST /driver/notifications/{id}/acknowledge`
Acknowledges receipt of a notification.

**Response:**
```json
{
    "message": "Notification acknowledged successfully",
    "notification_id": 23,
    "acknowledged_at": "2024-01-15T14:35:00Z"
}
```

#### `POST /driver/notifications/{id}/complete`
Marks a notification as completed.

**Response:**
```json
{
    "message": "Notification marked as completed",
    "notification_id": 23,
    "completed_at": "2024-01-15T14:40:00Z"
}
```

## 🧠 Proposed LLM Integration Endpoints

### 1. Intelligent Situation Analysis

#### `POST /ai/analyze-situation`
Uses LLM to analyze current train situation and recommend actions.

**Request:**
```json
{
    "train_id": "EXP001",
    "current_conditions": {
        "weather": {
            "condition": "heavy_rain",
            "visibility_km": 2.5,
            "wind_speed_kmph": 45
        },
        "track_status": {
            "condition": "maintenance_ahead",
            "affected_sections": ["SECTION_A_KM_50"],
            "maintenance_type": "signal_repair"
        },
        "traffic_density": "high",
        "time_of_day": "peak_hour"
    },
    "historical_data": {
        "similar_situations": 15,
        "average_delay_minutes": 12,
        "incident_rate": 0.08
    }
}
```

**Response:**
```json
{
    "analysis_id": "ai_001",
    "train_id": "EXP001",
    "recommended_decision": {
        "action": "reduce_speed",
        "parameters": {
            "new_speed_limit": 80,
            "recommended_section": "SECTION_A_KM_40_to_60"
        },
        "confidence_score": 0.87,
        "reasoning": "Heavy rainfall combined with upcoming signal maintenance requires speed reduction to 80 km/h for safety. Historical data shows 15 similar situations with average 12-minute delays when speed was reduced proactively."
    },
    "alternative_options": [
        {
            "action": "hold_train",
            "confidence_score": 0.65,
            "reasoning": "Complete stop until weather improves, but may cause significant delays",
            "estimated_delay_minutes": 25
        },
        {
            "action": "reroute",
            "confidence_score": 0.45,
            "reasoning": "Alternative route available but adds 15km distance",
            "parameters": {
                "new_route": "SECTION_B_alternate"
            }
        }
    ],
    "risk_assessment": {
        "safety_risk": "medium",
        "delay_risk": "low",
        "passenger_impact": "minimal"
    },
    "generated_at": "2024-01-15T14:30:00Z"
}
```

### 2. Predictive Critical Situation Detection

#### `POST /ai/predict-critical-situations`
Uses LLM to predict potential critical situations.

**Request:**
```json
{
    "time_horizon_minutes": 30,
    "current_trains": [
        {
            "train_id": "EXP001",
            "current_speed": 120,
            "location": "SECTION_A_KM_45",
            "destination": "New Delhi"
        },
        {
            "train_id": "PASS002",
            "current_speed": 85,
            "location": "SECTION_A_KM_35",
            "destination": "Mumbai CST"
        }
    ],
    "environmental_factors": {
        "weather_forecast": {
            "next_30_min": "heavy_rain_continuing",
            "visibility_trend": "decreasing"
        },
        "scheduled_maintenance": [
            {
                "section": "SECTION_A_KM_50",
                "start_time": "2024-01-15T15:00:00Z",
                "duration_minutes": 60
            }
        ]
    },
    "historical_patterns": {
        "similar_conditions_incidents": 8,
        "peak_hour_delay_probability": 0.35
    }
}
```

**Response:**
```json
{
    "prediction_id": "pred_001",
    "predicted_situations": [
        {
            "train_id": "EXP001",
            "predicted_issue": "potential_delay_due_to_maintenance",
            "probability": 0.73,
            "estimated_occurrence": "2024-01-15T15:05:00Z",
            "severity": "medium",
            "description": "Train EXP001 likely to encounter delays due to scheduled signal maintenance at SECTION_A_KM_50",
            "preventive_actions": [
                {
                    "action": "reduce_speed_early",
                    "timing": "5_minutes_before_maintenance_zone",
                    "effectiveness": 0.85
                },
                {
                    "action": "increase_headway",
                    "target_headway_seconds": 300,
                    "effectiveness": 0.70
                }
            ]
        },
        {
            "train_id": "PASS002",
            "predicted_issue": "weather_related_visibility_issue",
            "probability": 0.68,
            "estimated_occurrence": "2024-01-15T14:50:00Z",
            "severity": "high",
            "description": "Passenger train approaching area with deteriorating visibility conditions",
            "preventive_actions": [
                {
                    "action": "activate_enhanced_signaling",
                    "timing": "immediate",
                    "effectiveness": 0.90
                },
                {
                    "action": "reduce_speed_to_60",
                    "timing": "before_entering_low_visibility_zone",
                    "effectiveness": 0.95
                }
            ]
        }
    ],
    "system_recommendations": {
        "overall_risk_level": "medium",
        "suggested_monitoring_frequency": "every_5_minutes",
        "operator_alert_threshold": "probability_above_0.70"
    },
    "confidence_metadata": {
        "model_version": "v2.1",
        "training_data_similar_scenarios": 245,
        "prediction_accuracy_historical": 0.82
    },
    "generated_at": "2024-01-15T14:30:00Z"
}
```

### 3. Natural Language Query Interface

#### `POST /ai/query`
Processes natural language queries about the railway system.

**Request:**
```json
{
    "query": "What's the current status of all express trains heading to Delhi and are there any potential delays?",
    "context": {
        "user_role": "operator",
        "user_id": "OP123",
        "timestamp": "2024-01-15T14:30:00Z"
    },
    "include_predictions": true,
    "response_format": "detailed"
}
```

**Response:**
```json
{
    "query_id": "nlq_001",
    "response": {
        "summary": "Currently 2 express trains are heading to Delhi. EXP001 is on schedule but may face weather-related delays, while EXP005 is running 5 minutes late due to earlier congestion.",
        "detailed_analysis": {
            "trains_found": [
                {
                    "train_id": "EXP001",
                    "status": "ON_TIME",
                    "current_location": "SECTION_A_KM_45",
                    "eta_delhi": "2024-01-15T16:30:00Z",
                    "potential_issues": [
                        {
                            "type": "weather_delay",
                            "probability": 0.65,
                            "estimated_delay": "10-15 minutes"
                        }
                    ]
                },
                {
                    "train_id": "EXP005",
                    "status": "DELAYED",
                    "current_delay_minutes": 5,
                    "current_location": "SECTION_C_KM_25",
                    "eta_delhi": "2024-01-15T19:35:00Z",
                    "delay_reason": "Previous section congestion cleared"
                }
            ]
        },
        "suggested_actions": [
            {
                "train_id": "EXP001",
                "action": "monitor_weather_conditions",
                "priority": "medium",
                "reasoning": "Proactive monitoring will allow early intervention if conditions worsen"
            }
        ],
        "relevant_alerts": [
            {
                "type": "weather_warning",
                "message": "Heavy rainfall expected in SECTION_A for next 2 hours",
                "impact_level": "medium"
            }
        ]
    },
    "confidence": 0.92,
    "processing_time_ms": 1250,
    "generated_at": "2024-01-15T14:30:00Z"
}
```

### 4. AI-Assisted Decision Validation

#### `POST /ai/validate-decision`
Validates operator decisions before execution.

**Request:**
```json
{
    "proposed_decision": {
        "train_id": "EXP001",
        "decision": "emergency_stop",
        "reason": "Suspected track obstruction reported",
        "operator_id": "OP123"
    },
    "current_context": {
        "train_status": {
            "current_speed": 120,
            "location": "SECTION_A_KM_45",
            "passengers_on_board": 450
        },
        "surrounding_trains": [
            {
                "train_id": "PASS002",
                "distance_behind_km": 5,
                "current_speed": 85
            }
        ]
    }
}
```

**Response:**
```json
{
    "validation_result": {
        "recommendation": "proceed_with_caution",
        "safety_score": 0.85,
        "risk_assessment": {
            "immediate_safety_risk": "low",
            "passenger_impact": "high",
            "system_impact": "medium"
        },
        "suggested_modifications": [
            {
                "modification": "gradual_deceleration_instead_of_emergency_stop",
                "reasoning": "Gradual deceleration to 40 km/h would be safer for passengers while still addressing the reported obstruction",
                "safety_improvement": 0.15
            }
        ],
        "compliance_check": {
            "regulatory_compliance": "compliant",
            "safety_protocols": "partially_compliant",
            "recommendations": [
                "Verify obstruction report with track sensors",
                "Alert following trains of deceleration"
            ]
        }
    },
    "alternative_approaches": [
        {
            "approach": "controlled_deceleration",
            "parameters": {
                "target_speed": 40,
                "deceleration_rate": "standard"
            },
            "effectiveness": 0.90
        }
    ],
    "confidence": 0.88,
    "generated_at": "2024-01-15T14:30:00Z"
}
```

## 🔧 Integration Guidelines

### Authentication for AI Endpoints
```python
# Recommended header for AI endpoints
headers = {
    "Authorization": "Bearer <ai_service_token>",
    "Content-Type": "application/json",
    "X-Operator-ID": "OP123",  # For audit logging
    "X-Request-ID": "unique_request_id"  # For tracking
}
```

### Error Handling
```json
{
    "error": {
        "code": "AI_ANALYSIS_FAILED",
        "message": "Unable to analyze situation due to insufficient data",
        "details": {
            "missing_parameters": ["weather_data", "historical_context"],
            "retry_recommended": true,
            "retry_after_seconds": 30
        }
    }
}
```

### Rate Limiting
- AI endpoints should implement rate limiting
- Suggested limits:
  - `/ai/analyze-situation`: 10 requests/minute per operator
  - `/ai/predict-critical-situations`: 5 requests/minute per system
  - `/ai/query`: 20 requests/minute per user

### Performance Considerations
- AI endpoints should respond within 5 seconds for real-time operations
- Implement async processing for complex analyses
- Cache frequent queries for improved performance
- Provide streaming responses for long-running predictions

This API specification provides a comprehensive foundation for integrating LLM capabilities into the Railway Optimization System while maintaining compatibility with the existing operator-driver workflow.