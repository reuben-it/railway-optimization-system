# Railway Optimization System - Backend API

## 🚂 Overview

This is the backend API server for the Railway Optimization System, a real-time railway traffic management platform that supports operator-driver communication and decision-making workflows. The system is designed to be extended with LLM capabilities for intelligent railway optimization.

## 🏗️ Architecture

```
Backend Components:
├── app.py                           # Main FastAPI application
├── continuous_prediction_engine.py  # Real-time prediction system
├── train_state_manager.py          # Train state management
├── create_models.py                # Database models and setup
├── requirements.txt                # Python dependencies
└── data/                           # SQLite database storage
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FastAPI
- SQLite

### Installation & Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv env
   # Windows
   env\Scripts\activate
   # Linux/Mac
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize database:**
   ```bash
   python create_models.py
   ```

5. **Start the server:**
   ```bash
   python app.py
   ```

6. **Verify installation:**
   - API Documentation: http://localhost:8001/docs
   - Health Check: http://localhost:8001/

### Sample Data Setup

To populate the system with test data:
```bash
cd ../scripts
python create_sample_data.py
```

## 🔌 API Endpoints

### Core System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | System health and status |
| `GET` | `/trains` | List all registered trains |
| `POST` | `/trains/register` | Register a new train |
| `GET` | `/critical-situations` | Get current critical situations |

### Operator Workflow Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/operator/decisions` | Create operator decision |
| `GET` | `/operator/decisions` | Get all operator decisions |

### Driver Workflow Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/driver/notifications/{driver_id}` | Get driver notifications |
| `POST` | `/driver/notifications/{id}/acknowledge` | Acknowledge notification |
| `POST` | `/driver/notifications/{id}/complete` | Complete notification |

## 🤖 LLM Integration Points

### 1. Intelligent Decision Making (`/operator/decisions`)

**Current Implementation:**
- Operators manually create decisions based on train status
- Simple rule-based recommendations

**LLM Enhancement Opportunities:**
```python
# Suggested endpoint: POST /ai/analyze-situation
{
    "train_id": "EXP001",
    "current_conditions": {
        "weather": "heavy_rain",
        "track_status": "maintenance_ahead",
        "traffic_density": "high"
    },
    "historical_data": [...]
}

# LLM Response:
{
    "recommended_decision": "reduce_speed",
    "confidence": 0.87,
    "reasoning": "Heavy rainfall combined with upcoming maintenance requires speed reduction to 80 km/h for safety",
    "alternative_options": [
        {"decision": "hold_train", "confidence": 0.65, "reason": "..."},
        {"decision": "reroute", "confidence": 0.45, "reason": "..."}
    ]
}
```

### 2. Critical Situation Analysis (`/critical-situations`)

**Current Implementation:**
- Basic rule-based critical situation detection
- Simple threshold-based alerts

**LLM Enhancement Opportunities:**
```python
# Suggested endpoint: POST /ai/predict-critical-situations
{
    "time_horizon_minutes": 30,
    "trains": [...],
    "weather_forecast": {...},
    "historical_incidents": [...]
}

# LLM Response:
{
    "predicted_situations": [
        {
            "train_id": "EXP001",
            "predicted_issue": "potential_delay",
            "probability": 0.73,
            "estimated_time": "2024-01-15T14:30:00Z",
            "preventive_actions": ["reduce_speed", "increase_headway"]
        }
    ]
}
```

### 3. Natural Language Interface

**Integration Point: New endpoint for natural language queries**
```python
# Suggested endpoint: POST /ai/query
{
    "query": "What's the status of all express trains heading to Delhi?",
    "context": {
        "user_role": "operator",
        "user_id": "OP123"
    }
}

# LLM Response:
{
    "response": "Currently 3 express trains are heading to Delhi...",
    "relevant_trains": ["EXP001", "EXP005"],
    "suggested_actions": [...]
}
```

## 📊 Data Models

### Train Registration
```json
{
    "train_id": "EXP001",
    "train_type": "EXPRESS",
    "current_section": "SECTION_A",
    "from_station": "Mumbai Central",
    "to_station": "New Delhi",
    "scheduled_arrival": "2024-01-15T14:30:00Z",
    "block_length_km": 35.0,
    "speed_limit_kmph": 130,
    "rake_length_m": 450,
    "priority_level": 1,
    "headway_seconds": 240
}
```

### Operator Decision
```json
{
    "train_id": "EXP001",
    "decision": "reduce_speed",
    "new_speed_limit": 80,
    "reason": "Heavy rainfall causing poor visibility",
    "operator_id": "OP123",
    "priority": "high"
}
```

### Driver Notification
```json
{
    "id": 1,
    "decision_id": 1,
    "train_id": "EXP001",
    "message": "REDUCE SPEED to 80 km/h - Heavy rainfall causing poor visibility",
    "priority": "high",
    "status": "pending",
    "created_at": "2024-01-15T14:30:00Z"
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional configurations
DATABASE_URL=sqlite:///data/railway_system.db
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:8081,http://localhost:19006
```

### Database Schema
The system uses SQLite with the following main tables:
- `trains` - Train information and current status
- `operator_decisions` - Decisions made by operators
- `driver_notifications` - Notifications sent to drivers

## 🧪 Testing

### API Testing
Use the interactive documentation at http://localhost:8001/docs or test with curl:

```bash
# Health check
curl http://localhost:8001/

# Get all trains
curl http://localhost:8001/trains

# Register a train
curl -X POST http://localhost:8001/trains/register \
  -H "Content-Type: application/json" \
  -d '{"train_id": "TEST001", "train_type": "EXPRESS", ...}'
```

### Frontend Integration
The frontend React Native app connects to this backend:
- Operator Dashboard: Real-time train monitoring and decision making
- Driver Dashboard: Notification management and acknowledgment

## 🔐 Authentication

### Current Implementation
- Simple role-based access control
- Hardcoded credentials for demo:
  - Operators: `OP123/op@123`, `OP456/station@456`
  - Drivers: `DR123/dr@123`, `DR456/train@456`

### LLM Integration Considerations
- Consider implementing JWT tokens for AI service authentication
- Add rate limiting for AI endpoints
- Implement audit logging for AI-generated decisions

## 📈 Performance Considerations

- **Real-time Updates**: Current system polls every 15-30 seconds
- **Database**: SQLite suitable for development; consider PostgreSQL for production
- **Scalability**: Designed for horizontal scaling with separate services

## 🤝 Contributing & LLM Integration

### Recommended Approach for LLM Integration

1. **Create AI Service Module**
   ```python
   # ai_service.py
   class RailwayAIService:
       def analyze_train_situation(self, train_data):
           # LLM integration logic
           pass
       
       def predict_critical_situations(self, system_state):
           # Predictive analysis
           pass
       
       def generate_natural_language_response(self, query):
           # NL interface
           pass
   ```

2. **Add AI Endpoints to app.py**
   ```python
   @app.post("/ai/analyze-situation")
   async def analyze_situation(request: SituationAnalysisRequest):
       # Integrate with LLM service
       pass
   ```

3. **Extend Data Models**
   ```python
   # Add AI-related tables
   class AIDecision(Base):
       __tablename__ = "ai_decisions"
       # Store AI recommendations and confidence scores
   ```

4. **Testing Strategy**
   - Unit tests for AI service components
   - Integration tests with mock LLM responses
   - Performance tests for real-time AI analysis

### Development Workflow
1. Fork/clone the repository
2. Create feature branch for LLM integration
3. Implement AI service components
4. Add comprehensive tests
5. Update API documentation
6. Submit pull request

## 📞 Support

For questions or issues:
1. Check the API documentation at http://localhost:8001/docs
2. Review logs in the terminal where the server is running
3. Use the sample data generator for testing scenarios

## 📄 License

This project is part of the Railway Optimization System development.