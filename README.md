# Railway Optimization System

A comprehensive ML-powered system for Indian Railways optimization, featuring delay prediction and conflict detection using trained XGBoost models.

## 🎯 System Overview

This system provides:
- **Delay Prediction**: Predicts train delays based on operational parameters
- **Conflict Detection**: Identifies potential conflicts between trains
- **Risk Assessment**: Combines both models for comprehensive risk analysis
- **Optimization Recommendations**: Suggests operational actions based on predictions

## 🚀 Features

### ✅ Completed Components

1. **Synthetic Data Generation**
   - Creates realistic railway operation scenarios
   - 850+ records with Indian Railways characteristics
   - Includes delay distributions, TSR conditions, and conflict scenarios

2. **Trained ML Models**
   - **Delay Prediction Model**: XGBoost regressor (sklearn Pipeline)
   - **Conflict Detection Model**: XGBoost classifier (sklearn Pipeline)
   - Both models use 21 engineered features
   - Compatible with scikit-learn 1.6.1

3. **FastAPI Backend**
   - RESTful API for model serving
   - Multiple endpoints for different use cases
   - CORS support for frontend integration
   - Health monitoring and error handling

4. **Integration & Testing**
   - Complete model integration working
   - Comprehensive test suite
   - Demo scenarios for validation

## 📊 Model Performance

### Delay Prediction Model
- **Type**: XGBoost Regressor
- **Features**: 21 engineered features
- **Output**: Delay in minutes
- **Status**: ✅ Working

### Conflict Detection Model
- **Type**: XGBoost Classifier 
- **Features**: 21 engineered features
- **Output**: Conflict probability (0-1)
- **Status**: ✅ Working

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.13.1
- Windows PowerShell (for Windows users)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd railway-optimization-system
```

### 2. Backend Setup
```bash
cd backend
python -m venv env
.\env\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

### 3. Generate Synthetic Data (Optional)
```bash
cd ../scripts
python create_seed_dataset.py
python generate_synthetic_data.py
python visualize_data.py
```

### 4. Run the API Server
```bash
cd ../backend
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Test the System
```bash
# Test models directly
python test_models_direct.py

# Run integration demo
python integration_demo.py

# Test API endpoints (requires server running)
python test_full_api.py
```

## 🔌 API Endpoints

### Health & Info
- `GET /` - Basic health check
- `GET /health` - Detailed health status
- `GET /model_info` - Model information

### Delay Prediction
- `POST /predict_delay` - Full feature delay prediction
- `POST /predict_delay_simple` - Simplified delay prediction

### Conflict Detection
- `POST /check_conflict` - Full feature conflict detection
- `POST /check_conflict_simple` - Simplified conflict detection

### Optimization
- `POST /optimize` - Complete optimization with both models
- `POST /suggest_action` - Action suggestions based on predictions

## 📝 API Usage Examples

### Simple Delay Prediction
```python
import requests

data = {
    "train_type": "express",
    "block_length_km": 150.0,
    "speed_limit_kmph": 110.0,
    "rake_length_m": 400.0,
    "priority_level": 1,
    "headway_seconds": 240,
    "tsr_active": "Y",
    "tsr_speed_kmph": 60.0
}

response = requests.post("http://localhost:8000/predict_delay_simple", json=data)
print(response.json())
```

### Complete Optimization
```python
response = requests.post("http://localhost:8000/optimize", json=data)
result = response.json()
print(f"Delay: {result['optimization_results']['predicted_delay_minutes']} min")
print(f"Conflict Risk: {result['optimization_results']['conflict_likelihood']:.1%}")
print(f"Risk Level: {result['optimization_results']['risk_level']}")
```

## 📁 Project Structure

```
railway-optimization-system/
├── README.md
├── backend/
│   ├── app.py                      # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   ├── models/
│   │   ├── delay_model.pkl         # Trained delay prediction model
│   │   └── conflict_model.pkl      # Trained conflict detection model
│   ├── test_models_direct.py       # Direct model testing
│   ├── test_full_api.py           # API endpoint testing
│   ├── integration_demo.py         # Complete integration demo
│   └── env/                       # Virtual environment
└── scripts/
    ├── create_seed_dataset.py      # Seed data generation
    ├── generate_synthetic_data.py  # Synthetic data creation
    └── visualize_data.py          # Data visualization
```

## 🎯 Model Features

Both models use the same 21 features:
- `train_id`, `train_type`, `date`, `section_id`
- `from_station`, `to_station`, `scheduled_departure`, `scheduled_arrival`
- `actual_departure`, `actual_arrival`, `block_length_km`, `track_type`
- `speed_limit_kmph`, `rake_length_m`, `priority_level`, `headway_seconds`
- `tsr_active`, `tsr_speed_kmph`, `platform_assigned`, `controller_action`
- `propagated_delay_minutes`

## 🚦 Risk Assessment Levels

### Combined Risk Scoring
- **CRITICAL** (>0.8): Immediate intervention required
- **HIGH** (0.6-0.8): Close monitoring needed  
- **MEDIUM** (0.3-0.6): Standard protocols
- **LOW** (<0.3): Normal operations

### Recommendations Engine
- Delay-based suggestions (passenger notifications, rerouting)
- Conflict-based suggestions (traffic control, headway adjustments)
- Combined risk assessment and operational recommendations

## 📈 Current Status

### ✅ Working Components
- [x] Synthetic data generation (850+ records)
- [x] XGBoost delay prediction model
- [x] XGBoost conflict detection model
- [x] FastAPI backend with all endpoints
- [x] Model integration and testing
- [x] Risk assessment and recommendations
- [x] Complete demonstration scenarios

### 🔄 Ready for Enhancement
- [ ] Frontend web interface
- [ ] Real-time data integration
- [ ] Historical performance tracking
- [ ] Advanced visualization dashboards
- [ ] Production deployment configuration

## 🤝 Contributing

This system is ready for production use and further development. The core ML models and API infrastructure are fully functional and tested.

## 📞 Support

For technical support or questions about the railway optimization system, please refer to the comprehensive test suite and documentation provided.