# Railway Optimization System# Railway Optimization System



A comprehensive real-time railway traffic management system featuring role-based operator-driver communication, decision-making workflows, and ML-powered optimization capabilities.A comprehensive ML-powered system for Indian Railways optimization, featuring delay prediction and conflict detection using trained XGBoost models, with mobile app integration.



## 🚂 System Overview## 🎯 System Overview



This system provides:This system provides:

- **Role-Based Access Control**: Separate interfaces for operators and drivers- **Delay Prediction**: Predicts train delays based on operational parameters

- **Real-time Communication**: Operator decisions instantly notify drivers- **Conflict Detection**: Identifies potential conflicts between trains

- **Decision Management**: Comprehensive workflow for operational decisions- **Risk Assessment**: Combines both models for comprehensive risk analysis

- **Mobile App Interface**: React Native frontend with Expo Router- **Optimization Recommendations**: Suggests operational actions based on predictions

- **FastAPI Backend**: Robust API server with SQLite database- **Mobile App Interface**: React Native app for operators and drivers

- **ML Integration Ready**: Designed for LLM integration and intelligent decision-making

## 🚀 Features

## 🎯 Current Features

### ✅ Completed Components

### ✅ Production Ready Components

1. **Synthetic Data Generation**

1. **Role-Based Authentication System**   - Creates realistic railway operation scenarios

   - Operator accounts for traffic control and decision-making   - 850+ records with Indian Railways characteristics

   - Driver accounts for receiving and acknowledging notifications   - Includes delay distributions, TSR conditions, and conflict scenarios

   - Secure login with predefined test credentials

2. **Trained ML Models**

2. **Operator Dashboard**   - **Delay Prediction Model**: XGBoost regressor (sklearn Pipeline)

   - Real-time train monitoring and status tracking   - **Conflict Detection Model**: XGBoost classifier (sklearn Pipeline)

   - Critical situation detection and alerts   - Both models use 21 engineered features

   - Decision-making interface with multiple action types   - Compatible with scikit-learn 1.6.1

   - System health monitoring and prediction engine status

3. **FastAPI Backend**

3. **Driver Dashboard**   - RESTful API for model serving

   - Real-time notification system for operator decisions   - Multiple endpoints for different use cases

   - Priority-based notification management (normal/high/critical)   - CORS support for frontend integration

   - Acknowledgment and completion tracking   - Health monitoring and error handling

   - Notification history and status updates

4. **Integration & Testing**

4. **FastAPI Backend API**   - Complete model integration working

   - Train registration and management   - Comprehensive test suite

   - Operator decision recording and processing   - Demo scenarios for validation

   - Driver notification generation and tracking

   - Critical situation analysis and reporting## 📊 Model Performance

   - Comprehensive health monitoring

### Delay Prediction Model

5. **Real-time Communication**- **Type**: XGBoost Regressor

   - Automatic notification generation from operator decisions- **Features**: 21 engineered features

   - Live updates with configurable refresh intervals- **Output**: Delay in minutes

   - Status tracking throughout the workflow- **Status**: ✅ Working



## 🔧 Quick Start### Conflict Detection Model

- **Type**: XGBoost Classifier 

### Prerequisites- **Features**: 21 engineered features

- Python 3.8+- **Output**: Conflict probability (0-1)

- Node.js 16+- **Status**: ✅ Working

- React Native development environment

## 🛠️ Setup Instructions

### 1. Backend Setup

```bash### Prerequisites

cd backend- Python 3.13.1

python -m venv env- Windows PowerShell (for Windows users)

# Windows

env\Scripts\activate### 1. Clone and Setup

# Linux/Mac```bash

source env/bin/activategit clone <repository-url>

cd railway-optimization-system

pip install -r requirements.txt```

python create_models.py  # Initialize database

python app.py           # Start server on port 8001### 2. Backend Setup

``````bash

cd backend

### 2. Frontend Setuppython -m venv env

```bash.\env\Scripts\Activate.ps1  # Windows PowerShell

cd frontendpip install -r requirements.txt

npm install```

npx expo start          # Start development server

```### 3. Generate Synthetic Data (Optional)

```bash

### 3. Sample Data Setupcd ../scripts

```bashpython create_seed_dataset.py

cd scriptspython generate_synthetic_data.py

python create_sample_data.py  # Populate with test datapython visualize_data.py

``````



## 🔐 Test Credentials### 4. Run the API Server

```bash

### Operator Accountscd ../backend

- **OP123** / **op@123** - Primary operator accountpython app.py

- **OP456** / **station@456** - Secondary operator account# OR

uvicorn app:app --host 0.0.0.0 --port 8000

### Driver Accounts  ```

- **DR123** / **dr@123** - Primary driver account

- **DR456** / **train@456** - Secondary driver account### 5. Test the System

```bash

## 🔌 API Documentation# Test models directly

python test_models_direct.py

### Core Endpoints

- `GET /` - System health and status# Run integration demo

- `GET /trains` - List all registered trainspython integration_demo.py

- `POST /trains/register` - Register new train

- `GET /critical-situations` - Get critical situations# Test API endpoints (requires server running)

python test_full_api.py

### Operator Workflow```

- `POST /operator/decisions` - Create operator decision

- `GET /operator/decisions` - Get all decisions## 🔌 API Endpoints



### Driver Workflow### Health & Info

- `GET /driver/notifications/{driver_id}` - Get notifications- `GET /` - Basic health check

- `POST /driver/notifications/{id}/acknowledge` - Acknowledge notification- `GET /health` - Detailed health status

- `POST /driver/notifications/{id}/complete` - Complete notification- `GET /model_info` - Model information



**📖 Detailed API Documentation**: See `backend/API_REFERENCE.md`### Delay Prediction

- `POST /predict_delay` - Full feature delay prediction

## 📱 Mobile App Features- `POST /predict_delay_simple` - Simplified delay prediction



### Operator Interface### Conflict Detection

- **Train Monitoring**: Real-time status of all registered trains- `POST /check_conflict` - Full feature conflict detection

- **Decision Making**: Create decisions with multiple action types:- `POST /check_conflict_simple` - Simplified conflict detection

  - Reduce Speed (with new speed limit)

  - Allow Delay (with delay duration)### Optimization

  - Hold Train (stop at current location)- `POST /optimize` - Complete optimization with both models

  - Emergency Stop (immediate stop)- `POST /suggest_action` - Action suggestions based on predictions

- **Critical Alerts**: System-generated critical situation notifications

- **System Status**: Health monitoring and prediction engine status## 📝 API Usage Examples



### Driver Interface### Simple Delay Prediction

- **Notification Management**: Receive and manage operator decisions```python

- **Priority System**: Visual indicators for normal/high/critical notificationsimport requests

- **Action Tracking**: Acknowledge receipt and mark actions complete

- **Real-time Updates**: Auto-refresh for instant notification deliverydata = {

    "train_type": "express",

## 🤖 LLM Integration Ready    "block_length_km": 150.0,

    "speed_limit_kmph": 110.0,

The system is architected for easy LLM integration with suggested endpoints:    "rake_length_m": 400.0,

    "priority_level": 1,

### Proposed AI Features    "headway_seconds": 240,

- `POST /ai/analyze-situation` - Intelligent situation analysis    "tsr_active": "Y",

- `POST /ai/predict-critical-situations` - Predictive analytics    "tsr_speed_kmph": 60.0

- `POST /ai/query` - Natural language interface}

- `POST /ai/validate-decision` - Decision validation

response = requests.post("http://localhost:8000/predict_delay_simple", json=data)

**📋 Integration Guide**: See `backend/README.md` for detailed LLM integration guidelinesprint(response.json())

```

## 📁 Project Structure

### Complete Optimization

``````python

railway-optimization-system/response = requests.post("http://localhost:8000/optimize", json=data)

├── README.md                    # This fileresult = response.json()

├── backend/                     # FastAPI backendprint(f"Delay: {result['optimization_results']['predicted_delay_minutes']} min")

│   ├── app.py                  # Main applicationprint(f"Conflict Risk: {result['optimization_results']['conflict_likelihood']:.1%}")

│   ├── README.md               # Backend documentationprint(f"Risk Level: {result['optimization_results']['risk_level']}")

│   ├── API_REFERENCE.md        # Detailed API docs```

│   ├── requirements.txt        # Python dependencies

│   ├── create_models.py        # Database initialization## 📁 Project Structure

│   ├── continuous_prediction_engine.py

│   ├── train_state_manager.py```

│   └── data/                   # SQLite databaserailway-optimization-system/

├── frontend/                   # React Native app├── README.md

│   ├── app/                    # Expo Router structure├── backend/

│   │   ├── index.tsx          # Login screen│   ├── app.py                      # FastAPI application

│   │   ├── operator/          # Operator dashboard│   ├── requirements.txt            # Python dependencies

│   │   ├── driver/            # Driver dashboard│   ├── models/

│   │   └── _layout.tsx        # App layout│   │   ├── delay_model.pkl         # Trained delay prediction model

│   ├── services/              # API integration│   │   └── conflict_model.pkl      # Trained conflict detection model

│   └── package.json│   ├── test_models_direct.py       # Direct model testing

├── scripts/                   # Utility scripts│   ├── test_full_api.py           # API endpoint testing

│   ├── create_sample_data.py  # Test data generator│   ├── integration_demo.py         # Complete integration demo

│   ├── create_seed_dataset.py│   └── env/                       # Virtual environment

│   ├── generate_synthetic_data.py└── scripts/

│   └── visualize_data.py    ├── create_seed_dataset.py      # Seed data generation

└── data/                     # Data files and datasets    ├── generate_synthetic_data.py  # Synthetic data creation

```    └── visualize_data.py          # Data visualization

```

## 🎮 Usage Scenarios

## 🎯 Model Features

### Typical Operator Workflow

1. Login with operator credentials (OP123/op@123)Both models use the same 21 features:

2. Monitor real-time train status and critical situations- `train_id`, `train_type`, `date`, `section_id`

3. Select a train requiring intervention- `from_station`, `to_station`, `scheduled_departure`, `scheduled_arrival`

4. Choose appropriate decision (reduce speed, delay, hold, etc.)- `actual_departure`, `actual_arrival`, `block_length_km`, `track_type`

5. Provide reason and set priority level- `speed_limit_kmph`, `rake_length_m`, `priority_level`, `headway_seconds`

6. Submit decision - driver automatically notified- `tsr_active`, `tsr_speed_kmph`, `platform_assigned`, `controller_action`

- `propagated_delay_minutes`

### Typical Driver Workflow

1. Login with driver credentials (DR123/dr@123)## 🚦 Risk Assessment Levels

2. View pending notifications from operators

3. Acknowledge receipt of instructions### Combined Risk Scoring

4. Execute required actions- **CRITICAL** (>0.8): Immediate intervention required

5. Mark notifications as completed- **HIGH** (0.6-0.8): Close monitoring needed  

6. Monitor for new instructions- **MEDIUM** (0.3-0.6): Standard protocols

- **LOW** (<0.3): Normal operations

## 🚀 Current Status

### Recommendations Engine

### ✅ Production Ready- Delay-based suggestions (passenger notifications, rerouting)

- [x] Complete role-based authentication system- Conflict-based suggestions (traffic control, headway adjustments)

- [x] Real-time operator-driver communication- Combined risk assessment and operational recommendations

- [x] Mobile-responsive React Native interface

- [x] Comprehensive API with all workflows## 📈 Current Status

- [x] Database-backed persistence

- [x] Sample data and testing scenarios### ✅ Working Components

- [x] Logout functionality and session management- [x] Synthetic data generation (850+ records)

- [x] XGBoost delay prediction model

### 🔄 Ready for Enhancement- [x] XGBoost conflict detection model

- [ ] LLM integration for intelligent decision support- [x] FastAPI backend with all endpoints

- [ ] Advanced analytics and reporting- [x] Model integration and testing

- [ ] Historical performance tracking- [x] Risk assessment and recommendations

- [ ] Integration with real railway systems- [x] Complete demonstration scenarios

- [ ] Enhanced visualization and dashboards

- [ ] Multi-station and multi-route support### 🔄 Ready for Enhancement

- [ ] Frontend web interface

## 🤝 Contributing- [ ] Real-time data integration

- [ ] Historical performance tracking

This system provides a solid foundation for railway optimization with clear integration points for AI/ML enhancements. The architecture supports easy extension and customization for specific railway operational requirements.- [ ] Advanced visualization dashboards

- [ ] Production deployment configuration

**For LLM Integration**: See detailed guidelines in `backend/README.md`

## 🤝 Contributing

## 📞 Support

This system is ready for production use and further development. The core ML models and API infrastructure are fully functional and tested.

- **API Documentation**: Interactive docs at http://localhost:8001/docs

- **Backend Setup**: See `backend/README.md`## 📞 Support

- **API Reference**: See `backend/API_REFERENCE.md`

- **Sample Data**: Use `scripts/create_sample_data.py`For technical support or questions about the railway optimization system, please refer to the comprehensive test suite and documentation provided.
