import axios from 'axios';

// Base URL for the FastAPI backend
const BASE_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000, // 10 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for API responses
export interface HealthResponse {
  message: string;
  version: string;
  models_loaded: {
    delay_model: boolean;
    conflict_model: boolean;
  };
  prediction_engine: {
    running: boolean;
    total_trains: number;
  };
}

export interface TrainRegistration {
  train_id: string;
  train_type: string;
  current_section: string;
  from_station: string;
  to_station: string;
  scheduled_arrival: string;
  actual_delay?: number;
  block_length_km: number;
  speed_limit_kmph: number;
  rake_length_m: number;
  priority_level: number;
  headway_seconds: number;
  tsr_active?: string;
  tsr_speed_kmph?: number;
}

export interface TrainInfo {
  train_id: string;
  train_type: string;
  current_section: string;
  from_station: string;
  to_station: string;
  scheduled_arrival: string;
  actual_delay: number;
  state: string;
  last_conflict_prob: number;
  last_delay_pred: number;
  last_check: string;
}

export interface CriticalSituation {
  train_id: string;
  train_type: string;
  current_section: string;
  conflict_probability: number;
  delay_prediction: number;
  consecutive_critical: number;
  needs_immediate_attention: boolean;
}

export interface CriticalSituationsResponse {
  critical_situations: CriticalSituation[];
  total_critical: number;
  immediate_attention_needed: number;
}

export interface DelayPredictionRequest {
  train_type: string;
  block_length_km: number;
  speed_limit_kmph: number;
  rake_length_m: number;
  priority_level: number;
  headway_seconds: number;
  tsr_active?: string;
  tsr_speed_kmph?: number;
}

export interface DelayPredictionResponse {
  predicted_delay: number;
  input_features: DelayPredictionRequest;
  model_type: string;
}

export interface ConflictPredictionResponse {
  conflict_likelihood: number;
  estimated_delay: number;
  input_features: DelayPredictionRequest;
  model_type: string;
}

// New interfaces for operator-driver system
export interface OperatorDecision {
  train_id: string;
  decision: string; // allow_delay, reduce_speed, hold_train, emergency_stop
  delay_minutes?: number;
  new_speed_limit?: number;
  reason: string;
  operator_id: string;
  priority?: string; // normal, high, critical
}

export interface DriverNotification {
  notification_id: string;
  driver_id: string;
  train_id: string;
  message: string;
  decision_type: string;
  action_required: string;
  timestamp: string;
  status: string; // pending, acknowledged, completed
  operator_id: string;
  priority: string;
}

export interface OperatorDecisionResponse {
  message: string;
  decision_id: string;
  notification_id: string;
}

export interface NotificationsResponse {
  driver_id?: string;
  train_id?: string;
  notifications: DriverNotification[];
  pending_notifications?: number;
  total_notifications: number;
}

// API Service Class
class RailwayAPI {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get('/');
    return response.data;
  }

  // Train registration and management
  async registerTrain(train: TrainRegistration): Promise<any> {
    const response = await api.post('/trains/register', train);
    return response.data;
  }

  async getAllTrains(): Promise<TrainInfo[]> {
    const response = await api.get('/trains');
    // Handle both direct array and object with trains property
    return Array.isArray(response.data) ? response.data : (response.data.trains || []);
  }

  async getTrain(trainId: string): Promise<TrainInfo> {
    const response = await api.get(`/trains/${trainId}`);
    return response.data;
  }

  async updateTrain(trainId: string, updates: Partial<TrainRegistration>): Promise<any> {
    const response = await api.put(`/trains/${trainId}/update`, updates);
    return response.data;
  }

  async deleteTrain(trainId: string): Promise<any> {
    const response = await api.delete(`/trains/${trainId}`);
    return response.data;
  }

  // Critical situations
  async getCriticalSituations(): Promise<CriticalSituationsResponse> {
    const response = await api.get('/critical-situations');
    return response.data;
  }

  // Predictions
  async predictDelay(data: DelayPredictionRequest): Promise<DelayPredictionResponse> {
    const response = await api.post('/predict_delay_simple', data);
    return response.data;
  }

  async predictConflict(data: DelayPredictionRequest): Promise<ConflictPredictionResponse> {
    const response = await api.post('/check_conflict_simple', data);
    return response.data;
  }

  // Train actions
  async handleTrain(trainId: string): Promise<any> {
    const response = await api.post(`/trains/${trainId}/action`, {
      action: 'handle',
      operator_id: 'mobile_app',
      reason: 'Handled via mobile app'
    });
    return response.data;
  }

  async forceRecheck(trainId: string): Promise<any> {
    const response = await api.post(`/trains/${trainId}/force-recheck`);
    return response.data;
  }

  // System stats
  async getSystemStats(): Promise<any> {
    const response = await api.get('/system-stats');
    return response.data;
  }

  // Operator Decision Management
  async makeOperatorDecision(decision: OperatorDecision): Promise<OperatorDecisionResponse> {
    const response = await api.post('/operator/decisions', decision);
    return response.data;
  }

  async getOperatorDecisions(): Promise<any> {
    const response = await api.get('/operator/decisions');
    return response.data;
  }

  async getTrainDecisions(trainId: string): Promise<any> {
    const response = await api.get(`/operator/decisions/${trainId}`);
    return response.data;
  }

  // Driver Notification Management
  async getDriverNotifications(driverId: string): Promise<NotificationsResponse> {
    const response = await api.get(`/driver/notifications/${driverId}`);
    return response.data;
  }

  async getTrainNotifications(trainId: string): Promise<NotificationsResponse> {
    const response = await api.get(`/driver/notifications/train/${trainId}`);
    return response.data;
  }

  async acknowledgeNotification(notificationId: string): Promise<any> {
    const response = await api.post(`/driver/notifications/${notificationId}/acknowledge`);
    return response.data;
  }

  async completeNotification(notificationId: string): Promise<any> {
    const response = await api.post(`/driver/notifications/${notificationId}/complete`);
    return response.data;
  }

  async getAllNotifications(): Promise<NotificationsResponse> {
    const response = await api.get('/driver/notifications');
    return response.data;
  }
}

// Export singleton instance
export const railwayAPI = new RailwayAPI();

// Export error handling utility
export const handleAPIError = (error: any) => {
  if (error.response) {
    // Server responded with error status
    console.error('API Error:', error.response.data);
    return error.response.data.detail || 'Server error occurred';
  } else if (error.request) {
    // Network error
    console.error('Network Error:', error.request);
    return 'Unable to connect to server. Please check your connection.';
  } else {
    // Other error
    console.error('Error:', error.message);
    return 'An unexpected error occurred';
  }
};