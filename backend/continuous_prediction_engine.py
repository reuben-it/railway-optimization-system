"""
Continuous Prediction Engine for Railway DSS.
Manages continuous monitoring and prediction for multiple trains using ML models.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import pandas as pd
import logging
from datetime import datetime, timedelta
import threading
import json

from train_state_manager import TrainStateManager, TrainState, TrainStatus

logger = logging.getLogger(__name__)

@dataclass
class TrainData:
    """Represents current train operational data."""
    train_id: str
    train_type: str
    current_section: str
    from_station: str
    to_station: str
    scheduled_arrival: str
    actual_delay: float  # in minutes
    block_length_km: float
    speed_limit_kmph: float
    rake_length_m: float
    priority_level: int
    headway_seconds: int
    tsr_active: str
    tsr_speed_kmph: float
    last_updated: float
    
    def to_features_dict(self) -> Dict[str, Any]:
        """Convert train data to features format for ML models."""
        current_time = datetime.now()
        
        return {
            'train_id': self.train_id,
            'train_type': self.train_type,
            'date': current_time.strftime('%Y-%m-%d'),
            'section_id': self.current_section,
            'from_station': self.from_station,
            'to_station': self.to_station,
            'scheduled_departure': current_time.strftime('%H:%M:%S'),
            'scheduled_arrival': self.scheduled_arrival,
            'actual_departure': (current_time + timedelta(minutes=self.actual_delay)).strftime('%H:%M:%S'),
            'actual_arrival': (current_time + timedelta(hours=2, minutes=self.actual_delay)).strftime('%H:%M:%S'),
            'block_length_km': self.block_length_km,
            'track_type': 'double',  # Default for now
            'speed_limit_kmph': self.speed_limit_kmph,
            'rake_length_m': self.rake_length_m,
            'priority_level': self.priority_level,
            'headway_seconds': self.headway_seconds,
            'tsr_active': self.tsr_active,
            'tsr_speed_kmph': self.tsr_speed_kmph,
            'platform_assigned': f"P{self.priority_level}",  # Simple mapping
            'controller_action': 'none',
            'propagated_delay_minutes': self.actual_delay * 0.3  # Estimate
        }

@dataclass
class PredictionResult:
    """Result of ML model predictions for a train."""
    train_id: str
    conflict_probability: float
    delay_prediction: float
    timestamp: float
    model_confidence: float
    recommendations: List[str]

class ContinuousPredictionEngine:
    """
    Main engine for continuous train monitoring and prediction.
    
    Features:
    - Continuous monitoring of multiple trains
    - State-based prediction scheduling
    - ML model integration
    - Event-driven updates
    - Operator notification system
    """
    
    def __init__(self, 
                 delay_model,
                 conflict_model,
                 state_manager: TrainStateManager,
                 prediction_interval: float = 30,  # seconds
                 max_concurrent_predictions: int = 10):
        """
        Initialize the Continuous Prediction Engine.
        
        Args:
            delay_model: Trained delay prediction model
            conflict_model: Trained conflict detection model
            state_manager: Train state manager instance
            prediction_interval: How often to run the prediction loop (seconds)
            max_concurrent_predictions: Maximum concurrent prediction tasks
        """
        self.delay_model = delay_model
        self.conflict_model = conflict_model
        self.state_manager = state_manager
        self.prediction_interval = prediction_interval
        self.max_concurrent_predictions = max_concurrent_predictions
        
        # Train tracking
        self.trains: Dict[str, TrainData] = {}
        self.prediction_history: Dict[str, List[PredictionResult]] = {}
        
        # Engine state
        self.is_running = False
        self.engine_thread = None
        self.prediction_callbacks: List[Callable] = []
        
        # Statistics
        self.total_predictions = 0
        self.start_time = None
        
        logger.info(f"ContinuousPredictionEngine initialized with {prediction_interval}s interval")
    
    def add_train(self, train_data: TrainData):
        """Add a new train to the monitoring system."""
        self.trains[train_data.train_id] = train_data
        self.prediction_history[train_data.train_id] = []
        logger.info(f"Added train {train_data.train_id} to continuous monitoring")
    
    def update_train(self, train_id: str, **kwargs):
        """Update train data."""
        if train_id in self.trains:
            for key, value in kwargs.items():
                if hasattr(self.trains[train_id], key):
                    setattr(self.trains[train_id], key, value)
            
            self.trains[train_id].last_updated = time.time()
            
            # Force recheck if significant changes
            significant_changes = ['actual_delay', 'tsr_active', 'priority_level']
            if any(key in significant_changes for key in kwargs.keys()):
                self.state_manager.force_recheck(train_id, "significant data update")
            
            logger.debug(f"Updated train {train_id}: {kwargs}")
    
    def remove_train(self, train_id: str):
        """Remove a train from monitoring."""
        if train_id in self.trains:
            del self.trains[train_id]
            # Keep prediction history for analysis
            logger.info(f"Removed train {train_id} from monitoring")
    
    def predict_for_train(self, train_data: TrainData) -> PredictionResult:
        """
        Run ML predictions for a single train.
        
        Args:
            train_data: Train operational data
            
        Returns:
            Prediction results including conflict probability and delay prediction
        """
        try:
            # Prepare features
            features_dict = train_data.to_features_dict()
            features_df = pd.DataFrame([features_dict])
            
            # Delay prediction
            if self.delay_model is not None:
                delay_pred = self.delay_model.predict(features_df)[0]
                delay_pred = max(0, float(delay_pred))
            else:
                # Fallback heuristic
                delay_pred = self._fallback_delay_prediction(train_data)
            
            # Conflict prediction
            if self.conflict_model is not None:
                if hasattr(self.conflict_model, 'predict_proba'):
                    conflict_prob = self.conflict_model.predict_proba(features_df)[0][1]
                else:
                    conflict_prob = self.conflict_model.predict(features_df)[0]
                    conflict_prob = min(1.0, max(0.0, float(conflict_prob)))
            else:
                # Fallback heuristic
                conflict_prob = self._fallback_conflict_prediction(train_data)
            
            # Generate recommendations based on predictions
            recommendations = self._generate_recommendations(
                train_data, conflict_prob, delay_pred
            )
            
            # Calculate confidence (simplified)
            model_confidence = 0.85 if (self.delay_model and self.conflict_model) else 0.6
            
            result = PredictionResult(
                train_id=train_data.train_id,
                conflict_probability=float(conflict_prob),
                delay_prediction=delay_pred,
                timestamp=time.time(),
                model_confidence=model_confidence,
                recommendations=recommendations
            )
            
            self.total_predictions += 1
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for train {train_data.train_id}: {e}")
            # Return safe fallback
            return PredictionResult(
                train_id=train_data.train_id,
                conflict_probability=0.5,
                delay_prediction=5.0,
                timestamp=time.time(),
                model_confidence=0.1,
                recommendations=["Monitor closely due to prediction error"]
            )
    
    def _fallback_delay_prediction(self, train_data: TrainData) -> float:
        """Fallback delay prediction when model is not available."""
        base_delay = train_data.actual_delay
        
        # Add factors
        if train_data.tsr_active.upper() == 'Y':
            base_delay += 8
        if train_data.headway_seconds < 300:
            base_delay += 3
        if train_data.priority_level > 2:
            base_delay += 2
        
        return max(0, base_delay + 1)  # Add 1 minute base propagation
    
    def _fallback_conflict_prediction(self, train_data: TrainData) -> float:
        """Fallback conflict prediction when model is not available."""
        base_prob = 0.2  # Base probability
        
        # Increase based on factors
        if train_data.actual_delay > 10:
            base_prob += 0.3
        if train_data.headway_seconds < 240:
            base_prob += 0.2
        if train_data.priority_level == 1:  # High priority trains have more conflicts
            base_prob += 0.1
        
        return min(1.0, base_prob)
    
    def _generate_recommendations(self, train_data: TrainData, 
                                conflict_prob: float, delay_pred: float) -> List[str]:
        """Generate operational recommendations based on predictions."""
        recommendations = []
        
        # Delay-based recommendations
        if delay_pred > 15:
            recommendations.append(f"HIGH DELAY RISK: Consider rerouting {train_data.train_id}")
        elif delay_pred > 10:
            recommendations.append(f"MODERATE DELAY: Monitor {train_data.train_id} closely")
        elif delay_pred > 5:
            recommendations.append(f"MINOR DELAY: Inform passengers about {train_data.train_id}")
        
        # Conflict-based recommendations
        if conflict_prob > 0.8:
            recommendations.append(f"CRITICAL CONFLICT RISK: Immediate action required for {train_data.train_id}")
        elif conflict_prob > 0.6:
            recommendations.append(f"HIGH CONFLICT RISK: Increase headway for {train_data.train_id}")
        elif conflict_prob > 0.4:
            recommendations.append(f"MODERATE CONFLICT RISK: Monitor section around {train_data.train_id}")
        
        # Combined recommendations
        combined_risk = (delay_pred / 60.0) * 0.6 + conflict_prob * 0.4
        if combined_risk > 0.7:
            recommendations.append(f"OVERALL HIGH RISK: Priority handling needed for {train_data.train_id}")
        
        return recommendations if recommendations else ["Normal operations"]
    
    async def prediction_loop(self):
        """Main continuous prediction loop."""
        logger.info("Starting continuous prediction loop")
        
        while self.is_running:
            try:
                current_time = time.time()
                trains_to_check = []
                
                # Determine which trains need checking
                for train_id, train_data in self.trains.items():
                    if self.state_manager.should_check(train_id, current_time):
                        trains_to_check.append(train_data)
                
                logger.debug(f"Checking {len(trains_to_check)} trains in this cycle")
                
                # Process trains (limit concurrent predictions)
                semaphore = asyncio.Semaphore(self.max_concurrent_predictions)
                tasks = []
                
                for train_data in trains_to_check:
                    task = self._process_train_async(train_data, semaphore)
                    tasks.append(task)
                
                # Wait for all predictions to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Cleanup old trains
                self.state_manager.cleanup_old_trains()
                
                # Wait for next cycle
                await asyncio.sleep(self.prediction_interval)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(self.prediction_interval)
    
    async def _process_train_async(self, train_data: TrainData, semaphore):
        """Process a single train asynchronously."""
        async with semaphore:
            try:
                # Run prediction
                result = self.predict_for_train(train_data)
                
                # Update state manager
                new_state = self.state_manager.update_train(
                    train_data.train_id,
                    result.conflict_probability,
                    result.delay_prediction,
                    result.timestamp
                )
                
                # Store prediction history
                self.prediction_history[train_data.train_id].append(result)
                
                # Keep only last 100 predictions per train
                if len(self.prediction_history[train_data.train_id]) > 100:
                    self.prediction_history[train_data.train_id] = \
                        self.prediction_history[train_data.train_id][-100:]
                
                # Notify callbacks
                for callback in self.prediction_callbacks:
                    try:
                        await callback(train_data, result, new_state)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                logger.debug(f"Processed train {train_data.train_id}: "
                           f"conflict={result.conflict_probability:.3f}, "
                           f"delay={result.delay_prediction:.1f}, state={new_state.value}")
                
            except Exception as e:
                logger.error(f"Error processing train {train_data.train_id}: {e}")
    
    def start(self):
        """Start the continuous prediction engine."""
        if self.is_running:
            logger.warning("Prediction engine is already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start the async loop in a separate thread
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.prediction_loop())
            finally:
                loop.close()
        
        self.engine_thread = threading.Thread(target=run_loop, daemon=True)
        self.engine_thread.start()
        
        logger.info("Continuous prediction engine started")
    
    def stop(self):
        """Stop the continuous prediction engine."""
        self.is_running = False
        if self.engine_thread:
            self.engine_thread.join(timeout=5)
        logger.info("Continuous prediction engine stopped")
    
    def add_prediction_callback(self, callback: Callable):
        """Add a callback function to be called when predictions are made."""
        self.prediction_callbacks.append(callback)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the prediction engine."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            "engine_status": "running" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "total_trains": len(self.trains),
            "total_predictions": self.total_predictions,
            "predictions_per_minute": (self.total_predictions / (uptime / 60)) if uptime > 0 else 0,
            "state_manager_stats": self.state_manager.get_system_stats()
        }
    
    def get_train_predictions(self, train_id: str, limit: int = 10) -> List[PredictionResult]:
        """Get recent predictions for a specific train."""
        if train_id in self.prediction_history:
            return self.prediction_history[train_id][-limit:]
        return []
    
    def get_critical_situations(self) -> Dict[str, Any]:
        """Get current critical situations requiring attention."""
        critical_trains = self.state_manager.get_critical_trains()
        attention_needed = self.state_manager.get_trains_needing_attention()
        
        situations = []
        for train_id in critical_trains:
            if train_id in self.trains:
                train_data = self.trains[train_id]
                train_status = self.state_manager.get_train_status(train_id)
                
                situations.append({
                    "train_id": train_id,
                    "train_type": train_data.train_type,
                    "current_section": train_data.current_section,
                    "conflict_probability": train_status.last_conflict_prob,
                    "delay_prediction": train_status.last_delay_pred,
                    "consecutive_critical": train_status.consecutive_critical_count,
                    "needs_immediate_attention": train_id in attention_needed
                })
        
        return {
            "critical_situations": situations,
            "total_critical": len(critical_trains),
            "immediate_attention_needed": len(attention_needed)
        }