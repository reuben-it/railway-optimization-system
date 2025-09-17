"""
Train State Manager for Railway DSS Continuous Prediction System.
Manages train states (SAFE, WATCHLIST, CRITICAL) and cooldown periods.
"""

import time
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class TrainState(Enum):
    SAFE = "SAFE"
    WATCHLIST = "WATCHLIST"
    CRITICAL = "CRITICAL"

@dataclass
class TrainStatus:
    """Represents the current status of a train in the system."""
    train_id: str
    state: TrainState
    next_check_time: float
    last_conflict_prob: float
    last_delay_pred: float
    last_updated: float
    handled_action: Optional[str] = None
    handled_by: Optional[str] = None
    consecutive_critical_count: int = 0

class TrainStateManager:
    """
    Manages the state of trains in the continuous prediction system.
    
    Key Features:
    - Dynamic cooldown periods based on conflict probability
    - State transitions (SAFE -> WATCHLIST -> CRITICAL)
    - Override mechanisms for safety-critical situations
    - Operator action tracking
    """
    
    def __init__(self, 
                 base_cooldown: float = 600,  # 10 minutes in seconds
                 conflict_threshold: float = 0.7,
                 watchlist_threshold: float = 0.4,
                 critical_escalation_count: int = 3):
        """
        Initialize the Train State Manager.
        
        Args:
            base_cooldown: Base cooldown period in seconds (default: 10 minutes)
            conflict_threshold: Conflict probability above which train becomes CRITICAL
            watchlist_threshold: Conflict probability above which train becomes WATCHLIST
            critical_escalation_count: Number of consecutive CRITICAL predictions before forcing action
        """
        self.states: Dict[str, TrainStatus] = {}
        self.base_cooldown = base_cooldown
        self.conflict_threshold = conflict_threshold
        self.watchlist_threshold = watchlist_threshold
        self.critical_escalation_count = critical_escalation_count
        
        logger.info(f"TrainStateManager initialized: base_cooldown={base_cooldown}s, "
                   f"conflict_threshold={conflict_threshold}, watchlist_threshold={watchlist_threshold}")
    
    def update_train(self, train_id: str, conflict_prob: float, delay_pred: float, 
                    current_time: Optional[float] = None) -> TrainState:
        """
        Update a train's state based on current predictions.
        
        Args:
            train_id: Unique identifier for the train
            conflict_prob: Current conflict probability (0-1)
            delay_pred: Current delay prediction in minutes
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            The new state of the train
        """
        if current_time is None:
            current_time = time.time()
        
        # Get previous state if exists
        prev_status = self.states.get(train_id)
        prev_state = prev_status.state if prev_status else None
        
        # Determine new state based on conflict probability
        if conflict_prob >= self.conflict_threshold:
            new_state = TrainState.CRITICAL
            # Critical trains get immediate recheck
            next_check = current_time
            
            # Track consecutive critical predictions
            consecutive_count = (prev_status.consecutive_critical_count + 1 
                               if prev_status and prev_status.state == TrainState.CRITICAL 
                               else 1)
        
        elif conflict_prob >= self.watchlist_threshold:
            new_state = TrainState.WATCHLIST
            # Watchlist trains get shorter cooldown
            next_check = current_time + self.base_cooldown / 2
            consecutive_count = 0
        
        else:
            new_state = TrainState.SAFE
            # Safe trains get longer cooldown, inversely proportional to conflict prob
            cooldown_multiplier = 1 + (1 - conflict_prob) * 0.5  # 1.0 to 1.5x multiplier
            next_check = current_time + self.base_cooldown * cooldown_multiplier
            consecutive_count = 0
        
        # Create or update train status
        self.states[train_id] = TrainStatus(
            train_id=train_id,
            state=new_state,
            next_check_time=next_check,
            last_conflict_prob=conflict_prob,
            last_delay_pred=delay_pred,
            last_updated=current_time,
            handled_action=prev_status.handled_action if prev_status else None,
            handled_by=prev_status.handled_by if prev_status else None,
            consecutive_critical_count=consecutive_count
        )
        
        # Log state changes
        if prev_state != new_state:
            logger.info(f"Train {train_id} state changed: {prev_state} -> {new_state} "
                       f"(conflict: {conflict_prob:.3f}, delay: {delay_pred:.1f})")
        
        return new_state
    
    def should_check(self, train_id: str, current_time: Optional[float] = None) -> bool:
        """
        Determine if a train should be re-evaluated for predictions.
        
        Args:
            train_id: Unique identifier for the train
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            True if train should be checked, False otherwise
        """
        if current_time is None:
            current_time = time.time()
        
        # Always check trains not in the system yet
        if train_id not in self.states:
            return True
        
        train_status = self.states[train_id]
        
        # Always check CRITICAL trains
        if train_status.state == TrainState.CRITICAL:
            return True
        
        # Check if cooldown period has elapsed
        return current_time >= train_status.next_check_time
    
    def mark_train_handled(self, train_id: str, action: str, handled_by: str, 
                          current_time: Optional[float] = None):
        """
        Mark a train as handled by an operator action.
        
        Args:
            train_id: Unique identifier for the train
            action: Action taken (e.g., "hold", "pass", "reroute")
            handled_by: Identifier of who handled it (operator ID)
            current_time: Current timestamp (defaults to time.time())
        """
        if current_time is None:
            current_time = time.time()
        
        if train_id in self.states:
            self.states[train_id].handled_action = action
            self.states[train_id].handled_by = handled_by
            
            # If not critical, set to SAFE with cooldown
            if self.states[train_id].state != TrainState.CRITICAL:
                self.states[train_id].state = TrainState.SAFE
                self.states[train_id].next_check_time = current_time + self.base_cooldown
                self.states[train_id].consecutive_critical_count = 0
            
            logger.info(f"Train {train_id} marked as handled: {action} by {handled_by}")
    
    def force_recheck(self, train_id: str, reason: str = "manual override"):
        """
        Force immediate recheck of a train, bypassing cooldown.
        
        Args:
            train_id: Unique identifier for the train
            reason: Reason for forcing recheck
        """
        if train_id in self.states:
            self.states[train_id].next_check_time = time.time()
            logger.info(f"Forced recheck for train {train_id}: {reason}")
    
    def get_train_status(self, train_id: str) -> Optional[TrainStatus]:
        """Get the current status of a train."""
        return self.states.get(train_id)
    
    def get_trains_by_state(self, state: TrainState) -> list:
        """Get all trains in a specific state."""
        return [train_id for train_id, status in self.states.items() 
                if status.state == state]
    
    def get_critical_trains(self) -> list:
        """Get all trains in CRITICAL state."""
        return self.get_trains_by_state(TrainState.CRITICAL)
    
    def get_trains_needing_attention(self) -> list:
        """Get trains that have been CRITICAL for too long and need intervention."""
        critical_trains = []
        for train_id, status in self.states.items():
            if (status.state == TrainState.CRITICAL and 
                status.consecutive_critical_count >= self.critical_escalation_count):
                critical_trains.append(train_id)
        return critical_trains
    
    def cleanup_old_trains(self, max_age_hours: float = 24):
        """
        Remove trains that haven't been updated for a long time.
        
        Args:
            max_age_hours: Maximum age in hours before removing train from tracking
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_trains = [
            train_id for train_id, status in self.states.items()
            if current_time - status.last_updated > max_age_seconds
        ]
        
        for train_id in old_trains:
            del self.states[train_id]
            logger.info(f"Removed old train {train_id} from tracking")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the current state of the system."""
        total_trains = len(self.states)
        if total_trains == 0:
            return {"total_trains": 0}
        
        state_counts = {state.value: 0 for state in TrainState}
        conflict_probs = []
        delay_preds = []
        
        for status in self.states.values():
            state_counts[status.state.value] += 1
            conflict_probs.append(status.last_conflict_prob)
            delay_preds.append(status.last_delay_pred)
        
        return {
            "total_trains": total_trains,
            "state_distribution": state_counts,
            "avg_conflict_prob": sum(conflict_probs) / len(conflict_probs),
            "avg_delay_pred": sum(delay_preds) / len(delay_preds),
            "trains_needing_attention": len(self.get_trains_needing_attention()),
            "critical_trains": len(self.get_critical_trains())
        }