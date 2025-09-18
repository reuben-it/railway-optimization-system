"""
Auto-populate default trains on backend startup.
This module provides functionality to automatically register default trains
when the backend starts up, ensuring operators always see some train data.
"""

import logging
import time
from typing import List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default trains configuration
DEFAULT_TRAINS_CONFIG = [
    {
        "train_id": "EXP001",
        "train_type": "express",
        "current_section": "SEC001",
        "from_station": "NDLS",  # New Delhi
        "to_station": "BCT",     # Mumbai Central
        "scheduled_arrival": "14:30:00",
        "actual_delay": 5.2,
        "block_length_km": 150.0,
        "speed_limit_kmph": 130,
        "rake_length_m": 750,
        "priority_level": 1,
        "headway_seconds": 600,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "PASS202",
        "train_type": "passenger",
        "current_section": "SEC015",
        "from_station": "HWH",   # Howrah
        "to_station": "CSMT",    # Chhatrapati Shivaji Terminus
        "scheduled_arrival": "16:45:00",
        "actual_delay": 12.8,
        "block_length_km": 85.0,
        "speed_limit_kmph": 110,
        "rake_length_m": 600,
        "priority_level": 2,
        "headway_seconds": 420,
        "tsr_active": "Y",
        "tsr_speed_kmph": 80
    },
    {
        "train_id": "FRT303",
        "train_type": "freight",
        "current_section": "SEC032",
        "from_station": "KYN",   # Kalyan
        "to_station": "LKO",     # Lucknow
        "scheduled_arrival": "20:15:00",
        "actual_delay": 25.3,
        "block_length_km": 200.0,
        "speed_limit_kmph": 75,
        "rake_length_m": 1200,
        "priority_level": 4,
        "headway_seconds": 900,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "SUB404",
        "train_type": "suburban",
        "current_section": "SEC008",
        "from_station": "CST",   # Chhatrapati Shivaji Terminus
        "to_station": "KYN",     # Kalyan
        "scheduled_arrival": "11:20:00",
        "actual_delay": 2.1,
        "block_length_km": 45.0,
        "speed_limit_kmph": 90,
        "rake_length_m": 300,
        "priority_level": 3,
        "headway_seconds": 180,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "EXP505",
        "train_type": "express",
        "current_section": "SEC021",
        "from_station": "MAO",   # Madgaon
        "to_station": "PUNE",    # Pune
        "scheduled_arrival": "18:30:00",
        "actual_delay": 8.7,
        "block_length_km": 120.0,
        "speed_limit_kmph": 120,
        "rake_length_m": 650,
        "priority_level": 1,
        "headway_seconds": 480,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    },
    {
        "train_id": "SUB606",
        "train_type": "suburban",
        "current_section": "SEC003",
        "from_station": "CCG",   # Church Gate
        "to_station": "VR",      # Virar
        "scheduled_arrival": "10:45:00",
        "actual_delay": 0.8,
        "block_length_km": 25.0,
        "speed_limit_kmph": 80,
        "rake_length_m": 250,
        "priority_level": 3,
        "headway_seconds": 120,
        "tsr_active": "N",
        "tsr_speed_kmph": 0
    }
]

def auto_populate_trains(prediction_engine, max_retries=3):
    """
    Automatically populate the prediction engine with default trains.
    
    Args:
        prediction_engine: The ContinuousPredictionEngine instance
        max_retries: Maximum number of retry attempts per train
    
    Returns:
        int: Number of successfully registered trains
    """
    if not prediction_engine:
        logger.error("Prediction engine not available for auto-population")
        return 0
    
    logger.info(f"Auto-populating {len(DEFAULT_TRAINS_CONFIG)} default trains...")
    
    successful_count = 0
    failed_count = 0
    
    for train_config in DEFAULT_TRAINS_CONFIG:
        train_id = train_config["train_id"]
        
        # Skip if train already exists
        if train_id in prediction_engine.trains:
            logger.info(f"Train {train_id} already exists, skipping...")
            successful_count += 1
            continue
        
        # Try to register the train with retries
        for attempt in range(max_retries):
            try:
                # Import TrainData here to avoid circular imports
                from continuous_prediction_engine import TrainData
                
                train_data = TrainData(
                    train_id=train_config["train_id"],
                    train_type=train_config["train_type"],
                    current_section=train_config["current_section"],
                    from_station=train_config["from_station"],
                    to_station=train_config["to_station"],
                    scheduled_arrival=train_config["scheduled_arrival"],
                    actual_delay=train_config["actual_delay"],
                    block_length_km=train_config["block_length_km"],
                    speed_limit_kmph=train_config["speed_limit_kmph"],
                    rake_length_m=train_config["rake_length_m"],
                    priority_level=train_config["priority_level"],
                    headway_seconds=train_config["headway_seconds"],
                    tsr_active=train_config["tsr_active"],
                    tsr_speed_kmph=train_config["tsr_speed_kmph"],
                    last_updated=time.time()
                )
                
                prediction_engine.add_train(train_data)
                logger.info(f"✅ Successfully registered default train: {train_id} ({train_config['train_type']})")
                successful_count += 1
                break
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for train {train_id}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed to register train {train_id} after {max_retries} attempts")
                    failed_count += 1
                else:
                    time.sleep(1)  # Brief pause before retry
    
    logger.info(f"Auto-population complete: {successful_count} success, {failed_count} failed")
    
    if successful_count > 0:
        logger.info(f"🎉 {successful_count} default trains are now available in the operator dashboard!")
    
    return successful_count

def get_train_summary(prediction_engine):
    """
    Get a summary of currently registered trains.
    
    Args:
        prediction_engine: The ContinuousPredictionEngine instance
    
    Returns:
        dict: Summary information about registered trains
    """
    if not prediction_engine:
        return {"total": 0, "by_type": {}, "trains": []}
    
    trains = []
    type_counts = {}
    
    for train_id, train_data in prediction_engine.trains.items():
        train_info = {
            "train_id": train_id,
            "train_type": train_data.train_type,
            "current_section": train_data.current_section,
            "actual_delay": train_data.actual_delay,
            "from_station": train_data.from_station,
            "to_station": train_data.to_station
        }
        trains.append(train_info)
        
        # Count by type
        train_type = train_data.train_type
        type_counts[train_type] = type_counts.get(train_type, 0) + 1
    
    return {
        "total": len(trains),
        "by_type": type_counts,
        "trains": trains
    }

def ensure_minimum_trains(prediction_engine, min_trains=3):
    """
    Ensure there are at least a minimum number of trains in the system.
    If below the threshold, auto-populate some default trains.
    
    Args:
        prediction_engine: The ContinuousPredictionEngine instance
        min_trains: Minimum number of trains to maintain
    
    Returns:
        bool: True if minimum threshold is met
    """
    if not prediction_engine:
        return False
    
    current_count = len(prediction_engine.trains)
    
    if current_count >= min_trains:
        logger.info(f"Sufficient trains already registered: {current_count}")
        return True
    
    logger.info(f"Only {current_count} trains registered, need minimum {min_trains}")
    logger.info("Auto-populating default trains to meet minimum threshold...")
    
    populated_count = auto_populate_trains(prediction_engine)
    new_total = current_count + populated_count
    
    if new_total >= min_trains:
        logger.info(f"✅ Minimum threshold met: {new_total} trains now registered")
        return True
    else:
        logger.warning(f"⚠️ Still below minimum: {new_total} trains (need {min_trains})")
        return False

# Export the main functions
__all__ = [
    'auto_populate_trains',
    'get_train_summary', 
    'ensure_minimum_trains',
    'DEFAULT_TRAINS_CONFIG'
]