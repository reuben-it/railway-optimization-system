"""
Synthetic Data Generator for Railway Optimization System.

This script takes the seed dataset and generates variations to create
a larger synthetic dataset for training ML models.
"""

import csv
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Define constants
SEED_FILE = '../data/raw/seed_sample.csv'  # Change to actual seed file when ready
OUTPUT_DIR = '../data/synthetic'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define realistic distributions based on IR patterns
delay_distributions = {
    'express': {'mean': 8, 'std': 5, 'max': 60},
    'passenger': {'mean': 12, 'std': 8, 'max': 90},
    'freight': {'mean': 20, 'std': 15, 'max': 180},
    'suburban': {'mean': 5, 'std': 3, 'max': 30}
}

# Define parameters
tsr_probability = 0.15  # 15% sections have TSR
platform_conflict_rate = 0.20  # 20% at major stations

def load_seed_data(file_path=SEED_FILE):
    """Load the seed dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} seed records from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading seed data: {e}")
        return None

def calculate_propagation(delay_minutes, headway_seconds, priority_level):
    """Calculate realistic propagation of delays to downstream trains."""
    # Higher priority trains cause more propagation
    priority_factor = 1 + (1 / priority_level)
    
    # Shorter headway means more propagation
    headway_factor = 600 / max(headway_seconds, 180)  # Normalize against 10-minute headway
    
    # Calculate propagation with some randomness
    base_propagation = delay_minutes * priority_factor * headway_factor * random.uniform(0.3, 0.7)
    
    # Ensure propagation doesn't exceed original delay (unlikely in real scenarios)
    return min(delay_minutes, max(0, round(base_propagation)))

def parse_time(time_str):
    """Parse time string to hours and minutes."""
    hours, minutes, _ = time_str.split(':')
    return int(hours), int(minutes)

def add_time(time_str, minutes_to_add):
    """Add minutes to a time string."""
    hours, minutes = parse_time(time_str)
    
    total_minutes = hours * 60 + minutes + minutes_to_add
    new_hours = (total_minutes // 60) % 24
    new_minutes = total_minutes % 60
    
    return f"{new_hours:02d}:{new_minutes:02d}:00"

def generate_synthetic_scenario(base_row):
    """Generate a synthetic variation of a seed data row."""
    new_row = base_row.copy()
    
    # Vary delays based on train type
    train_type = base_row['train_type']
    if random.random() < 0.3:  # 30% chance of delay
        delay = np.random.normal(
            delay_distributions[train_type]['mean'],
            delay_distributions[train_type]['std']
        )
        new_row['delay_minutes'] = max(0, min(round(delay), delay_distributions[train_type]['max']))
    
    # If delay exists, calculate actual departure and arrival times
    if new_row['delay_minutes'] > 0:
        new_row['actual_departure'] = add_time(new_row['scheduled_departure'], new_row['delay_minutes'])
        new_row['actual_arrival'] = add_time(new_row['scheduled_arrival'], new_row['delay_minutes'])
    
    # Add TSR conditions
    if random.random() < tsr_probability:
        new_row['tsr_active'] = 'Y'
        new_row['tsr_speed_kmph'] = random.choice([30, 45, 60])
    else:
        new_row['tsr_active'] = 'N'
        new_row['tsr_speed_kmph'] = 0
    
    # Generate propagated delays
    if new_row['delay_minutes'] > 0:
        new_row['propagated_delay_minutes'] = calculate_propagation(
            new_row['delay_minutes'],
            new_row['headway_seconds'],
            new_row['priority_level']
        )
    
    # Randomize platform assignment at major stations
    if random.random() < 0.1:  # Platform change in 10% cases
        new_row['platform_assigned'] = f"P{random.randint(1,10)}"
    
    # Controller action based on delay and conflicts
    if new_row['delay_minutes'] > 20:
        new_row['controller_action'] = random.choice(['reschedule', 'divert'])
    elif new_row['delay_minutes'] > 5:
        new_row['controller_action'] = random.choice(['hold', 'none', 'divert'])
    else:
        new_row['controller_action'] = 'none'
    
    # Adjust conflict likelihood
    if new_row['delay_minutes'] > 15:
        new_row['conflict_likelihood'] = min(1.0, new_row['conflict_likelihood'] + random.uniform(0.2, 0.4))
    
    return new_row

def generate_synthetic_scenarios(base_row, num_variations=100):
    """Generate multiple synthetic scenarios from one base row."""
    scenarios = []
    for i in range(num_variations):
        scenarios.append(generate_synthetic_scenario(base_row))
    return scenarios

def create_train_journey(train_id, route_sections, base_delay=0, train_type='express'):
    """Generate complete journey across multiple sections."""
    journey = []
    cumulative_delay = base_delay
    
    # Assign a random date for the journey
    today = datetime.now()
    random_days = random.randint(-10, 10)
    journey_date = (today + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    # Generate base departure time
    base_hour = random.randint(0, 23)
    base_minute = random.choice([0, 10, 15, 20, 30, 40, 45, 50])
    current_time = f"{base_hour:02d}:{base_minute:02d}:00"
    
    for section in route_sections:
        # Delay can accumulate or recover
        delay_change = random.choice([-2, -1, 0, 0, 1, 2, 3])
        cumulative_delay = max(0, cumulative_delay + delay_change)
        
        # Calculate travel time based on section length and train type
        section_length = section['length']
        speed_range = {
            'express': (90, 130),
            'passenger': (70, 100),
            'freight': (50, 70),
            'suburban': (60, 80)
        }
        
        speed = random.randint(*speed_range[train_type])
        travel_time_minutes = (section_length / speed) * 60
        
        # Calculate scheduled arrival
        scheduled_arrival = add_time(current_time, round(travel_time_minutes))
        
        # TSR impact
        tsr_active = 'N'
        tsr_speed = 0
        if random.random() < tsr_probability:
            tsr_active = 'Y'
            tsr_speed = random.choice([30, 45, 60])
            # Recalculate travel time with TSR
            tsr_distance = section_length * 0.3  # Assume TSR applies to 30% of section
            normal_distance = section_length * 0.7
            normal_time = (normal_distance / speed) * 60
            tsr_time = (tsr_distance / tsr_speed) * 60
            travel_time_minutes = normal_time + tsr_time
            scheduled_arrival = add_time(current_time, round(travel_time_minutes))
        
        # Apply delay to actual times
        actual_departure = current_time
        if cumulative_delay > 0:
            actual_departure = add_time(current_time, cumulative_delay)
        
        actual_arrival = scheduled_arrival
        if cumulative_delay > 0:
            actual_arrival = add_time(scheduled_arrival, cumulative_delay)
        
        # Priority and rake length based on train type
        priority_map = {'express': 1, 'passenger': 2, 'freight': 3, 'suburban': 2}
        rake_length_map = {
            'express': (500, 750),
            'passenger': (400, 600),
            'freight': (800, 1500),
            'suburban': (300, 500)
        }
        
        section_data = {
            'train_id': train_id,
            'train_type': train_type,
            'date': journey_date,
            'section_id': section['id'],
            'from_station': section['from'],
            'to_station': section['to'],
            'scheduled_departure': current_time,
            'scheduled_arrival': scheduled_arrival,
            'actual_departure': actual_departure,
            'actual_arrival': actual_arrival,
            'delay_minutes': cumulative_delay,
            'block_length_km': section_length,
            'track_type': random.choice(['single', 'double', 'electrified']),
            'speed_limit_kmph': speed,
            'rake_length_m': random.randint(*rake_length_map[train_type]),
            'priority_level': priority_map[train_type],
            'headway_seconds': random.randint(180, 600),
            'tsr_active': tsr_active,
            'tsr_speed_kmph': tsr_speed,
            'platform_assigned': f"P{random.randint(1,8)}",
            'controller_action': 'none',  # Will be updated when adding conflicts
            'conflict_likelihood': round(random.random() * 0.3, 2),  # Base likelihood
            'propagated_delay_minutes': 0  # Will be updated when adding conflicts
        }
        
        journey.append(section_data)
        
        # Update current time for next section
        current_time = scheduled_arrival
    
    return journey

def find_trains_in_section(dataset, section_id, time_str, window_minutes=10):
    """Find trains in the same section within the time window."""
    trains_in_section = []
    
    for idx, row in dataset.iterrows():
        if row['section_id'] == section_id:
            row_time = datetime.strptime(row['scheduled_departure'], '%H:%M:%S')
            check_time = datetime.strptime(time_str, '%H:%M:%S')
            
            time_diff = abs((row_time - check_time).total_seconds() / 60)
            
            if time_diff <= window_minutes:
                trains_in_section.append(row)
    
    return trains_in_section

def check_headway_violation(trains):
    """Check if trains violate minimum headway."""
    if len(trains) < 2:
        return False
    
    # Sort by scheduled departure
    sorted_trains = sorted(trains, key=lambda x: x['scheduled_departure'])
    
    for i in range(len(sorted_trains) - 1):
        t1 = datetime.strptime(sorted_trains[i]['scheduled_departure'], '%H:%M:%S')
        t2 = datetime.strptime(sorted_trains[i+1]['scheduled_departure'], '%H:%M:%S')
        
        diff_seconds = (t2 - t1).total_seconds()
        min_headway = min(sorted_trains[i]['headway_seconds'], sorted_trains[i+1]['headway_seconds'])
        
        if diff_seconds < min_headway:
            return True
    
    return False

def assign_precedence(trains):
    """Assign precedence based on train priority."""
    if len(trains) < 2:
        return 'none'
    
    # Sort by priority (lower number = higher priority)
    sorted_trains = sorted(trains, key=lambda x: x['priority_level'])
    
    if sorted_trains[0]['priority_level'] < sorted_trains[1]['priority_level']:
        return 'prioritize'
    else:
        return random.choice(['hold', 'divert'])

def inject_conflicts(dataset):
    """Add realistic conflict scenarios to the dataset."""
    df = pd.DataFrame(dataset)
    
    for idx, row in df.iterrows():
        nearby_trains = find_trains_in_section(
            df, 
            row['section_id'], 
            row['scheduled_departure'],
            window_minutes=10
        )
        
        if len(nearby_trains) > 1:
            # Apply headway rules
            if check_headway_violation(nearby_trains):
                df.at[idx, 'conflict_likelihood'] = round(0.7 + random.random() * 0.3, 2)
                df.at[idx, 'controller_action'] = assign_precedence(nearby_trains)
                
                # If this train is delayed, it may cause propagation
                if row['delay_minutes'] > 0:
                    for other_idx in [t for t in range(len(nearby_trains)) if nearby_trains[t]['train_id'] != row['train_id']]:
                        other_train_idx = df[df['train_id'] == nearby_trains[other_idx]['train_id']].index
                        if len(other_train_idx) > 0:
                            prop_delay = calculate_propagation(
                                row['delay_minutes'],
                                row['headway_seconds'],
                                row['priority_level']
                            )
                            df.at[other_train_idx[0], 'propagated_delay_minutes'] = prop_delay
    
    return df.to_dict('records')

def validate_synthetic_data(df):
    """Ensure physical feasibility of the data."""
    valid = True
    issues = []
    
    # Check headway constraints
    if any(df['headway_seconds'] < 180):
        valid = False
        issues.append("Some records have headway less than 3 minutes")
    
    # Check speed limits with TSR
    tsr_rows = df[df['tsr_active'] == 'Y']
    if any(tsr_rows['tsr_speed_kmph'] > tsr_rows['speed_limit_kmph']):
        valid = False
        issues.append("Some TSR speeds exceed section speed limits")
    
    # Check delay propagation logic
    if any(df['propagated_delay_minutes'] > df['delay_minutes'] * 2):
        valid = False
        issues.append("Some propagated delays exceed twice the original delay")
    
    return valid, issues

def add_temporal_patterns(dataset):
    """Add temporal patterns like peak hour congestion."""
    df = pd.DataFrame(dataset)
    
    # Convert departure time to hours for analysis
    df['departure_hour'] = df['scheduled_departure'].apply(lambda x: int(x.split(':')[0]))
    
    # Define peak hours (morning and evening)
    morning_peak = range(7, 11)  # 7 AM to 10 AM
    evening_peak = range(17, 21)  # 5 PM to 8 PM
    
    # Increase delay and conflicts during peak hours
    for idx, row in df.iterrows():
        hour = row['departure_hour']
        
        if hour in morning_peak or hour in evening_peak:
            # Increase delay probability and magnitude during peak hours
            if random.random() < 0.4:  # 40% chance of additional peak delay
                peak_delay = random.randint(5, 15)
                df.at[idx, 'delay_minutes'] += peak_delay
                
                # Update actual times
                df.at[idx, 'actual_departure'] = add_time(row['scheduled_departure'], row['delay_minutes'] + peak_delay)
                df.at[idx, 'actual_arrival'] = add_time(row['scheduled_arrival'], row['delay_minutes'] + peak_delay)
                
                # Increase conflict likelihood
                df.at[idx, 'conflict_likelihood'] = min(1.0, row['conflict_likelihood'] + 0.2)
                
                # More likely to need controller intervention
                if random.random() < 0.7:
                    df.at[idx, 'controller_action'] = random.choice(['hold', 'divert', 'reschedule'])
    
    # Remove the temporary column
    df = df.drop('departure_hour', axis=1)
    
    return df.to_dict('records')

def add_weather_impacts(dataset, monsoon_probability=0.15):
    """Add weather impacts like monsoon delays."""
    df = pd.DataFrame(dataset)
    
    # Randomly select days affected by monsoon
    all_dates = df['date'].unique()
    monsoon_dates = random.sample(
        list(all_dates),
        k=int(len(all_dates) * monsoon_probability)
    )
    
    # Add monsoon impacts
    for idx, row in df.iterrows():
        if row['date'] in monsoon_dates:
            # 60% chance of weather impact on monsoon days
            if random.random() < 0.6:
                # Add significant delays during monsoon
                weather_delay = random.randint(15, 45)
                df.at[idx, 'delay_minutes'] += weather_delay
                
                # Update actual times
                df.at[idx, 'actual_departure'] = add_time(row['scheduled_departure'], row['delay_minutes'])
                df.at[idx, 'actual_arrival'] = add_time(row['scheduled_arrival'], row['delay_minutes'])
                
                # Almost certainly will cause propagation
                df.at[idx, 'propagated_delay_minutes'] = max(
                    row['propagated_delay_minutes'],
                    int(weather_delay * random.uniform(0.6, 0.9))
                )
                
                # Higher conflict likelihood and controller intervention
                df.at[idx, 'conflict_likelihood'] = min(1.0, 0.7 + random.random() * 0.3)
                df.at[idx, 'controller_action'] = 'reschedule'
    
    return df.to_dict('records')

def create_module_specific_datasets(synthetic_df):
    """Create specific datasets for different ML modules."""
    
    # For Prediction Module
    prediction_data = synthetic_df[['delay_minutes', 'train_type', 'tsr_active', 
                                 'headway_seconds', 'conflict_likelihood', 
                                 'section_id', 'block_length_km']]
    
    # Save prediction data
    prediction_data.to_csv(f'{OUTPUT_DIR}/prediction_train.csv', index=False)
    print(f"Created prediction dataset with {len(prediction_data)} records")
    
    # For time-series sequence data
    sequences = synthetic_df.sort_values(by=['train_id', 'scheduled_departure'])
    sequences.to_csv(f'{OUTPUT_DIR}/time_series_data.csv', index=False)
    print(f"Created time-series sequence dataset")
    
    # Create metadata
    metadata = {
        'version': '1.0',
        'total_synthetic': len(synthetic_df),
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'distributions': delay_distributions
    }
    
    with open(f'{OUTPUT_DIR}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Created metadata file")

def generate_synthetic_data():
    """Main function to generate synthetic data."""
    # Load seed data
    seed_data = load_seed_data()
    if seed_data is None:
        print("Failed to load seed data. Exiting.")
        return
    
    # Generate synthetic scenarios from seed data
    print("Generating synthetic variations from seed data...")
    synthetic_data = []
    variations_per_seed = 50  # Adjust based on needs
    
    for _, row in seed_data.iterrows():
        variations = generate_synthetic_scenarios(row.to_dict(), variations_per_seed)
        synthetic_data.extend(variations)
    
    print(f"Generated {len(synthetic_data)} synthetic records from seed data")
    
    # Define sections for creating train journeys
    sections = []
    station_codes = ['NDLS', 'CNB', 'ALD', 'MGS', 'GAYA', 'DHN', 'ASN', 'HWH']
    for i in range(len(station_codes) - 1):
        sections.append({
            'id': f'SEC{i+1:03d}',
            'from': station_codes[i],
            'to': station_codes[i+1],
            'length': random.randint(50, 200)
        })
    
    # Generate complete train journeys
    print("Generating complete train journeys...")
    journey_data = []
    
    for i in range(50):  # Generate 50 train journeys
        train_type = random.choice(['express', 'passenger', 'freight', 'suburban'])
        train_id = f"T{random.randint(10000, 99999)}"
        base_delay = random.choice([0, 0, 0, 5, 10, 15])  # 50% on-time, 50% with initial delay
        
        journey = create_train_journey(train_id, sections, base_delay, train_type)
        journey_data.extend(journey)
    
    print(f"Generated {len(journey_data)} journey records")
    
    # Combine all data
    all_data = synthetic_data + journey_data
    
    # Inject conflicts
    print("Injecting train conflicts and interactions...")
    all_data = inject_conflicts(all_data)
    
    # Add temporal patterns
    print("Adding temporal patterns...")
    all_data = add_temporal_patterns(all_data)
    
    # Add weather impacts
    print("Adding weather impacts...")
    all_data = add_weather_impacts(all_data)
    
    # Convert to DataFrame for validation and export
    synthetic_df = pd.DataFrame(all_data)
    
    # Validate the data
    print("Validating synthetic data...")
    valid, issues = validate_synthetic_data(synthetic_df)
    if not valid:
        print(f"Warning: Data validation issues found: {issues}")
    else:
        print("Data validation successful!")
    
    # Export the full synthetic dataset
    output_file = f'{OUTPUT_DIR}/synthetic_data_v1.csv'
    synthetic_df.to_csv(output_file, index=False)
    print(f"Exported {len(synthetic_df)} synthetic records to {output_file}")
    
    # Create module-specific datasets
    create_module_specific_datasets(synthetic_df)
    
    print("\nSynthetic data generation completed successfully!")
    print(f"Total records generated: {len(synthetic_df)}")

if __name__ == "__main__":
    print("Starting synthetic data generation...")
    generate_synthetic_data()
    print("Process completed!")