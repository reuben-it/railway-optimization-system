"""
Create the base seed dataset for railway optimization system.
This script generates a template CSV file with the required structure.
"""

import csv
import os
import random
from datetime import datetime, timedelta

# Ensure data directories exist
os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../data/synthetic', exist_ok=True)

# Define the CSV structure
csv_columns = [
    'train_id', 'train_type', 'date', 'section_id', 'from_station', 'to_station',
    'scheduled_departure', 'scheduled_arrival', 'actual_departure', 'actual_arrival',
    'delay_minutes', 'block_length_km', 'track_type', 'speed_limit_kmph',
    'rake_length_m', 'priority_level', 'headway_seconds', 'tsr_active', 'tsr_speed_kmph',
    'platform_assigned', 'controller_action', 'conflict_likelihood', 'propagated_delay_minutes'
]

# Sample station codes for Indian Railways
station_codes = [
    'NDLS', 'BCT', 'MAS', 'HWH', 'SBC', 'JP', 'ADI', 'NGP', 'CNB', 'LKO',
    'PNBE', 'BPL', 'ALD', 'GHY', 'VSKP', 'CSTM', 'KYN', 'ASR', 'TPTY', 'SC'
]

# Sample sections (from station to station)
sections = []
for i in range(0, len(station_codes)-1):
    sections.append({
        'id': f'SEC{i+1:03d}',
        'from': station_codes[i],
        'to': station_codes[i+1],
        'length': random.randint(30, 150)  # km
    })

# Train types and their properties
train_types = {
    'express': {'priority': 1, 'speed': (90, 130), 'rake_length': (500, 750)},
    'passenger': {'priority': 2, 'speed': (70, 100), 'rake_length': (400, 600)},
    'freight': {'priority': 3, 'speed': (50, 70), 'rake_length': (800, 1500)},
    'suburban': {'priority': 2, 'speed': (60, 80), 'rake_length': (300, 500)}
}

# Track types
track_types = ['single', 'double', 'electrified']

# Controller actions
controller_actions = ['hold', 'divert', 'reschedule', 'none']

def create_template_csv():
    """Create an empty CSV file with the column headers."""
    filepath = '../data/raw/seed_template.csv'
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
    print(f"Created template CSV at {filepath}")

def generate_random_sample():
    """Generate a single random sample row for demonstration."""
    train_type = random.choice(list(train_types.keys()))
    properties = train_types[train_type]
    section = random.choice(sections)
    
    # Base time values
    base_date = datetime.now().strftime('%Y-%m-%d')
    base_hour = random.randint(0, 23)
    base_minute = random.choice([0, 15, 30, 45])
    
    # Calculate scheduled times
    speed = random.randint(*properties['speed'])
    duration_hours = section['length'] / speed
    duration_minutes = duration_hours * 60
    
    scheduled_departure = f"{base_hour:02d}:{base_minute:02d}:00"
    
    # Calculate arrival time
    dep_hour, dep_minute = base_hour, base_minute
    arr_minute = dep_minute + int(duration_minutes % 60)
    arr_hour = dep_hour + int(duration_minutes // 60)
    
    if arr_minute >= 60:
        arr_hour += 1
        arr_minute -= 60
    if arr_hour >= 24:
        arr_hour -= 24
    
    scheduled_arrival = f"{arr_hour:02d}:{arr_minute:02d}:00"
    
    # Random delay (70% chance of being on time)
    delay = 0
    if random.random() > 0.7:
        delay = random.randint(1, 30)
    
    # Calculate actual times based on delay
    actual_departure = scheduled_departure
    if delay > 0:
        dep_hour, dep_minute = int(scheduled_departure.split(':')[0]), int(scheduled_departure.split(':')[1])
        dep_minute += delay
        if dep_minute >= 60:
            dep_hour += 1
            dep_minute -= 60
        if dep_hour >= 24:
            dep_hour -= 24
        actual_departure = f"{dep_hour:02d}:{dep_minute:02d}:00"
    
    # Assume same delay for arrival
    actual_arrival = scheduled_arrival
    if delay > 0:
        arr_hour, arr_minute = int(scheduled_arrival.split(':')[0]), int(scheduled_arrival.split(':')[1])
        arr_minute += delay
        if arr_minute >= 60:
            arr_hour += 1
            arr_minute -= 60
        if arr_hour >= 24:
            arr_hour -= 24
        actual_arrival = f"{arr_hour:02d}:{arr_minute:02d}:00"
    
    # TSR data
    tsr_active = 'N'
    tsr_speed = 0
    if random.random() < 0.15:  # 15% chance of TSR
        tsr_active = 'Y'
        tsr_speed = random.choice([30, 45, 60])
    
    # Platform assignment
    platform = f"P{random.randint(1,8)}"
    
    # Propagated delay (usually less than or equal to original delay)
    propagated_delay = 0
    if delay > 0:
        propagated_delay = int(delay * random.random())
    
    return {
        'train_id': f"T{random.randint(10000, 99999)}",
        'train_type': train_type,
        'date': base_date,
        'section_id': section['id'],
        'from_station': section['from'],
        'to_station': section['to'],
        'scheduled_departure': scheduled_departure,
        'scheduled_arrival': scheduled_arrival,
        'actual_departure': actual_departure,
        'actual_arrival': actual_arrival,
        'delay_minutes': delay,
        'block_length_km': section['length'],
        'track_type': random.choice(track_types),
        'speed_limit_kmph': random.randint(*properties['speed']),
        'rake_length_m': random.randint(*properties['rake_length']),
        'priority_level': properties['priority'],
        'headway_seconds': random.randint(180, 600),
        'tsr_active': tsr_active,
        'tsr_speed_kmph': tsr_speed,
        'platform_assigned': platform,
        'controller_action': random.choice(controller_actions),
        'conflict_likelihood': round(random.random(), 2),
        'propagated_delay_minutes': propagated_delay
    }

def generate_sample_seed_data(num_samples=10):
    """Generate a few sample rows to demonstrate the data structure."""
    filepath = '../data/raw/seed_sample.csv'
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
        for _ in range(num_samples):
            writer.writerow(generate_random_sample())
            
    print(f"Created sample seed data with {num_samples} rows at {filepath}")

if __name__ == "__main__":
    create_template_csv()
    generate_sample_seed_data(10)  # Generate 10 sample rows
    print("\nScript completed successfully!")
    print("Next steps:")
    print("1. Review the sample data in ../data/raw/seed_sample.csv")
    print("2. Create your 150 manual entries across different scenarios")
    print("3. Run the synthetic data generator script")