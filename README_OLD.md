# Railway Optimization System

A machine learning system for optimizing railway operations using synthetic data for initial training and validation. This project generates realistic railway operation data that can be used to train and test ML models for predicting delays, optimizing schedules, and resolving conflicts in railway networks.

## Project Structure

- `/data`: Contains both raw seed data and generated synthetic datasets
  - `/data/raw`: Base seed dataset files
  - `/data/synthetic`: Generated synthetic datasets
  - `/data/visualizations`: Visualization outputs and statistics
- `/scripts`: Python scripts for generating and processing data
- `/models`: ML models for railway optimization (to be added)

## Data Generation Process

This project uses a synthetic data generation approach to create realistic railway operation datasets for training machine learning models. The data generation follows these steps:

1. Create a base seed dataset with manual entries across different scenarios
2. Generate synthetic variations using statistical distributions
3. Build time-series sequences for train journeys
4. Add conflict scenarios and interactions between trains
5. Validate data against physical and operational constraints
6. Export datasets for different ML modules

## Synthetic Data Features

The generated dataset contains the following key features:

- **Train Information**: Train ID, type (express, passenger, freight, suburban), and priority level
- **Schedule Information**: Scheduled and actual departure/arrival times across sections
- **Infrastructure Details**: Section lengths, track types, speed limits, platforms
- **Delay Patterns**: Primary delays and propagated delays
- **Temporary Speed Restrictions (TSR)**: Impact of speed restrictions on operations
- **Controller Actions**: Decisions made to resolve conflicts (hold, divert, reschedule)
- **Conflict Likelihood**: Probability of conflicts between trains

## Dataset Statistics (from Generated Sample)

- Total Records: 850
- Average Delay: 7.08 minutes
- Maximum Delay: 54 minutes
- On-time Percentage: 42.47%
- TSR Affected Percentage: 13.88%

### Train Type Distribution
- Passenger: 35.1%
- Suburban: 32.6%
- Freight: 20.8%
- Express: 11.5%

### Controller Action Distribution
- Hold: 27.6%
- Divert: 27.4%
- None: 23.9%
- Prioritize: 16.5%
- Reschedule: 4.6%

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, seaborn

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn
   ```

### Data Generation

1. Generate seed data template:
   ```
   python scripts/create_seed_dataset.py
   ```

2. Generate synthetic data:
   ```
   python scripts/generate_synthetic_data.py
   ```

3. Visualize the data:
   ```
   python scripts/visualize_data.py
   ```

## Future Work

1. Integration with real data from Indian Railways systems
2. Expansion of ML models for:
   - Delay prediction
   - Conflict resolution
   - Schedule optimization
   - Network capacity analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.