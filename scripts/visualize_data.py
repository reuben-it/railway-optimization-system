"""
Data visualization for the Railway Optimization System synthetic data.
Provides insights and visualizations to understand the generated data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np

# Define paths
SYNTHETIC_DATA = '../data/synthetic/synthetic_data_v1.csv'
OUTPUT_DIR = '../data/visualizations'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load the synthetic data."""
    try:
        df = pd.read_csv(SYNTHETIC_DATA)
        print(f"Loaded {len(df)} records from {SYNTHETIC_DATA}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_delay_distribution(df):
    """Plot distribution of delays by train type."""
    plt.figure(figsize=(12, 6))
    
    # Create a boxplot of delays by train type
    ax = sns.boxplot(x='train_type', y='delay_minutes', data=df)
    ax.set_title('Distribution of Delays by Train Type')
    ax.set_xlabel('Train Type')
    ax.set_ylabel('Delay (minutes)')
    
    plt.savefig(f"{OUTPUT_DIR}/delay_distribution.png")
    print(f"Saved delay distribution plot to {OUTPUT_DIR}/delay_distribution.png")

def plot_delay_histogram(df):
    """Plot histogram of delays."""
    plt.figure(figsize=(12, 6))
    
    # Create a histogram of delays
    ax = sns.histplot(df['delay_minutes'], bins=30, kde=True)
    ax.set_title('Distribution of Train Delays')
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Count')
    
    plt.savefig(f"{OUTPUT_DIR}/delay_histogram.png")
    print(f"Saved delay histogram plot to {OUTPUT_DIR}/delay_histogram.png")

def plot_conflict_likelihood(df):
    """Plot conflict likelihood vs delay."""
    plt.figure(figsize=(10, 6))
    
    # Create a scatterplot of conflict likelihood vs delay
    ax = sns.scatterplot(x='delay_minutes', y='conflict_likelihood', 
                         hue='train_type', alpha=0.6, data=df)
    ax.set_title('Conflict Likelihood vs Delay Minutes')
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Conflict Likelihood')
    
    plt.savefig(f"{OUTPUT_DIR}/conflict_likelihood.png")
    print(f"Saved conflict likelihood plot to {OUTPUT_DIR}/conflict_likelihood.png")

def plot_controller_actions(df):
    """Plot distribution of controller actions."""
    plt.figure(figsize=(10, 6))
    
    # Count controller actions
    action_counts = df['controller_action'].value_counts()
    
    # Create a bar chart
    ax = sns.barplot(x=action_counts.index, y=action_counts.values)
    ax.set_title('Distribution of Controller Actions')
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/controller_actions.png")
    print(f"Saved controller actions plot to {OUTPUT_DIR}/controller_actions.png")

def plot_tsr_impact(df):
    """Plot impact of TSR on delays."""
    plt.figure(figsize=(10, 6))
    
    # Create a boxplot of delays by TSR status
    ax = sns.boxplot(x='tsr_active', y='delay_minutes', data=df)
    ax.set_title('Impact of TSR on Delays')
    ax.set_xlabel('TSR Active')
    ax.set_ylabel('Delay (minutes)')
    
    plt.savefig(f"{OUTPUT_DIR}/tsr_impact.png")
    print(f"Saved TSR impact plot to {OUTPUT_DIR}/tsr_impact.png")

def generate_summary_statistics(df):
    """Generate and save summary statistics."""
    # Convert NumPy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    # Overall summary
    summary = {
        'total_records': int(len(df)),
        'train_types': {k: int(v) for k, v in df['train_type'].value_counts().to_dict().items()},
        'avg_delay': float(df['delay_minutes'].mean()),
        'max_delay': int(df['delay_minutes'].max()),
        'on_time_percentage': float((df['delay_minutes'] == 0).mean() * 100),
        'tsr_percentage': float((df['tsr_active'] == 'Y').mean() * 100),
        'controller_actions': {k: int(v) for k, v in df['controller_action'].value_counts().to_dict().items()}
    }
    
    # Delay by train type
    delay_by_type = df.groupby('train_type')['delay_minutes'].agg(['mean', 'median', 'max']).to_dict()
    # Convert the nested dictionary with numpy values to Python native types
    delay_by_type_serializable = {}
    for stat, values in delay_by_type.items():
        delay_by_type_serializable[stat] = {k: float(v) for k, v in values.items()}
    
    summary['delay_by_train_type'] = delay_by_type_serializable
    
    # Save summary to json
    with open(f"{OUTPUT_DIR}/summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Saved summary statistics to {OUTPUT_DIR}/summary_statistics.json")
    
    # Also print summary to console
    print("\nSummary Statistics:")
    print(f"Total Records: {summary['total_records']}")
    print(f"Average Delay: {summary['avg_delay']:.2f} minutes")
    print(f"Maximum Delay: {summary['max_delay']} minutes")
    print(f"On-time Percentage: {summary['on_time_percentage']:.2f}%")
    print(f"TSR Affected Percentage: {summary['tsr_percentage']:.2f}%")
    
    print("\nTrain Type Distribution:")
    for train_type, count in summary['train_types'].items():
        print(f"  {train_type}: {count} records ({count/summary['total_records']*100:.1f}%)")
    
    print("\nController Action Distribution:")
    for action, count in summary['controller_actions'].items():
        print(f"  {action}: {count} records ({count/summary['total_records']*100:.1f}%)")

def main():
    """Main function to generate all visualizations."""
    print("Loading synthetic data...")
    df = load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Generating visualizations...")
    plot_delay_distribution(df)
    plot_delay_histogram(df)
    plot_conflict_likelihood(df)
    plot_controller_actions(df)
    plot_tsr_impact(df)
    
    print("Generating summary statistics...")
    generate_summary_statistics(df)
    
    print("All visualizations and statistics completed!")

if __name__ == "__main__":
    print("Starting data visualization process...")
    main()
    print("Process completed!")