"""
Test just data collection and preprocessing
"""
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from part1_data_collection import DataCollector
from part2_preprocessing import DataPreprocessor, DataSplitter

print("\n" + "=" * 70)
print("DATA PIPELINE TEST")
print("=" * 70)

start_time = datetime.now()

# Stage 1: Data Collection (short period)
print("\n[1/3] Data Collection (2024-2025)...")
collector = DataCollector(start_date='2024-01-01', end_date='2025-12-31')
raw_data = collector.collect_all_data()
print(f"✓ Raw data: {raw_data.shape}")

# Stage 2: Preprocessing
print("\n[2/3] Preprocessing...")
preprocessor = DataPreprocessor(raw_data)
processed_data, stationarity = preprocessor.preprocess()
print(f"✓ Processed data: {processed_data.shape}")

# Stage 3: Data Splitting
print("\n[3/3] Splitting...")
splitter = DataSplitter()
train, val, test = splitter.split(processed_data)
print(f"✓ Split: Train={train.shape[0]}, Val={val.shape[0]}, Test={test.shape[0]}")

duration = (datetime.now() - start_time).total_seconds()
print(f"\n✓ Completed in {duration:.1f}s")
print("\nColumns available:")
print(f"  {list(processed_data.columns[:10])}")
