"""preprocess_data.py

Add technical indicators and VIX to the raw OHLCV data. This version
handles VIX separately to avoid column mismatch issues.

Usage:
    # First download VIX separately
    python download_vix.py
    
    # Then run preprocessing
    python preprocess_data_with_vix.py

This script:
1. Loads raw_data.csv and vix_data.csv (if available)
2. Adds technical indicators using FinRL
3. Merges VIX data by date
4. Adds turbulence features
5. Saves processed data
"""

import pandas as pd
import numpy as np
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS
import os

print("=" * 70)
print("PREPROCESSING STOCK DATA WITH VIX")
print("=" * 70)

# --- Step 1: Load raw data ---
print("\n[1/5] Loading raw market data...")
try:
    df_raw = pd.read_csv("./data/raw_data.csv")
    print(f"[OK] Loaded {len(df_raw)} rows from ./data/raw_data.csv")
except FileNotFoundError:
    print("[ERROR] raw_data.csv not found.")
    print("Please run download_data.py first.")
    exit(1)

# --- Step 2: Load VIX data (optional) ---
print("\n[2/5] Loading VIX data...")
vix_available = False
try:
    vix_df = pd.read_csv("./data/vix_data.csv")
    print(f"[OK] Loaded VIX data: {len(vix_df)} rows")
    vix_available = True
except FileNotFoundError:
    print("[WARNING] vix_data.csv not found")
    print("  Run 'python download_vix.py' to add VIX data")
    print("  Continuing without VIX...")

# --- Step 3: Validate data ---
print("\n[3/5] Validating data format...")
required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
missing_cols = [col for col in required_columns if col not in df_raw.columns]

if missing_cols:
    print(f"[ERROR] Missing required columns: {missing_cols}")
    exit(1)

print(f"[OK] Data validation passed")
print(f"  - Tickers: {df_raw['tic'].nunique()} ({', '.join(sorted(df_raw['tic'].unique()))})")
print(f"  - Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")

# --- Step 4: Add technical indicators (without VIX) ---
print(f"\n[4/5] Calculating technical indicators...")
print(f"Indicators: {INDICATORS}")

# Use FinRL FeatureEngineer WITHOUT VIX (we'll add it manually)
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=False,  # We'll add VIX manually to avoid column mismatch
    use_turbulence=True,
    user_defined_feature=False,
)

try:
    processed_df = fe.preprocess_data(df_raw)
    print("[OK] Successfully added technical indicators and turbulence")
except Exception as e:
    print(f"[ERROR] Error during preprocessing: {str(e)}")
    print("\nTrying without turbulence...")
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed_df = fe.preprocess_data(df_raw)
    print("[OK] Successfully added technical indicators")

# --- Step 5: Merge VIX data ---
print(f"\n[5/5] Merging VIX data...")
if vix_available:
    try:
        # Ensure date columns are in the same format
        processed_df['date'] = processed_df['date'].astype(str)
        vix_df['date'] = vix_df['date'].astype(str)
        
        # Merge VIX by date (broadcast VIX value to all tickers on same date)
        processed_df = processed_df.merge(vix_df, on='date', how='left')
        
        # Check if merge was successful
        if 'vix' in processed_df.columns:
            # Forward fill missing VIX values (for dates when market is open but VIX isn't)
            processed_df['vix'] = processed_df.groupby('tic')['vix'].fillna(method='ffill')
            
            # Count how many NaN values remain
            vix_nan_count = processed_df['vix'].isna().sum()
            if vix_nan_count > 0:
                print(f"[WARNING] {vix_nan_count} rows with missing VIX (will be filled)")
                # Backward fill any remaining NaNs at the start
                processed_df['vix'] = processed_df.groupby('tic')['vix'].fillna(method='bfill')
            
            print(f"[OK] VIX data merged successfully")
            print(f"  VIX range: {processed_df['vix'].min():.2f} to {processed_df['vix'].max():.2f}")
        else:
            print("[ERROR] VIX merge failed - column not found after merge")
            vix_available = False
            
    except Exception as e:
        print(f"[WARNING] Could not merge VIX: {str(e)}")
        vix_available = False
else:
    print("[SKIP] Skipping VIX merge (data not available)")

# Clean up and sort
processed_df = processed_df.copy()
processed_df = processed_df.sort_values(['date', 'tic']).reset_index(drop=True)

# Remove rows with NaN values (common in first few rows due to indicators)
initial_rows = len(processed_df)
processed_df = processed_df.dropna()
dropped_rows = initial_rows - len(processed_df)
if dropped_rows > 0:
    print(f"\n[INFO] Dropped {dropped_rows} rows with NaN values (normal for technical indicators)")

# --- Save processed data ---
print(f"\nSaving processed data...")
output_path = "./data/processed_data.csv"
processed_df.to_csv(output_path, index=False)
print(f"[OK] Saved to {output_path}")

# --- Verification ---
print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)
print(f"Final dataset: {len(processed_df)} rows × {len(processed_df.columns)} columns")
print(f"\nColumns ({len(processed_df.columns)}):")
print(processed_df.columns.tolist())

# Show sample data
print("\nSample data (AAPL - last 5 days):")
sample_cols = ['date', 'close', 'rsi_30', 'macd']
if 'vix' in processed_df.columns:
    sample_cols.append('vix')
if 'turbulence' in processed_df.columns:
    sample_cols.append('turbulence')

if 'AAPL' in processed_df['tic'].values:
    print(processed_df[processed_df.tic == 'AAPL'][sample_cols].tail())
else:
    print(processed_df[sample_cols].tail())

# Summary statistics
print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)
print(f"[OK] Technical Indicators: {len(INDICATORS)}")
print(f"[OK] VIX: {'Yes' if vix_available and 'vix' in processed_df.columns else 'No'}")
print(f"[OK] Turbulence: {'Yes' if 'turbulence' in processed_df.columns else 'No'}")
print(f"[OK] Total Features: {len(processed_df.columns) - 7}  (+ 7 base columns)")

print("\n[OK] Preprocessing complete! Ready for RL training.")
print("=" * 70)