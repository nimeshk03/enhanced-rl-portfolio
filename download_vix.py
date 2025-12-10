"""download_vix.py

Download VIX (Volatility Index) data separately to avoid column mismatch
issues during preprocessing. This creates a VIX CSV that can be merged
with the main data.

Usage:
    python download_vix.py

Output:
    ./data/vix_data.csv - VIX closing prices by date
"""

import os
import pandas as pd
import yfinance as yf

# Ensure data directory exists
if not os.path.exists("./data"):
    os.makedirs("./data")

# Match the date range from your main data (extended for enhanced training)
START_DATE = '2015-01-01'
END_DATE = '2025-11-30'

print("=" * 60)
print("DOWNLOADING VIX DATA")
print("=" * 60)

try:
    print(f"\nDownloading VIX from {START_DATE} to {END_DATE}...")
    
    # Download VIX data
    vix_data = yf.download(
        "^VIX",
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True
    )
    
    if vix_data.empty:
        print("[ERROR] No VIX data returned")
        exit(1)
    
    # Reset index to make Date a column
    vix_data = vix_data.reset_index()
    
    # Handle MultiIndex columns if present
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = [col[0] if isinstance(col, tuple) else col 
                           for col in vix_data.columns]
    
    # Standardize column names
    vix_data.columns = [str(col).lower() for col in vix_data.columns]
    
    # Rename date column if needed
    if 'date' not in vix_data.columns:
        date_col = [col for col in vix_data.columns if 'date' in col.lower() or col == 'index']
        if date_col:
            vix_data.rename(columns={date_col[0]: 'date'}, inplace=True)
    
    # Convert date to string format YYYY-MM-DD
    vix_data['date'] = pd.to_datetime(vix_data['date']).dt.strftime('%Y-%m-%d')
    
    # We only need date and close for VIX
    if 'close' in vix_data.columns:
        vix_clean = vix_data[['date', 'close']].copy()
        vix_clean.rename(columns={'close': 'vix'}, inplace=True)
    else:
        print("[ERROR] No 'close' column found in VIX data")
        print(f"Available columns: {vix_data.columns.tolist()}")
        exit(1)
    
    # Save to CSV
    output_path = "./data/vix_data.csv"
    vix_clean.to_csv(output_path, index=False)
    
    print(f"[OK] VIX data downloaded: {len(vix_clean)} rows")
    print(f"[OK] Saved to {output_path}")
    print(f"  Date range: {vix_clean['date'].min()} to {vix_clean['date'].max()}")
    
    print("\nSample VIX data (last 5 days):")
    print(vix_clean.tail())
    
    print("\n[OK] VIX download complete!")
    print("=" * 60)
    
except Exception as e:
    print(f"[ERROR] Error downloading VIX: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)