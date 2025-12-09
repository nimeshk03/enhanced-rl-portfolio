"""download_data.py

Download historical OHLCV (and volume) data for a selected list of
tickers using yfinance and save the consolidated CSV to `./data/raw_data.csv`.

This script is intentionally minimal and structured as a simple
standalone script that can be run with:

    python download_data.py

The script performs the following steps:
1. Ensure the `./data` directory exists.
2. Define the ticker list and date range.
3. Fetch data via yfinance.
4. Sort and save the combined data to CSV.
5. Print a short verification summary to the console.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime


# Create a data directory if it doesn't already exist.
if not os.path.exists("./data"):
    os.makedirs("./data")


# --- Configuration ---
# A small, representative list of equities / ETFs for portfolio experiments.
STOCK_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Large-cap tech
    'JPM', 'BAC',                              # Financials
    'GLD', 'TLT', 'SPY'                       # Commodities / bonds / market ETF
]

# Date range for historical download (YYYY-MM-DD)
START_DATE = '2019-01-01'
END_DATE = '2025-10-30'


def download_with_yfinance_directly():
    """
    Download data directly using yfinance.
    This handles the MultiIndex column issue when downloading single tickers.
    """
    print(f"Downloading data for: {STOCK_TICKERS}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("-" * 60)
    
    all_data = []
    successful_tickers = []
    failed_tickers = []
    
    for ticker in STOCK_TICKERS:
        try:
            print(f"Downloading {ticker}...", end=" ")
            
            # Download data for individual ticker
            ticker_data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            if ticker_data.empty:
                print(f"[ERROR] No data returned")
                failed_tickers.append(ticker)
                continue
            
            # Reset index to make 'Date' a column
            ticker_data = ticker_data.reset_index()
            
            # Handle MultiIndex columns (when downloading single ticker)
            # yfinance returns columns like ('Close', 'AAPL') for single tickers
            if isinstance(ticker_data.columns, pd.MultiIndex):
                # Flatten the MultiIndex - just take the first level
                ticker_data.columns = [col[0] if isinstance(col, tuple) else col 
                                      for col in ticker_data.columns]
            
            # Rename columns to lowercase to match FinRL convention
            ticker_data.columns = [str(col).lower() for col in ticker_data.columns]
            
            # Add ticker column
            ticker_data['tic'] = ticker
            
            # Ensure we have a 'date' column
            if 'date' not in ticker_data.columns:
                # The index might have been reset with a different name
                date_candidates = [col for col in ticker_data.columns 
                                 if 'date' in col.lower() or col == 'index']
                if date_candidates:
                    ticker_data.rename(columns={date_candidates[0]: 'date'}, inplace=True)
            
            # Convert date to string format YYYY-MM-DD
            ticker_data['date'] = pd.to_datetime(ticker_data['date']).dt.strftime('%Y-%m-%d')
            
            # Select and reorder columns to match FinRL format
            # Check which columns we actually have
            available_cols = ticker_data.columns.tolist()
            expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
            
            # Only select columns that exist
            cols_to_keep = [col for col in expected_cols if col in available_cols]
            ticker_data = ticker_data[cols_to_keep]
            
            all_data.append(ticker_data)
            successful_tickers.append(ticker)
            print(f"[OK] ({len(ticker_data)} rows)")
            
        except Exception as e:
            print(f"[ERROR] {str(e)[:100]}")
            failed_tickers.append(ticker)
            import traceback
            traceback.print_exc()  # Print full error for debugging
    
    if not all_data:
        print("\n[ERROR] No data was successfully downloaded!")
        print("This might be a network connectivity issue from within Docker.")
        print("\nTry running this script directly on your host machine:")
        print("  python download_data.py")
        sys.exit(1)
    
    # Combine all dataframes
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date then ticker
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    
    print("\n" + "=" * 60)
    print(f"[OK] Successfully downloaded: {len(successful_tickers)}/{len(STOCK_TICKERS)} tickers")
    if successful_tickers:
        print(f"  Success: {', '.join(successful_tickers)}")
    if failed_tickers:
        print(f"  Failed: {', '.join(failed_tickers)}")
    print("=" * 60)
    
    return df


def main():
    """Main entry point: download, save, and verify data."""
    
    # Use direct yfinance download
    df = download_with_yfinance_directly()
    
    # Save raw downloaded data for later reproducibility / processing.
    out_path = "./data/raw_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Data saved to '{out_path}'")

    # --- Quick verification / sanity checks ---
    print("\n--- DATA VERIFICATION ---")
    print(f"Shape of dataframe: {df.shape}")
    print(f"Unique tickers found: {sorted(df.tic.unique())}")
    print(f"Date range: {df.date.min()} to {df.date.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())


if __name__ == '__main__':
    main()