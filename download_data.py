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
# Extended for enhanced training (was 2019-01-01)
START_DATE = '2015-01-01'
END_DATE = '2025-11-30'


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


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate downloaded data meets quality requirements.
    
    Returns:
        True if validation passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    issues = []
    
    # Check 1: All tickers present
    missing_tickers = set(STOCK_TICKERS) - set(df['tic'].unique())
    if missing_tickers:
        issues.append(f"Missing tickers: {missing_tickers}")
    print(f"[{'OK' if not missing_tickers else 'FAIL'}] Tickers: {len(df['tic'].unique())}/{len(STOCK_TICKERS)}")
    
    # Check 2: Date range coverage
    date_min = pd.to_datetime(df['date'].min())
    date_max = pd.to_datetime(df['date'].max())
    expected_start = pd.to_datetime(START_DATE)
    
    # Allow 5 days tolerance for market holidays
    if (date_min - expected_start).days > 5:
        issues.append(f"Data starts too late: {date_min} (expected ~{expected_start})")
    print(f"[{'OK' if (date_min - expected_start).days <= 5 else 'WARN'}] Start date: {date_min.strftime('%Y-%m-%d')}")
    print(f"[OK] End date: {date_max.strftime('%Y-%m-%d')}")
    
    # Check 3: Trading days count
    trading_days = df['date'].nunique()
    years = (date_max - date_min).days / 365
    expected_days = int(years * 252)  # ~252 trading days per year
    
    if trading_days < expected_days * 0.9:  # Allow 10% tolerance
        issues.append(f"Too few trading days: {trading_days} (expected ~{expected_days})")
    print(f"[{'OK' if trading_days >= expected_days * 0.9 else 'WARN'}] Trading days: {trading_days} (expected ~{expected_days})")
    
    # Check 4: No missing values in critical columns
    critical_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} null values")
                print(f"[WARN] Column '{col}': {null_count} null values")
    
    # Check 5: Data per ticker
    print("\nRows per ticker:")
    ticker_counts = df.groupby('tic').size()
    for ticker, count in ticker_counts.items():
        status = 'OK' if count >= expected_days * 0.9 else 'WARN'
        print(f"  [{status}] {ticker}: {count} rows")
    
    # Summary
    print("\n" + "-" * 60)
    if issues:
        print("VALIDATION WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nData downloaded but has warnings. Review above.")
        return False
    else:
        print("VALIDATION PASSED - All checks OK")
        return True


def main():
    """Main entry point: download, save, and verify data."""
    
    # Use direct yfinance download
    df = download_with_yfinance_directly()
    
    # Save raw downloaded data for later reproducibility / processing.
    out_path = "./data/raw_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Data saved to '{out_path}'")

    # --- Quick verification / sanity checks ---
    print("\n--- DATA SUMMARY ---")
    print(f"Shape of dataframe: {df.shape}")
    print(f"Unique tickers found: {sorted(df.tic.unique())}")
    print(f"Date range: {df.date.min()} to {df.date.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    # Run validation
    validation_passed = validate_data(df)
    
    # Return exit code based on validation
    return 0 if validation_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)