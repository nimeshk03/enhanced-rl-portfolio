"""track_performance.py

Track and log daily performance for your 5-day experiment.

Usage:
    # Log today's performance (run at end of each trading day)
    python track_performance.py --log
    
    # View all logged performance
    python track_performance.py --view
    
    # Reset tracking (start fresh)
    python track_performance.py --reset

This creates a CSV file tracking:
- Daily portfolio value
- Daily P&L
- Cumulative return
- Positions held
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from alpaca_trade_api import REST
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

TRACKING_FILE = "./performance_log.csv"
INITIAL_CAPITAL = 100000


def get_account_snapshot(api):
    """Get current account snapshot."""
    account = api.get_account()
    positions = api.list_positions()
    
    position_summary = []
    for pos in positions:
        position_summary.append(f"{pos.symbol}:{pos.qty}")
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'portfolio_value': float(account.portfolio_value),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power),
        'equity': float(account.equity),
        'num_positions': len(positions),
        'positions': ', '.join(position_summary) if position_summary else 'None',
    }


def log_performance():
    """Log current performance to CSV."""
    print("Logging today's performance...")
    
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    snapshot = get_account_snapshot(api)
    
    # Calculate returns
    snapshot['daily_return_pct'] = 0
    snapshot['cumulative_return_pct'] = ((snapshot['portfolio_value'] / INITIAL_CAPITAL) - 1) * 100
    
    # Load existing data or create new
    if os.path.exists(TRACKING_FILE):
        df = pd.read_csv(TRACKING_FILE)
        
        # Check if today already logged
        if snapshot['date'] in df['date'].values:
            print(f"[WARNING] Today ({snapshot['date']}) already logged. Updating...")
            df = df[df['date'] != snapshot['date']]
        
        # Calculate daily return from previous day
        if len(df) > 0:
            prev_value = df.iloc[-1]['portfolio_value']
            snapshot['daily_return_pct'] = ((snapshot['portfolio_value'] / prev_value) - 1) * 100
        
        df = pd.concat([df, pd.DataFrame([snapshot])], ignore_index=True)
    else:
        df = pd.DataFrame([snapshot])
    
    # Save
    df.to_csv(TRACKING_FILE, index=False)
    
    print(f"\n[OK] Logged performance for {snapshot['date']}")
    print(f"  Portfolio Value: ${snapshot['portfolio_value']:,.2f}")
    print(f"  Daily Return: {snapshot['daily_return_pct']:+.2f}%")
    print(f"  Cumulative Return: {snapshot['cumulative_return_pct']:+.2f}%")
    print(f"  Positions: {snapshot['num_positions']}")
    print(f"\n  Data saved to: {TRACKING_FILE}")


def view_performance():
    """View logged performance history."""
    if not os.path.exists(TRACKING_FILE):
        print("[ERROR] No performance data yet. Run with --log first.")
        return
    
    df = pd.read_csv(TRACKING_FILE)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE TRACKING - 5 DAY EXPERIMENT")
    print("=" * 70)
    
    print(f"\n{'Day':<5} {'Date':<12} {'Portfolio':>14} {'Daily':>10} {'Cumulative':>12} {'Positions':>10}")
    print("-" * 70)
    
    for i, row in df.iterrows():
        print(f"{i+1:<5} {row['date']:<12} ${row['portfolio_value']:>12,.2f} "
              f"{row['daily_return_pct']:>+9.2f}% {row['cumulative_return_pct']:>+11.2f}% "
              f"{row['num_positions']:>10}")
    
    print("-" * 70)
    
    # Summary stats
    if len(df) > 0:
        total_return = df.iloc[-1]['cumulative_return_pct']
        avg_daily = df['daily_return_pct'].mean()
        best_day = df['daily_return_pct'].max()
        worst_day = df['daily_return_pct'].min()
        
        print(f"\nSUMMARY ({len(df)} days logged)")
        print(f"   Starting Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"   Current Value:    ${df.iloc[-1]['portfolio_value']:,.2f}")
        print(f"   Total Return:     {total_return:+.2f}%")
        print(f"   Avg Daily Return: {avg_daily:+.2f}%")
        print(f"   Best Day:         {best_day:+.2f}%")
        print(f"   Worst Day:        {worst_day:+.2f}%")
        
        # Compare to SPY (rough estimate)
        days_elapsed = len(df)
        if days_elapsed >= 5:
            print(f"\n5-DAY EXPERIMENT COMPLETE!")
            print(f"   Your Return: {total_return:+.2f}%")
    
    print("=" * 70)


def reset_tracking():
    """Reset tracking data."""
    if os.path.exists(TRACKING_FILE):
        os.remove(TRACKING_FILE)
        print("[OK] Performance tracking reset.")
    else:
        print("No tracking file to reset.")


def main():
    parser = argparse.ArgumentParser(description='Track daily trading performance')
    parser.add_argument('--log', action='store_true', help='Log today\'s performance')
    parser.add_argument('--view', action='store_true', help='View performance history')
    parser.add_argument('--reset', action='store_true', help='Reset tracking data')
    args = parser.parse_args()
    
    if args.log:
        log_performance()
    elif args.view:
        view_performance()
    elif args.reset:
        reset_tracking()
    else:
        # Default: show status and view
        view_performance()


if __name__ == "__main__":
    main()