"""day_trading.py

Trade during market hours and automatically stop when market closes.

Usage:
    python day_trading.py --interval 60

This script:
1. Checks if market is open
2. If open: runs trading cycles at specified interval
3. Automatically stops when market closes
4. Logs all activity with timestamps

Perfect for: "Start in the morning, let it run, auto-stops at market close"
"""

import os
import sys
import argparse
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Alpaca API
from alpaca_trade_api import REST, TimeFrame

# Import from paper_trading
from paper_trading import AlpacaTrader, TradingAgent, run_single_trade, print_account_status

# Import configuration
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    STOCK_TICKERS, MODEL_PATH
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Day Trading - Auto-stop at market close')
    parser.add_argument('--interval', type=int, default=60,
                        help='Minutes between trades (default: 60)')
    return parser.parse_args()


def get_market_status(api):
    """Get detailed market status."""
    clock = api.get_clock()
    return {
        'is_open': clock.is_open,
        'next_open': clock.next_open,
        'next_close': clock.next_close,
        'timestamp': datetime.now()
    }


def wait_for_market_open(trader):
    """Wait until market opens."""
    print("\nWaiting for market to open...")
    
    while True:
        market = get_market_status(trader.api)
        
        if market['is_open']:
            print("Market is now OPEN!")
            return True
        
        # Calculate time until open
        now = datetime.now(market['next_open'].tzinfo)
        time_until_open = (market['next_open'] - now).total_seconds()
        
        if time_until_open > 0:
            hours = int(time_until_open // 3600)
            minutes = int((time_until_open % 3600) // 60)
            print(f"  Market opens in {hours}h {minutes}m (at {market['next_open'].strftime('%H:%M %Z')})")
            
            # Sleep for shorter intervals as we get closer
            if time_until_open > 3600:
                time.sleep(600)  # Check every 10 min if > 1 hour away
            elif time_until_open > 300:
                time.sleep(60)   # Check every 1 min if > 5 min away
            else:
                time.sleep(30)   # Check every 30 sec if close
        else:
            time.sleep(10)  # Brief sleep and recheck


def run_day_trading(trader, agent, tickers, interval_minutes):
    """
    Run trading throughout the day until market closes.
    """
    trade_count = 0
    start_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("DAY TRADING SESSION STARTED")
    print(f"Interval: Every {interval_minutes} minutes")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Will auto-stop when market closes")
    print("=" * 60)
    
    while True:
        # Check market status
        market = get_market_status(trader.api)
        
        if not market['is_open']:
            print("\n" + "=" * 60)
            print("MARKET CLOSED - Stopping trading session")
            print("=" * 60)
            break
        
        # Calculate time until close
        now = datetime.now(market['next_close'].tzinfo)
        time_until_close = (market['next_close'] - now).total_seconds()
        hours_left = time_until_close / 3600
        
        print(f"\nMarket closes in {hours_left:.1f} hours")
        
        # Run a trading cycle
        trade_count += 1
        print(f"\n--- Trade Cycle #{trade_count} ---")
        
        try:
            run_single_trade(trader, agent, tickers)
        except Exception as e:
            print(f"Error during trade cycle: {e}")
        
        # Check if market will close before next interval
        if time_until_close < (interval_minutes * 60):
            print(f"\nLess than {interval_minutes} min until close. Running final check...")
            time.sleep(60)  # Wait 1 minute and check again
            continue
        
        # Wait for next interval
        print(f"\nSleeping for {interval_minutes} minutes until next trade...")
        print(f"   Next trade at: {(datetime.now() + __import__('datetime').timedelta(minutes=interval_minutes)).strftime('%H:%M:%S')}")
        
        # Sleep in chunks to allow checking market status
        for _ in range(interval_minutes):
            time.sleep(60)  # Sleep 1 minute at a time
            
            # Quick market check every minute
            if not trader.api.get_clock().is_open:
                print("\nMarket closed during wait period")
                break
    
    # End of day summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("DAY TRADING SESSION SUMMARY")
    print("=" * 60)
    print(f"   Session Duration: {duration}")
    print(f"   Total Trade Cycles: {trade_count}")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final account status
    print_account_status(trader)
    
    return trade_count


def main():
    args = parse_arguments()
    
    print("=" * 60)
    print("DAY TRADING MODE")
    print("   Trades during market hours, auto-stops at close")
    print("=" * 60)
    
    # Initialize trader
    trader = AlpacaTrader(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    
    # Initialize agent
    agent = TradingAgent(MODEL_PATH, STOCK_TICKERS)
    
    # Check market status
    market = get_market_status(trader.api)
    
    if not market['is_open']:
        print(f"\nMarket is currently CLOSED")
        print(f"Next open: {market['next_open']}")
        
        # Check if market opens today (within 12 hours)
        from datetime import timezone
        now = datetime.now(market['next_open'].tzinfo)
        hours_until_open = (market['next_open'] - now).total_seconds() / 3600
        
        if hours_until_open > 12:
            # Weekend or holiday - exit to save dyno hours
            print(f"\nMarket won't open for {hours_until_open:.1f} hours (weekend/holiday)")
            print("Exiting to save resources. Scheduler will retry next trading day.")
            sys.exit(0)
        
        print("\nAuto-waiting for market to open...")
        wait_for_market_open(trader)
    
    # Run day trading
    trade_count = run_day_trading(trader, agent, STOCK_TICKERS, args.interval)
    
    print("\nDay trading session complete!")
    print(f"Total trades executed: {trade_count}")


if __name__ == "__main__":
    main()