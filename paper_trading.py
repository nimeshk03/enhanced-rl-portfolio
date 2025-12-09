"""paper_trading.py

Automated Paper Trading with Alpaca API.

This script:
1. Connects to Alpaca Paper Trading API
2. Fetches real-time market data
3. Uses our trained RL agent to make decisions
4. Executes trades automatically
5. Logs all activity

Usage:
    # Run once (single trading decision)
    python paper_trading.py --mode single
    
    # Run continuously during market hours
    python paper_trading.py --mode continuous
    
    # Check account status only
    python paper_trading.py --mode status

Requirements:
    pip install alpaca-trade-api
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Alpaca API
from alpaca_trade_api import REST, TimeFrame

# For loading our trained model
from stable_baselines3 import PPO

# Technical indicators
try:
    from stockstats import StockDataFrame as Sdf
except ImportError:
    print("Installing stockstats...")
    os.system("pip install stockstats --break-system-packages -q")
    from stockstats import StockDataFrame as Sdf

# Import configuration
try:
    from config import (
        ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
        STOCK_TICKERS, MODEL_PATH, MAX_DRAWDOWN_THRESHOLD, VIX_THRESHOLD
    )
except ImportError:
    print("ERROR: config.py not found!")
    print("Please create config.py with your Alpaca API credentials.")
    print("See config_template.py for an example.")
    sys.exit(1)

# =============================================================================
# ALPACA CLIENT
# =============================================================================

class AlpacaTrader:
    """
    Handles all interactions with the Alpaca API.
    """
    
    def __init__(self, api_key, secret_key, base_url):
        """Initialize Alpaca API connection."""
        self.api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        self.base_url = base_url
        
        # Verify connection
        try:
            account = self.api.get_account()
            print(f"[OK] Connected to Alpaca ({base_url})")
            print(f"  Account Status: {account.status}")
            print(f"  Buying Power: ${float(account.buying_power):,.2f}")
            print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        except Exception as e:
            print(f"[FAIL] Failed to connect to Alpaca: {e}")
            sys.exit(1)
    
    def get_account(self):
        """Get account information."""
        return self.api.get_account()
    
    def get_positions(self):
        """Get current positions."""
        return self.api.list_positions()
    
    def get_position(self, symbol):
        """Get position for a specific symbol."""
        try:
            return self.api.get_position(symbol)
        except:
            return None
    
    def is_market_open(self):
        """Check if market is currently open."""
        clock = self.api.get_clock()
        return clock.is_open
    
    def get_market_hours(self):
        """Get market open/close times for today."""
        clock = self.api.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": clock.next_close
        }
    
    def get_historical_data(self, symbols, days=60):
        """
        Fetch historical price data for technical indicator calculation.
        We need ~60 days of history to calculate indicators like 60-day SMA.
        
        Note: Free Alpaca accounts must use 'iex' feed instead of 'sip'.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Use IEX feed for free accounts (instead of default SIP)
                bars = self.api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex'  # FREE: Use IEX data feed
                ).df
                
                if len(bars) > 0:
                    bars = bars.reset_index()
                    bars['tic'] = symbol
                    bars.columns = [c.lower() for c in bars.columns]
                    
                    # Rename timestamp to date
                    if 'timestamp' in bars.columns:
                        bars['date'] = pd.to_datetime(bars['timestamp']).dt.strftime('%Y-%m-%d')
                        bars = bars.drop('timestamp', axis=1)
                    
                    all_data.append(bars)
                    
            except Exception as e:
                print(f"  Warning: Could not fetch data for {symbol}: {e}")
        
        if not all_data:
            return None
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return df
    
    def get_latest_prices(self, symbols):
        """Get latest prices for symbols using IEX feed (free tier)."""
        prices = {}
        for symbol in symbols:
            try:
                # Use IEX feed for free accounts
                quote = self.api.get_latest_trade(symbol, feed='iex')
                prices[symbol] = quote.price
            except Exception as e:
                print(f"  Warning: Could not get price for {symbol}: {e}")
                prices[symbol] = None
        return prices
    
    def submit_order(self, symbol, qty, side, order_type='market'):
        """
        Submit an order to Alpaca.
        
        Args:
            symbol: Stock ticker
            qty: Number of shares (positive)
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
        """
        if qty <= 0:
            return None
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type=order_type,
                time_in_force='day'
            )
            return order
        except Exception as e:
            print(f"  [FAIL] Order failed for {symbol}: {e}")
            return None
    
    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.api.cancel_all_orders()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_technical_indicators(df):
    """
    Add technical indicators to match our training data.
    
    Indicators: macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, 
                close_30_sma, close_60_sma
    """
    # Process each ticker separately
    ticker_dfs = []
    
    for tic in df['tic'].unique():
        tic_df = df[df['tic'] == tic].copy()
        
        # Use stockstats for indicator calculation
        stock = Sdf.retype(tic_df.copy())
        
        # Calculate indicators
        tic_df['macd'] = stock['macd']
        tic_df['boll_ub'] = stock['boll_ub']
        tic_df['boll_lb'] = stock['boll_lb']
        tic_df['rsi_30'] = stock['rsi_30']
        tic_df['cci_30'] = stock['cci_30']
        tic_df['dx_30'] = stock['dx_30']
        tic_df['close_30_sma'] = stock['close_30_sma']
        tic_df['close_60_sma'] = stock['close_60_sma']
        
        ticker_dfs.append(tic_df)
    
    result = pd.concat(ticker_dfs, ignore_index=True)
    result = result.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Fill NaN values (from indicator warm-up period)
    result = result.fillna(method='ffill').fillna(0)
    
    return result


def add_vix_and_turbulence(df):
    """Add VIX and turbulence indicators."""
    # For simplicity, we'll use a placeholder VIX value
    # In production, you'd fetch real VIX data
    df['vix'] = 20.0  # Placeholder - could fetch from ^VIX
    df['turbulence'] = 0.0  # Placeholder
    return df


def prepare_state(df, tickers):
    """
    Prepare the state vector for our RL agent.
    
    State format (matching training):
    [cash, stock_prices (10), holdings (10), technical_indicators (10 * 10)]
    
    For prediction, we don't have cash/holdings from the environment,
    so we'll use a simplified version focusing on market data.
    """
    # Get latest data for each ticker
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()
    
    # Ensure tickers are in correct order
    latest_data = latest_data.set_index('tic').loc[tickers].reset_index()
    
    # Feature columns (matching training)
    feature_cols = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 
                    'dx_30', 'close_30_sma', 'close_60_sma', 'turbulence', 'vix']
    
    # Build state vector
    # Placeholder for cash and holdings (will be filled by actual portfolio state)
    state = []
    
    # Prices
    prices = latest_data['close'].tolist()
    
    # Technical indicators for each stock
    tech_indicators = []
    for col in feature_cols:
        tech_indicators.extend(latest_data[col].tolist())
    
    return prices, tech_indicators, latest_data


# =============================================================================
# TRADING LOGIC
# =============================================================================

class TradingAgent:
    """
    Wrapper for our trained RL model to make trading decisions.
    """
    
    def __init__(self, model_path, tickers):
        """Load the trained model."""
        self.tickers = tickers
        self.stock_dim = len(tickers)
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        try:
            self.model = PPO.load(model_path)
            print("[OK] Model loaded successfully")
        except Exception as e:
            print(f"[FAIL] Failed to load model: {e}")
            sys.exit(1)
    
    def get_action(self, state):
        """
        Get trading action from the model.
        
        Returns: Array of actions for each stock [-1, 1]
                 Negative = sell, Positive = buy, ~0 = hold
        """
        state_array = np.array(state).reshape(1, -1)
        action, _ = self.model.predict(state_array, deterministic=True)
        return action[0]
    
    def interpret_actions(self, actions, prices, portfolio_value, current_holdings, cash, buying_power):
        """
        Convert model actions to actual trade orders.
        
        Args:
            actions: Model output [-1, 1] for each stock
            prices: Current prices for each stock
            portfolio_value: Total portfolio value
            current_holdings: Dict of {symbol: shares}
            cash: Current cash (can be negative with margin)
            buying_power: Available buying power
        
        Returns:
            List of trade orders: [{'symbol': 'AAPL', 'side': 'buy', 'qty': 10}, ...]
        """
        orders = []
        hmax = 100  # Max shares per trade (matching training)
        
        # Calculate max position value (20% of portfolio per stock)
        max_position_value = portfolio_value * 0.20
        
        # Minimum buying power to keep in reserve
        min_buying_power_reserve = 5000
        
        for i, (ticker, action) in enumerate(zip(self.tickers, actions)):
            current_shares = current_holdings.get(ticker, 0)
            price = prices[i]
            
            if price is None or price <= 0:
                continue
            
            current_position_value = current_shares * price
            
            # Scale action to shares
            target_shares_delta = int(action * hmax)
            
            if target_shares_delta > 0:
                # BUY signal - but check constraints
                
                # Skip if we don't have enough buying power
                order_cost = target_shares_delta * price
                if buying_power < (order_cost + min_buying_power_reserve):
                    # Reduce order size to fit buying power
                    affordable_shares = int((buying_power - min_buying_power_reserve) / price)
                    if affordable_shares <= 0:
                        continue  # Skip this buy
                    target_shares_delta = min(target_shares_delta, affordable_shares)
                
                # Skip if position would exceed max size
                new_position_value = (current_shares + target_shares_delta) * price
                if new_position_value > max_position_value:
                    # Calculate how many shares we can add
                    max_additional_value = max_position_value - current_position_value
                    if max_additional_value <= 0:
                        continue  # Position already at max
                    target_shares_delta = min(target_shares_delta, int(max_additional_value / price))
                
                if target_shares_delta > 0:
                    orders.append({
                        'symbol': ticker,
                        'side': 'buy',
                        'qty': min(target_shares_delta, hmax),
                        'action_value': action
                    })
                    # Update buying power estimate for next iteration
                    buying_power -= target_shares_delta * price
                    
            elif target_shares_delta < 0:
                # SELL signal
                sell_qty = min(abs(target_shares_delta), current_shares)
                if sell_qty > 0:
                    orders.append({
                        'symbol': ticker,
                        'side': 'sell',
                        'qty': sell_qty,
                        'action_value': action
                    })
        
        return orders


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def print_account_status(trader):
    """Print current account status."""
    account = trader.get_account()
    positions = trader.get_positions()
    
    print("\n" + "=" * 60)
    print("ACCOUNT STATUS")
    print("=" * 60)
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Day's P&L: ${float(account.equity) - float(account.last_equity):,.2f}")
    
    if positions:
        print(f"\nPositions ({len(positions)}):")
        print(f"  {'Symbol':<8} {'Qty':>8} {'Price':>10} {'Value':>12} {'P&L':>10}")
        print("  " + "-" * 50)
        for pos in positions:
            print(f"  {pos.symbol:<8} {int(float(pos.qty)):>8} "
                  f"${float(pos.current_price):>9.2f} "
                  f"${float(pos.market_value):>11.2f} "
                  f"${float(pos.unrealized_pl):>9.2f}")
    else:
        print("\nNo open positions")
    
    print("=" * 60)


def run_single_trade(trader, agent, tickers):
    """Execute a single trading cycle."""
    
    print("\n" + "=" * 60)
    print(f"TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check market status
    market = trader.get_market_hours()
    if not market['is_open']:
        print(f"[WARNING] Market is CLOSED")
        print(f"  Next open: {market['next_open']}")
        return
    
    print("[OK] Market is OPEN")
    
    # Get account info
    account = trader.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    
    print(f"\nPortfolio: ${portfolio_value:,.2f} (Cash: ${cash:,.2f}, Buying Power: ${buying_power:,.2f})")
    
    # Get current holdings
    positions = trader.get_positions()
    current_holdings = {p.symbol: int(float(p.qty)) for p in positions}
    print(f"Current positions: {current_holdings}")
    
    # Fetch market data
    print("\nFetching market data...")
    df = trader.get_historical_data(tickers, days=70)
    
    if df is None or len(df) == 0:
        print("[FAIL] Failed to fetch market data")
        return
    
    print(f"[OK] Fetched {len(df)} rows of data")
    
    # Add technical indicators
    print("Calculating technical indicators...")
    df = add_technical_indicators(df)
    df = add_vix_and_turbulence(df)
    print("[OK] Indicators calculated")
    
    # Prepare state for model
    prices, tech_indicators, latest_data = prepare_state(df, tickers)
    
    # Build full state vector (matching training format)
    # State: [cash, prices(10), holdings(10), tech_indicators(100)]
    holdings_list = [current_holdings.get(t, 0) for t in tickers]
    state = [cash] + prices + holdings_list + tech_indicators
    
    print(f"\nState vector size: {len(state)}")
    
    # Get model prediction
    print("Getting model prediction...")
    actions = agent.get_action(state)
    
    print(f"\nModel Actions:")
    for ticker, action, price in zip(tickers, actions, prices):
        direction = "BUY " if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
        print(f"  {ticker}: {action:+.3f} ({direction}) @ ${price:.2f}")
    
    # Convert to trade orders
    orders = agent.interpret_actions(actions, prices, portfolio_value, current_holdings, cash, float(account.buying_power))
    
    if not orders:
        print("\n→ No trades to execute")
        return
    
    # Execute trades
    print(f"\nEXECUTING {len(orders)} TRADES:")
    print("-" * 40)
    
    for order in orders:
        print(f"  {order['side'].upper():4} {order['qty']:3} {order['symbol']:<5}", end=" ")
        
        result = trader.submit_order(
            symbol=order['symbol'],
            qty=order['qty'],
            side=order['side']
        )
        
        if result:
            print(f"[OK] Order submitted (ID: {result.id[:8]}...)")
        else:
            print("[FAIL]")
    
    print("-" * 40)
    print("[OK] Trading cycle complete")


def run_continuous(trader, agent, tickers, interval_minutes=60):
    """Run trading continuously during market hours."""
    
    print("\n" + "=" * 60)
    print("CONTINUOUS TRADING MODE")
    print(f"Checking every {interval_minutes} minutes during market hours")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    while True:
        try:
            market = trader.get_market_hours()
            
            if market['is_open']:
                run_single_trade(trader, agent, tickers)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M')}] Market closed. "
                      f"Next open: {market['next_open']}")
            
            # Wait for next cycle
            print(f"\nSleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n\nStopping continuous trading...")
            break


def main():
    parser = argparse.ArgumentParser(description='RL Portfolio Paper Trading')
    parser.add_argument('--mode', type=str, default='status',
                        choices=['status', 'single', 'continuous'],
                        help='Trading mode')
    parser.add_argument('--interval', type=int, default=60,
                        help='Minutes between trades in continuous mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL PORTFOLIO MANAGER - PAPER TRADING")
    print("=" * 60)
    
    # Initialize Alpaca trader
    trader = AlpacaTrader(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    
    if args.mode == 'status':
        print_account_status(trader)
        return
    
    # Initialize trading agent
    agent = TradingAgent(MODEL_PATH, STOCK_TICKERS)
    
    if args.mode == 'single':
        run_single_trade(trader, agent, STOCK_TICKERS)
        print_account_status(trader)
        
    elif args.mode == 'continuous':
        run_continuous(trader, agent, STOCK_TICKERS, args.interval)
    
    print("\n[OK] Done")


if __name__ == "__main__":
    main()