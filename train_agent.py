"""train_agent.py

Train a Deep Reinforcement Learning agent for portfolio management.
This script uses PPO (Proximal Policy Optimization) from Stable-Baselines3
with FinRL's stock trading environment.

Usage:
    python train_agent.py

This script:
1. Loads processed data with technical indicators
2. Splits data into train/trade periods
3. Creates the stock trading environment
4. Trains PPO agent
5. Saves the trained model

The environment follows OpenAI Gym interface:
- State Space: [cash, stock_holdings, stock_prices, technical_indicators]
- Action Space: Continuous [-1, 1] for each stock (sell to buy)
- Reward: Change in portfolio value (can be modified to Sharpe ratio)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# FinRL imports
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS

# Stable-Baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEP RL PORTFOLIO MANAGER - TRAINING")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
DATA_PATH = "./data/processed_data.csv"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"

# Time periods for train/test split
# Extended training data (2015-2024) for enhanced model
TRAIN_START = "2015-01-01"
TRAIN_END = "2024-06-30"    # ~9.5 years of training data
TRADE_START = "2024-07-01"
TRADE_END = "2025-11-30"    # ~17 months of testing

# Environment configuration
ENV_CONFIG = {
    "hmax": 100,              # Maximum number of shares to trade per action
    "initial_amount": 100000,  # Starting portfolio value ($100k)
    "buy_cost_pct": 0.001,    # 0.1% transaction cost for buying
    "sell_cost_pct": 0.001,   # 0.1% transaction cost for selling
    "reward_scaling": 1e-4,   # Scale rewards for numerical stability
}

# PPO Hyperparameters (tuned for financial trading)
PPO_CONFIG = {
    "learning_rate": 3e-4,    # Learning rate
    "n_steps": 2048,          # Steps per update (captures ~8 trading days patterns)
    "batch_size": 128,        # Mini-batch size for PPO updates
    "n_epochs": 10,           # Number of epochs per update
    "gamma": 0.99,            # Discount factor (high = long-term focus)
    "gae_lambda": 0.95,       # GAE lambda for advantage estimation
    "clip_range": 0.2,        # PPO clip range
    "ent_coef": 0.01,         # Entropy coefficient (exploration)
    "verbose": 1,             # Print training progress
}

# Training configuration
TOTAL_TIMESTEPS = 200_000    # Total training steps (adjust based on compute)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directories():
    """Create necessary directories for models and results."""
    for directory in [MODEL_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[OK] Created directory: {directory}")


def load_and_split_data(data_path, train_start, train_end, trade_start, trade_end):
    """
    Load processed data and split into training and trading periods.
    
    Walk-forward validation approach:
    - Train on historical data (2019-2023)
    - Test on unseen future data (2024-2025)
    
    IMPORTANT: FinRL requires data to be sorted by (date, tic) and each date
    must have exactly the same number of tickers.
    """
    print("\n[1/5] Loading and splitting data...")
    
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df)} rows from {data_path}")
    
    # Ensure date column is string for comparison
    df['date'] = df['date'].astype(str)
    
    # CRITICAL: Sort by date first, then by ticker
    # FinRL iterates through data expecting this exact order
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Verify each date has the same number of tickers (required by FinRL)
    tickers_per_date = df.groupby('date')['tic'].count()
    if tickers_per_date.nunique() != 1:
        print(f"[WARNING] Inconsistent ticker counts per date")
        print(f"  Min: {tickers_per_date.min()}, Max: {tickers_per_date.max()}")
        
        # Filter to only dates that have all tickers
        expected_ticker_count = df['tic'].nunique()
        valid_dates = tickers_per_date[tickers_per_date == expected_ticker_count].index
        df = df[df['date'].isin(valid_dates)].reset_index(drop=True)
        print(f"  Filtered to {len(valid_dates)} dates with all {expected_ticker_count} tickers")
    
    # Split data
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
    trade_df = df[(df['date'] >= trade_start) & (df['date'] <= trade_end)].copy()
    
    # Add 'day' column - FinRL uses this internally for indexing
    # Each unique date gets a sequential number starting from 0
    train_dates = sorted(train_df['date'].unique())
    train_date_to_day = {date: i for i, date in enumerate(train_dates)}
    train_df['day'] = train_df['date'].map(train_date_to_day)
    
    trade_dates = sorted(trade_df['date'].unique())
    trade_date_to_day = {date: i for i, date in enumerate(trade_dates)}
    trade_df['day'] = trade_df['date'].map(trade_date_to_day)
    
    # CRITICAL: FinRL uses df.loc[day, :] to access data
    # This requires the 'day' column to be the INDEX, not just a column
    # When indexed by 'day', df.loc[0, :] returns all rows where day=0 (all 10 stocks)
    train_df = train_df.set_index('day')
    trade_df = trade_df.set_index('day')
    
    print(f"[OK] Training period: {train_start} to {train_end}")
    print(f"  - {len(train_df)} rows, {train_df['date'].nunique()} trading days")
    print(f"[OK] Trading period: {trade_start} to {trade_end}")
    print(f"  - {len(trade_df)} rows, {trade_df['date'].nunique()} trading days")
    
    # Verify we have all tickers in both periods
    train_tickers = set(train_df['tic'].unique())
    trade_tickers = set(trade_df['tic'].unique())
    
    if train_tickers != trade_tickers:
        print(f"[WARNING] Ticker mismatch between train and trade periods")
    else:
        print(f"[OK] All {len(train_tickers)} tickers present in both periods")
    
    # Debug: Show data structure
    print(f"\n  Data structure check:")
    print(f"  - Tickers per day: {len(train_df.loc[0])}")  # day is now the index
    print(f"  - First day (index 0) tickers: {train_df.loc[0]['tic'].tolist()}")
    print(f"  - Day index range (train): 0 to {train_df.index.max()}")
    
    return train_df, trade_df


def get_feature_columns(df):
    """
    Identify which columns are technical indicators/features.
    These will be part of the state space for the RL agent.
    """
    # Base columns that are NOT features
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
    
    # Everything else is a feature
    feature_cols = [col for col in df.columns if col not in base_cols]
    
    return feature_cols


def create_environment(df, env_config, feature_columns):
    """
    Create the stock trading environment.
    
    The FinRL StockTradingEnv implements:
    - State: [cash, holdings, prices, features] 
    - Action: Continuous allocation weights
    - Reward: Portfolio value change
    """
    print("\n[2/5] Creating trading environment...")
    
    # Get unique tickers (sorted for consistency)
    tickers = sorted(df['tic'].unique())
    stock_dim = len(tickers)
    
    # State space dimension:
    # cash (1) + holdings (stock_dim) + prices (stock_dim) + features (stock_dim * num_features)
    state_space = 1 + 2 * stock_dim + len(feature_columns) * stock_dim
    
    print(f"  - Number of stocks: {stock_dim}")
    print(f"  - Tickers: {tickers}")
    print(f"  - Number of features per stock: {len(feature_columns)}")
    print(f"  - State space dimension: {state_space}")
    print(f"  - Action space dimension: {stock_dim} (continuous)")
    
    # Initial stock holdings - start with 0 shares of each stock
    # The agent will use the initial cash to build positions
    num_stock_shares = [0] * stock_dim
    
    # Environment kwargs
    env_kwargs = {
        "hmax": env_config["hmax"],
        "initial_amount": env_config["initial_amount"],
        "num_stock_shares": num_stock_shares,  # Required: initial holdings (start with 0)
        "buy_cost_pct": [env_config["buy_cost_pct"]] * stock_dim,
        "sell_cost_pct": [env_config["sell_cost_pct"]] * stock_dim,
        "reward_scaling": env_config["reward_scaling"],
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": feature_columns,
        "action_space": stock_dim,
        "print_verbosity": 1,  # Reduced verbosity during training
        "day": 0,  # Starting day index
        "initial": True,  # This is initial setup
    }
    
    # Verify data format before creating environment
    print(f"\n  Verifying data format...")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Index name: {df.index.name}")  # Should be 'day'
    print(f"  - 'tic' column present: {'tic' in df.columns}")
    
    # Check first day's data (day=0 should have all 10 stocks)
    first_day_data = df.loc[0]
    print(f"  - First day (index 0) has {len(first_day_data)} rows")
    print(f"  - df.loc[0, :] type: {type(first_day_data)}")  # Should be DataFrame, not Series
    
    # Create environment
    env = StockTradingEnv(df=df, **env_kwargs)
    
    print(f"[OK] Environment created successfully")
    print(f"  - Initial portfolio value: ${env_config['initial_amount']:,}")
    print(f"  - Transaction costs: {env_config['buy_cost_pct']*100}% buy, {env_config['sell_cost_pct']*100}% sell")
    
    return env, env_kwargs


def train_ppo_agent(env, ppo_config, total_timesteps, model_dir):
    """
    Train PPO agent using Stable-Baselines3.
    
    PPO (Proximal Policy Optimization) is well-suited for:
    - Continuous action spaces (portfolio weights)
    - Sample efficiency
    - Stable training
    """
    print("\n[3/5] Training PPO agent...")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Learning rate: {ppo_config['learning_rate']}")
    print(f"  - Batch size: {ppo_config['batch_size']}")
    print("-" * 70)
    
    # Wrap environment in DummyVecEnv for Stable-Baselines3 compatibility
    env_train = DummyVecEnv([lambda: env])
    
    # Create PPO model
    # Note: tensorboard_log is set to None if tensorboard is not installed
    try:
        import tensorboard
        tb_log_path = f"{model_dir}/tensorboard/"
        print(f"  - TensorBoard logging enabled: {tb_log_path}")
    except ImportError:
        tb_log_path = None
        print(f"  - TensorBoard not installed, logging disabled")
    
    model = PPO(
        "MlpPolicy",           # Multi-layer perceptron policy
        env_train,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
        verbose=ppo_config["verbose"],
        tensorboard_log=tb_log_path  # None if tensorboard not installed
    )
    
    # Train the model
    print("\nStarting training...")
    start_time = datetime.now()
    
    model.learn(total_timesteps=total_timesteps)
    
    training_time = datetime.now() - start_time
    print("-" * 70)
    print(f"[OK] Training completed in {training_time}")
    
    # Save the trained model
    model_path = f"{model_dir}/ppo_portfolio_agent"
    model.save(model_path)
    print(f"[OK] Model saved to {model_path}.zip")
    
    return model


def backtest_agent(model, trade_df, env_kwargs, results_dir):
    """
    Backtest the trained agent on unseen data.
    
    This simulates how the agent would perform in live trading
    during the test period (2024-2025).
    """
    print("\n[4/5] Backtesting on test period...")
    
    # Create test environment
    env_trade = StockTradingEnv(df=trade_df, **env_kwargs)
    env_trade = DummyVecEnv([lambda: env_trade])
    
    # Run backtest
    obs = env_trade.reset()
    
    portfolio_values = [ENV_CONFIG["initial_amount"]]
    actions_taken = []
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_trade.step(action)
        
        # Extract portfolio value from info
        if isinstance(info, list) and len(info) > 0:
            if 'terminal_observation' in info[0]:
                # Episode ended
                pass
        
        actions_taken.append(action[0])
    
    # Get final results from the environment
    # The env tracks portfolio values internally
    print(f"[OK] Backtest completed")
    
    return actions_taken


def calculate_metrics(portfolio_values, risk_free_rate=0.02):
    """
    Calculate key performance metrics for the portfolio.
    
    Metrics:
    - Total Return: Overall percentage gain/loss
    - Sharpe Ratio: Risk-adjusted return (target > 1.0)
    - Max Drawdown: Largest peak-to-trough decline (target < 25%)
    - Volatility: Standard deviation of returns
    """
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # Annualization factor (252 trading days)
    annual_factor = np.sqrt(252)
    
    metrics = {
        "Total Return (%)": ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100,
        "Annual Return (%)": ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(returns)) - 1) * 100,
        "Sharpe Ratio": (returns.mean() - risk_free_rate/252) / returns.std() * annual_factor if returns.std() > 0 else 0,
        "Max Drawdown (%)": ((pd.Series(portfolio_values).cummax() - portfolio_values) / pd.Series(portfolio_values).cummax()).max() * 100,
        "Volatility (%)": returns.std() * annual_factor * 100,
        "Win Rate (%)": (returns > 0).sum() / len(returns) * 100
    }
    
    return metrics


def main():
    """Main training pipeline."""
    
    # Setup
    create_directories()
    
    # Load data
    train_df, trade_df = load_and_split_data(
        DATA_PATH, TRAIN_START, TRAIN_END, TRADE_START, TRADE_END
    )
    
    # Get feature columns
    feature_columns = get_feature_columns(train_df)
    print(f"\nFeatures being used ({len(feature_columns)}):")
    print(f"  {feature_columns}")
    
    # Create training environment
    train_env, env_kwargs = create_environment(train_df, ENV_CONFIG, feature_columns)
    
    # Train agent
    model = train_ppo_agent(train_env, PPO_CONFIG, TOTAL_TIMESTEPS, MODEL_DIR)
    
    # Backtest
    actions = backtest_agent(model, trade_df, env_kwargs, RESULTS_DIR)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"[OK] Model saved to: {MODEL_DIR}/ppo_portfolio_agent.zip")
    print(f"\nNext steps:")
    print(f"  1. Run 'python evaluate_agent.py' for detailed backtesting")
    print(f"  2. Compare against SPY benchmark")
    print(f"  3. (Optional) Install tensorboard for training visualization")
    print("=" * 70)


if __name__ == "__main__":
    main()