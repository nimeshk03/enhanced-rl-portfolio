"""run_experiment.py

Unified experiment runner for testing different training configurations.
Each experiment is saved with a unique name for easy comparison.

Usage:
    python run_experiment.py --name baseline --timesteps 200000
    python run_experiment.py --name longer_training --timesteps 500000
    python run_experiment.py --name high_lr --timesteps 200000 --lr 0.001

This script:
1. Trains a PPO agent with specified configuration
2. Runs backtest on test period
3. Compares against SPY benchmark
4. Saves all results to experiments/{experiment_name}/
5. Updates the experiments summary CSV
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# FinRL imports
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "./data/processed_data.csv"
EXPERIMENTS_DIR = "./experiments"

# Time periods
TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31"
TRADE_START = "2024-01-01"
TRADE_END = "2025-10-30"

# Default environment config
DEFAULT_ENV_CONFIG = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
}

# Default PPO config
DEFAULT_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RL trading experiment')
    
    # Experiment identification
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name (e.g., baseline, longer_training)')
    parser.add_argument('--description', type=str, default='',
                        help='Description of what this experiment tests')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Total training timesteps')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C'],
                        help='RL algorithm to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--n_steps', type=int, default=2048,
                        help='Steps per update')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    
    return parser.parse_args()


def create_experiment_dir(experiment_name):
    """Create directory for this experiment."""
    exp_dir = f"{EXPERIMENTS_DIR}/{experiment_name}"
    if os.path.exists(exp_dir):
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"{EXPERIMENTS_DIR}/{experiment_name}_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    
    return exp_dir


def load_and_prepare_data(data_path, start_date, end_date):
    """Load and prepare data with day index."""
    df = pd.read_csv(data_path)
    df['date'] = df['date'].astype(str)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Filter date range
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Create day index
    dates = sorted(filtered_df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    filtered_df['day'] = filtered_df['date'].map(date_to_day)
    
    # Set day as index (required by FinRL)
    filtered_df = filtered_df.set_index('day')
    
    return filtered_df, dates


def get_feature_columns(df):
    """Get technical indicator columns."""
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
    return [col for col in df.columns if col not in base_cols]


def create_environment(df, env_config, feature_columns):
    """Create trading environment."""
    tickers = sorted(df['tic'].unique())
    stock_dim = len(tickers)
    state_space = 1 + 2 * stock_dim + len(feature_columns) * stock_dim
    num_stock_shares = [0] * stock_dim
    
    env_kwargs = {
        "hmax": env_config["hmax"],
        "initial_amount": env_config["initial_amount"],
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": [env_config["buy_cost_pct"]] * stock_dim,
        "sell_cost_pct": [env_config["sell_cost_pct"]] * stock_dim,
        "reward_scaling": env_config["reward_scaling"],
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": feature_columns,
        "action_space": stock_dim,
        "print_verbosity": 500,
        "day": 0,
        "initial": True,
    }
    
    env = StockTradingEnv(df=df, **env_kwargs)
    return env, env_kwargs


def train_agent(env, algorithm, ppo_config, total_timesteps, seed, model_dir):
    """Train the RL agent."""
    env_train = DummyVecEnv([lambda: env])
    
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=ppo_config["learning_rate"],
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gamma=ppo_config["gamma"],
            gae_lambda=ppo_config["gae_lambda"],
            clip_range=ppo_config["clip_range"],
            ent_coef=ppo_config["ent_coef"],
            seed=seed,
            verbose=1,
        )
    else:  # A2C
        model = A2C(
            "MlpPolicy",
            env_train,
            learning_rate=ppo_config["learning_rate"],
            n_steps=ppo_config["n_steps"],
            gamma=ppo_config["gamma"],
            gae_lambda=ppo_config["gae_lambda"],
            ent_coef=ppo_config["ent_coef"],
            seed=seed,
            verbose=1,
        )
    
    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps)
    training_time = datetime.now() - start_time
    
    # Save model
    model_path = f"{model_dir}/trained_agent"
    model.save(model_path)
    
    return model, training_time


def run_backtest(model, trade_df, env_config, feature_columns):
    """Run backtest and return portfolio values."""
    tickers = sorted(trade_df['tic'].unique())
    stock_dim = len(tickers)
    state_space = 1 + 2 * stock_dim + len(feature_columns) * stock_dim
    num_stock_shares = [0] * stock_dim
    
    env_kwargs = {
        "hmax": env_config["hmax"],
        "initial_amount": env_config["initial_amount"],
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": [env_config["buy_cost_pct"]] * stock_dim,
        "sell_cost_pct": [env_config["sell_cost_pct"]] * stock_dim,
        "reward_scaling": env_config["reward_scaling"],
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": feature_columns,
        "action_space": stock_dim,
        "print_verbosity": 1000,
        "day": 0,
        "initial": True,
    }
    
    backtest_env = StockTradingEnv(df=trade_df, **env_kwargs)
    obs, _ = backtest_env.reset()
    
    done = False
    while not done:
        obs_array = np.array(obs).reshape(1, -1)
        action, _ = model.predict(obs_array, deterministic=True)
        obs, reward, done, truncated, info = backtest_env.step(action[0])
    
    return backtest_env.asset_memory, backtest_env.date_memory


def download_spy_benchmark(start_date, end_date, initial_amount):
    """Download SPY benchmark data."""
    try:
        spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
        
        if spy_data.empty:
            return None, None
        
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] for col in spy_data.columns]
        
        spy_data = spy_data.reset_index()
        spy_data['Date'] = pd.to_datetime(spy_data['Date']).dt.strftime('%Y-%m-%d')
        
        initial_price = spy_data['Close'].iloc[0]
        shares_bought = initial_amount / initial_price
        spy_data['portfolio_value'] = shares_bought * spy_data['Close']
        
        return spy_data['portfolio_value'].tolist(), spy_data['Date'].tolist()
    
    except Exception as e:
        print(f"[WARNING] Error downloading SPY: {e}")
        return None, None


def calculate_metrics(portfolio_values, name="Strategy"):
    """Calculate performance metrics."""
    if portfolio_values is None or len(portfolio_values) < 2:
        return {"Strategy": name, "Error": "Insufficient data"}
    
    pv = pd.Series(portfolio_values)
    returns = pv.pct_change().dropna()
    
    total_days = len(pv) - 1
    if total_days == 0:
        return {"Strategy": name, "Error": "No data"}
    
    total_return = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
    annual_return = ((pv.iloc[-1] / pv.iloc[0]) ** (252 / total_days) - 1) * 100
    
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100 if daily_vol > 0 else 0
    
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate / 252
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    cumulative = pv / pv.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    
    return {
        "Strategy": name,
        "Initial Value": pv.iloc[0],
        "Final Value": pv.iloc[-1],
        "Total Return (%)": total_return,
        "Annual Return (%)": annual_return,
        "Annual Volatility (%)": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_drawdown,
        "Win Rate (%)": win_rate,
        "Trading Days": total_days,
    }


def create_visualizations(agent_values, spy_values, exp_dir, experiment_name):
    """Create and save performance visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Experiment: {experiment_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Portfolio Value Comparison
    ax1 = axes[0, 0]
    ax1.plot(agent_values, label='RL Agent', color='blue', linewidth=2)
    if spy_values:
        min_len = min(len(agent_values), len(spy_values))
        ax1.plot(spy_values[:min_len], label='SPY', color='orange', linewidth=2, linestyle='--')
    ax1.axhline(y=100000, color='gray', linestyle=':', alpha=0.7)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    ax2 = axes[0, 1]
    pv = pd.Series(agent_values)
    cumulative = pv / pv.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown Analysis')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Daily Returns Distribution
    ax3 = axes[1, 0]
    returns = pv.pct_change().dropna() * 100
    ax3.hist(returns, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.3f}%')
    ax3.set_title('Daily Returns Distribution')
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Returns
    ax4 = axes[1, 1]
    agent_cum = (pv / pv.iloc[0] - 1) * 100
    ax4.plot(agent_cum, label='RL Agent', color='blue', linewidth=2)
    if spy_values:
        spy_pv = pd.Series(spy_values[:len(agent_values)])
        spy_cum = (spy_pv / spy_pv.iloc[0] - 1) * 100
        ax4.plot(spy_cum, label='SPY', color='orange', linewidth=2, linestyle='--')
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    ax4.set_title('Cumulative Returns')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{exp_dir}/performance_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def save_experiment_results(exp_dir, experiment_name, args, agent_metrics, spy_metrics, 
                           agent_values, training_time, config):
    """Save all experiment results."""
    
    # Save configuration
    config_data = {
        "experiment_name": experiment_name,
        "description": args.description,
        "timestamp": datetime.now().isoformat(),
        "training_time_seconds": training_time.total_seconds(),
        "algorithm": args.algorithm,
        "total_timesteps": args.timesteps,
        "seed": args.seed,
        "ppo_config": config,
        "train_period": f"{TRAIN_START} to {TRAIN_END}",
        "test_period": f"{TRADE_START} to {TRADE_END}",
    }
    
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Save metrics
    metrics_df = pd.DataFrame([agent_metrics])
    if spy_metrics and "Error" not in spy_metrics:
        metrics_df = pd.concat([metrics_df, pd.DataFrame([spy_metrics])], ignore_index=True)
    metrics_df.to_csv(f"{exp_dir}/metrics.csv", index=False)
    
    # Save portfolio values
    portfolio_df = pd.DataFrame({'portfolio_value': agent_values})
    portfolio_df.to_csv(f"{exp_dir}/portfolio_values.csv", index=False)
    
    # Update master experiments summary
    update_experiments_summary(experiment_name, args, agent_metrics, spy_metrics, training_time)


def update_experiments_summary(experiment_name, args, agent_metrics, spy_metrics, training_time):
    """Update the master experiments summary CSV."""
    summary_path = f"{EXPERIMENTS_DIR}/experiments_summary.csv"
    
    # Create summary row
    spy_return = spy_metrics.get("Total Return (%)", 0) if spy_metrics and "Error" not in spy_metrics else 0
    
    summary_row = {
        "Experiment": experiment_name,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Algorithm": args.algorithm,
        "Timesteps": args.timesteps,
        "Learning Rate": args.lr,
        "Total Return (%)": round(agent_metrics.get("Total Return (%)", 0), 2),
        "Sharpe Ratio": round(agent_metrics.get("Sharpe Ratio", 0), 3),
        "Max Drawdown (%)": round(agent_metrics.get("Max Drawdown (%)", 0), 2),
        "vs SPY (%)": round(agent_metrics.get("Total Return (%)", 0) - spy_return, 2),
        "Training Time (min)": round(training_time.total_seconds() / 60, 1),
        "Description": args.description,
    }
    
    # Load existing or create new
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Updated experiments summary: {summary_path}")


def print_results(experiment_name, agent_metrics, spy_metrics, training_time):
    """Print experiment results."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT RESULTS: {experiment_name}")
    print("=" * 70)
    
    print(f"\nTraining Time: {training_time}")
    
    print(f"\n{'Metric':<25} {'RL Agent':>15} {'SPY':>15} {'Diff':>10}")
    print("-" * 70)
    
    metrics_to_show = [
        ("Total Return (%)", True),
        ("Annual Return (%)", True),
        ("Sharpe Ratio", True),
        ("Sortino Ratio", True),
        ("Max Drawdown (%)", False),  # False = lower is better
        ("Win Rate (%)", True),
    ]
    
    for metric, higher_better in metrics_to_show:
        agent_val = agent_metrics.get(metric, 0)
        spy_val = spy_metrics.get(metric, 0) if spy_metrics and "Error" not in spy_metrics else 0
        diff = agent_val - spy_val
        
        # Determine if agent won
        if higher_better:
            winner = "[OK]" if diff > 0 else ""
        else:
            winner = "[OK]" if diff > 0 else ""  # For drawdown, less negative is better
        
        print(f"{metric:<25} {agent_val:>15.2f} {spy_val:>15.2f} {diff:>+10.2f} {winner}")
    
    print("=" * 70)


def main():
    """Main experiment runner."""
    args = parse_arguments()
    
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: {args.name}")
    print("=" * 70)
    print(f"Description: {args.description}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed: {args.seed}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.name)
    print(f"\nExperiment directory: {exp_dir}")
    
    # Build config from arguments
    ppo_config = {
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": args.ent_coef,
    }
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df, train_dates = load_and_prepare_data(DATA_PATH, TRAIN_START, TRAIN_END)
    trade_df, trade_dates = load_and_prepare_data(DATA_PATH, TRADE_START, TRADE_END)
    feature_columns = get_feature_columns(train_df)
    print(f"[OK] Training data: {len(train_df)} rows, {len(train_dates)} days")
    print(f"[OK] Test data: {len(trade_df)} rows, {len(trade_dates)} days")
    
    # Create environment
    print("\n[2/6] Creating environment...")
    train_env, env_kwargs = create_environment(train_df, DEFAULT_ENV_CONFIG, feature_columns)
    print("[OK] Environment created")
    
    # Train agent
    print(f"\n[3/6] Training {args.algorithm} agent...")
    print("-" * 70)
    model, training_time = train_agent(
        train_env, args.algorithm, ppo_config, 
        args.timesteps, args.seed, f"{exp_dir}/models"
    )
    print("-" * 70)
    print(f"[OK] Training completed in {training_time}")
    
    # Run backtest
    print("\n[4/6] Running backtest...")
    agent_values, agent_dates = run_backtest(model, trade_df, DEFAULT_ENV_CONFIG, feature_columns)
    print(f"[OK] Backtest completed")
    print(f"  Initial: ${agent_values[0]:,.2f}")
    print(f"  Final: ${agent_values[-1]:,.2f}")
    
    # Download benchmark
    print("\n[5/6] Downloading SPY benchmark...")
    spy_values, spy_dates = download_spy_benchmark(TRADE_START, TRADE_END, DEFAULT_ENV_CONFIG["initial_amount"])
    if spy_values:
        print(f"[OK] SPY benchmark downloaded")
    
    # Calculate metrics
    agent_metrics = calculate_metrics(agent_values, "RL Agent")
    spy_metrics = calculate_metrics(spy_values, "SPY") if spy_values else None
    
    # Create visualizations
    print("\n[6/6] Saving results...")
    create_visualizations(agent_values, spy_values, exp_dir, args.name)
    
    # Save all results
    save_experiment_results(
        exp_dir, args.name, args, agent_metrics, spy_metrics,
        agent_values, training_time, ppo_config
    )
    
    # Print results
    print_results(args.name, agent_metrics, spy_metrics, training_time)
    
    print(f"\n[OK] All results saved to: {exp_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()