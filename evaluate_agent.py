"""evaluate_agent.py

Comprehensive evaluation of the trained RL portfolio agent.
This script compares the agent's performance against the SPY benchmark
and calculates detailed performance metrics.

Usage:
    python evaluate_agent.py

This script:
1. Loads the trained PPO agent
2. Runs backtest on the test period (2024-2025)
3. Downloads SPY benchmark data for comparison
4. Calculates performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
5. Generates visualizations (equity curve, drawdown chart)
6. Saves results to CSV files
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FinRL imports
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# For downloading benchmark data
import yfinance as yf

print("=" * 70)
print("RL PORTFOLIO AGENT - COMPREHENSIVE EVALUATION")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "./data/processed_data.csv"
MODEL_PATH = "./models/ppo_portfolio_agent.zip"
RESULTS_DIR = "./results"

# Test period
TRADE_START = "2024-01-01"
TRADE_END = "2025-10-30"

# Environment config (must match training)
ENV_CONFIG = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directories():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"[OK] Created directory: {RESULTS_DIR}")


def load_and_prepare_data(data_path, trade_start, trade_end):
    """Load and prepare data for backtesting."""
    print("\n[1/6] Loading data...")
    
    df = pd.read_csv(data_path)
    df['date'] = df['date'].astype(str)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Filter to test period
    trade_df = df[(df['date'] >= trade_start) & (df['date'] <= trade_end)].copy()
    
    # Create day index
    trade_dates = sorted(trade_df['date'].unique())
    trade_date_to_day = {date: i for i, date in enumerate(trade_dates)}
    trade_df['day'] = trade_df['date'].map(trade_date_to_day)
    
    # Set day as index (required by FinRL)
    trade_df = trade_df.set_index('day')
    
    print(f"[OK] Loaded test data: {len(trade_df)} rows")
    print(f"  Period: {trade_start} to {trade_end}")
    print(f"  Trading days: {len(trade_dates)}")
    
    return trade_df, trade_dates


def get_feature_columns(df):
    """Get technical indicator columns."""
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
    return [col for col in df.columns if col not in base_cols]


def create_test_environment(df, env_config, feature_columns):
    """Create test environment."""
    print("\n[2/6] Creating test environment...")
    
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
        "print_verbosity": 500,  # Less verbose during eval
        "day": 0,
        "initial": True,
    }
    
    env = StockTradingEnv(df=df, **env_kwargs)
    print(f"[OK] Environment created")
    
    return env, env_kwargs


def run_backtest(model, env, trade_df, env_config, feature_columns):
    """Run the agent through the test environment and collect results."""
    print("\n[3/6] Running backtest...")
    
    # We need to create a fresh environment for backtesting
    # because the environment tracks state internally
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
        "print_verbosity": 500,
        "day": 0,
        "initial": True,
    }
    
    # Create fresh environment
    backtest_env = StockTradingEnv(df=trade_df, **env_kwargs)
    
    # Run without DummyVecEnv to access internal state directly
    obs, _ = backtest_env.reset()
    
    done = False
    step_count = 0
    
    while not done:
        # Model expects observation in a specific shape
        obs_array = np.array(obs).reshape(1, -1)
        action, _states = model.predict(obs_array, deterministic=True)
        obs, reward, done, truncated, info = backtest_env.step(action[0])
        step_count += 1
    
    # Get results directly from the environment
    portfolio_values = backtest_env.asset_memory
    dates = backtest_env.date_memory
    
    print(f"[OK] Backtest completed: {step_count} steps")
    print(f"  Portfolio values collected: {len(portfolio_values)}")
    print(f"  Initial: ${portfolio_values[0]:,.2f}")
    print(f"  Final: ${portfolio_values[-1]:,.2f}")
    
    return portfolio_values, dates


def download_benchmark(start_date, end_date, initial_amount):
    """Download SPY benchmark and calculate portfolio values."""
    print("\n[4/6] Downloading SPY benchmark...")
    
    try:
        spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
        
        if spy_data.empty:
            print("[WARNING] Could not download SPY data")
            return None, None
        
        # Handle MultiIndex columns
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] for col in spy_data.columns]
        
        spy_data = spy_data.reset_index()
        spy_data['Date'] = pd.to_datetime(spy_data['Date']).dt.strftime('%Y-%m-%d')
        
        # Calculate buy-and-hold returns
        initial_price = spy_data['Close'].iloc[0]
        shares_bought = initial_amount / initial_price
        spy_data['portfolio_value'] = shares_bought * spy_data['Close']
        
        print(f"[OK] SPY benchmark downloaded: {len(spy_data)} days")
        print(f"  Initial SPY price: ${initial_price:.2f}")
        print(f"  Shares bought: {shares_bought:.2f}")
        
        return spy_data['portfolio_value'].tolist(), spy_data['Date'].tolist()
    
    except Exception as e:
        print(f"[WARNING] Error downloading SPY: {e}")
        return None, None


def calculate_metrics(portfolio_values, name="Strategy"):
    """Calculate comprehensive performance metrics."""
    
    # Ensure we have valid data
    if portfolio_values is None or len(portfolio_values) < 2:
        print(f"[WARNING] Insufficient data for {name}")
        return {"Strategy": name, "Error": "Insufficient data"}
    
    pv = pd.Series(portfolio_values)
    returns = pv.pct_change().dropna()
    
    # Check if we have enough data points
    n_days = len(returns)
    if n_days == 0:
        print(f"[WARNING] No return data for {name}")
        return {"Strategy": name, "Error": "No return data"}
    
    # Basic metrics
    total_return = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
    
    # Annualized metrics (252 trading days)
    # Use total days in portfolio values, not just returns
    total_days = len(pv) - 1  # -1 because first day is starting point
    if total_days > 0:
        annual_return = ((pv.iloc[-1] / pv.iloc[0]) ** (252 / total_days) - 1) * 100
    else:
        annual_return = 0
    
    # Volatility
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100 if daily_vol > 0 else 0
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate / 252
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Sortino Ratio (only considers downside volatility)
    downside_returns = returns[returns < 0]
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Maximum Drawdown
    cumulative = pv / pv.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win Rate
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    
    # Calmar Ratio (annual return / max drawdown)
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        "Strategy": name,
        "Initial Value ($)": f"{pv.iloc[0]:,.0f}",
        "Final Value ($)": f"{pv.iloc[-1]:,.0f}",
        "Total Return (%)": f"{total_return:.2f}",
        "Annual Return (%)": f"{annual_return:.2f}",
        "Annual Volatility (%)": f"{annual_vol:.2f}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Sortino Ratio": f"{sortino:.3f}",
        "Max Drawdown (%)": f"{max_drawdown:.2f}",
        "Win Rate (%)": f"{win_rate:.2f}",
        "Calmar Ratio": f"{calmar:.3f}",
        "Trading Days": total_days,
    }
    
    return metrics


def create_visualizations(agent_values, agent_dates, spy_values, spy_dates, results_dir):
    """Create and save performance visualizations."""
    print("\n[5/6] Generating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RL Portfolio Agent - Performance Analysis', fontsize=14, fontweight='bold')
    
    # --- Plot 1: Portfolio Value Comparison ---
    ax1 = axes[0, 0]
    ax1.plot(agent_values, label='RL Agent', color='blue', linewidth=2)
    if spy_values is not None:
        # Align SPY values to agent timeline
        min_len = min(len(agent_values), len(spy_values))
        ax1.plot(spy_values[:min_len], label='SPY (Buy & Hold)', color='orange', linewidth=2, linestyle='--')
    ax1.axhline(y=100000, color='gray', linestyle=':', alpha=0.7, label='Initial Investment')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(agent_values))
    
    # --- Plot 2: Drawdown ---
    ax2 = axes[0, 1]
    agent_pv = pd.Series(agent_values)
    cumulative = agent_pv / agent_pv.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown Analysis')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(drawdown))
    
    # --- Plot 3: Daily Returns Distribution ---
    ax3 = axes[1, 0]
    returns = agent_pv.pct_change().dropna() * 100
    ax3.hist(returns, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.3f}%')
    ax3.set_title('Daily Returns Distribution')
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Cumulative Returns Comparison ---
    ax4 = axes[1, 1]
    agent_cum_returns = (agent_pv / agent_pv.iloc[0] - 1) * 100
    ax4.plot(agent_cum_returns, label='RL Agent', color='blue', linewidth=2)
    
    if spy_values is not None:
        spy_pv = pd.Series(spy_values[:len(agent_values)])
        spy_cum_returns = (spy_pv / spy_pv.iloc[0] - 1) * 100
        ax4.plot(spy_cum_returns, label='SPY', color='orange', linewidth=2, linestyle='--')
    
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    ax4.set_title('Cumulative Returns')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, len(agent_cum_returns))
    
    plt.tight_layout()
    
    # Save figure
    plot_path = f"{results_dir}/performance_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved visualization: {plot_path}")
    
    return plot_path


def save_results(agent_metrics, spy_metrics, agent_values, agent_dates, results_dir):
    """Save results to CSV files."""
    print("\n[6/6] Saving results...")
    
    # Save metrics comparison
    metrics_df = pd.DataFrame([agent_metrics, spy_metrics]) if spy_metrics else pd.DataFrame([agent_metrics])
    metrics_path = f"{results_dir}/performance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[OK] Saved metrics: {metrics_path}")
    
    # Save portfolio values
    portfolio_df = pd.DataFrame({
        'date': agent_dates,
        'portfolio_value': agent_values
    })
    portfolio_path = f"{results_dir}/portfolio_values.csv"
    portfolio_df.to_csv(portfolio_path, index=False)
    print(f"[OK] Saved portfolio values: {portfolio_path}")
    
    return metrics_path, portfolio_path


def print_comparison(agent_metrics, spy_metrics):
    """Print a nice comparison table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    metrics_to_show = [
        "Initial Value ($)",
        "Final Value ($)",
        "Total Return (%)",
        "Annual Return (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Win Rate (%)",
    ]
    
    print(f"\n{'Metric':<25} {'RL Agent':>15} {'SPY Benchmark':>15} {'Winner':>10}")
    print("-" * 70)
    
    for metric in metrics_to_show:
        agent_val = agent_metrics.get(metric, "N/A")
        spy_val = spy_metrics.get(metric, "N/A") if spy_metrics else "N/A"
        
        # Determine winner (higher is better, except drawdown)
        winner = ""
        try:
            a = float(agent_val.replace(",", "").replace("%", ""))
            s = float(spy_val.replace(",", "").replace("%", "")) if spy_val != "N/A" else None
            if s is not None:
                if "Drawdown" in metric:
                    winner = "Agent" if a > s else "SPY"  # Less negative is better
                else:
                    winner = "Agent" if a > s else "SPY"
        except:
            pass
        
        print(f"{metric:<25} {agent_val:>15} {spy_val:>15} {winner:>10}")
    
    print("=" * 70)


def main():
    """Main evaluation pipeline."""
    
    create_directories()
    
    # Load data
    trade_df, trade_dates = load_and_prepare_data(DATA_PATH, TRADE_START, TRADE_END)
    
    # Get feature columns
    feature_columns = get_feature_columns(trade_df)
    
    # Create environment
    env, env_kwargs = create_test_environment(trade_df, ENV_CONFIG, feature_columns)
    
    # Load trained model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("[OK] Model loaded")
    
    # Run backtest (pass trade_df and config for fresh environment creation)
    agent_values, agent_dates = run_backtest(model, env, trade_df, ENV_CONFIG, feature_columns)
    
    # Download SPY benchmark
    spy_values, spy_dates = download_benchmark(
        TRADE_START, TRADE_END, ENV_CONFIG["initial_amount"]
    )
    
    # Calculate metrics
    agent_metrics = calculate_metrics(agent_values, "RL Agent")
    spy_metrics = calculate_metrics(spy_values, "SPY (Buy & Hold)") if spy_values else None
    
    # Create visualizations
    plot_path = create_visualizations(
        agent_values, agent_dates,
        spy_values, spy_dates,
        RESULTS_DIR
    )
    
    # Save results
    save_results(agent_metrics, spy_metrics, agent_values, agent_dates, RESULTS_DIR)
    
    # Print comparison
    print_comparison(agent_metrics, spy_metrics)
    
    # Final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to '{RESULTS_DIR}/':")
    print(f"  - performance_analysis.png (visualizations)")
    print(f"  - performance_metrics.csv (detailed metrics)")
    print(f"  - portfolio_values.csv (daily portfolio values)")
    
    # Highlight key achievements
    sharpe = float(agent_metrics["Sharpe Ratio"])
    max_dd = float(agent_metrics["Max Drawdown (%)"].replace(",", ""))
    total_return = float(agent_metrics["Total Return (%)"].replace(",", ""))
    
    print(f"\nKEY RESULTS:")
    print(f"   Total Return: {total_return:.1f}%", end="")
    print(" [OK]" if total_return > 0 else "")
    
    print(f"   Sharpe Ratio: {sharpe:.3f}", end="")
    print(" [OK] (Target: >1.0)" if sharpe > 1.0 else " (Target: >1.0)")
    
    print(f"   Max Drawdown: {max_dd:.1f}%", end="")
    print(" [OK] (Target: >-25%)" if max_dd > -25 else " (Target: >-25%)")
    
    if spy_metrics:
        spy_return = float(spy_metrics["Total Return (%)"].replace(",", ""))
        outperformance = total_return - spy_return
        print(f"   vs SPY: {'+' if outperformance > 0 else ''}{outperformance:.1f}%", end="")
        print(" [OK] (Beat benchmark!)" if outperformance > 0 else "")
    
    print("=" * 70)


if __name__ == "__main__":
    main()