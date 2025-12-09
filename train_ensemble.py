"""train_ensemble.py

Train an ensemble of RL agents for more robust portfolio management.
This script trains multiple PPO agents with different random seeds and
combines their predictions for more stable trading decisions.

Why Ensemble?
- Reduces variance in predictions
- More robust to market conditions
- Diversifies "model risk" (different agents learn different patterns)

Usage:
    python train_ensemble.py --n_agents 5 --timesteps 500000

This script:
1. Trains N agents with different random seeds
2. Saves each agent separately
3. Creates an ensemble predictor that averages actions
4. Backtests the ensemble vs individual agents vs SPY
5. Saves comprehensive results
"""

import os
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf

print("=" * 70)
print("ENSEMBLE RL PORTFOLIO MANAGER")
print("=" * 70)

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

# Environment config
ENV_CONFIG = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
}

# Best PPO config from experiments (high entropy)
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,  # High entropy - our best finding!
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ensemble of RL agents')
    parser.add_argument('--n_agents', type=int, default=5,
                        help='Number of agents in ensemble')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Training timesteps per agent')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1011',
                        help='Comma-separated random seeds')
    return parser.parse_args()


def load_and_prepare_data(data_path, start_date, end_date):
    """Load and prepare data with day index."""
    df = pd.read_csv(data_path)
    df['date'] = df['date'].astype(str)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    dates = sorted(filtered_df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    filtered_df['day'] = filtered_df['date'].map(date_to_day)
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
        "print_verbosity": 1000,
        "day": 0,
        "initial": True,
    }
    
    return env_kwargs


def train_single_agent(train_df, env_kwargs, ppo_config, timesteps, seed, model_path):
    """Train a single PPO agent."""
    env = StockTradingEnv(df=train_df, **env_kwargs)
    env_train = DummyVecEnv([lambda: env])
    
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
        verbose=0,  # Quiet training
    )
    
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    
    return model


def run_single_agent_backtest(model, trade_df, env_kwargs):
    """Run backtest for a single agent."""
    env = StockTradingEnv(df=trade_df, **env_kwargs)
    obs, _ = env.reset()
    
    done = False
    while not done:
        obs_array = np.array(obs).reshape(1, -1)
        action, _ = model.predict(obs_array, deterministic=True)
        obs, reward, done, truncated, info = env.step(action[0])
    
    return env.asset_memory, env.date_memory


def run_ensemble_backtest(models, trade_df, env_kwargs):
    """
    Run backtest with ensemble of agents.
    
    The ensemble averages the actions from all agents at each timestep,
    resulting in more diversified and stable trading decisions.
    """
    env = StockTradingEnv(df=trade_df, **env_kwargs)
    obs, _ = env.reset()
    
    done = False
    actions_history = []
    
    while not done:
        obs_array = np.array(obs).reshape(1, -1)
        
        # Get predictions from all agents
        all_actions = []
        for model in models:
            action, _ = model.predict(obs_array, deterministic=True)
            all_actions.append(action[0])
        
        # Average the actions (ensemble prediction)
        ensemble_action = np.mean(all_actions, axis=0)
        actions_history.append({
            'individual': all_actions,
            'ensemble': ensemble_action
        })
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
    
    return env.asset_memory, env.date_memory, actions_history


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
        "Total Return (%)": round(total_return, 2),
        "Annual Return (%)": round(annual_return, 2),
        "Annual Volatility (%)": round(annual_vol, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Trading Days": total_days,
    }


def create_ensemble_visualization(ensemble_values, individual_values, spy_values, 
                                  seeds, exp_dir):
    """Create visualization comparing ensemble vs individual agents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ensemble vs Individual Agents Performance', fontsize=14, fontweight='bold')
    
    # Plot 1: All agents comparison
    ax1 = axes[0, 0]
    for i, (values, seed) in enumerate(zip(individual_values, seeds)):
        ax1.plot(values, alpha=0.5, linewidth=1, label=f'Agent {seed}')
    ax1.plot(ensemble_values, color='blue', linewidth=2.5, label='Ensemble')
    if spy_values:
        min_len = min(len(ensemble_values), len(spy_values))
        ax1.plot(spy_values[:min_len], color='orange', linewidth=2, linestyle='--', label='SPY')
    ax1.axhline(y=100000, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('Portfolio Values: Ensemble vs Individual Agents')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final returns comparison
    ax2 = axes[0, 1]
    final_returns = [(v[-1] / v[0] - 1) * 100 for v in individual_values]
    ensemble_return = (ensemble_values[-1] / ensemble_values[0] - 1) * 100
    spy_return = (spy_values[-1] / spy_values[0] - 1) * 100 if spy_values else 0
    
    x_labels = [f'Agent\n{s}' for s in seeds] + ['Ensemble', 'SPY']
    returns = final_returns + [ensemble_return, spy_return]
    colors = ['lightblue'] * len(seeds) + ['blue', 'orange']
    
    bars = ax2.bar(x_labels, returns, color=colors, edgecolor='black')
    ax2.axhline(y=spy_return, color='orange', linestyle='--', alpha=0.7)
    ax2.set_title('Total Returns Comparison')
    ax2.set_ylabel('Total Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ret:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Drawdown comparison
    ax3 = axes[1, 0]
    
    # Ensemble drawdown
    pv = pd.Series(ensemble_values)
    cumulative = pv / pv.iloc[0]
    running_max = cumulative.cummax()
    ensemble_dd = (cumulative - running_max) / running_max * 100
    ax3.plot(ensemble_dd, color='blue', linewidth=2, label='Ensemble')
    
    # Individual agent drawdowns (lighter)
    for i, values in enumerate(individual_values):
        pv = pd.Series(values)
        cumulative = pv / pv.iloc[0]
        running_max = cumulative.cummax()
        dd = (cumulative - running_max) / running_max * 100
        ax3.plot(dd, alpha=0.3, linewidth=1)
    
    ax3.fill_between(range(len(ensemble_dd)), ensemble_dd, 0, color='blue', alpha=0.2)
    ax3.set_title('Drawdown: Ensemble vs Individual Agents')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk-Return scatter
    ax4 = axes[1, 1]
    
    for i, (values, seed) in enumerate(zip(individual_values, seeds)):
        metrics = calculate_metrics(values, f"Agent {seed}")
        ax4.scatter(abs(metrics["Max Drawdown (%)"]), metrics["Total Return (%)"],
                   s=100, alpha=0.6, label=f'Agent {seed}')
    
    ens_metrics = calculate_metrics(ensemble_values, "Ensemble")
    ax4.scatter(abs(ens_metrics["Max Drawdown (%)"]), ens_metrics["Total Return (%)"],
               s=200, color='blue', marker='*', label='Ensemble', zorder=5)
    
    if spy_values:
        spy_metrics = calculate_metrics(spy_values, "SPY")
        ax4.scatter(abs(spy_metrics["Max Drawdown (%)"]), spy_metrics["Total Return (%)"],
                   s=200, color='orange', marker='^', label='SPY', zorder=5)
    
    ax4.set_title('Risk-Return Profile')
    ax4.set_xlabel('Max Drawdown (%) - Lower is Better')
    ax4.set_ylabel('Total Return (%) - Higher is Better')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{exp_dir}/ensemble_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    args = parse_arguments()
    seeds = [int(s) for s in args.seeds.split(',')][:args.n_agents]
    
    print(f"\nConfiguration:")
    print(f"  Number of agents: {len(seeds)}")
    print(f"  Seeds: {seeds}")
    print(f"  Timesteps per agent: {args.timesteps:,}")
    print(f"  Total training steps: {args.timesteps * len(seeds):,}")
    
    # Create experiment directory
    exp_dir = f"{EXPERIMENTS_DIR}/ensemble_{len(seeds)}agents_{args.timesteps//1000}k"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_df, _ = load_and_prepare_data(DATA_PATH, TRAIN_START, TRAIN_END)
    trade_df, _ = load_and_prepare_data(DATA_PATH, TRADE_START, TRADE_END)
    feature_columns = get_feature_columns(train_df)
    env_kwargs = create_environment(train_df, ENV_CONFIG, feature_columns)
    print(f"[OK] Data loaded")
    
    # Train ensemble
    print(f"\n[2/5] Training {len(seeds)} agents...")
    print("-" * 70)
    
    models = []
    training_times = []
    
    for i, seed in enumerate(seeds):
        print(f"  Training Agent {i+1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
        start_time = datetime.now()
        
        model_path = f"{exp_dir}/models/agent_seed_{seed}"
        model = train_single_agent(
            train_df, env_kwargs, PPO_CONFIG, 
            args.timesteps, seed, model_path
        )
        models.append(model)
        
        elapsed = datetime.now() - start_time
        training_times.append(elapsed)
        print(f"[OK] ({elapsed})")
    
    total_training_time = sum(training_times, datetime.now() - datetime.now())
    print("-" * 70)
    print(f"[OK] All agents trained in {sum(t.total_seconds() for t in training_times)/60:.1f} minutes")
    
    # Run individual backtests
    print(f"\n[3/5] Running backtests...")
    
    individual_results = []
    individual_values = []
    
    for i, (model, seed) in enumerate(zip(models, seeds)):
        values, dates = run_single_agent_backtest(model, trade_df, env_kwargs)
        individual_values.append(values)
        metrics = calculate_metrics(values, f"Agent_{seed}")
        individual_results.append(metrics)
        print(f"  Agent {seed}: Return={metrics['Total Return (%)']:.1f}%, "
              f"Sharpe={metrics['Sharpe Ratio']:.3f}, MaxDD={metrics['Max Drawdown (%)']:.1f}%")
    
    # Run ensemble backtest
    print(f"\n  Running ensemble backtest...")
    
    # Reset env_kwargs for trade data
    env_kwargs = create_environment(trade_df, ENV_CONFIG, feature_columns)
    ensemble_values, ensemble_dates, actions_history = run_ensemble_backtest(
        models, trade_df, env_kwargs
    )
    ensemble_metrics = calculate_metrics(ensemble_values, "Ensemble")
    
    print(f"  Ensemble: Return={ensemble_metrics['Total Return (%)']:.1f}%, "
          f"Sharpe={ensemble_metrics['Sharpe Ratio']:.3f}, MaxDD={ensemble_metrics['Max Drawdown (%)']:.1f}%")
    
    # Download SPY benchmark
    print(f"\n[4/5] Downloading SPY benchmark...")
    spy_values, spy_dates = download_spy_benchmark(TRADE_START, TRADE_END, ENV_CONFIG["initial_amount"])
    spy_metrics = calculate_metrics(spy_values, "SPY") if spy_values else None
    if spy_values:
        print(f"[OK] SPY: Return={spy_metrics['Total Return (%)']:.1f}%")
    
    # Create visualizations and save results
    print(f"\n[5/5] Saving results...")
    
    # Create visualization
    plot_path = create_ensemble_visualization(
        ensemble_values, individual_values, spy_values, seeds, exp_dir
    )
    print(f"[OK] Saved visualization: {plot_path}")
    
    # Save all metrics
    all_metrics = individual_results + [ensemble_metrics]
    if spy_metrics:
        all_metrics.append(spy_metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{exp_dir}/metrics.csv", index=False)
    
    # Save configuration
    config = {
        "experiment_type": "ensemble",
        "n_agents": len(seeds),
        "seeds": seeds,
        "timesteps_per_agent": args.timesteps,
        "total_timesteps": args.timesteps * len(seeds),
        "ppo_config": PPO_CONFIG,
        "training_time_minutes": sum(t.total_seconds() for t in training_times) / 60,
    }
    
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save ensemble portfolio values
    portfolio_df = pd.DataFrame({
        'ensemble': ensemble_values,
        **{f'agent_{seed}': values for seed, values in zip(seeds, individual_values)}
    })
    portfolio_df.to_csv(f"{exp_dir}/portfolio_values.csv", index=False)
    
    # Update experiments summary
    spy_return = spy_metrics['Total Return (%)'] if spy_metrics else 48.59  # Use known value
    
    summary_row = {
        "Experiment": f"ensemble_{len(seeds)}agents",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Algorithm": f"PPO Ensemble ({len(seeds)} agents)",
        "Timesteps": args.timesteps * len(seeds),
        "Learning Rate": PPO_CONFIG["learning_rate"],
        "Total Return (%)": ensemble_metrics["Total Return (%)"],
        "Sharpe Ratio": ensemble_metrics["Sharpe Ratio"],
        "Max Drawdown (%)": ensemble_metrics["Max Drawdown (%)"],
        "vs SPY (%)": round(ensemble_metrics["Total Return (%)"] - spy_return, 2),
        "Training Time (min)": round(sum(t.total_seconds() for t in training_times) / 60, 1),
        "Description": f"Ensemble of {len(seeds)} PPO agents with high entropy",
    }
    
    summary_path = f"{EXPERIMENTS_DIR}/experiments_summary.csv"
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_path, index=False)
    
    # Print final results
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)
    
    print(f"\n{'Strategy':<15} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'vs SPY':>10}")
    print("-" * 70)
    
    for metrics in individual_results:
        name = metrics['Strategy']
        vs_spy = metrics['Total Return (%)'] - spy_return
        print(f"{name:<15} {metrics['Total Return (%)']:>9.1f}% {metrics['Sharpe Ratio']:>10.3f} "
              f"{metrics['Max Drawdown (%)']:>9.1f}% {vs_spy:>+9.1f}%")
    
    print("-" * 70)
    vs_spy = ensemble_metrics['Total Return (%)'] - spy_return
    print(f"{'ENSEMBLE':<15} {ensemble_metrics['Total Return (%)']:>9.1f}% {ensemble_metrics['Sharpe Ratio']:>10.3f} "
          f"{ensemble_metrics['Max Drawdown (%)']:>9.1f}% {vs_spy:>+9.1f}%")
    
    if spy_metrics:
        print(f"{'SPY':<15} {spy_metrics['Total Return (%)']:>9.1f}% {spy_metrics['Sharpe Ratio']:>10.3f} "
              f"{spy_metrics['Max Drawdown (%)']:>9.1f}% {0:>+9.1f}%")
    
    print("=" * 70)
    
    # Highlight ensemble benefits
    avg_individual_return = np.mean([m['Total Return (%)'] for m in individual_results])
    avg_individual_dd = np.mean([m['Max Drawdown (%)'] for m in individual_results])
    
    print(f"\nENSEMBLE BENEFITS:")
    print(f"   Avg Individual Return: {avg_individual_return:.1f}%")
    print(f"   Ensemble Return: {ensemble_metrics['Total Return (%)']:.1f}%")
    print(f"   Avg Individual MaxDD: {avg_individual_dd:.1f}%")
    print(f"   Ensemble MaxDD: {ensemble_metrics['Max Drawdown (%)']:.1f}%")
    
    print(f"\n[OK] All results saved to: {exp_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()