"""
Train final production model locally with progress bar.
Uses CPU (faster for PPO MLP policy).

Usage:
    pip install stable-baselines3[extra] gymnasium pandas numpy
    python train_final_local.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.enhanced_portfolio_env import EnhancedPortfolioEnv
from src.data.enhanced_processor import EnhancedDataProcessor

# ============ CONFIGURATION ============
EXPERIMENT_NAME = "final_production_model"
TOTAL_TIMESTEPS = 1_500_000

# Optuna-tuned hyperparameters (Trial 21, Sharpe 2.28)
PPO_CONFIG = {
    "learning_rate": 0.000812,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.992,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0024,
    "vf_coef": 0.428,
    "max_grad_norm": 0.769,
}

POLICY_KWARGS = {"net_arch": [256, 256]}

TECH_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'vix', 'turbulence'
]


def prepare_env_data(df):
    """Add day index required by EnhancedPortfolioEnv."""
    df = df.copy()
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    dates = sorted(df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    df['day'] = df['date'].map(date_to_day)
    return df.set_index('day')


def make_env(data, mode="train"):
    return EnhancedPortfolioEnv(
        df=data,
        stock_dim=10,
        hmax=100,
        initial_amount=100000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        tech_indicator_list=TECH_INDICATORS,
        sentiment_feature_list=[],
        include_sentiment=False,
        normalize_obs=True,
        mode=mode,
    )


def main():
    print("=" * 60)
    print("FINAL PRODUCTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    processor = EnhancedDataProcessor(
        price_path="data/processed_data.csv",
        sentiment_path="data/historical_sentiment_complete.csv"
    )
    train_df, test_df = processor.get_train_test_split(
        train_end="2024-06-30",
        test_start="2024-07-01"
    )
    
    train_data = prepare_env_data(train_df)
    test_data = prepare_env_data(test_df)
    
    print(f"Train: {len(train_data)} records, {train_data.index.nunique()} days")
    print(f"Test: {len(test_data)} records, {test_data.index.nunique()} days")
    
    # Create environment
    train_env = DummyVecEnv([lambda: make_env(train_data, "train")])
    
    # Setup experiment directory
    exp_dir = f"experiments/{EXPERIMENT_NAME}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create model
    print(f"\nTraining {TOTAL_TIMESTEPS:,} timesteps on CPU...")
    print("Estimated time: ~35-45 minutes\n")
    
    model = PPO(
        "MlpPolicy",
        train_env,
        **PPO_CONFIG,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        device="cpu",
    )
    
    # Train with progress bar
    start_time = datetime.now()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    training_time = datetime.now() - start_time
    
    # Save model
    model_path = f"{exp_dir}/ppo_final_production.zip"
    model.save(model_path)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {training_time}")
    print(f"Model: {model_path}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_env = make_env(test_data, "test")
    
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
    
    stats = test_env.get_portfolio_stats()
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Sharpe Ratio:  {stats['sharpe_ratio']:.3f}")
    print(f"Total Return:  {stats['total_return']*100:.2f}%")
    print(f"Max Drawdown:  {stats['max_drawdown']*100:.2f}%")
    print(f"Total Trades:  {stats['total_trades']}")
    print(f"Final Value:   ${stats['final_value']:,.2f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "training_time": str(training_time),
        "config": {
            "include_sentiment": False,
            "timesteps": TOTAL_TIMESTEPS,
            "learning_rate": PPO_CONFIG["learning_rate"],
            "batch_size": PPO_CONFIG["batch_size"],
            "net_arch": POLICY_KWARGS["net_arch"],
        },
        "metrics": {
            "sharpe_ratio": float(stats['sharpe_ratio']),
            "total_return": float(stats['total_return']),
            "max_drawdown": float(stats['max_drawdown']),
            "total_trades": int(stats['total_trades']),
            "final_value": float(stats['final_value']),
        }
    }
    
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Copy to models folder
    os.makedirs("models", exist_ok=True)
    import shutil
    shutil.copy(model_path, "models/ppo_final_production.zip")
    
    print(f"\nResults saved to {exp_dir}/results.json")
    print(f"Model copied to models/ppo_final_production.zip")
    print("\nUpdate config.py:")
    print('MODEL_PATH = "./models/ppo_final_production.zip"')


if __name__ == "__main__":
    main()
