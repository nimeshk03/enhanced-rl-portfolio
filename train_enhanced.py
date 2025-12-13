"""
train_enhanced.py

Train an Enhanced Deep RL agent with sentiment features for portfolio management.
This script uses PPO with the EnhancedPortfolioEnv that includes sentiment data.

Usage:
    # Local training (CPU - slower)
    python train_enhanced.py
    
    # With custom config
    python train_enhanced.py --timesteps 500000 --experiment enhanced_test
    
    # For full training, use the Colab notebook: notebooks/train_enhanced.ipynb

This script:
1. Loads and processes price + sentiment data
2. Creates enhanced environment with sentiment features
3. Trains PPO agent with larger network for more features
4. Evaluates on test period
5. Saves model and experiment results
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# Local imports
from src.env.enhanced_portfolio_env import EnhancedPortfolioEnv
from src.data.enhanced_processor import EnhancedDataProcessor, ProcessorConfig

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
PRICE_DATA_PATH = "./data/processed_data.csv"
SENTIMENT_DATA_PATH = "./data/historical_sentiment_complete.csv"
MODEL_DIR = "./models"
EXPERIMENTS_DIR = "./experiments"

# Time periods
TRAIN_START = "2015-01-01"
TRAIN_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-11-30"

# Environment configuration
ENV_CONFIG = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
    "sentiment_reward_weight": 0.0,  # Can enable for sentiment-aligned rewards
}

# Enhanced PPO configuration (larger network for more features)
ENHANCED_PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,  # Higher entropy for exploration with more features
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# Network architecture (larger for sentiment features)
POLICY_KWARGS = {
    "net_arch": dict(pi=[128, 128], vf=[128, 128]),
}

# Training configuration
DEFAULT_TIMESTEPS = 500_000  # Use 1.5M for full training


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directories(experiment_name: str) -> Dict[str, str]:
    """Create directories for experiment."""
    paths = {
        "model_dir": MODEL_DIR,
        "experiment_dir": os.path.join(EXPERIMENTS_DIR, experiment_name),
        "logs_dir": os.path.join(EXPERIMENTS_DIR, experiment_name, "logs"),
        "checkpoints_dir": os.path.join(EXPERIMENTS_DIR, experiment_name, "checkpoints"),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def load_and_process_data(
    normalize: bool = True,
    normalization_window: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load and process price + sentiment data.
    
    Returns:
        Tuple of (train_df, test_df, feature_info)
    """
    print("\n[1/5] Loading and processing data...")
    
    config = ProcessorConfig(
        normalize_features=normalize,
        normalization_window=normalization_window,
        start_date=TRAIN_START,
    )
    
    processor = EnhancedDataProcessor(
        price_path=PRICE_DATA_PATH,
        sentiment_path=SENTIMENT_DATA_PATH,
        config=config,
    )
    
    # Get train/test split
    train_df, test_df = processor.get_train_test_split(
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    
    feature_info = processor.get_feature_info()
    
    print(f"[OK] Data loaded and processed")
    print(f"  - Train: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  - Test: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")
    print(f"  - Features: {feature_info['n_tech_indicators']} tech + {feature_info['n_sentiment_features']} sentiment")
    
    return train_df, test_df, feature_info


def create_environment(
    df: pd.DataFrame,
    feature_info: Dict[str, Any],
    include_sentiment: bool = True,
    normalize_obs: bool = True,
) -> EnhancedPortfolioEnv:
    """
    Create enhanced portfolio environment.
    
    Args:
        df: Processed DataFrame
        feature_info: Feature information from processor
        include_sentiment: Whether to include sentiment features
        normalize_obs: Whether to normalize observations (recommended for training)
        
    Returns:
        EnhancedPortfolioEnv instance
    """
    # Prepare data with day index
    df = df.copy()
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    dates = sorted(df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    df['day'] = df['date'].map(date_to_day)
    df = df.set_index('day')
    
    # Get feature lists
    tech_indicators = feature_info['tech_indicators']
    sentiment_features = feature_info['sentiment_features'] if include_sentiment else []
    
    env = EnhancedPortfolioEnv(
        df=df,
        stock_dim=feature_info['n_tickers'],
        tech_indicator_list=tech_indicators,
        sentiment_feature_list=sentiment_features,
        include_sentiment=include_sentiment,
        normalize_obs=normalize_obs,
        print_verbosity=0,
        **ENV_CONFIG,
    )
    
    return env


def train_enhanced_agent(
    train_env: EnhancedPortfolioEnv,
    eval_env: Optional[EnhancedPortfolioEnv],
    total_timesteps: int,
    paths: Dict[str, str],
    use_tensorboard: bool = True,
) -> PPO:
    """
    Train PPO agent on enhanced environment.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment (optional)
        total_timesteps: Total training timesteps
        paths: Directory paths
        use_tensorboard: Whether to use TensorBoard logging
        
    Returns:
        Trained PPO model
    """
    print("\n[3/5] Training enhanced PPO agent...")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Learning rate: {ENHANCED_PPO_CONFIG['learning_rate']}")
    print(f"  - Network architecture: {POLICY_KWARGS['net_arch']}")
    print("-" * 70)
    
    # Wrap environments
    train_env_wrapped = DummyVecEnv([lambda: Monitor(train_env)])
    
    # TensorBoard logging
    tb_log_path = None
    if use_tensorboard:
        try:
            import tensorboard
            tb_log_path = paths["logs_dir"]
            print(f"  - TensorBoard logging: {tb_log_path}")
        except ImportError:
            print("  - TensorBoard not installed, logging disabled")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env_wrapped,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=tb_log_path,
        **ENHANCED_PPO_CONFIG,
    )
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=paths["checkpoints_dir"],
        name_prefix="ppo_enhanced",
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (if eval env provided)
    if eval_env is not None:
        eval_env_wrapped = DummyVecEnv([lambda: Monitor(eval_env)])
        eval_callback = EvalCallback(
            eval_env_wrapped,
            best_model_save_path=paths["experiment_dir"],
            log_path=paths["logs_dir"],
            eval_freq=25000,
            n_eval_episodes=1,
            deterministic=True,
        )
        callbacks.append(eval_callback)
    
    callback_list = CallbackList(callbacks) if callbacks else None
    
    # Train
    print("\nStarting training...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )
    
    training_time = datetime.now() - start_time
    print("-" * 70)
    print(f"[OK] Training completed in {training_time}")
    
    return model


def evaluate_agent(
    model: PPO,
    test_env: EnhancedPortfolioEnv,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate trained agent on test environment.
    
    Args:
        model: Trained PPO model
        test_env: Test environment (used directly, not wrapped)
        deterministic: Use deterministic actions
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n[4/5] Evaluating on test period...")
    
    # Run evaluation directly on environment (not wrapped)
    # This ensures we get stats from the actual environment used
    obs, info = test_env.reset()
    done = False
    
    while not done:
        # Model expects batch dimension
        action, _ = model.predict(obs.reshape(1, -1), deterministic=deterministic)
        action = action[0]  # Remove batch dimension
        obs, reward, done, truncated, info = test_env.step(action)
    
    # Get portfolio statistics
    stats = test_env.get_portfolio_stats()
    
    print(f"[OK] Evaluation completed")
    print(f"  - Final value: ${stats['final_value']:,.2f}")
    print(f"  - Total return: {stats['total_return']*100:.2f}%")
    print(f"  - Sharpe ratio: {stats['sharpe_ratio']:.3f}")
    print(f"  - Max drawdown: {stats['max_drawdown']*100:.2f}%")
    
    return stats


def save_experiment_results(
    model: PPO,
    stats: Dict[str, Any],
    feature_info: Dict[str, Any],
    paths: Dict[str, str],
    experiment_name: str,
    total_timesteps: int,
    training_time: Optional[str] = None,
) -> str:
    """
    Save model and experiment results.
    
    Args:
        model: Trained model
        stats: Evaluation statistics
        feature_info: Feature information
        paths: Directory paths
        experiment_name: Name of experiment
        total_timesteps: Training timesteps
        training_time: Training duration string
        
    Returns:
        Path to saved model
    """
    print("\n[5/5] Saving results...")
    
    # Save model
    model_path = os.path.join(paths["model_dir"], f"ppo_{experiment_name}")
    model.save(model_path)
    print(f"  - Model saved: {model_path}.zip")
    
    # Save experiment config and results
    experiment_results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "training_time": training_time,
        "total_timesteps": total_timesteps,
        "config": {
            "env_config": ENV_CONFIG,
            "ppo_config": ENHANCED_PPO_CONFIG,
            "policy_kwargs": POLICY_KWARGS,
            "train_period": f"{TRAIN_START} to {TRAIN_END}",
            "test_period": f"{TEST_START} to {TEST_END}",
        },
        "feature_info": feature_info,
        "results": {
            "final_value": float(stats['final_value']),
            "total_return": float(stats['total_return']),
            "sharpe_ratio": float(stats['sharpe_ratio']),
            "max_drawdown": float(stats['max_drawdown']),
            "total_trades": int(stats['total_trades']),
            "total_cost": float(stats['total_cost']),
        },
    }
    
    results_path = os.path.join(paths["experiment_dir"], "results.json")
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"  - Results saved: {results_path}")
    
    # Save to experiments summary
    summary_path = os.path.join(EXPERIMENTS_DIR, "experiments_summary.csv")
    summary_row = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "timesteps": total_timesteps,
        "sharpe_ratio": stats['sharpe_ratio'],
        "total_return": stats['total_return'],
        "max_drawdown": stats['max_drawdown'],
        "n_features": feature_info['total_features'],
    }
    
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    
    summary_df.to_csv(summary_path, index=False)
    print(f"  - Summary updated: {summary_path}")
    
    return model_path


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train Enhanced RL Portfolio Agent")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--experiment", type=str, default="enhanced_v1",
                        help="Experiment name")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Train without sentiment features (baseline)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip feature normalization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENHANCED RL PORTFOLIO MANAGER - TRAINING")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Include sentiment: {not args.no_sentiment}")
    
    # Set seed
    np.random.seed(args.seed)
    
    # Create directories
    paths = create_directories(args.experiment)
    
    # Load data
    train_df, test_df, feature_info = load_and_process_data(
        normalize=not args.no_normalize,
    )
    
    # Create environments
    print("\n[2/5] Creating environments...")
    include_sentiment = not args.no_sentiment
    
    train_env = create_environment(train_df, feature_info, include_sentiment)
    test_env = create_environment(test_df, feature_info, include_sentiment)
    
    print(f"[OK] Environments created")
    print(f"  - State space: {train_env.state_space}")
    print(f"  - Action space: {train_env.action_space.shape[0]} stocks")
    
    # Train
    start_time = datetime.now()
    model = train_enhanced_agent(
        train_env=train_env,
        eval_env=None,  # Skip eval callback for faster training
        total_timesteps=args.timesteps,
        paths=paths,
    )
    training_time = str(datetime.now() - start_time)
    
    # Evaluate
    stats = evaluate_agent(model, test_env)
    
    # Save
    model_path = save_experiment_results(
        model=model,
        stats=stats,
        feature_info=feature_info,
        paths=paths,
        experiment_name=args.experiment,
        total_timesteps=args.timesteps,
        training_time=training_time,
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model: {model_path}.zip")
    print(f"Results: {paths['experiment_dir']}/results.json")
    print(f"\nPerformance:")
    print(f"  - Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
    print(f"  - Total Return: {stats['total_return']*100:.2f}%")
    print(f"  - Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"\nNext steps:")
    print(f"  1. Compare with baseline: python train_enhanced.py --no-sentiment --experiment baseline_v1")
    print(f"  2. View TensorBoard: tensorboard --logdir {paths['logs_dir']}")
    print(f"  3. For full training (1.5M steps), use: notebooks/train_enhanced.ipynb on Colab")
    print("=" * 70)


if __name__ == "__main__":
    main()
