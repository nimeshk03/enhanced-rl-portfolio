"""
Ablation Study Runner

Systematically tests different sentiment feature combinations to understand
which features contribute most to model performance.

Usage:
    python -m src.experiments.ablation --config baseline --seed 42
    python -m src.experiments.ablation --config all --seeds 42 123 456
    python -m src.experiments.ablation --run-all
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.enhanced_portfolio_env import EnhancedPortfolioEnv
from src.data.enhanced_processor import EnhancedDataProcessor, ProcessorConfig


def load_ablation_config(config_path: str = "configs/ablation_configs.yaml") -> Dict:
    """Load ablation configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def prepare_df_for_env(df: pd.DataFrame) -> pd.DataFrame:
    """Add day index for environment."""
    df = df.copy()
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    dates = sorted(df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    df['day'] = df['date'].map(date_to_day)
    return df.set_index('day')


def run_ablation_experiment(
    config_name: str,
    seed: int,
    ablation_config: Dict,
    output_dir: str = "experiments/ablation_results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment.
    
    Args:
        config_name: Name of the configuration (baseline, score_only, core_3, all_sentiment)
        seed: Random seed for reproducibility
        ablation_config: Full ablation configuration dict
        output_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dictionary with experiment results
    """
    # Get configuration
    defaults = ablation_config['defaults']
    config = ablation_config['configurations'][config_name]
    
    experiment_name = f"{config_name}_seed{seed}"
    
    if verbose:
        print("=" * 70)
        print(f"ABLATION EXPERIMENT: {experiment_name}")
        print("=" * 70)
        print(f"Config: {config['description']}")
        print(f"Sentiment features: {config['sentiment_features']}")
        print(f"Seed: {seed}")
    
    # Set seed
    set_seed(seed)
    
    # Load and process data
    if verbose:
        print("\nLoading data...")
    
    processor_config = ProcessorConfig(
        normalize_features=True,
        normalization_window=60,
    )
    
    processor = EnhancedDataProcessor(
        price_path='data/processed_data.csv',
        sentiment_path='data/historical_sentiment_complete.csv',
        config=processor_config,
    )
    
    train_df, test_df = processor.get_train_test_split(
        train_end=defaults['train_end'],
        test_start=defaults['test_start'],
        test_end=defaults['test_end'],
    )
    
    feature_info = processor.get_feature_info()
    
    # Prepare dataframes
    train_df_indexed = prepare_df_for_env(train_df)
    test_df_indexed = prepare_df_for_env(test_df)
    
    if verbose:
        print(f"Train: {len(train_df)} records, Test: {len(test_df)} records")
    
    # Create environment with specified sentiment features
    env_config = defaults['env_config']
    
    train_env = EnhancedPortfolioEnv(
        df=train_df_indexed,
        stock_dim=feature_info['n_tickers'],
        tech_indicator_list=feature_info['tech_indicators'],
        sentiment_feature_list=config['sentiment_features'],
        include_sentiment=config['include_sentiment'],
        normalize_obs=False,
        print_verbosity=0,
        **env_config,
    )
    
    if verbose:
        print(f"State space: {train_env.state_space}")
    
    # Create output directory
    exp_output_dir = Path(output_dir) / experiment_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wrap environment
    train_env_wrapped = DummyVecEnv([lambda: Monitor(train_env)])
    
    # Create PPO model
    ppo_config = defaults['ppo_config'].copy()
    ppo_config['seed'] = seed
    
    policy_kwargs = {
        "net_arch": dict(
            pi=defaults['policy_kwargs']['net_arch'],
            vf=defaults['policy_kwargs']['net_arch']
        ),
    }
    
    model = PPO(
        "MlpPolicy",
        train_env_wrapped,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(exp_output_dir / "logs"),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0,
        **ppo_config,
    )
    
    # Train
    start_time = datetime.now()
    if verbose:
        print(f"\nTraining for {defaults['total_timesteps']:,} timesteps...")
    
    model.learn(
        total_timesteps=defaults['total_timesteps'],
        progress_bar=verbose,
    )
    
    training_time = datetime.now() - start_time
    
    # Save model
    model_path = exp_output_dir / f"model_{experiment_name}.zip"
    model.save(str(model_path))
    
    if verbose:
        print(f"\nTraining completed in {training_time}")
        print(f"Model saved to {model_path}")
    
    # Evaluate on test set
    if verbose:
        print("\nEvaluating on test period...")
    
    test_env = EnhancedPortfolioEnv(
        df=test_df_indexed,
        stock_dim=feature_info['n_tickers'],
        tech_indicator_list=feature_info['tech_indicators'],
        sentiment_feature_list=config['sentiment_features'],
        include_sentiment=config['include_sentiment'],
        normalize_obs=False,
        print_verbosity=0,
        **env_config,
    )
    
    obs, info = test_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        action = action[0]
        obs, reward, done, truncated, info = test_env.step(action)
    
    stats = test_env.get_portfolio_stats()
    
    if verbose:
        print(f"\nResults:")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"  Total Return: {stats['total_return']*100:.2f}%")
        print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
        print(f"  Total Trades: {stats['total_trades']}")
    
    # Save results
    results = {
        "experiment_name": experiment_name,
        "config_name": config_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "training_time": str(training_time),
        "total_timesteps": defaults['total_timesteps'],
        "config": {
            "description": config['description'],
            "include_sentiment": config['include_sentiment'],
            "sentiment_features": config['sentiment_features'],
            "env_config": env_config,
            "ppo_config": {k: v for k, v in ppo_config.items() if k != 'seed'},
        },
        "results": {
            "final_value": float(stats['final_value']),
            "total_return": float(stats['total_return']),
            "sharpe_ratio": float(stats['sharpe_ratio']),
            "max_drawdown": float(stats['max_drawdown']),
            "total_trades": int(stats['total_trades']),
            "total_cost": float(stats['total_cost']),
        },
    }
    
    results_path = exp_output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"Results saved to {results_path}")
    
    return results


def run_all_ablations(
    ablation_config: Dict,
    output_dir: str = "experiments/ablation_results",
    configs: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run all ablation experiments.
    
    Args:
        ablation_config: Full ablation configuration
        output_dir: Directory to save results
        configs: List of config names to run (default: all)
        seeds: List of seeds to use (default: from config)
        
    Returns:
        DataFrame with all results
    """
    if configs is None:
        configs = list(ablation_config['configurations'].keys())
    
    if seeds is None:
        seeds = ablation_config['defaults']['seeds']
    
    all_results = []
    total_runs = len(configs) * len(seeds)
    current_run = 0
    
    print(f"\nRunning {total_runs} ablation experiments")
    print(f"Configs: {configs}")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    
    for config_name in configs:
        for seed in seeds:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Running {config_name} with seed {seed}")
            
            try:
                results = run_ablation_experiment(
                    config_name=config_name,
                    seed=seed,
                    ablation_config=ablation_config,
                    output_dir=output_dir,
                    verbose=True,
                )
                all_results.append(results)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            "config": r['config_name'],
            "seed": r['seed'],
            "sharpe_ratio": r['results']['sharpe_ratio'],
            "total_return": r['results']['total_return'],
            "max_drawdown": r['results']['max_drawdown'],
            "total_trades": r['results']['total_trades'],
            "training_time": r['training_time'],
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = Path(output_dir) / "ablation_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    
    # Print aggregated results
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    
    agg = df.groupby('config').agg({
        'sharpe_ratio': ['mean', 'std'],
        'total_return': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
    }).round(4)
    
    print(agg)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--config",
        type=str,
        choices=['baseline', 'score_only', 'core_3', 'all_sentiment', 'all'],
        default='all',
        help="Configuration to run (or 'all' for all configs)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to use (overrides config)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=None,
        help="List of seeds to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/ablation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/ablation_configs.yaml",
        help="Path to ablation config file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    ablation_config = load_ablation_config(args.config_file)
    
    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = ablation_config['defaults']['seeds']
    
    # Run experiments
    if args.config == 'all':
        run_all_ablations(
            ablation_config=ablation_config,
            output_dir=args.output_dir,
            seeds=seeds,
        )
    else:
        for seed in seeds:
            run_ablation_experiment(
                config_name=args.config,
                seed=seed,
                ablation_config=ablation_config,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
