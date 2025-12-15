"""
Hyperparameter Search with Optuna

Uses Optuna to find optimal PPO hyperparameters for the enhanced portfolio model.
Runs shorter training (300k steps) for faster iteration, then validates top configs.

Usage:
    python -m src.experiments.hyperparameter_search --n-trials 25
    python -m src.experiments.hyperparameter_search --n-trials 10 --quick
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.enhanced_portfolio_env import EnhancedPortfolioEnv
from src.data.enhanced_processor import EnhancedDataProcessor, ProcessorConfig


class TrialEvalCallback(BaseCallback):
    """
    Callback for reporting intermediate values to Optuna for pruning.
    """
    def __init__(self, trial: optuna.Trial, eval_env, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.best_sharpe = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Quick evaluation
            sharpe = self._evaluate()
            self.eval_idx += 1
            
            # Report to Optuna
            self.trial.report(sharpe, self.eval_idx)
            
            # Prune if needed
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                
        return True
    
    def _evaluate(self) -> float:
        """Run quick evaluation and return Sharpe ratio."""
        obs, _ = self.eval_env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
            obs, _, done, _, _ = self.eval_env.step(action[0])
        
        stats = self.eval_env.get_portfolio_stats()
        return stats['sharpe_ratio']


def prepare_df_for_env(df: pd.DataFrame) -> pd.DataFrame:
    """Add day index for environment."""
    df = df.copy()
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    dates = sorted(df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    df['day'] = df['date'].map(date_to_day)
    return df.set_index('day')


def create_objective(
    train_df_indexed: pd.DataFrame,
    test_df_indexed: pd.DataFrame,
    feature_info: Dict,
    timesteps: int = 300_000,
    eval_freq: int = 50_000,
) -> callable:
    """
    Create Optuna objective function.
    
    Args:
        train_df_indexed: Training data
        test_df_indexed: Test data for evaluation
        feature_info: Feature information from processor
        timesteps: Training timesteps per trial
        eval_freq: Evaluation frequency for pruning
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximize Sharpe ratio."""
        
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.3, 0.7)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        net_arch_size = trial.suggest_categorical("net_arch_size", [64, 128, 256])
        
        # Ensure batch_size <= n_steps
        if batch_size > n_steps:
            batch_size = n_steps
        
        # Environment config
        env_config = {
            "hmax": 100,
            "initial_amount": 100000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "reward_scaling": 1e-4,
            "sentiment_reward_weight": 0.0,
        }
        
        # Create training environment
        train_env = EnhancedPortfolioEnv(
            df=train_df_indexed,
            stock_dim=feature_info['n_tickers'],
            tech_indicator_list=feature_info['tech_indicators'],
            sentiment_feature_list=feature_info['sentiment_features'],
            include_sentiment=True,
            normalize_obs=False,
            print_verbosity=0,
            **env_config,
        )
        
        # Create evaluation environment
        eval_env = EnhancedPortfolioEnv(
            df=test_df_indexed,
            stock_dim=feature_info['n_tickers'],
            tech_indicator_list=feature_info['tech_indicators'],
            sentiment_feature_list=feature_info['sentiment_features'],
            include_sentiment=True,
            normalize_obs=False,
            print_verbosity=0,
            **env_config,
        )
        
        # Wrap training environment
        train_env_wrapped = DummyVecEnv([lambda: Monitor(train_env)])
        
        # Create model
        policy_kwargs = {
            "net_arch": dict(pi=[net_arch_size, net_arch_size], vf=[net_arch_size, net_arch_size]),
        }
        
        model = PPO(
            "MlpPolicy",
            train_env_wrapped,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0,
        )
        
        # Create pruning callback
        eval_callback = TrialEvalCallback(
            trial=trial,
            eval_env=eval_env,
            eval_freq=eval_freq,
        )
        
        try:
            # Train
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                progress_bar=False,
            )
            
            # Final evaluation
            obs, _ = eval_env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                obs, _, done, _, _ = eval_env.step(action[0])
            
            stats = eval_env.get_portfolio_stats()
            sharpe = stats['sharpe_ratio']
            
            # Store additional metrics
            trial.set_user_attr("total_return", stats['total_return'])
            trial.set_user_attr("max_drawdown", stats['max_drawdown'])
            trial.set_user_attr("total_trades", stats['total_trades'])
            
            return sharpe
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf
    
    return objective


def run_hyperparameter_search(
    n_trials: int = 25,
    timesteps: int = 300_000,
    output_dir: str = "experiments/optuna_study",
    study_name: str = "ppo_enhanced_tuning",
    quick: bool = False,
) -> optuna.Study:
    """
    Run Optuna hyperparameter search.
    
    Args:
        n_trials: Number of trials to run
        timesteps: Training timesteps per trial (reduced for speed)
        output_dir: Directory to save results
        study_name: Name for the Optuna study
        quick: If True, use even fewer timesteps for testing
        
    Returns:
        Optuna study object
    """
    if quick:
        timesteps = 100_000
        eval_freq = 25_000
    else:
        eval_freq = 50_000
    
    print("=" * 70)
    print("HYPERPARAMETER SEARCH WITH OPTUNA")
    print("=" * 70)
    print(f"Trials: {n_trials}")
    print(f"Timesteps per trial: {timesteps:,}")
    print(f"Estimated time: ~{n_trials * timesteps / 300_000 * 10:.0f} minutes")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Load data
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
        train_end="2024-06-30",
        test_start="2024-07-01",
        test_end="2025-11-30",
    )
    
    feature_info = processor.get_feature_info()
    
    train_df_indexed = prepare_df_for_env(train_df)
    test_df_indexed = prepare_df_for_env(test_df)
    
    print(f"Train: {len(train_df)} records, Test: {len(test_df)} records")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create study
    storage = f"sqlite:///{output_path / 'optuna_study.db'}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    
    # Create objective
    objective = create_objective(
        train_df_indexed=train_df_indexed,
        test_df_indexed=test_df_indexed,
        feature_info=feature_info,
        timesteps=timesteps,
        eval_freq=eval_freq,
    )
    
    # Run optimization
    print(f"\nStarting optimization...")
    start_time = datetime.now()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    total_time = datetime.now() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time}")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    print(f"\nBest trial:")
    print(f"  Sharpe Ratio: {study.best_value:.4f}")
    print(f"  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    if study.best_trial.user_attrs:
        print(f"  Additional metrics:")
        for key, value in study.best_trial.user_attrs.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    # Save best hyperparameters
    best_params = {
        "best_sharpe": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "total_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
        "timesteps_per_trial": timesteps,
    }
    
    if study.best_trial.user_attrs:
        best_params["best_metrics"] = study.best_trial.user_attrs
    
    with open(output_path / "best_hyperparameters.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print top 5 trials
    print("\n" + "=" * 70)
    print("TOP 5 TRIALS")
    print("=" * 70)
    
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    
    for idx, row in trials_df.iterrows():
        print(f"\nTrial {int(row['number'])}: Sharpe = {row['value']:.4f}")
        for col in trials_df.columns:
            if col.startswith('params_'):
                param_name = col.replace('params_', '')
                print(f"  {param_name}: {row[col]}")
    
    # Save full results
    trials_df_full = study.trials_dataframe()
    trials_df_full.to_csv(output_path / "all_trials.csv", index=False)
    
    return study


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search with Optuna")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Training timesteps per trial"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/optuna_study",
        help="Output directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer timesteps (for testing)"
    )
    
    args = parser.parse_args()
    
    run_hyperparameter_search(
        n_trials=args.n_trials,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
