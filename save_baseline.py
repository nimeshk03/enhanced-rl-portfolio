"""save_baseline.py

Save the baseline experiment results that we already ran.
This creates the proper experiment structure for comparison.

Run once to establish baseline, then use run_experiment.py for new experiments.
"""

import os
import json
import pandas as pd
from datetime import datetime

# Create baseline experiment directory
EXPERIMENTS_DIR = "./experiments"
BASELINE_DIR = f"{EXPERIMENTS_DIR}/baseline_200k"

os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(f"{BASELINE_DIR}/models", exist_ok=True)

# Copy existing model
import shutil
if os.path.exists("./models/ppo_portfolio_agent.zip"):
    shutil.copy("./models/ppo_portfolio_agent.zip", f"{BASELINE_DIR}/models/trained_agent.zip")
    print("[OK] Copied trained model")

# Copy existing results
if os.path.exists("./results/portfolio_values.csv"):
    shutil.copy("./results/portfolio_values.csv", f"{BASELINE_DIR}/portfolio_values.csv")
    print("[OK] Copied portfolio values")

if os.path.exists("./results/performance_analysis.png"):
    shutil.copy("./results/performance_analysis.png", f"{BASELINE_DIR}/performance_analysis.png")
    print("[OK] Copied performance chart")

# Save configuration
config = {
    "experiment_name": "baseline_200k",
    "description": "Baseline PPO agent with 200k timesteps - initial experiment",
    "timestamp": datetime.now().isoformat(),
    "training_time_seconds": 332.5,  # ~5 min 32 sec from original run
    "algorithm": "PPO",
    "total_timesteps": 200000,
    "seed": 42,
    "ppo_config": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    "train_period": "2019-01-01 to 2023-12-31",
    "test_period": "2024-01-01 to 2025-10-30",
}

with open(f"{BASELINE_DIR}/config.json", 'w') as f:
    json.dump(config, f, indent=2)
print("[OK] Saved configuration")

# Save metrics (from our evaluation results)
metrics = {
    "Strategy": "RL Agent",
    "Initial Value": 100000,
    "Final Value": 148285,
    "Total Return (%)": 48.28,
    "Annual Return (%)": 24.21,
    "Annual Volatility (%)": 23.5,  # Approximate
    "Sharpe Ratio": 1.016,
    "Sortino Ratio": 1.409,
    "Max Drawdown (%)": -26.15,
    "Win Rate (%)": 57.42,
    "Trading Days": 458,
}

spy_metrics = {
    "Strategy": "SPY",
    "Initial Value": 100000,
    "Final Value": 148589,
    "Total Return (%)": 48.59,
    "Annual Return (%)": 24.35,
    "Annual Volatility (%)": 18.5,  # Approximate
    "Sharpe Ratio": 1.266,
    "Sortino Ratio": 1.598,
    "Max Drawdown (%)": -18.76,
    "Win Rate (%)": 58.30,
    "Trading Days": 458,
}

metrics_df = pd.DataFrame([metrics, spy_metrics])
metrics_df.to_csv(f"{BASELINE_DIR}/metrics.csv", index=False)
print("[OK] Saved metrics")

# Create/update experiments summary
summary_path = f"{EXPERIMENTS_DIR}/experiments_summary.csv"

summary_row = {
    "Experiment": "baseline_200k",
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "Algorithm": "PPO",
    "Timesteps": 200000,
    "Learning Rate": 0.0003,
    "Total Return (%)": 48.28,
    "Sharpe Ratio": 1.016,
    "Max Drawdown (%)": -26.15,
    "vs SPY (%)": -0.31,
    "Training Time (min)": 5.5,
    "Description": "Baseline - initial PPO agent",
}

summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(summary_path, index=False)
print(f"[OK] Created experiments summary: {summary_path}")

print("\n" + "=" * 60)
print("BASELINE SAVED")
print("=" * 60)
print(f"Location: {BASELINE_DIR}/")
print("\nBaseline Results:")
print(f"  Total Return: 48.28%")
print(f"  Sharpe Ratio: 1.016")
print(f"  Max Drawdown: -26.15%")
print(f"  vs SPY: -0.31%")
print("\nNext: Run experiments with run_experiment.py")
print("=" * 60)