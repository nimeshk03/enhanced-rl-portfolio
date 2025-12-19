# Deep Reinforcement Learning Portfolio Manager

A sophisticated trading agent that learns optimal portfolio allocation strategies using Deep Reinforcement Learning (PPO/A2C), technical indicators, and market sentiment features.

## Project Overview

This project demonstrates the integration of:
- **Reinforcement Learning**: PPO and A2C algorithms for portfolio optimization
- **Technical Analysis**: 8+ indicators (MACD, RSI, Bollinger Bands, etc.)
- **Risk Management**: VIX volatility index and turbulence indicators
- **Real Market Data**: Yahoo Finance historical data for 10 diverse assets

## Dataset

**Tickers** (10 assets for diversification):
- **Tech**: AAPL, MSFT, GOOGL, AMZN, NVDA
- **Financials**: JPM, BAC
- **Safe Havens**: GLD (Gold), TLT (Bonds), SPY (S&P 500)

**Date Range**: 2015-01-02 to 2025-11-28 (11 years, includes COVID crash)

**Features** (17 total):
- Base OHLCV data (7 columns)
- Technical indicators (8): MACD, Bollinger Bands, RSI-30, CCI-30, DX-30, SMA-30, SMA-60
- Market stress indicators (2): VIX, Turbulence Index

**Total Dataset**: 27,440 rows × 17 columns

## Quick Start

### Prerequisites
- Docker installed
- ~500MB disk space
- Internet connection for data download
- (Optional) Alpaca API account for paper trading

### Setup & Data Pipeline

```bash
# 1. Clone the repository
git clone https://github.com/nimeshk03/rl-portfolio-manager.git
cd rl-portfolio-manager

# 2. Build Docker image
# Linux/macOS:
sudo docker build -t rl_portfolio .
# Windows (PowerShell):
docker build -t rl_portfolio .
# Windows (CMD):
docker build -t rl_portfolio .

# 3. Download market data
# Linux/macOS:
sudo docker run --rm --network host -v $(pwd):/app rl_portfolio python download_data.py
# Windows (PowerShell):
docker run --rm --network host -v ${PWD}:/app rl_portfolio python download_data.py
# Windows (CMD):
docker run --rm --network host -v %cd%:/app rl_portfolio python download_data.py

# 4. Download VIX volatility data
# Linux/macOS:
sudo docker run --rm --network host -v $(pwd):/app rl_portfolio python download_vix.py
# Windows (PowerShell):
docker run --rm --network host -v ${PWD}:/app rl_portfolio python download_vix.py
# Windows (CMD):
docker run --rm --network host -v %cd%:/app rl_portfolio python download_vix.py

# 5. Process data and add technical indicators
# Linux/macOS:
sudo docker run --rm --network host -v $(pwd):/app rl_portfolio python preprocess_data.py
# Windows (PowerShell):
docker run --rm --network host -v ${PWD}:/app rl_portfolio python preprocess_data.py
# Windows (CMD):
docker run --rm --network host -v %cd%:/app rl_portfolio python preprocess_data.py
```

### Expected Output
```
./data/raw_data.csv       - 17,170 rows of OHLCV data
./data/vix_data.csv       - 1,717 rows of VIX data
./data/processed_data.csv - 17,170 rows × 17 columns (ready for RL)
```

### Training & Evaluation Pipeline

Once data is prepared, train and evaluate the RL agent:

```bash
# 6. Train PPO agent (takes ~5-10 min depending on timesteps)
# Linux/macOS:
sudo docker run --rm -v $(pwd):/app rl_portfolio python train_agent.py
# Windows (PowerShell):
docker run --rm -v ${PWD}:/app rl_portfolio python train_agent.py
# Windows (CMD):
docker run --rm -v %cd%:/app rl_portfolio python train_agent.py

# 7. Evaluate agent and compare against SPY benchmark
# Linux/macOS:
sudo docker run --rm -v $(pwd):/app rl_portfolio python evaluate_agent.py
# Windows (PowerShell):
docker run --rm -v ${PWD}:/app rl_portfolio python evaluate_agent.py
# Windows (CMD):
docker run --rm -v %cd%:/app rl_portfolio python evaluate_agent.py

# 8. Save results to experiment tracking log
# Linux/macOS:
sudo docker run --rm -v $(pwd):/app rl_portfolio python save_baseline.py
# Windows (PowerShell):
docker run --rm -v ${PWD}:/app rl_portfolio python save_baseline.py
# Windows (CMD):
docker run --rm -v %cd%:/app rl_portfolio python save_baseline.py

# OR: Run comprehensive experiments with custom hyperparameters
# Linux/macOS:
sudo docker run --rm -v $(pwd):/app rl_portfolio python run_experiment.py \
    --name my_experiment \
    --description "Testing custom hyperparameters" \
    --algorithm PPO \
    --timesteps 500000 \
    --lr 0.0003 \
    --ent_coef 0.05

# Windows (PowerShell - use backtick ` for line continuation):
docker run --rm -v ${PWD}:/app rl_portfolio python run_experiment.py `
    --name my_experiment `
    --description "Testing custom hyperparameters" `
    --algorithm PPO `
    --timesteps 500000 `
    --lr 0.0003 `
    --ent_coef 0.05

# Windows (CMD - put everything on one line):
docker run --rm -v %cd%:/app rl_portfolio python run_experiment.py --name my_experiment --description "Testing custom hyperparameters" --algorithm PPO --timesteps 500000 --lr 0.0003 --ent_coef 0.05
```

**Output Files**:
- `models/ppo_portfolio_agent.zip` - Trained model weights
- `experiments/{experiment_name}/` - Complete experiment results
  - `config.json` - All hyperparameters and settings
  - `metrics.csv` - Performance metrics vs SPY
  - `portfolio_values.csv` - Daily portfolio values
  - `performance_analysis.png` - Visualizations
  - `models/trained_agent.zip` - Trained model
- `experiments/experiments_summary.csv` - Master tracking log

## Paper Trading & Deployment

### Setup Alpaca Paper Trading

1. **Get API Credentials** (free):
   - Sign up at [alpaca.markets](https://alpaca.markets/)
   - Navigate to Paper Trading dashboard
   - Generate API keys (free paper trading account)

2. **Configure the Application**:
```bash
# Copy the example config
cp config_example.py config.py

# Edit config.py with your credentials
nano config.py  # or use any editor

# Add your API keys:
ALPACA_API_KEY = "YOUR_ACTUAL_KEY"
ALPACA_SECRET_KEY = "YOUR_ACTUAL_SECRET"
```

3. **Install Additional Dependencies**:
```bash
# Alpaca API and Streamlit for dashboard
pip install alpaca-trade-api streamlit plotly
```

### Running Paper Trading

The RL agent can trade automatically using your trained model:

```bash
# Check account status
python paper_trading.py --mode status

# Execute a single trading cycle (manual)
python paper_trading.py --mode single

# Run continuously during market hours (automated)
python paper_trading.py --mode continuous --interval 60

# Or with Docker:
# Linux/macOS:
sudo docker run --rm -v $(pwd):/app rl_portfolio python paper_trading.py --mode single
# Windows (PowerShell):
docker run --rm -v ${PWD}:/app rl_portfolio python paper_trading.py --mode single
# Windows (CMD):
docker run --rm -v %cd%:/app rl_portfolio python paper_trading.py --mode single
```

**What it does**:
- Fetches real-time market data from Alpaca
- Calculates technical indicators (MACD, RSI, Bollinger Bands, etc.)
- Uses your trained RL agent to make decisions
- Executes trades automatically via Alpaca API
- Logs all activity and performance
- Respects market hours (only trades when market is open)

### Live Dashboard

Monitor your paper trading in real-time with Streamlit:

```bash
# Launch the dashboard
streamlit run dashboard.py

# Opens in browser at http://localhost:8501
```

**Dashboard Features**:
- Real-time portfolio value and P&L
- Current positions with unrealized gains/losses
- Portfolio history charts
- Trade history and execution log
- Performance metrics vs SPY benchmark
- Risk metrics (drawdown, Sharpe ratio)

### Trading Configuration

Edit `config.py` to customize:

```python
# Which stocks to trade (must match training)
STOCK_TICKERS = ['AAPL', 'AMZN', 'BAC', 'GLD', 'GOOGL', 
                 'JPM', 'MSFT', 'NVDA', 'SPY', 'TLT']

# Risk management
MAX_DRAWDOWN_THRESHOLD = 0.25  # Stop if drawdown > 25%
VIX_THRESHOLD = 30             # Reduce exposure when VIX > 30

# Model selection (use your best model)
MODEL_PATH = "./models/ppo_sentiment_tuned.zip"
```

### Safety Features

**Important Safeguards**:
- Uses **Paper Trading only** (fake money, no risk)
- API credentials in `config.py` (gitignored, never committed)
- Respects market hours (no after-hours trading)
- Transaction cost simulation (0.1% per trade)
- Max position sizing limits
- Drawdown monitoring

**From Paper to Live Trading**:
- Test thoroughly with paper trading first (recommended: 30+ days)
- Monitor dashboard daily for anomalies
- When ready, change `ALPACA_BASE_URL` to live endpoint
- Start with small capital in live trading

## Project Structure

```
rl_portfolio/
├── Dockerfile              # Python 3.10 with FinRL, stable-baselines3
├── requirements.txt        # Pinned dependencies (alpha-vantage==2.3.1)
├── download_data.py        # Download OHLCV data from Yahoo Finance
├── download_vix.py         # Download VIX volatility index
├── preprocess_data.py      # Add technical indicators & features
├── train_agent.py          # Train PPO agent for portfolio management
├── evaluate_agent.py       # Backtest and evaluate trained agent vs SPY
├── save_baseline.py        # Save baseline results to experiments log
├── run_experiment.py       # Unified experiment runner with hyperparameter control
├── train_ensemble.py       # Train ensemble of multiple agents
├── paper_trading.py        # Automated paper trading with Alpaca API
├── dashboard.py            # Streamlit dashboard for monitoring trades
├── config.py               # API credentials (gitignored - see config_example.py)
├── config_example.py       # Template for API configuration
├── data/                   # Generated data files (gitignored)
│   ├── raw_data.csv
│   ├── vix_data.csv
│   └── processed_data.csv
├── models/                 # Trained RL agents (gitignored)
│   └── ppo_portfolio_agent.zip
├── src/                    # Source modules for enhanced features
│   ├── __init__.py
│   ├── sentiment/          # Modular sentiment integration
│   │   ├── __init__.py
│   │   ├── provider.py     # Abstract SentimentDataProvider interface
│   │   ├── csv_provider.py # CsvFileProvider for historical data
│   │   ├── api_provider.py # FinBertApiProvider for live inference
│   │   ├── aggregator.py   # Daily aggregation utilities
│   │   └── features.py     # Sentiment feature engineering
│   ├── data/               # Data collection and processing
│   │   ├── __init__.py
│   │   ├── news_collector.py      # Multi-source news collection
│   │   ├── sentiment_inference.py # FinBERT sentiment inference
│   │   ├── sentiment_proxy.py     # Proxy sentiment for sparse periods
│   │   └── enhanced_processor.py  # Data processing for RL training
│   └── env/                # Enhanced trading environments
│       ├── __init__.py
│       └── enhanced_portfolio_env.py  # Gymnasium env with sentiment
├── notebooks/              # Jupyter notebooks
│   ├── generate_historical_sentiment.ipynb  # Colab/Kaggle GPU inference
│   └── train_enhanced.ipynb                 # Enhanced model training on Colab
├── tests/                  # Unit tests (141 total)
│   ├── __init__.py
│   ├── test_sentiment_provider.py   # 21 provider tests
│   ├── test_sentiment_features.py   # 23 feature tests
│   ├── test_news_collector.py       # 20 news collector tests
│   ├── test_sentiment_inference.py  # 17 inference tests
│   ├── test_sentiment_proxy.py      # 22 proxy tests
│   ├── test_enhanced_env.py         # 19 environment tests
│   └── test_enhanced_processor.py   # 19 processor tests
├── experiments/            # Experiment tracking and results
│   ├── experiments_summary.csv
│   └── {experiment_name}/
│       ├── config.json
│       ├── metrics.csv
│       ├── portfolio_values.csv
│       ├── performance_analysis.png
│       └── models/ (gitignored)
└── README.md
```

## Technical Details

### Data Download Strategy
- Uses `yfinance` directly (not FinRL wrapper) for reliability
- Handles MultiIndex columns properly
- Robust error handling for network issues
- Works within Docker with `--network host`

### VIX Integration
- Separate download to avoid FinRL column mismatch bug
- Merged by date (broadcasts to all tickers)
- Forward/backward fill for missing dates
- Range: 11.54 to 82.69 (includes COVID volatility spike)

### Feature Engineering
- **FinRL FeatureEngineer**: Automated technical indicator calculation
- **Turbulence Index**: Market stress detection (uses Mahalanobis distance)
- **Data Cleaning**: NaN removal from indicator edge effects
- **Sorted Output**: By date then ticker for reproducibility

## Known Issues & Solutions

### Issue: Dependency conflicts with `alpaca-trade-api`
**Solution**: Pin `alpha-vantage==2.3.1` in requirements.txt

### Issue: VIX column mismatch in FinRL
**Solution**: Download VIX separately, merge manually by date

### Issue: Docker network timeouts
**Solution**: Use `--network host` flag for data downloads

## Next Steps

- [x] Train RL agent (PPO/A2C)
- [x] Implement walk-forward validation
- [x] Backtest on test set (2024-2025)
- [x] Calculate performance metrics (Sharpe, max drawdown)
- [x] Run hyperparameter optimization experiments
- [x] Compare PPO vs A2C algorithms
- [x] Test different training durations and entropy coefficients
- [x] Test seed sensitivity (42 vs 456)
- [x] Experiment with ensemble methods (3 & 5 agents)
- [x] Deploy paper trading via Alpaca API
- [x] Build Streamlit dashboard for visualization
- [ ] Implement alternative reward functions (Sortino, Calmar)
- [ ] Try more advanced ensemble techniques (weighted voting, meta-learning)
- [ ] Add real-time VIX-based position sizing
- [ ] Deploy to production with live trading

## Development

### Running Scripts Locally (without Docker)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run scripts
python download_data.py
python download_vix.py
python preprocess_data.py
```

### Rebuilding Docker Image
```bash
# After modifying Dockerfile or requirements.txt
# Linux/macOS:
sudo docker build -t rl_portfolio .
# Windows:
docker build -t rl_portfolio .
```

## Results Summary

### Latest Results (December 2025)

#### Production Model (Sentiment-Enhanced)

**PPO with Optuna Hyperparameters + Sentiment** (`ppo_sentiment_tuned`)
- **Total Return**: 50.26% (Test Period)
- **Sharpe Ratio**: 1.60 (Target: >1.5) - **ACHIEVED**
- **Max Drawdown**: -19.06% (Target: <25%) - **ACHIEVED**
- **Final Portfolio Value**: $150,260 (from $100k)
- **Training Time**: ~25 minutes (800k timesteps)

**Model Configuration:**
| Parameter | Value |
|-----------|-------|
| learning_rate | 8.1e-4 |
| batch_size | 64 |
| ent_coef | 0.0024 |
| net_arch | [256, 256] |
| gamma | 0.992 |
| include_sentiment | True |
| normalize_obs | False |

#### Hyperparameter Tuning (Optuna)

**Best Trial (Trial 21)** - Validation Sharpe 2.28
- Used for production model hyperparameters
- 25 trials with TPE sampler and median pruning
- Search space: learning_rate, batch_size, ent_coef, net_arch, gamma

#### Ablation Study Results

Tested 4 configurations x 3 seeds = 12 experiments:

| Config | Mean Sharpe | Mean Return | Sentiment Features |
|--------|-------------|-------------|-------------------|
| **baseline** | **1.627** | **47.8%** | 0 (tech only) |
| all_sentiment | 1.431 | 43.8% | 6 |
| score_only | 1.380 | 41.6% | 1 |
| core_3 | 1.140 | 36.8% | 3 |

**Note:** While ablation showed baseline winning, the production model uses sentiment features because:
1. The autonomous trading system architecture requires sentiment integration
2. With proper Optuna-tuned hyperparameters, sentiment model achieves Sharpe 1.60
3. Sentiment features enable future RAG safety layer integration

---

### Previous Best (Original Training)

**PPO with 1M Timesteps + High Entropy (Seed 42)** (`ppo_1m_high_entropy`)
- **Total Return**: 86.94% (2024-2025, different test period)
- **Sharpe Ratio**: 1.617 (Target: >1.0) - ACHIEVED
- **Max Drawdown**: -21.62% (Target: <25%) - ACHIEVED
- **vs SPY**: +38.35% (significantly outperformed benchmark!)
- **Training Time**: 23.9 minutes

**Key Achievement**: All targets exceeded with significant margin!

### Alternative Strong Performers

**PPO with Seed 456** (`best_seed_456_1m`)
- **Total Return**: 66.62%
- **Sharpe Ratio**: 1.406
- **Max Drawdown**: -21.75% (achieved)
- **vs SPY**: +18.03%
- Shows seed dependency but still strong performance

### Comparison to S&P 500 (SPY)
- **SPY Return**: +48.59%
- **SPY Sharpe Ratio**: ~0.95
- **Best RL Agent Outperformance**: +38.35% absolute return
- **Risk-Adjusted Performance**: 70% better Sharpe ratio (1.617 vs 0.95)

### Key Insights
- **Higher entropy coefficient** (0.05) reduced over-concentration in single assets  
- **Longer training** (1M timesteps) significantly improved performance  
- **PPO outperformed A2C** in this environment with proper hyperparameters  
- **Risk management**: Best agents had lowest drawdown despite highest returns  
- **Seed sensitivity**: Performance varies with random seed (66-87% returns)  
- **Ensemble methods** (note): Simple averaging underperformed single best agent  

## All Experimental Runs

Complete experiment history tracked in `experiments/experiments_summary.csv`:

| Experiment | Algorithm | Timesteps | LR | Return | Sharpe | Max DD | vs SPY | Time (min) |
|-----------|-----------|-----------|-----|--------|--------|--------|--------|-----------|
| **ppo_1m_high_entropy** | PPO (seed 42) | 1,000,000 | 3e-4 | **86.94%** | **1.617** | **-21.62%** | **+38.35%** | 23.9 |
| best_seed_456_1m | PPO (seed 456) | 1,000,000 | 3e-4 | 66.62% | 1.406 | -21.75% | +18.03% | 23.2 |
| ppo_high_entropy | PPO | 500,000 | 3e-4 | 62.91% | 1.333 | -23.06% | +14.32% | 12.0 |
| longer_training_500k | PPO | 500,000 | 3e-4 | 60.93% | 1.07 | -30.30% | +12.34% | 13.6 |
| a2c_500k | A2C | 500,000 | 3e-4 | 54.84% | 1.018 | -28.89% | +6.25% | 10.0 |
| ensemble_3agents | PPO Ensemble (3) | 1,500,000 | 3e-4 | 54.10% | 0.932 | -33.02% | +5.51% | 34.9 |
| baseline_200k | PPO | 200,000 | 3e-4 | 48.28% | 1.016 | -26.15% | -0.31% | 5.5 |
| ensemble_5agents | PPO Ensemble (5) | 2,500,000 | 3e-4 | 48.53% | 0.828 | -33.34% | -0.06% | 57.6 |
| lower_lr | PPO | 500,000 | 1e-4 | 48.02% | 0.982 | -27.70% | -0.56% | 12.2 |

### Experiment Insights

1. **Training Duration Matters**: 1M timesteps >> 500K > 200K
2. **Entropy is Critical**: Higher entropy (0.05 vs 0.01) prevents over-concentration
3. **PPO > A2C**: PPO achieved better results with same timesteps
4. **Learning Rate**: Default 3e-4 worked better than lower 1e-4
5. **Risk-Return Tradeoff**: Best agents had both high returns AND low drawdown
6. **Seed Dependency**: Performance varies significantly (66-87% returns) based on random seed
7. **Ensemble Underperformance**: Simple averaging of agents performed worse than single best agent
   - Possible reasons: Diversity loss, over-smoothing, conflicting strategies

## Performance Metrics

**Achieved Benchmarks (Best Agent)**:
- **Sharpe Ratio**: 1.617 (Target: >1.0) - **Exceeded by 62%**
- **Max Drawdown**: -21.62% (Target: <25%) - **Beat target**
- **Annual Return**: ~87% (Target: Beat SPY by 5%+) - **Beat SPY by 38%**
- **Risk-Adjusted Returns**: 70% better than SPY

## Enhanced Training (December 2025)

### Hyperparameter Tuning with Optuna

Used Optuna TPE sampler with median pruning to search hyperparameter space:

```bash
python -m src.experiments.hyperparameter_search --n-trials 25
```

**Search Space:**
- learning_rate: 1e-5 to 1e-3 (log scale)
- batch_size: [32, 64, 128]
- ent_coef: 0.001 to 0.1 (log scale)
- net_arch_size: [64, 128, 256]
- n_steps, n_epochs, gamma, vf_coef, max_grad_norm

**Results:** 25 trials, best Sharpe 2.28 (Trial 21)

### Ablation Study

Tested which sentiment features contribute to performance:

```bash
python -m src.experiments.ablation --config all
```

**Configurations tested:**
- `baseline`: No sentiment (technical indicators only)
- `score_only`: Just sentiment_score
- `core_3`: score + news_count + sentiment_proxy
- `all_sentiment`: All 6 sentiment features

**Finding:** Baseline outperforms sentiment models. See `experiments/ablation_analysis_report.md`.

## Technologies Used

- **FinRL**: Financial Reinforcement Learning framework
- **Stable-Baselines3**: State-of-the-art RL algorithms (PPO, A2C)
- **Optuna**: Hyperparameter optimization
- **yfinance**: Yahoo Finance data downloader
- **pandas**: Data manipulation
- **Docker**: Reproducible environment

## License

MIT License - Feel free to use for learning and portfolio projects

## Acknowledgments

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Financial RL framework
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- Yahoo Finance - Market data provider

---

**Status**: Data pipeline [DONE] | Training [DONE] | Hyperparameter Tuning [DONE] | Ablation Study [DONE] | Backtesting [DONE] | Production Model Ready

### Project Milestones

| Phase | Status | Description |
|-------|--------|-------------|
| Data Pipeline | DONE | 10 assets, 11 years, 17 features |
| Initial Training | DONE | 10 experiments, best Sharpe 1.617 |
| Sentiment Enhancement | DONE | 6 sentiment features added |
| Hyperparameter Tuning | DONE | Optuna 25 trials, best Sharpe 2.28 |
| Ablation Study | DONE | 12 experiments, analyzed tradeoffs |
| Production Model | DONE | Sharpe 1.60, Return 50.26% |
| Agentic System Integration | PLANNED | LangGraph + RAG + QuantRisk MCP |

### Next: Autonomous Portfolio System

This RL agent will be integrated into a larger autonomous trading system:
- **LangGraph Orchestration**: Multi-agent workflow coordination
- **RAG Safety Layer**: SEC filing analysis for qualitative risk
- **QuantRisk MCP**: Monte Carlo VaR/CVaR for quantitative risk
- **Deployment**: Heroku (main app) + Supabase (vector store) + HuggingFace (sentiment)

See: `autonomous-portfolio-system-plan.md` for full architecture.

Last updated: 2025-12-19