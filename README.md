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

**Date Range**: 2019-01-02 to 2025-10-29 (6+ years, includes COVID crash)

**Features** (17 total):
- Base OHLCV data (7 columns)
- Technical indicators (8): MACD, Bollinger Bands, RSI-30, CCI-30, DX-30, SMA-30, SMA-60
- Market stress indicators (2): VIX, Turbulence Index

**Total Dataset**: 17,170 rows × 17 columns

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
MODEL_PATH = "./experiments/ppo_1m_high_entropy/models/trained_agent"
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
├── experiments/            # Experiment tracking and results (10 total)
│   ├── experiments_summary.csv
│   ├── baseline_200k/
│   ├── longer_training_500k/
│   ├── lower_lr/
│   ├── a2c_500k/
│   ├── ppo_high_entropy/
│   ├── ppo_1m_high_entropy/  # Best performer (seed 42)
│   ├── best_seed_456_1m/     Strong alternative (seed 456)
│   ├── ensemble_3agents_500k/
│   └── ensemble_5agents_500k/
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

### Best Performing Agent

**PPO with 1M Timesteps + High Entropy (Seed 42)** (`ppo_1m_high_entropy`)
- **Total Return**: 86.94% (2024-2025)
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

## Technologies Used

- **FinRL**: Financial Reinforcement Learning framework
- **Stable-Baselines3**: State-of-the-art RL algorithms (PPO, A2C)
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

**Status**: Data pipeline [DONE] | Training (10 experiments) [DONE] | Backtesting [DONE] | Paper Trading Deployed | Dashboard Live

Last updated: 2025-12-01