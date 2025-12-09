"""config.py

Alpaca API Configuration for Paper Trading.

IMPORTANT: 
- Never commit this file with real keys to git!
- Add config.py to your .gitignore file

To get your keys:
1. Sign up at https://alpaca.markets/
2. Go to Paper Trading dashboard
3. Generate API keys
"""

# =============================================================================
# ALPACA API CREDENTIALS
# =============================================================================

# Paper Trading API (fake money - safe to test)
ALPACA_API_KEY = "YOUR_API_KEY"          # Replace with your key
ALPACA_SECRET_KEY = "YOUR_API_SECRET"    # Replace with your secret
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint (library adds /v2)

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Stock tickers to trade (must match training data)
STOCK_TICKERS = ['AAPL', 'AMZN', 'BAC', 'GLD', 'GOOGL', 
                 'JPM', 'MSFT', 'NVDA', 'SPY', 'TLT']

# Trading parameters
INITIAL_CAPITAL = 100000      # Starting paper money ($100k)
MAX_POSITION_SIZE = 0.20      # Max 20% of portfolio in single stock
TRADING_FREQUENCY = "daily"   # How often to rebalance

# Model path (our best performing model)
MODEL_PATH = "./experiments/ensemble_5agents_500k/models/agent_seed_456"

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Stop trading if drawdown exceeds this threshold
MAX_DRAWDOWN_THRESHOLD = 0.25  # 25% max drawdown

# VIX threshold - reduce positions when volatility is high
VIX_THRESHOLD = 30  # Reduce exposure when VIX > 30