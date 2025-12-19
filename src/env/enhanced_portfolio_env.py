"""
Enhanced Portfolio Environment

A Gymnasium-compatible portfolio trading environment that supports:
1. Technical indicators (baseline features)
2. Sentiment features (optional, configurable)
3. Sentiment-aware reward shaping (optional)
4. Backward compatibility with baseline training

The environment follows the OpenAI Gym interface:
- State Space: [cash, holdings, prices, technical_indicators, sentiment_features]
- Action Space: Continuous [-1, 1] for each stock (sell to buy)
- Reward: Change in portfolio value (optionally sentiment-adjusted)

Usage:
    # With sentiment
    env = EnhancedPortfolioEnv.from_csv(
        price_data_path='data/processed_data.csv',
        sentiment_data_path='data/historical_sentiment_complete.csv',
        include_sentiment=True,
    )
    
    # Without sentiment (baseline compatible)
    env = EnhancedPortfolioEnv.from_csv(
        price_data_path='data/processed_data.csv',
        include_sentiment=False,
    )
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPortfolioEnv(gym.Env):
    """
    Enhanced Portfolio Trading Environment with Sentiment Integration.
    
    This environment extends the standard portfolio trading setup with:
    - Optional sentiment features in the observation space
    - Configurable reward shaping based on sentiment alignment
    - Support for both training and evaluation modes
    
    State Space Components:
    1. Cash balance (1 dim)
    2. Stock holdings (n_stocks dims)
    3. Stock prices (n_stocks dims)
    4. Technical indicators (n_stocks * n_tech_indicators dims)
    5. Sentiment features (n_stocks * n_sentiment_features dims) [optional]
    
    Action Space:
    - Continuous actions in [-1, 1] for each stock
    - Negative = sell, Positive = buy, magnitude = intensity
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = 100,
        initial_amount: float = 100000,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        tech_indicator_list: Optional[List[str]] = None,
        sentiment_feature_list: Optional[List[str]] = None,
        include_sentiment: bool = True,
        sentiment_reward_weight: float = 0.0,
        turbulence_threshold: Optional[float] = None,
        print_verbosity: int = 1,
        day: int = 0,
        initial: bool = True,
        mode: str = "train",
        render_mode: Optional[str] = None,
        normalize_obs: bool = True,
    ):
        """
        Initialize the Enhanced Portfolio Environment.
        
        Args:
            df: DataFrame with price data, technical indicators, and optionally sentiment
            stock_dim: Number of stocks in the portfolio
            hmax: Maximum number of shares to trade per action
            initial_amount: Starting portfolio value
            buy_cost_pct: Transaction cost for buying (as decimal)
            sell_cost_pct: Transaction cost for selling (as decimal)
            reward_scaling: Scale factor for rewards
            tech_indicator_list: List of technical indicator column names
            sentiment_feature_list: List of sentiment feature column names
            include_sentiment: Whether to include sentiment in observations
            sentiment_reward_weight: Weight for sentiment-aligned reward bonus (0 = disabled)
            turbulence_threshold: Threshold for turbulence-based risk management
            print_verbosity: Verbosity level (0=silent, 1=normal, 2=verbose)
            day: Starting day index
            initial: Whether this is initial setup
            mode: 'train' or 'trade'
            render_mode: Rendering mode for gymnasium compatibility
            normalize_obs: Whether to normalize observations (recommended for new training)
        """
        super().__init__()
        
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.sentiment_feature_list = sentiment_feature_list or []
        self.include_sentiment = include_sentiment
        self.sentiment_reward_weight = sentiment_reward_weight
        self.turbulence_threshold = turbulence_threshold
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.mode = mode
        self.render_mode = render_mode
        self.normalize_obs = normalize_obs
        
        # Get unique tickers
        self.tickers = sorted(df['tic'].unique())
        assert len(self.tickers) == stock_dim, f"Expected {stock_dim} tickers, got {len(self.tickers)}"
        
        # Calculate state space dimension
        self.n_tech_features = len(self.tech_indicator_list)
        self.n_sentiment_features = len(self.sentiment_feature_list) if include_sentiment else 0
        
        # State: cash + holdings + prices + tech_indicators + sentiment
        self.state_space = (
            1 +  # cash
            stock_dim +  # holdings
            stock_dim +  # prices
            stock_dim * self.n_tech_features +  # technical indicators
            stock_dim * self.n_sentiment_features  # sentiment features
        )
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(stock_dim,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.day = day
        self.terminal = False
        self.data = self._get_day_data(self.day)
        
        # Portfolio state
        self.cash = initial_amount
        self.holdings = np.zeros(stock_dim)
        self.portfolio_value = initial_amount
        self.portfolio_value_history = [initial_amount]
        
        # Trading records
        self.trades = 0
        self.cost = 0
        self.rewards_history = []
        
        # Get max day from data
        if isinstance(df.index, pd.MultiIndex):
            self.max_day = df.index.get_level_values(0).max()
        else:
            self.max_day = df.index.max()
        
        if print_verbosity >= 1:
            logger.info(f"EnhancedPortfolioEnv initialized:")
            logger.info(f"  - Stocks: {stock_dim}")
            logger.info(f"  - Tech features: {self.n_tech_features}")
            logger.info(f"  - Sentiment features: {self.n_sentiment_features}")
            logger.info(f"  - State space: {self.state_space}")
            logger.info(f"  - Include sentiment: {include_sentiment}")
            logger.info(f"  - Sentiment reward weight: {sentiment_reward_weight}")
    
    def _get_day_data(self, day: int) -> pd.DataFrame:
        """Get data for a specific day."""
        try:
            data = self.df.loc[day]
            if isinstance(data, pd.Series):
                # Single row, convert to DataFrame
                data = data.to_frame().T
            return data.sort_values('tic').reset_index(drop=True)
        except KeyError:
            logger.warning(f"Day {day} not found in data")
            return pd.DataFrame()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct the observation state vector.
        
        State = [cash, holdings, prices, tech_indicators, sentiment_features]
        
        If normalize_obs=True (recommended for new training):
        - Cash: normalized by initial_amount (centered around 0)
        - Holdings: normalized by hmax (typical range 0-1)
        - Prices: log-normalized and scaled (typical range -1 to 1)
        - Tech/Sentiment: already normalized by processor (range -3 to 3)
        
        If normalize_obs=False (for backward compatibility with old models):
        - Raw cash, holdings, and prices are used
        """
        # Get current prices (ensure float64 for numpy operations)
        prices = np.array(self.data['close'].values, dtype=np.float64)
        
        if self.normalize_obs:
            # Normalize cash: ratio to initial amount, centered at 0
            obs_cash = (self.cash / self.initial_amount) - 1.0
            # Normalize holdings: ratio to hmax
            obs_holdings = self.holdings / self.hmax
            # Normalize prices: log-scale relative to mean price
            mean_price = np.mean(prices) if np.mean(prices) > 0 else 1.0
            obs_prices = np.log(prices / mean_price + 1e-8)
        else:
            # Raw values (backward compatible with old models)
            obs_cash = self.cash
            obs_holdings = self.holdings
            obs_prices = prices
        
        # Get technical indicators (already normalized by processor)
        tech_features = []
        for indicator in self.tech_indicator_list:
            if indicator in self.data.columns:
                tech_features.extend(self.data[indicator].values)
            else:
                tech_features.extend([0.0] * self.stock_dim)
        
        # Get sentiment features (if enabled, already normalized)
        sentiment_features = []
        if self.include_sentiment:
            for feature in self.sentiment_feature_list:
                if feature in self.data.columns:
                    sentiment_features.extend(self.data[feature].values)
                else:
                    sentiment_features.extend([0.0] * self.stock_dim)
        
        # Construct state
        state = np.concatenate([
            [obs_cash],
            obs_holdings,
            obs_prices,
            tech_features,
            sentiment_features,
        ]).astype(np.float32)
        
        return state
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        prices = np.array(self.data['close'].values, dtype=np.float64)
        stock_value = np.sum(self.holdings * prices)
        return self.cash + stock_value
    
    def _execute_trades(self, actions: np.ndarray) -> float:
        """
        Execute trading actions and return transaction costs.
        
        Actions are in [-1, 1]:
        - Negative: sell (magnitude * hmax shares)
        - Positive: buy (magnitude * hmax shares)
        """
        prices = np.array(self.data['close'].values, dtype=np.float64)
        total_cost = 0.0
        
        for i, action in enumerate(actions):
            if action > 0:  # Buy
                # Calculate shares to buy
                shares_to_buy = min(
                    int(action * self.hmax),
                    int(self.cash / (prices[i] * (1 + self.buy_cost_pct)))
                )
                if shares_to_buy > 0:
                    cost = shares_to_buy * prices[i] * (1 + self.buy_cost_pct)
                    self.cash -= cost
                    self.holdings[i] += shares_to_buy
                    total_cost += shares_to_buy * prices[i] * self.buy_cost_pct
                    self.trades += 1
                    
            elif action < 0:  # Sell
                # Calculate shares to sell
                shares_to_sell = min(
                    int(abs(action) * self.hmax),
                    int(self.holdings[i])
                )
                if shares_to_sell > 0:
                    revenue = shares_to_sell * prices[i] * (1 - self.sell_cost_pct)
                    self.cash += revenue
                    self.holdings[i] -= shares_to_sell
                    total_cost += shares_to_sell * prices[i] * self.sell_cost_pct
                    self.trades += 1
        
        return total_cost
    
    def _calculate_sentiment_reward_bonus(self, actions: np.ndarray) -> float:
        """
        Calculate bonus/penalty based on sentiment alignment.
        
        Rewards actions that align with sentiment:
        - Buying when sentiment is positive
        - Selling when sentiment is negative
        """
        if not self.include_sentiment or self.sentiment_reward_weight == 0:
            return 0.0
        
        # Get sentiment scores (use first sentiment feature as primary)
        if 'sentiment_score' in self.data.columns:
            sentiments = self.data['sentiment_score'].values
        elif len(self.sentiment_feature_list) > 0 and self.sentiment_feature_list[0] in self.data.columns:
            sentiments = self.data[self.sentiment_feature_list[0]].values
        else:
            return 0.0
        
        # Calculate alignment score
        # Positive sentiment + buy action = aligned
        # Negative sentiment + sell action = aligned
        alignment = np.sum(actions * sentiments)
        
        return alignment * self.sentiment_reward_weight
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Array of actions for each stock [-1, 1]
            
        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Clip actions to valid range
        actions = np.clip(actions, -1, 1)
        
        # Store previous portfolio value
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # Execute trades
        transaction_cost = self._execute_trades(actions)
        self.cost += transaction_cost
        
        # Move to next day
        self.day += 1
        
        # Check if terminal
        if self.day > self.max_day:
            self.terminal = True
        else:
            self.data = self._get_day_data(self.day)
            if len(self.data) == 0:
                self.terminal = True
        
        # Calculate new portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        self.portfolio_value_history.append(self.portfolio_value)
        
        # Calculate reward
        portfolio_return = self.portfolio_value - prev_portfolio_value
        sentiment_bonus = self._calculate_sentiment_reward_bonus(actions)
        reward = (portfolio_return + sentiment_bonus) * self.reward_scaling
        self.rewards_history.append(reward)
        
        # Get new state
        state = self._get_state()
        
        # Info dict
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "trades": self.trades,
            "cost": self.cost,
            "day": self.day,
        }
        
        return state, reward, self.terminal, False, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset state
        self.day = 0
        self.terminal = False
        self.data = self._get_day_data(self.day)
        
        # Reset portfolio
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
        self.portfolio_value = self.initial_amount
        self.portfolio_value_history = [self.initial_amount]
        
        # Reset records
        self.trades = 0
        self.cost = 0
        self.rewards_history = []
        
        state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
        }
        
        return state, info
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            print(f"Day: {self.day}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Holdings: {self.holdings}")
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio performance statistics."""
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_drawdown = np.min(values / np.maximum.accumulate(values) - 1)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": values[-1],
            "total_trades": self.trades,
            "total_cost": self.cost,
        }
    
    @classmethod
    def from_csv(
        cls,
        price_data_path: str,
        sentiment_data_path: Optional[str] = None,
        include_sentiment: bool = True,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        **kwargs,
    ) -> "EnhancedPortfolioEnv":
        """
        Create environment from CSV files.
        
        Args:
            price_data_path: Path to price/technical indicator data
            sentiment_data_path: Path to sentiment data (optional)
            include_sentiment: Whether to include sentiment features
            train_start: Start date for training period
            train_end: End date for training period
            **kwargs: Additional arguments for environment
            
        Returns:
            EnhancedPortfolioEnv instance
        """
        logger.info(f"Loading price data from {price_data_path}")
        price_df = pd.read_csv(price_data_path)
        price_df['date'] = price_df['date'].astype(str)
        
        # Load sentiment data if provided
        if include_sentiment and sentiment_data_path and os.path.exists(sentiment_data_path):
            logger.info(f"Loading sentiment data from {sentiment_data_path}")
            sentiment_df = pd.read_csv(sentiment_data_path)
            sentiment_df['date'] = sentiment_df['date'].astype(str)
            
            # Rename ticker column if needed
            if 'ticker' in sentiment_df.columns and 'tic' not in sentiment_df.columns:
                sentiment_df = sentiment_df.rename(columns={'ticker': 'tic'})
            
            # Filter to only numeric sentiment columns (exclude string columns like 'sentiment_source')
            non_feature_cols = ['date', 'tic', 'ticker']
            sentiment_cols = []
            for col in sentiment_df.columns:
                if col not in non_feature_cols:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(sentiment_df[col]):
                        sentiment_cols.append(col)
                    else:
                        logger.info(f"Skipping non-numeric column: {col}")
            
            # Keep only numeric sentiment columns for merge
            merge_cols = ['date', 'tic'] + sentiment_cols
            sentiment_df_numeric = sentiment_df[merge_cols]
            
            # Merge sentiment with price data
            df = price_df.merge(
                sentiment_df_numeric,
                on=['date', 'tic'],
                how='left'
            )
            
            # Fill missing sentiment with 0 (neutral)
            for col in sentiment_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            logger.info(f"Merged data shape: {df.shape}")
        else:
            df = price_df
            sentiment_cols = []
            if include_sentiment:
                logger.warning("Sentiment data not found, proceeding without sentiment")
                include_sentiment = False
        
        # Filter by date range
        if train_start:
            df = df[df['date'] >= train_start]
        if train_end:
            df = df[df['date'] <= train_end]
        
        # Sort and prepare data
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Create day index
        dates = sorted(df['date'].unique())
        date_to_day = {date: i for i, date in enumerate(dates)}
        df['day'] = df['date'].map(date_to_day)
        df = df.set_index('day')
        
        # Identify feature columns
        base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
        sentiment_feature_cols = sentiment_cols if include_sentiment else []
        tech_indicator_cols = [c for c in df.columns if c not in base_cols + sentiment_feature_cols]
        
        # Get stock dimension
        stock_dim = df['tic'].nunique()
        
        logger.info(f"Creating environment with {stock_dim} stocks")
        logger.info(f"Technical indicators: {len(tech_indicator_cols)}")
        logger.info(f"Sentiment features: {len(sentiment_feature_cols)}")
        
        return cls(
            df=df,
            stock_dim=stock_dim,
            tech_indicator_list=tech_indicator_cols,
            sentiment_feature_list=sentiment_feature_cols,
            include_sentiment=include_sentiment,
            **kwargs,
        )


def create_enhanced_environment(
    price_data_path: str,
    sentiment_data_path: Optional[str] = None,
    include_sentiment: bool = True,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    hmax: int = 100,
    initial_amount: float = 100000,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    reward_scaling: float = 1e-4,
    sentiment_reward_weight: float = 0.0,
    print_verbosity: int = 1,
) -> EnhancedPortfolioEnv:
    """
    Factory function to create an enhanced portfolio environment.
    
    This is a convenience function that wraps EnhancedPortfolioEnv.from_csv()
    with common default parameters.
    
    Args:
        price_data_path: Path to price/technical indicator data
        sentiment_data_path: Path to sentiment data
        include_sentiment: Whether to include sentiment features
        train_start: Start date for training period
        train_end: End date for training period
        hmax: Maximum shares per trade
        initial_amount: Starting portfolio value
        buy_cost_pct: Buy transaction cost
        sell_cost_pct: Sell transaction cost
        reward_scaling: Reward scaling factor
        sentiment_reward_weight: Weight for sentiment-aligned rewards
        print_verbosity: Verbosity level
        
    Returns:
        EnhancedPortfolioEnv instance
    """
    return EnhancedPortfolioEnv.from_csv(
        price_data_path=price_data_path,
        sentiment_data_path=sentiment_data_path,
        include_sentiment=include_sentiment,
        train_start=train_start,
        train_end=train_end,
        hmax=hmax,
        initial_amount=initial_amount,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=reward_scaling,
        sentiment_reward_weight=sentiment_reward_weight,
        print_verbosity=print_verbosity,
    )
