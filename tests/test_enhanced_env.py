"""
Tests for Enhanced Portfolio Environment

Tests the EnhancedPortfolioEnv class:
- Environment initialization
- State space construction
- Action execution
- Reward calculation
- Sentiment integration
- Backward compatibility
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.enhanced_portfolio_env import (
    EnhancedPortfolioEnv,
    create_enhanced_environment,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample price data with technical indicators."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    tickers = ["AAPL", "MSFT", "SPY"]
    
    records = []
    for date in dates:
        for ticker in tickers:
            base_price = {"AAPL": 150, "MSFT": 350, "SPY": 450}[ticker]
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "open": base_price + np.random.randn() * 2,
                "high": base_price + np.random.randn() * 2 + 1,
                "low": base_price + np.random.randn() * 2 - 1,
                "close": base_price + np.random.randn() * 2,
                "volume": np.random.randint(1000000, 5000000),
                "macd": np.random.randn() * 0.5,
                "rsi_30": 50 + np.random.randn() * 10,
                "cci_30": np.random.randn() * 50,
                "vix": 15 + np.random.randn() * 3,
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    tickers = ["AAPL", "MSFT", "SPY"]
    
    records = []
    for date in dates:
        for ticker in tickers:
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "sentiment_score": np.random.uniform(-0.5, 0.5),
                "sentiment_std": np.random.uniform(0, 0.3),
                "news_count": np.random.randint(0, 10),
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def prepared_data(sample_price_data, sample_sentiment_data):
    """Prepare merged and indexed data."""
    # Merge
    df = sample_price_data.merge(
        sample_sentiment_data,
        on=['date', 'tic'],
        how='left'
    )
    
    # Sort and index
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    dates = sorted(df['date'].unique())
    date_to_day = {date: i for i, date in enumerate(dates)}
    df['day'] = df['date'].map(date_to_day)
    df = df.set_index('day')
    
    return df


# =============================================================================
# Environment Initialization Tests
# =============================================================================

class TestEnvironmentInit:
    """Tests for environment initialization."""
    
    def test_basic_init(self, prepared_data):
        """Test basic environment initialization."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'vix'],
            sentiment_feature_list=['sentiment_score', 'sentiment_std', 'news_count'],
            include_sentiment=True,
            print_verbosity=0,
        )
        
        assert env is not None
        assert env.stock_dim == 3
    
    def test_state_space_with_sentiment(self, prepared_data):
        """Test state space includes sentiment features."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'vix'],
            sentiment_feature_list=['sentiment_score', 'sentiment_std', 'news_count'],
            include_sentiment=True,
            print_verbosity=0,
        )
        
        # State = cash(1) + holdings(3) + prices(3) + tech(3*4) + sentiment(3*3)
        expected_state_space = 1 + 3 + 3 + 12 + 9
        assert env.state_space == expected_state_space
        assert env.observation_space.shape[0] == expected_state_space
    
    def test_state_space_without_sentiment(self, prepared_data):
        """Test state space without sentiment features."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'vix'],
            sentiment_feature_list=['sentiment_score'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        # State = cash(1) + holdings(3) + prices(3) + tech(3*4) + sentiment(0)
        expected_state_space = 1 + 3 + 3 + 12 + 0
        assert env.state_space == expected_state_space
    
    def test_action_space(self, prepared_data):
        """Test action space dimensions."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        assert env.action_space.shape[0] == 3
        assert env.action_space.low[0] == -1
        assert env.action_space.high[0] == 1


# =============================================================================
# Reset and Step Tests
# =============================================================================

class TestResetAndStep:
    """Tests for reset and step methods."""
    
    def test_reset(self, prepared_data):
        """Test environment reset."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        assert obs is not None
        assert obs.shape[0] == env.state_space
        assert info['portfolio_value'] == env.initial_amount
        assert info['cash'] == env.initial_amount
    
    def test_step(self, prepared_data):
        """Test environment step."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert obs.shape[0] == env.state_space
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_episode_completion(self, prepared_data):
        """Test running a full episode."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if steps > 100:  # Safety limit
                break
        
        assert done
        assert steps > 0
        assert len(env.portfolio_value_history) > 1


# =============================================================================
# Trading Logic Tests
# =============================================================================

class TestTradingLogic:
    """Tests for trading execution logic."""
    
    def test_buy_action(self, prepared_data):
        """Test buying stocks."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            hmax=10,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        initial_cash = env.cash
        
        # Buy action for first stock
        action = np.array([1.0, 0.0, 0.0])
        obs, reward, done, truncated, info = env.step(action)
        
        # Should have bought some shares
        assert env.holdings[0] > 0
        assert env.cash < initial_cash
    
    def test_sell_action(self, prepared_data):
        """Test selling stocks."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            hmax=10,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # First buy
        action = np.array([1.0, 0.0, 0.0])
        env.step(action)
        holdings_after_buy = env.holdings[0]
        
        # Then sell
        action = np.array([-1.0, 0.0, 0.0])
        env.step(action)
        
        # Should have sold some shares
        assert env.holdings[0] < holdings_after_buy
    
    def test_transaction_costs(self, prepared_data):
        """Test transaction costs are applied."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            hmax=10,
            buy_cost_pct=0.01,  # 1% cost
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # Execute trades
        action = np.array([1.0, 1.0, 1.0])
        env.step(action)
        
        # Should have accumulated costs
        assert env.cost > 0


# =============================================================================
# Sentiment Integration Tests
# =============================================================================

class TestSentimentIntegration:
    """Tests for sentiment feature integration."""
    
    def test_sentiment_in_state(self, prepared_data):
        """Test sentiment features appear in state."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            sentiment_feature_list=['sentiment_score'],
            include_sentiment=True,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # State should be larger with sentiment
        # cash(1) + holdings(3) + prices(3) + tech(3*1) + sentiment(3*1)
        expected = 1 + 3 + 3 + 3 + 3
        assert obs.shape[0] == expected
    
    def test_sentiment_reward_bonus(self, prepared_data):
        """Test sentiment-aligned reward bonus."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            sentiment_feature_list=['sentiment_score'],
            include_sentiment=True,
            sentiment_reward_weight=0.1,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # Get sentiment values
        sentiments = env.data['sentiment_score'].values
        
        # Action aligned with sentiment
        aligned_action = np.sign(sentiments)
        obs1, reward1, _, _, _ = env.step(aligned_action)
        
        # Reset and try opposite action
        env.reset()
        opposite_action = -np.sign(sentiments)
        obs2, reward2, _, _, _ = env.step(opposite_action)
        
        # Aligned action should generally get higher reward
        # (not always due to market movements, but sentiment bonus helps)
        # Just verify rewards are different
        assert reward1 != reward2
    
    def test_no_sentiment_backward_compatible(self, prepared_data):
        """Test environment works without sentiment (backward compatible)."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'vix'],
            sentiment_feature_list=[],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # Should work without sentiment
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, float)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunction:
    """Tests for create_enhanced_environment factory."""
    
    def test_from_csv(self, sample_price_data, sample_sentiment_data, tmp_path):
        """Test creating environment from CSV files."""
        # Save test data
        price_path = tmp_path / "prices.csv"
        sentiment_path = tmp_path / "sentiment.csv"
        
        sample_price_data.to_csv(price_path, index=False)
        sample_sentiment_data.to_csv(sentiment_path, index=False)
        
        # Create environment
        env = EnhancedPortfolioEnv.from_csv(
            price_data_path=str(price_path),
            sentiment_data_path=str(sentiment_path),
            include_sentiment=True,
        )
        
        assert env is not None
        assert env.stock_dim == 3
        assert env.include_sentiment
    
    def test_from_csv_without_sentiment(self, sample_price_data, tmp_path):
        """Test creating environment without sentiment data."""
        price_path = tmp_path / "prices.csv"
        sample_price_data.to_csv(price_path, index=False)
        
        env = EnhancedPortfolioEnv.from_csv(
            price_data_path=str(price_path),
            include_sentiment=False,
        )
        
        assert env is not None
        assert not env.include_sentiment
    
    def test_factory_function(self, sample_price_data, sample_sentiment_data, tmp_path):
        """Test create_enhanced_environment factory function."""
        price_path = tmp_path / "prices.csv"
        sentiment_path = tmp_path / "sentiment.csv"
        
        sample_price_data.to_csv(price_path, index=False)
        sample_sentiment_data.to_csv(sentiment_path, index=False)
        
        env = create_enhanced_environment(
            price_data_path=str(price_path),
            sentiment_data_path=str(sentiment_path),
            include_sentiment=True,
            initial_amount=50000,
        )
        
        assert env is not None
        assert env.initial_amount == 50000


# =============================================================================
# Portfolio Statistics Tests
# =============================================================================

class TestPortfolioStats:
    """Tests for portfolio statistics."""
    
    def test_get_portfolio_stats(self, prepared_data):
        """Test portfolio statistics calculation."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        
        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        stats = env.get_portfolio_stats()
        
        assert 'total_return' in stats
        assert 'sharpe_ratio' in stats
        assert 'max_drawdown' in stats
        assert 'final_value' in stats
        assert 'total_trades' in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the enhanced environment."""
    
    def test_full_episode_with_sentiment(self, prepared_data):
        """Test running a full episode with sentiment features."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'vix'],
            sentiment_feature_list=['sentiment_score', 'sentiment_std', 'news_count'],
            include_sentiment=True,
            sentiment_reward_weight=0.05,
            print_verbosity=0,
        )
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        stats = env.get_portfolio_stats()
        
        assert stats['final_value'] > 0
        assert env.trades > 0
    
    def test_gymnasium_compatibility(self, prepared_data):
        """Test environment is gymnasium compatible."""
        env = EnhancedPortfolioEnv(
            df=prepared_data,
            stock_dim=3,
            tech_indicator_list=['macd'],
            include_sentiment=False,
            print_verbosity=0,
        )
        
        # Check gymnasium interface
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')
        
        # Check reset returns tuple
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # Check step returns 5 values
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
