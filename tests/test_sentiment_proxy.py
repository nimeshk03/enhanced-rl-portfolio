"""
Tests for Sentiment Proxy Module

Tests the proxy sentiment generation pipeline:
- VIX-based sentiment
- Returns momentum sentiment
- Coverage analysis
- Blending real and proxy sentiment
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sentiment_proxy import (
    compute_vix_sentiment,
    compute_returns_sentiment,
    compute_sector_sentiment,
    analyze_coverage,
    identify_sparse_periods,
    generate_proxy_sentiment,
    blend_sentiment,
    generate_complete_sentiment,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_vix_series():
    """Sample VIX data."""
    return pd.Series([12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 18.0])


@pytest.fixture
def sample_returns_series():
    """Sample returns data."""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01])


@pytest.fixture
def sample_market_data():
    """Sample market data DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "SPY"]:
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "close": 100 + np.random.randn() * 5,
                "vix": 15 + np.random.randn() * 3,
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_real_sentiment():
    """Sample real sentiment data (partial coverage)."""
    # Only cover last 5 days for 2 tickers
    dates = pd.bdate_range("2024-01-01", periods=20)[-5:]
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT"]:
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sentiment_score": np.random.uniform(-0.5, 0.5),
                "news_count": np.random.randint(1, 5),
            })
    return pd.DataFrame(records)


# =============================================================================
# VIX Sentiment Tests
# =============================================================================

class TestVixSentiment:
    """Tests for VIX-based sentiment."""
    
    def test_low_vix_positive(self, sample_vix_series):
        """Test low VIX produces positive sentiment."""
        sentiment = compute_vix_sentiment(sample_vix_series)
        
        # VIX=12 should give positive sentiment
        assert sentiment.iloc[0] > 0
    
    def test_high_vix_negative(self, sample_vix_series):
        """Test high VIX produces negative sentiment."""
        sentiment = compute_vix_sentiment(sample_vix_series)
        
        # VIX=35 should give negative sentiment
        assert sentiment.iloc[5] < 0
    
    def test_sentiment_range(self, sample_vix_series):
        """Test sentiment is in valid range."""
        sentiment = compute_vix_sentiment(sample_vix_series)
        
        assert sentiment.min() >= -1
        assert sentiment.max() <= 1
    
    def test_monotonic_relationship(self):
        """Test higher VIX = lower sentiment."""
        vix = pd.Series([10, 15, 20, 25, 30, 35])
        sentiment = compute_vix_sentiment(vix)
        
        # Should be monotonically decreasing
        for i in range(len(sentiment) - 1):
            assert sentiment.iloc[i] >= sentiment.iloc[i + 1]


# =============================================================================
# Returns Sentiment Tests
# =============================================================================

class TestReturnsSentiment:
    """Tests for returns-based sentiment."""
    
    def test_positive_returns_positive_sentiment(self):
        """Test positive returns produce positive sentiment."""
        returns = pd.Series([0.02, 0.01, 0.03, 0.02, 0.01])
        sentiment = compute_returns_sentiment(returns, lookback=3)
        
        # Should be mostly positive
        assert sentiment.iloc[-1] > 0
    
    def test_negative_returns_negative_sentiment(self):
        """Test negative returns produce negative sentiment."""
        returns = pd.Series([-0.02, -0.01, -0.03, -0.02, -0.01])
        sentiment = compute_returns_sentiment(returns, lookback=3)
        
        # Should be mostly negative
        assert sentiment.iloc[-1] < 0
    
    def test_sentiment_range(self, sample_returns_series):
        """Test sentiment is clipped to valid range."""
        sentiment = compute_returns_sentiment(sample_returns_series)
        
        assert sentiment.min() >= -1
        assert sentiment.max() <= 1
    
    def test_lookback_effect(self):
        """Test lookback parameter affects smoothing."""
        returns = pd.Series([0.05, -0.05, 0.05, -0.05, 0.05])
        
        short_lookback = compute_returns_sentiment(returns, lookback=1)
        long_lookback = compute_returns_sentiment(returns, lookback=5)
        
        # Longer lookback should be smoother (lower variance)
        assert short_lookback.std() >= long_lookback.std()


# =============================================================================
# Coverage Analysis Tests
# =============================================================================

class TestCoverageAnalysis:
    """Tests for coverage analysis functions."""
    
    def test_analyze_coverage(self, sample_market_data, sample_real_sentiment):
        """Test coverage analysis."""
        coverage = analyze_coverage(sample_real_sentiment, sample_market_data)
        
        assert "ticker" in coverage.columns
        assert "coverage_pct" in coverage.columns
        assert len(coverage) == 3  # AAPL, MSFT, SPY
    
    def test_coverage_percentage_valid(self, sample_market_data, sample_real_sentiment):
        """Test coverage percentages are valid."""
        coverage = analyze_coverage(sample_real_sentiment, sample_market_data)
        
        assert (coverage["coverage_pct"] >= 0).all()
        assert (coverage["coverage_pct"] <= 100).all()
    
    def test_identify_sparse_periods(self, sample_market_data, sample_real_sentiment):
        """Test sparse period identification."""
        sparse = identify_sparse_periods(sample_real_sentiment, sample_market_data)
        
        assert "is_sparse" in sparse.columns
        # Should have some sparse periods (SPY has no real sentiment)
        assert sparse["is_sparse"].sum() > 0


# =============================================================================
# Proxy Generation Tests
# =============================================================================

class TestProxyGeneration:
    """Tests for proxy sentiment generation."""
    
    def test_generate_proxy_sentiment(self, sample_market_data):
        """Test proxy sentiment generation."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        assert len(proxy) > 0
        assert "sentiment_proxy" in proxy.columns
        assert "ticker" in proxy.columns
    
    def test_proxy_covers_all_tickers(self, sample_market_data):
        """Test proxy covers all tickers."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        assert set(proxy["ticker"].unique()) == set(tickers)
    
    def test_proxy_sentiment_range(self, sample_market_data):
        """Test proxy sentiment is in valid range."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        assert proxy["sentiment_proxy"].min() >= -1
        assert proxy["sentiment_proxy"].max() <= 1
    
    def test_proxy_components_present(self, sample_market_data):
        """Test all proxy components are present."""
        tickers = ["AAPL"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        assert "vix_component" in proxy.columns
        assert "returns_component" in proxy.columns
        assert "sector_component" in proxy.columns


# =============================================================================
# Blending Tests
# =============================================================================

class TestBlendSentiment:
    """Tests for sentiment blending."""
    
    def test_blend_with_real_sentiment(self, sample_market_data, sample_real_sentiment):
        """Test blending real and proxy sentiment."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        blended = blend_sentiment(sample_real_sentiment, proxy)
        
        assert len(blended) > 0
        assert "sentiment_score" in blended.columns
        assert "sentiment_source" in blended.columns
    
    def test_blend_marks_sources(self, sample_market_data, sample_real_sentiment):
        """Test blending marks real vs proxy sources."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        blended = blend_sentiment(sample_real_sentiment, proxy)
        
        # Should have both real and proxy sources
        assert "real" in blended["sentiment_source"].values
        assert "proxy" in blended["sentiment_source"].values
    
    def test_blend_with_empty_real(self, sample_market_data):
        """Test blending with no real sentiment."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        empty_real = pd.DataFrame()
        
        blended = blend_sentiment(empty_real, proxy)
        
        # All should be proxy
        assert (blended["sentiment_source"] == "proxy").all()
    
    def test_blend_weight_effect(self, sample_market_data, sample_real_sentiment):
        """Test real_weight parameter affects blending."""
        tickers = ["AAPL", "MSFT", "SPY"]
        proxy = generate_proxy_sentiment(sample_market_data, tickers)
        
        high_weight = blend_sentiment(sample_real_sentiment, proxy, real_weight=0.9)
        low_weight = blend_sentiment(sample_real_sentiment, proxy, real_weight=0.1)
        
        # Different weights should produce different results
        # (for rows with real sentiment)
        real_rows_high = high_weight[high_weight["sentiment_source"] == "real"]
        real_rows_low = low_weight[low_weight["sentiment_source"] == "real"]
        
        if len(real_rows_high) > 0 and len(real_rows_low) > 0:
            # Scores should differ due to different blending weights
            assert not np.allclose(
                real_rows_high["sentiment_score"].values,
                real_rows_low["sentiment_score"].values
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete sentiment generation."""
    
    def test_generate_complete_sentiment(self, sample_market_data, sample_real_sentiment, tmp_path):
        """Test complete sentiment generation pipeline."""
        # Save test data
        market_path = tmp_path / "market.csv"
        sentiment_path = tmp_path / "sentiment.csv"
        output_path = tmp_path / "complete.csv"
        
        sample_market_data.to_csv(market_path, index=False)
        sample_real_sentiment.to_csv(sentiment_path, index=False)
        
        # Run pipeline
        result = generate_complete_sentiment(
            sentiment_path=str(sentiment_path),
            market_data_path=str(market_path),
            output_path=str(output_path),
        )
        
        assert len(result) > 0
        assert output_path.exists()
    
    def test_100_percent_coverage(self, sample_market_data, sample_real_sentiment, tmp_path):
        """Test complete sentiment achieves 100% coverage."""
        market_path = tmp_path / "market.csv"
        sentiment_path = tmp_path / "sentiment.csv"
        output_path = tmp_path / "complete.csv"
        
        sample_market_data.to_csv(market_path, index=False)
        sample_real_sentiment.to_csv(sentiment_path, index=False)
        
        result = generate_complete_sentiment(
            sentiment_path=str(sentiment_path),
            market_data_path=str(market_path),
            output_path=str(output_path),
        )
        
        # Should have one record per date-ticker combination
        expected_count = sample_market_data.groupby(["date", "tic"]).ngroups
        assert len(result) == expected_count
    
    def test_plan_criteria(self, tmp_path):
        """Test criteria from implementation plan."""
        # Create test data spanning multiple years
        dates = pd.bdate_range("2015-01-01", "2024-12-31", freq="B")[:100]  # Subset for speed
        tickers = ["AAPL", "MSFT", "SPY"]
        
        records = []
        for date in dates:
            for ticker in tickers:
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "tic": ticker,
                    "close": 100 + np.random.randn() * 5,
                    "vix": 18 + np.random.randn() * 5,
                })
        
        market_data = pd.DataFrame(records)
        market_path = tmp_path / "market.csv"
        sentiment_path = tmp_path / "sentiment.csv"
        output_path = tmp_path / "complete.csv"
        
        market_data.to_csv(market_path, index=False)
        
        # Empty real sentiment
        pd.DataFrame(columns=["date", "ticker", "sentiment_score", "news_count"]).to_csv(
            sentiment_path, index=False
        )
        
        result = generate_complete_sentiment(
            sentiment_path=str(sentiment_path),
            market_data_path=str(market_path),
            output_path=str(output_path),
        )
        
        # Verify criteria
        assert set(["AAPL", "MSFT", "SPY"]).issubset(set(result["ticker"].unique()))
        assert result["sentiment_score"].between(-1, 1).all()
        assert len(result) == len(dates) * len(tickers)  # 100% coverage


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
