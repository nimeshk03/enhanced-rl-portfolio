"""
Tests for Sentiment Feature Engineering Module

Tests the feature computation pipeline:
- compute_sentiment_features: Main feature engineering function
- merge_sentiment_with_market_data: Merge with OHLCV data
- normalize_features: Feature normalization
- get_feature_statistics: Feature statistics
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment.features import (
    compute_sentiment_features,
    merge_sentiment_with_market_data,
    normalize_features,
    get_feature_statistics,
    DEFAULT_SECTOR_MAPPING,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_raw_sentiment():
    """Create sample raw sentiment data (multiple headlines per day)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]:
            # 2-4 headlines per ticker per day
            n_headlines = np.random.randint(2, 5)
            for _ in range(n_headlines):
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sentiment_score": np.random.uniform(-0.5, 0.8),
                    "sentiment_confidence": np.random.uniform(0.6, 1.0),
                })
    return pd.DataFrame(records)


@pytest.fixture
def sample_daily_sentiment():
    """Create sample daily aggregated sentiment."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]:
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sentiment_score": np.random.uniform(-0.3, 0.5),
                "news_count": np.random.randint(1, 10),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_market_data():
    """Create sample market OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]:
            base_price = {"AAPL": 180, "MSFT": 380, "GOOGL": 140, "JPM": 170, "SPY": 480}[ticker]
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "open": base_price + np.random.uniform(-2, 2),
                "high": base_price + np.random.uniform(0, 3),
                "low": base_price + np.random.uniform(-3, 0),
                "close": base_price + np.random.uniform(-2, 2),
                "volume": np.random.randint(1000000, 10000000),
            })
    return pd.DataFrame(records)


# =============================================================================
# compute_sentiment_features Tests
# =============================================================================

class TestComputeSentimentFeatures:
    """Tests for compute_sentiment_features function."""
    
    def test_basic_feature_computation(self, sample_raw_sentiment):
        """Test that all required features are computed."""
        result = compute_sentiment_features(sample_raw_sentiment)
        
        required_columns = [
            "date", "ticker", "sentiment_score", "sentiment_std",
            "sentiment_momentum", "news_volume", "sector_sentiment", "market_sentiment"
        ]
        
        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"
    
    def test_output_schema(self, sample_raw_sentiment):
        """Test output DataFrame schema."""
        result = compute_sentiment_features(sample_raw_sentiment)
        
        # Check data types
        assert result["date"].dtype == object  # string
        assert result["ticker"].dtype == object
        assert np.issubdtype(result["sentiment_score"].dtype, np.floating)
        assert np.issubdtype(result["sentiment_std"].dtype, np.floating)
        assert np.issubdtype(result["sentiment_momentum"].dtype, np.floating)
    
    def test_sentiment_score_range(self, sample_raw_sentiment):
        """Test that sentiment scores are in valid range."""
        result = compute_sentiment_features(sample_raw_sentiment)
        
        # Sentiment score should be between -1 and 1
        assert result["sentiment_score"].min() >= -1.0
        assert result["sentiment_score"].max() <= 1.0
    
    def test_daily_aggregation(self, sample_raw_sentiment):
        """Test that multiple headlines per day are aggregated."""
        result = compute_sentiment_features(sample_raw_sentiment)
        
        # Should have one row per ticker per day
        unique_combinations = result.groupby(["date", "ticker"]).size()
        assert all(unique_combinations == 1)
    
    def test_ticker_filtering(self, sample_raw_sentiment):
        """Test filtering to specific tickers."""
        result = compute_sentiment_features(
            sample_raw_sentiment,
            tickers=["AAPL", "MSFT"]
        )
        
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}
    
    def test_sector_sentiment_computation(self, sample_raw_sentiment):
        """Test sector sentiment is computed."""
        result = compute_sentiment_features(sample_raw_sentiment)
        
        # Sector sentiment should exist and be numeric
        assert "sector_sentiment" in result.columns
        assert not result["sector_sentiment"].isna().all()
    
    def test_market_sentiment_computation(self, sample_raw_sentiment):
        """Test market sentiment (SPY proxy) is computed."""
        result = compute_sentiment_features(sample_raw_sentiment)
        
        # Market sentiment should exist
        assert "market_sentiment" in result.columns
        assert not result["market_sentiment"].isna().all()
    
    def test_momentum_computation(self, sample_raw_sentiment):
        """Test sentiment momentum is computed correctly."""
        result = compute_sentiment_features(
            sample_raw_sentiment,
            short_window=3,
            long_window=7
        )
        
        # Momentum should be bounded (difference of MAs)
        assert "sentiment_momentum" in result.columns
        # Momentum should be reasonable (not extreme)
        assert result["sentiment_momentum"].abs().max() < 2.0
    
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_df = pd.DataFrame(columns=["date", "ticker", "sentiment_score"])
        result = compute_sentiment_features(empty_df)
        
        assert len(result) == 0
        assert "sentiment_score" in result.columns
    
    def test_custom_sector_mapping(self, sample_raw_sentiment):
        """Test custom sector mapping."""
        custom_mapping = {
            "AAPL": "consumer",
            "MSFT": "enterprise",
            "GOOGL": "advertising",
            "JPM": "banking",
            "SPY": "index",
        }
        
        result = compute_sentiment_features(
            sample_raw_sentiment,
            sector_mapping=custom_mapping
        )
        
        # Should still compute sector sentiment
        assert "sector_sentiment" in result.columns
    
    def test_no_missing_values(self, sample_raw_sentiment):
        """Test that output has no missing values when fill_missing=True."""
        result = compute_sentiment_features(
            sample_raw_sentiment,
            fill_missing=True
        )
        
        feature_cols = ["sentiment_score", "sentiment_std", "sentiment_momentum",
                       "news_volume", "sector_sentiment", "market_sentiment"]
        
        for col in feature_cols:
            assert result[col].isna().sum() == 0, f"Column {col} has missing values"


# =============================================================================
# merge_sentiment_with_market_data Tests
# =============================================================================

class TestMergeSentimentWithMarketData:
    """Tests for merge_sentiment_with_market_data function."""
    
    def test_basic_merge(self, sample_market_data, sample_daily_sentiment):
        """Test basic merge of sentiment with market data."""
        sentiment_features = compute_sentiment_features(sample_daily_sentiment)
        
        result = merge_sentiment_with_market_data(
            sample_market_data,
            sentiment_features,
            date_column="date",
            ticker_column="tic"
        )
        
        # Should have all market data rows
        assert len(result) == len(sample_market_data)
        
        # Should have sentiment columns
        assert "sentiment_score" in result.columns
        assert "sentiment_momentum" in result.columns
    
    def test_preserves_market_columns(self, sample_market_data, sample_daily_sentiment):
        """Test that market data columns are preserved."""
        sentiment_features = compute_sentiment_features(sample_daily_sentiment)
        
        result = merge_sentiment_with_market_data(
            sample_market_data,
            sentiment_features,
            date_column="date",
            ticker_column="tic"
        )
        
        # All original market columns should exist
        for col in ["date", "tic", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
    
    def test_empty_sentiment(self, sample_market_data):
        """Test merge with empty sentiment data."""
        empty_sentiment = pd.DataFrame(columns=[
            "date", "ticker", "sentiment_score", "sentiment_std",
            "sentiment_momentum", "news_volume", "sector_sentiment", "market_sentiment"
        ])
        
        result = merge_sentiment_with_market_data(
            sample_market_data,
            empty_sentiment,
            date_column="date",
            ticker_column="tic"
        )
        
        # Should have all market rows with neutral sentiment
        assert len(result) == len(sample_market_data)
        assert all(result["sentiment_score"] == 0.0)
    
    def test_forward_fill_missing_dates(self, sample_market_data):
        """Test that missing sentiment dates are forward-filled."""
        # Create sentiment with gaps
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        sentiment_df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates[::2]],  # Every other day
            "ticker": ["AAPL"] * 5,
            "sentiment_score": [0.5] * 5,
            "news_count": [3] * 5,
        })
        
        sentiment_features = compute_sentiment_features(sentiment_df)
        
        # Market data for AAPL only
        market_df = sample_market_data[sample_market_data["tic"] == "AAPL"].head(10)
        
        result = merge_sentiment_with_market_data(
            market_df,
            sentiment_features,
            date_column="date",
            ticker_column="tic"
        )
        
        # Should have no NaN in sentiment columns (forward filled)
        assert result["sentiment_score"].isna().sum() == 0


# =============================================================================
# normalize_features Tests
# =============================================================================

class TestNormalizeFeatures:
    """Tests for normalize_features function."""
    
    def test_zscore_normalization(self, sample_raw_sentiment):
        """Test z-score normalization."""
        features = compute_sentiment_features(sample_raw_sentiment)
        normalized = normalize_features(features, method="zscore")
        
        # Z-scores should be clipped to [-3, 3] by default
        for col in ["sentiment_score", "sentiment_momentum"]:
            assert normalized[col].min() >= -3.0
            assert normalized[col].max() <= 3.0
    
    def test_minmax_normalization(self, sample_raw_sentiment):
        """Test min-max normalization."""
        features = compute_sentiment_features(sample_raw_sentiment)
        normalized = normalize_features(features, method="minmax")
        
        # Min-max should be in [0, 1]
        for col in ["sentiment_score", "sentiment_momentum"]:
            assert normalized[col].min() >= 0.0
            assert normalized[col].max() <= 1.0
    
    def test_custom_columns(self, sample_raw_sentiment):
        """Test normalizing specific columns."""
        features = compute_sentiment_features(sample_raw_sentiment)
        
        # Only normalize sentiment_score
        normalized = normalize_features(
            features,
            feature_columns=["sentiment_score"],
            method="zscore"
        )
        
        # sentiment_score should be normalized
        assert normalized["sentiment_score"].mean() < 0.1  # Close to 0
        
        # Other columns should be unchanged
        assert (normalized["sentiment_momentum"] == features["sentiment_momentum"]).all()
    
    def test_invalid_method(self, sample_raw_sentiment):
        """Test that invalid method raises error."""
        features = compute_sentiment_features(sample_raw_sentiment)
        
        with pytest.raises(ValueError):
            normalize_features(features, method="invalid")


# =============================================================================
# get_feature_statistics Tests
# =============================================================================

class TestGetFeatureStatistics:
    """Tests for get_feature_statistics function."""
    
    def test_statistics_computed(self, sample_raw_sentiment):
        """Test that statistics are computed for all features."""
        features = compute_sentiment_features(sample_raw_sentiment)
        stats = get_feature_statistics(features)
        
        expected_features = [
            "sentiment_score", "sentiment_std", "sentiment_momentum",
            "news_volume", "sector_sentiment", "market_sentiment"
        ]
        
        for feature in expected_features:
            assert feature in stats
            assert "mean" in stats[feature]
            assert "std" in stats[feature]
            assert "min" in stats[feature]
            assert "max" in stats[feature]
    
    def test_statistics_values(self, sample_raw_sentiment):
        """Test that statistics values are reasonable."""
        features = compute_sentiment_features(sample_raw_sentiment)
        stats = get_feature_statistics(features)
        
        # Mean should be between min and max
        for feature, feature_stats in stats.items():
            assert feature_stats["min"] <= feature_stats["mean"] <= feature_stats["max"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full feature pipeline."""
    
    def test_full_pipeline(self, sample_raw_sentiment, sample_market_data):
        """Test complete pipeline from raw sentiment to merged features."""
        # Step 1: Compute features
        features = compute_sentiment_features(sample_raw_sentiment)
        
        # Step 2: Normalize
        normalized = normalize_features(features, method="zscore")
        
        # Step 3: Merge with market data
        merged = merge_sentiment_with_market_data(
            sample_market_data,
            normalized,
            date_column="date",
            ticker_column="tic"
        )
        
        # Step 4: Get statistics
        stats = get_feature_statistics(merged)
        
        # Verify complete pipeline
        assert len(merged) == len(sample_market_data)
        assert "sentiment_score" in merged.columns
        assert "close" in merged.columns
        assert len(stats) > 0
    
    def test_plan_test_criteria(self):
        """Test the exact criteria from the implementation plan."""
        # This is the test from the plan
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d"),
            "ticker": ["AAPL"] * 30,
            "sentiment_score": [0.5] * 30,
            "news_count": [5] * 30,
        })
        
        result = compute_sentiment_features(df)
        
        required = [
            "sentiment_score", "sentiment_std", "sentiment_momentum",
            "news_volume", "sector_sentiment", "market_sentiment"
        ]
        
        for col in required:
            assert col in result.columns, f"Missing {col}"
        
        print("All sentiment features computed successfully!")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
