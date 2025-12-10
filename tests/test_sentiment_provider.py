"""
Tests for Sentiment Data Provider Interface

Tests the abstract interface and concrete implementations:
- SentimentDataProvider (abstract base class)
- CsvFileProvider (historical CSV data)
- FinBertApiProvider (live API placeholder)
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment.provider import SentimentDataProvider
from src.sentiment.csv_provider import CsvFileProvider
from src.sentiment.api_provider import FinBertApiProvider
from src.sentiment.aggregator import (
    aggregate_daily_sentiment,
    add_rolling_sentiment,
    compute_sentiment_momentum,
    normalize_news_volume,
    compute_sector_sentiment,
    compute_market_sentiment,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_sentiment_csv(tmp_path):
    """Create a temporary CSV file with sample sentiment data."""
    data = {
        "date": [
            "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03",
            "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05",
        ],
        "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        "sentiment_score": [0.5, 0.3, -0.2, 0.1, 0.7, 0.4, 0.2, -0.1],
        "sentiment_confidence": [0.9, 0.8, 0.7, 0.85, 0.95, 0.75, 0.8, 0.6],
        "news_count": [5, 3, 2, 4, 6, 2, 3, 5],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_sentiment.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_raw_sentiment():
    """Create sample raw sentiment data (multiple headlines per day)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "SPY"]:
            # 3-5 headlines per ticker per day
            n_headlines = np.random.randint(3, 6)
            for _ in range(n_headlines):
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sentiment_score": np.random.uniform(-1, 1),
                    "sentiment_confidence": np.random.uniform(0.5, 1.0),
                })
    return pd.DataFrame(records)


# =============================================================================
# CsvFileProvider Tests
# =============================================================================

class TestCsvFileProvider:
    """Tests for CsvFileProvider."""
    
    def test_load_valid_csv(self, sample_sentiment_csv):
        """Test loading a valid CSV file."""
        provider = CsvFileProvider(sample_sentiment_csv)
        
        assert provider.is_available()
        assert "AAPL" in provider.supported_tickers
        assert "MSFT" in provider.supported_tickers
        assert provider.provider_name.startswith("CsvFileProvider")
    
    def test_get_sentiment_date_range(self, sample_sentiment_csv):
        """Test filtering by date range."""
        provider = CsvFileProvider(sample_sentiment_csv)
        
        result = provider.get_sentiment(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-02",
            end_date="2024-01-03",
        )
        
        assert len(result) == 4  # 2 tickers x 2 days
        assert result["date"].min() == "2024-01-02"
        assert result["date"].max() == "2024-01-03"
    
    def test_get_sentiment_single_ticker(self, sample_sentiment_csv):
        """Test filtering by single ticker."""
        provider = CsvFileProvider(sample_sentiment_csv)
        
        result = provider.get_sentiment(
            tickers=["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-05",
        )
        
        assert len(result) == 4  # 4 days for AAPL
        assert all(result["ticker"] == "AAPL")
    
    def test_get_latest_sentiment(self, sample_sentiment_csv):
        """Test getting latest sentiment."""
        provider = CsvFileProvider(sample_sentiment_csv)
        
        result = provider.get_latest_sentiment(["AAPL", "MSFT"])
        
        assert len(result) == 2  # One per ticker
        assert set(result["ticker"]) == {"AAPL", "MSFT"}
        assert all(result["date"] == "2024-01-05")  # Latest date
    
    def test_missing_file(self, tmp_path):
        """Test handling of missing CSV file."""
        provider = CsvFileProvider(str(tmp_path / "nonexistent.csv"))
        
        assert not provider.is_available()
        assert provider.supported_tickers == []
        
        # Should return empty DataFrame, not raise error
        result = provider.get_sentiment(["AAPL"], "2024-01-01", "2024-01-31")
        assert len(result) == 0
    
    def test_invalid_date_range(self, sample_sentiment_csv):
        """Test validation of invalid date range."""
        provider = CsvFileProvider(sample_sentiment_csv)
        
        with pytest.raises(ValueError):
            provider.get_sentiment(
                tickers=["AAPL"],
                start_date="2024-01-31",
                end_date="2024-01-01",  # End before start
            )
    
    def test_coverage_info(self, sample_sentiment_csv):
        """Test coverage info method."""
        provider = CsvFileProvider(sample_sentiment_csv)
        
        info = provider.get_coverage_info()
        
        assert "provider" in info
        assert "start_date" in info
        assert "end_date" in info
        assert "total_records" in info
        assert info["total_records"] == 8


# =============================================================================
# FinBertApiProvider Tests
# =============================================================================

class TestFinBertApiProvider:
    """Tests for FinBertApiProvider."""
    
    def test_fallback_to_neutral(self):
        """Test that provider returns neutral sentiment when API unavailable."""
        provider = FinBertApiProvider(fallback_to_neutral=True)
        
        # API not configured, should return neutral
        assert not provider.is_available()
        
        result = provider.get_sentiment(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-01-05",
        )
        
        # Should have data (neutral fallback)
        assert len(result) > 0
        # All scores should be 0 (neutral)
        assert all(result["sentiment_score"] == 0.0)
        # Confidence should be 0 (indicating fallback)
        assert all(result["sentiment_confidence"] == 0.0)
    
    def test_get_latest_sentiment_fallback(self):
        """Test latest sentiment with fallback."""
        provider = FinBertApiProvider(fallback_to_neutral=True)
        
        result = provider.get_latest_sentiment(["AAPL", "MSFT"])
        
        assert len(result) > 0
        assert "AAPL" in result["ticker"].values
        assert "MSFT" in result["ticker"].values
    
    def test_no_fallback_returns_empty(self):
        """Test that empty DataFrame returned when fallback disabled."""
        provider = FinBertApiProvider(fallback_to_neutral=False)
        
        result = provider.get_sentiment(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-05",
        )
        
        assert len(result) == 0
    
    def test_provider_name(self):
        """Test provider name reflects status."""
        provider = FinBertApiProvider()
        
        assert "fallback" in provider.provider_name.lower()
    
    def test_supported_tickers(self):
        """Test supported tickers list."""
        provider = FinBertApiProvider()
        
        tickers = provider.supported_tickers
        assert "AAPL" in tickers
        assert "SPY" in tickers


# =============================================================================
# Aggregator Tests
# =============================================================================

class TestAggregator:
    """Tests for sentiment aggregation utilities."""
    
    def test_aggregate_daily_sentiment(self, sample_raw_sentiment):
        """Test daily aggregation of raw sentiment."""
        result = aggregate_daily_sentiment(sample_raw_sentiment)
        
        # Should have one row per ticker per day
        assert "sentiment_mean" in result.columns
        assert "sentiment_std" in result.columns
        assert "news_count" in result.columns
        assert "positive_ratio" in result.columns
        assert "negative_ratio" in result.columns
        
        # Check aggregation worked
        assert result["news_count"].min() >= 3  # At least 3 headlines per day
    
    def test_add_rolling_sentiment(self, sample_raw_sentiment):
        """Test rolling sentiment computation."""
        agg = aggregate_daily_sentiment(sample_raw_sentiment)
        result = add_rolling_sentiment(agg, windows=[3, 5])
        
        assert "sentiment_3d_ma" in result.columns
        assert "sentiment_5d_ma" in result.columns
    
    def test_compute_sentiment_momentum(self, sample_raw_sentiment):
        """Test sentiment momentum computation."""
        agg = aggregate_daily_sentiment(sample_raw_sentiment)
        result = compute_sentiment_momentum(agg, short_window=3, long_window=5)
        
        assert "sentiment_momentum" in result.columns
        # Momentum should be bounded (short MA - long MA)
        assert result["sentiment_momentum"].abs().max() <= 2.0
    
    def test_normalize_news_volume_zscore(self, sample_raw_sentiment):
        """Test z-score normalization of news volume."""
        agg = aggregate_daily_sentiment(sample_raw_sentiment)
        result = normalize_news_volume(agg, method="zscore")
        
        assert "news_volume_norm" in result.columns
    
    def test_normalize_news_volume_log(self, sample_raw_sentiment):
        """Test log normalization of news volume."""
        agg = aggregate_daily_sentiment(sample_raw_sentiment)
        result = normalize_news_volume(agg, method="log")
        
        assert "news_volume_norm" in result.columns
        assert all(result["news_volume_norm"] >= 0)  # log1p is always >= 0
    
    def test_compute_sector_sentiment(self, sample_raw_sentiment):
        """Test sector sentiment computation."""
        agg = aggregate_daily_sentiment(sample_raw_sentiment)
        result = compute_sector_sentiment(agg)
        
        assert "sector_sentiment" in result.columns
    
    def test_compute_market_sentiment(self, sample_raw_sentiment):
        """Test market sentiment computation."""
        agg = aggregate_daily_sentiment(sample_raw_sentiment)
        result = compute_market_sentiment(agg, market_ticker="SPY")
        
        assert "market_sentiment" in result.columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the sentiment module."""
    
    def test_full_pipeline(self, sample_sentiment_csv):
        """Test full pipeline from CSV to features."""
        # Load from CSV
        provider = CsvFileProvider(sample_sentiment_csv)
        
        # Get sentiment
        sentiment = provider.get_sentiment(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        
        # Verify output schema
        assert "date" in sentiment.columns
        assert "ticker" in sentiment.columns
        assert "sentiment_score" in sentiment.columns
        
        # Check data types
        assert sentiment["sentiment_score"].dtype in [np.float64, np.float32]
    
    def test_provider_interchangeability(self, sample_sentiment_csv):
        """Test that providers can be used interchangeably."""
        csv_provider = CsvFileProvider(sample_sentiment_csv)
        api_provider = FinBertApiProvider(fallback_to_neutral=True)
        
        tickers = ["AAPL", "MSFT"]
        start = "2024-01-02"
        end = "2024-01-05"
        
        # Both should return DataFrames with same schema
        csv_result = csv_provider.get_sentiment(tickers, start, end)
        api_result = api_provider.get_sentiment(tickers, start, end)
        
        # Same columns
        assert set(csv_result.columns) == set(api_result.columns)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
