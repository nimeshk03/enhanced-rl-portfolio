"""
Tests for Sentiment Inference Module

Tests the sentiment inference pipeline:
- SentimentInferenceEngine
- aggregate_to_daily_sentiment
- generate_historical_sentiment
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sentiment_inference import (
    SentimentInferenceEngine,
    aggregate_to_daily_sentiment,
    generate_historical_sentiment,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_headlines():
    """Sample headlines for testing."""
    return [
        "Apple stock surges on strong earnings report",
        "Microsoft shares fall after disappointing guidance",
        "Google announces new AI product launch",
        "Amazon stock drops amid market volatility",
        "NVIDIA rallies on chip demand surge",
    ]


@pytest.fixture
def sample_news_df():
    """Sample news DataFrame."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "SPY"]:
            for i in range(np.random.randint(1, 4)):
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "headline": f"{ticker} stock news {i} for {date.strftime('%Y-%m-%d')}",
                    "source": "Test",
                })
    return pd.DataFrame(records)


@pytest.fixture
def sample_sentiment_df():
    """Sample sentiment DataFrame with scores."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "SPY"]:
            for i in range(np.random.randint(2, 5)):
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "headline": f"{ticker} news {i}",
                    "sentiment_score": np.random.uniform(-0.5, 0.8),
                    "sentiment_label": np.random.choice(["positive", "negative", "neutral"]),
                    "sentiment_confidence": np.random.uniform(0.6, 0.95),
                })
    return pd.DataFrame(records)


# =============================================================================
# SentimentInferenceEngine Tests
# =============================================================================

class TestSentimentInferenceEngine:
    """Tests for SentimentInferenceEngine."""
    
    def test_initialization_with_fallback(self):
        """Test engine initializes with simple fallback."""
        engine = SentimentInferenceEngine(use_simple_fallback=True)
        assert engine._backend in ["transformers", "simple"]
    
    def test_predict_batch_simple(self, sample_headlines):
        """Test batch prediction with simple backend."""
        engine = SentimentInferenceEngine(use_simple_fallback=True)
        
        # Force simple backend for testing
        engine._backend = "simple"
        
        results = engine.predict_batch(sample_headlines)
        
        assert len(results) == len(sample_headlines)
        for result in results:
            assert "sentiment_score" in result
            assert "sentiment_label" in result
            assert "confidence" in result
            assert -1 <= result["sentiment_score"] <= 1
            assert result["sentiment_label"] in ["positive", "negative", "neutral"]
    
    def test_simple_backend_positive_detection(self):
        """Test simple backend detects positive sentiment."""
        engine = SentimentInferenceEngine(use_simple_fallback=True)
        engine._backend = "simple"
        
        positive_texts = [
            "Stock surges on strong earnings",
            "Company beats expectations, shares rally",
            "Bullish outlook drives gains",
        ]
        
        results = engine.predict_batch(positive_texts)
        
        # Most should be positive
        positive_count = sum(1 for r in results if r["sentiment_score"] > 0)
        assert positive_count >= 2
    
    def test_simple_backend_negative_detection(self):
        """Test simple backend detects negative sentiment."""
        engine = SentimentInferenceEngine(use_simple_fallback=True)
        engine._backend = "simple"
        
        negative_texts = [
            "Stock plunges on weak earnings",
            "Company misses expectations, shares tumble",
            "Bearish outlook drives losses",
        ]
        
        results = engine.predict_batch(negative_texts)
        
        # Most should be negative
        negative_count = sum(1 for r in results if r["sentiment_score"] < 0)
        assert negative_count >= 2
    
    def test_process_news_file(self, sample_news_df, tmp_path):
        """Test processing news file."""
        # Save sample news
        input_path = tmp_path / "news.csv"
        sample_news_df.to_csv(input_path, index=False)
        
        # Process
        engine = SentimentInferenceEngine(use_simple_fallback=True)
        engine._backend = "simple"
        
        result = engine.process_news_file(str(input_path))
        
        assert len(result) == len(sample_news_df)
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
        assert "sentiment_confidence" in result.columns
    
    def test_process_news_file_with_output(self, sample_news_df, tmp_path):
        """Test processing news file with output path."""
        input_path = tmp_path / "news.csv"
        output_path = tmp_path / "sentiment.csv"
        sample_news_df.to_csv(input_path, index=False)
        
        engine = SentimentInferenceEngine(use_simple_fallback=True)
        engine._backend = "simple"
        
        result = engine.process_news_file(str(input_path), str(output_path))
        
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(result)


# =============================================================================
# aggregate_to_daily_sentiment Tests
# =============================================================================

class TestAggregateToDailySentiment:
    """Tests for aggregate_to_daily_sentiment function."""
    
    def test_basic_aggregation(self, sample_sentiment_df):
        """Test basic daily aggregation."""
        result = aggregate_to_daily_sentiment(sample_sentiment_df)
        
        # Should have one row per date-ticker combination
        expected_combinations = sample_sentiment_df.groupby(["date", "ticker"]).ngroups
        assert len(result) == expected_combinations
    
    def test_output_columns(self, sample_sentiment_df):
        """Test output has required columns."""
        result = aggregate_to_daily_sentiment(sample_sentiment_df)
        
        required_cols = [
            "date", "ticker", "sentiment_score", "sentiment_std",
            "news_count", "positive_ratio", "negative_ratio"
        ]
        
        for col in required_cols:
            assert col in result.columns
    
    def test_sentiment_score_range(self, sample_sentiment_df):
        """Test aggregated scores are in valid range."""
        result = aggregate_to_daily_sentiment(sample_sentiment_df)
        
        assert result["sentiment_score"].min() >= -1.0
        assert result["sentiment_score"].max() <= 1.0
    
    def test_news_count_positive(self, sample_sentiment_df):
        """Test news count is positive."""
        result = aggregate_to_daily_sentiment(sample_sentiment_df)
        
        assert (result["news_count"] > 0).all()
    
    def test_ratios_valid(self, sample_sentiment_df):
        """Test positive/negative ratios are valid."""
        result = aggregate_to_daily_sentiment(sample_sentiment_df)
        
        assert (result["positive_ratio"] >= 0).all()
        assert (result["positive_ratio"] <= 1).all()
        assert (result["negative_ratio"] >= 0).all()
        assert (result["negative_ratio"] <= 1).all()
    
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_df = pd.DataFrame(columns=["date", "ticker", "sentiment_score"])
        result = aggregate_to_daily_sentiment(empty_df)
        
        assert len(result) == 0
        assert "sentiment_score" in result.columns
    
    def test_save_output(self, sample_sentiment_df, tmp_path):
        """Test saving output to file."""
        output_path = tmp_path / "daily_sentiment.csv"
        
        result = aggregate_to_daily_sentiment(
            sample_sentiment_df,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(result)


# =============================================================================
# generate_historical_sentiment Tests
# =============================================================================

class TestGenerateHistoricalSentiment:
    """Tests for generate_historical_sentiment function."""
    
    def test_full_pipeline(self, sample_news_df, tmp_path):
        """Test full sentiment generation pipeline."""
        # Save sample news
        news_path = tmp_path / "news.csv"
        output_path = tmp_path / "sentiment.csv"
        sample_news_df.to_csv(news_path, index=False)
        
        # Run pipeline
        result = generate_historical_sentiment(
            news_path=str(news_path),
            output_path=str(output_path),
            use_gpu=False,
        )
        
        assert len(result) > 0
        assert output_path.exists()
        assert "sentiment_score" in result.columns
        assert "ticker" in result.columns
    
    def test_output_format(self, sample_news_df, tmp_path):
        """Test output format matches expected schema."""
        news_path = tmp_path / "news.csv"
        output_path = tmp_path / "sentiment.csv"
        sample_news_df.to_csv(news_path, index=False)
        
        result = generate_historical_sentiment(
            news_path=str(news_path),
            output_path=str(output_path),
        )
        
        # Check all required columns
        required = ["date", "ticker", "sentiment_score", "sentiment_std", "news_count"]
        for col in required:
            assert col in result.columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for sentiment inference pipeline."""
    
    def test_end_to_end_pipeline(self, tmp_path):
        """Test complete pipeline from news to daily sentiment."""
        # Create sample news data
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        records = []
        for date in dates:
            for ticker in ["AAPL", "MSFT", "SPY"]:
                n_headlines = np.random.randint(1, 5)
                for i in range(n_headlines):
                    sentiment = np.random.choice(["surges", "falls", "steady"])
                    records.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "headline": f"{ticker} stock {sentiment} on market news",
                        "source": "Test",
                    })
        
        news_df = pd.DataFrame(records)
        news_path = tmp_path / "news.csv"
        output_path = tmp_path / "sentiment.csv"
        news_df.to_csv(news_path, index=False)
        
        # Run full pipeline
        result = generate_historical_sentiment(
            news_path=str(news_path),
            output_path=str(output_path),
        )
        
        # Verify results
        assert len(result) > 0
        assert result["date"].nunique() == 20
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT", "SPY"}
        assert result["sentiment_score"].between(-1, 1).all()
    
    def test_plan_test_criteria(self, tmp_path):
        """Test the criteria from the implementation plan."""
        # Generate sample data that meets criteria
        dates = pd.bdate_range("2015-01-01", "2024-12-31")
        records = []
        for date in dates[:100]:  # Use subset for speed
            for ticker in ["AAPL", "MSFT", "SPY"]:
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "headline": f"{ticker} stock news",
                    "source": "Test",
                })
        
        news_df = pd.DataFrame(records)
        news_path = tmp_path / "news.csv"
        output_path = tmp_path / "sentiment.csv"
        news_df.to_csv(news_path, index=False)
        
        result = generate_historical_sentiment(
            news_path=str(news_path),
            output_path=str(output_path),
        )
        
        # Verify criteria
        assert set(["AAPL", "MSFT", "SPY"]).issubset(set(result["ticker"].unique()))
        assert result["sentiment_score"].between(-1, 1).all()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
