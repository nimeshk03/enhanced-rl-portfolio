"""
Tests for News Data Collection Module

Tests the news collection infrastructure:
- NewsCollector abstract interface
- YahooFinanceCollector
- GDELTCollector
- MarketProxySentimentGenerator
- Coverage report generation
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.news_collector import (
    NewsCollector,
    YahooFinanceCollector,
    GDELTCollector,
    MarketProxySentimentGenerator,
    collect_historical_news,
    generate_coverage_report,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_news_df():
    """Create sample news DataFrame."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "SPY"]:
            # 1-3 headlines per ticker per day
            n_headlines = np.random.randint(1, 4)
            for i in range(n_headlines):
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "headline": f"{ticker} news headline {i} for {date.strftime('%Y-%m-%d')}",
                    "source": "Test Source",
                    "url": f"https://example.com/{ticker}/{date.strftime('%Y%m%d')}/{i}",
                })
    return pd.DataFrame(records)


@pytest.fixture
def sample_market_data(tmp_path):
    """Create sample market data file."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    records = []
    for date in dates:
        for ticker in ["AAPL", "MSFT", "SPY"]:
            base_price = {"AAPL": 180, "MSFT": 380, "SPY": 480}[ticker]
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "open": base_price + np.random.uniform(-2, 2),
                "high": base_price + np.random.uniform(0, 3),
                "low": base_price + np.random.uniform(-3, 0),
                "close": base_price + np.random.uniform(-2, 2),
                "volume": np.random.randint(1000000, 10000000),
                "vix": np.random.uniform(12, 25),
            })
    
    df = pd.DataFrame(records)
    market_file = tmp_path / "processed_data.csv"
    df.to_csv(market_file, index=False)
    return str(market_file)


@pytest.fixture
def mock_yfinance_news():
    """Create mock yfinance news response."""
    return [
        {
            "title": "Apple announces new product",
            "publisher": "Reuters",
            "link": "https://example.com/apple-news-1",
            "providerPublishTime": int(datetime(2024, 6, 15, 10, 30).timestamp()),
        },
        {
            "title": "Apple stock rises on earnings",
            "publisher": "Bloomberg",
            "link": "https://example.com/apple-news-2",
            "providerPublishTime": int(datetime(2024, 6, 14, 14, 0).timestamp()),
        },
    ]


# =============================================================================
# YahooFinanceCollector Tests
# =============================================================================

class TestYahooFinanceCollector:
    """Tests for YahooFinanceCollector."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = YahooFinanceCollector(delay_seconds=0.5)
        assert collector.delay_seconds == 0.5
        assert collector.source_name == "Yahoo Finance"
    
    def test_rate_limit_info(self):
        """Test rate limit info."""
        collector = YahooFinanceCollector()
        info = collector.get_rate_limit_info()
        
        assert "requests_per_minute" in info
        assert "delay_seconds" in info
        assert "historical_limit" in info
    
    @patch("yfinance.Ticker")
    def test_collect_with_mock(self, mock_ticker_class, mock_yfinance_news):
        """Test collection with mocked yfinance."""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.news = mock_yfinance_news
        mock_ticker_class.return_value = mock_ticker
        
        collector = YahooFinanceCollector(delay_seconds=0)
        
        # Patch the lazy import
        with patch.object(collector, "_get_yfinance") as mock_yf:
            mock_yf_module = MagicMock()
            mock_yf_module.Ticker = mock_ticker_class
            mock_yf.return_value = mock_yf_module
            
            result = collector.collect(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31"
            )
        
        # Verify structure
        assert isinstance(result, pd.DataFrame)
        expected_cols = ["date", "ticker", "headline", "source", "url"]
        for col in expected_cols:
            assert col in result.columns
    
    def test_empty_news_handling(self):
        """Test handling when no news is returned."""
        collector = YahooFinanceCollector(delay_seconds=0)
        
        with patch.object(collector, "_get_yfinance") as mock_yf:
            mock_yf_module = MagicMock()
            mock_ticker = MagicMock()
            mock_ticker.news = []  # Empty news
            mock_yf_module.Ticker.return_value = mock_ticker
            mock_yf.return_value = mock_yf_module
            
            result = collector.collect(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31"
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# =============================================================================
# GDELTCollector Tests
# =============================================================================

class TestGDELTCollector:
    """Tests for GDELTCollector."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = GDELTCollector(delay_seconds=1.0, max_records_per_query=100)
        assert collector.delay_seconds == 1.0
        assert collector.max_records == 100
        assert collector.source_name == "GDELT"
    
    def test_rate_limit_info(self):
        """Test rate limit info."""
        collector = GDELTCollector()
        info = collector.get_rate_limit_info()
        
        assert "requests_per_minute" in info
        assert "historical_limit" in info
        assert info["historical_limit"] == "2015+"
    
    def test_company_name_mapping(self):
        """Test company name lookup."""
        collector = GDELTCollector()
        
        assert collector._get_company_name("AAPL") == "Apple"
        assert collector._get_company_name("MSFT") == "Microsoft"
        assert collector._get_company_name("UNKNOWN") == "UNKNOWN"
    
    def test_query_building(self):
        """Test GDELT query construction."""
        collector = GDELTCollector()
        query = collector._build_query("AAPL", "Apple")
        
        # Query uses company name + stock for GDELT compatibility
        assert "Apple" in query
        assert "stock" in query
    
    @patch("requests.get")
    def test_collect_with_mock(self, mock_get):
        """Test collection with mocked API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Apple stock news",
                    "seendate": "20240615120000",
                    "domain": "reuters.com",
                    "url": "https://reuters.com/apple",
                },
            ]
        }
        mock_get.return_value = mock_response
        
        collector = GDELTCollector(delay_seconds=0)
        result = collector.collect(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"
        assert result.iloc[0]["date"] == "2024-06-15"
    
    @patch("requests.get")
    def test_api_error_handling(self, mock_get):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        collector = GDELTCollector(delay_seconds=0)
        result = collector.collect(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# =============================================================================
# MarketProxySentimentGenerator Tests
# =============================================================================

class TestMarketProxySentimentGenerator:
    """Tests for MarketProxySentimentGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = MarketProxySentimentGenerator("./data/test.csv")
        assert generator.market_data_path == "./data/test.csv"
    
    def test_generate_with_valid_data(self, sample_market_data):
        """Test proxy generation with valid market data."""
        generator = MarketProxySentimentGenerator(sample_market_data)
        
        result = generator.generate(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-03-31"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert "sentiment_proxy" in result.columns
        assert "ticker" in result.columns
        assert "date" in result.columns
        
        # Sentiment proxy should be in [-1, 1]
        assert result["sentiment_proxy"].min() >= -1.0
        assert result["sentiment_proxy"].max() <= 1.0
    
    def test_generate_missing_file(self):
        """Test handling of missing market data file."""
        generator = MarketProxySentimentGenerator("./nonexistent.csv")
        
        result = generator.generate(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_proxy_source_label(self, sample_market_data):
        """Test that proxy data is labeled correctly."""
        generator = MarketProxySentimentGenerator(sample_market_data)
        
        result = generator.generate(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-31"
        )
        
        if len(result) > 0:
            assert all(result["source"] == "Market Proxy")
            assert all(result["headline"] == "[Market Proxy]")


# =============================================================================
# collect_historical_news Tests
# =============================================================================

class TestCollectHistoricalNews:
    """Tests for collect_historical_news function."""
    
    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "news_output"
        
        with patch("src.data.news_collector.YahooFinanceCollector") as mock_collector:
            mock_instance = MagicMock()
            mock_instance.collect.return_value = pd.DataFrame(columns=[
                "date", "ticker", "headline", "source", "url"
            ])
            mock_collector.return_value = mock_instance
            
            collect_historical_news(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                sources=["yahoo"],
                output_dir=str(output_dir),
            )
        
        assert output_dir.exists()
    
    def test_unknown_source_warning(self, tmp_path, caplog):
        """Test warning for unknown source."""
        import logging
        caplog.set_level(logging.WARNING)
        
        result = collect_historical_news(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            sources=["unknown_source"],
            output_dir=str(tmp_path),
        )
        
        assert "Unknown source" in caplog.text


# =============================================================================
# generate_coverage_report Tests
# =============================================================================

class TestGenerateCoverageReport:
    """Tests for generate_coverage_report function."""
    
    def test_report_generation(self, sample_news_df, tmp_path):
        """Test coverage report generation."""
        output_path = tmp_path / "coverage_report.csv"
        
        report = generate_coverage_report(
            news_df=sample_news_df,
            tickers=["AAPL", "MSFT", "SPY"],
            start_date="2024-01-01",
            end_date="2024-02-28",
            output_path=str(output_path),
        )
        
        assert isinstance(report, pd.DataFrame)
        assert output_path.exists()
        
        # Check report columns
        expected_cols = [
            "ticker", "total_headlines", "unique_dates",
            "date_coverage_pct", "avg_headlines_per_day"
        ]
        for col in expected_cols:
            assert col in report.columns
    
    def test_report_includes_summary(self, sample_news_df, tmp_path):
        """Test that report includes TOTAL summary row."""
        output_path = tmp_path / "coverage_report.csv"
        
        report = generate_coverage_report(
            news_df=sample_news_df,
            tickers=["AAPL", "MSFT", "SPY"],
            start_date="2024-01-01",
            end_date="2024-02-28",
            output_path=str(output_path),
        )
        
        assert "TOTAL" in report["ticker"].values
    
    def test_empty_news_handling(self, tmp_path):
        """Test report generation with empty news data."""
        output_path = tmp_path / "coverage_report.csv"
        empty_df = pd.DataFrame(columns=["date", "ticker", "headline", "source", "url"])
        
        report = generate_coverage_report(
            news_df=empty_df,
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path=str(output_path),
        )
        
        assert isinstance(report, pd.DataFrame)
        # All coverage should be 0
        assert report[report["ticker"] != "TOTAL"]["date_coverage_pct"].sum() == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for news collection pipeline."""
    
    def test_full_pipeline_mock(self, tmp_path):
        """Test full pipeline with mocked collectors."""
        output_dir = tmp_path / "news"
        
        # Create mock news data
        mock_news = pd.DataFrame({
            "date": ["2024-06-15", "2024-06-16"],
            "ticker": ["AAPL", "AAPL"],
            "headline": ["News 1", "News 2"],
            "source": ["Test", "Test"],
            "url": ["http://test1", "http://test2"],
        })
        
        with patch("src.data.news_collector.YahooFinanceCollector") as mock_yahoo:
            mock_instance = MagicMock()
            mock_instance.collect.return_value = mock_news
            mock_yahoo.return_value = mock_instance
            
            # Collect news
            news_df = collect_historical_news(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                sources=["yahoo"],
                output_dir=str(output_dir),
            )
            
            # Generate report
            report = generate_coverage_report(
                news_df=news_df,
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                output_path=str(output_dir / "report.csv"),
            )
        
        assert len(news_df) == 2
        assert isinstance(report, pd.DataFrame)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
