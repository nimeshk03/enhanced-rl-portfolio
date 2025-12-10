"""
Data Collection and Processing Module

Provides utilities for collecting historical financial news data
and running sentiment inference.

Components:
- news_collector.py: Multi-source news data collection
- sentiment_inference.py: FinBERT sentiment inference engine
"""

from src.data.news_collector import (
    NewsCollector,
    YahooFinanceCollector,
    GDELTCollector,
    MarketProxySentimentGenerator,
    collect_historical_news,
    generate_coverage_report,
    generate_sample_news_data,
)

from src.data.sentiment_inference import (
    SentimentInferenceEngine,
    aggregate_to_daily_sentiment,
    generate_historical_sentiment,
)

__all__ = [
    # News collection
    "NewsCollector",
    "YahooFinanceCollector",
    "GDELTCollector",
    "MarketProxySentimentGenerator",
    "collect_historical_news",
    "generate_coverage_report",
    "generate_sample_news_data",
    # Sentiment inference
    "SentimentInferenceEngine",
    "aggregate_to_daily_sentiment",
    "generate_historical_sentiment",
]
