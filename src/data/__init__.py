"""
Data Collection Module

Provides utilities for collecting historical financial news data
for sentiment analysis.

Components:
- news_collector.py: Multi-source news data collection
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

__all__ = [
    "NewsCollector",
    "YahooFinanceCollector",
    "GDELTCollector",
    "MarketProxySentimentGenerator",
    "collect_historical_news",
    "generate_coverage_report",
    "generate_sample_news_data",
]
