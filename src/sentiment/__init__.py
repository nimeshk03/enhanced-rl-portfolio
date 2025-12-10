"""
Sentiment Analysis Module

Provides modular sentiment data integration for the RL portfolio manager.
Designed for loose coupling - can be replaced with different sentiment sources.

Components:
- provider.py: Abstract base class for sentiment data providers
- csv_provider.py: Load sentiment from CSV files (historical data)
- api_provider.py: Fetch sentiment from live API (placeholder for FinBERT)
- aggregator.py: Utilities for aggregating raw sentiment to daily features
- features.py: Compute derived sentiment features
"""

from src.sentiment.provider import SentimentDataProvider
from src.sentiment.csv_provider import CsvFileProvider
from src.sentiment.api_provider import FinBertApiProvider

__all__ = [
    "SentimentDataProvider",
    "CsvFileProvider", 
    "FinBertApiProvider",
]
