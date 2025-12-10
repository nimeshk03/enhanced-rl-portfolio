"""
Abstract Base Class for Sentiment Data Providers

This module defines the interface that all sentiment data providers must implement.
This abstraction allows the portfolio manager to work with different sentiment sources:
- CSV files (historical data)
- Live API (FinBERT inference)
- Future: LangGraph agent integration

Design Principles:
- Loose coupling: Portfolio manager doesn't depend on specific sentiment implementation
- Easy testing: Can mock providers for unit tests
- Extensible: Add new providers without changing existing code
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import date
import pandas as pd


class SentimentDataProvider(ABC):
    """
    Abstract base class for sentiment data providers.
    
    All sentiment providers must implement these methods to ensure
    consistent interface across different data sources.
    
    Expected DataFrame Schema:
    - date: str (YYYY-MM-DD)
    - ticker: str (e.g., 'AAPL')
    - sentiment_score: float (-1 to 1, negative to positive)
    - sentiment_confidence: float (0 to 1, optional)
    - news_count: int (number of articles, optional)
    """
    
    @abstractmethod
    def get_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Retrieve sentiment data for specified tickers and date range.
        
        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns:
            - date: str
            - ticker: str  
            - sentiment_score: float (-1 to 1)
            - sentiment_confidence: float (optional)
            - news_count: int (optional)
            
        Raises:
            ValueError: If date range is invalid
            FileNotFoundError: If data source is unavailable
        """
        pass
    
    @abstractmethod
    def get_latest_sentiment(
        self,
        tickers: List[str],
    ) -> pd.DataFrame:
        """
        Get the most recent sentiment for specified tickers.
        
        Used for live trading/inference scenarios.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            DataFrame with latest sentiment for each ticker
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the sentiment data source is available.
        
        Returns:
            True if data source is accessible, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider for logging/debugging."""
        pass
    
    @property
    @abstractmethod
    def supported_tickers(self) -> List[str]:
        """Return list of tickers this provider has data for."""
        pass
    
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Validate that date range is valid.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start > end:
                raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
            return True
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def get_coverage_info(self) -> Dict[str, Any]:
        """
        Get information about data coverage.
        
        Returns:
            Dictionary with coverage metadata:
            - start_date: Earliest available date
            - end_date: Latest available date
            - tickers: List of available tickers
            - total_records: Total number of records
        """
        return {
            "provider": self.provider_name,
            "tickers": self.supported_tickers,
            "available": self.is_available(),
        }
