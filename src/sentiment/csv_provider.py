"""
CSV File Sentiment Data Provider

Loads pre-computed sentiment data from CSV files.
Used for historical backtesting and training.

Expected CSV format:
- date: YYYY-MM-DD
- ticker: Stock symbol
- sentiment_score: Float (-1 to 1)
- sentiment_confidence: Float (0 to 1, optional)
- news_count: Integer (optional)
"""

import os
from typing import List, Optional
import pandas as pd

from src.sentiment.provider import SentimentDataProvider


class CsvFileProvider(SentimentDataProvider):
    """
    Load sentiment data from CSV files.
    
    This provider is used for:
    - Historical backtesting with pre-computed sentiment
    - Training RL models on historical sentiment data
    - Testing and development
    
    Example usage:
        provider = CsvFileProvider("./data/historical_sentiment.csv")
        sentiment = provider.get_sentiment(
            tickers=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
    """
    
    def __init__(
        self,
        csv_path: str,
        date_column: str = "date",
        ticker_column: str = "ticker",
        score_column: str = "sentiment_score",
        confidence_column: Optional[str] = "sentiment_confidence",
        count_column: Optional[str] = "news_count",
    ):
        """
        Initialize CSV provider.
        
        Args:
            csv_path: Path to the sentiment CSV file
            date_column: Name of the date column
            ticker_column: Name of the ticker column
            score_column: Name of the sentiment score column
            confidence_column: Name of confidence column (optional)
            count_column: Name of news count column (optional)
        """
        self.csv_path = csv_path
        self.date_column = date_column
        self.ticker_column = ticker_column
        self.score_column = score_column
        self.confidence_column = confidence_column
        self.count_column = count_column
        
        self._data: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and validate CSV data."""
        if not os.path.exists(self.csv_path):
            print(f"[WARNING] Sentiment CSV not found: {self.csv_path}")
            self._data = None
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            
            # Validate required columns
            required = [self.date_column, self.ticker_column, self.score_column]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Standardize column names
            df = df.rename(columns={
                self.date_column: "date",
                self.ticker_column: "ticker",
                self.score_column: "sentiment_score",
            })
            
            # Add optional columns if present
            if self.confidence_column and self.confidence_column in df.columns:
                df = df.rename(columns={self.confidence_column: "sentiment_confidence"})
            else:
                df["sentiment_confidence"] = 1.0  # Default confidence
                
            if self.count_column and self.count_column in df.columns:
                df = df.rename(columns={self.count_column: "news_count"})
            else:
                df["news_count"] = 1  # Default count
            
            # Ensure date is string format
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            # Sort by date and ticker
            df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
            
            self._data = df
            print(f"[OK] Loaded sentiment data: {len(df)} records")
            print(f"     Tickers: {df['ticker'].nunique()}")
            print(f"     Date range: {df['date'].min()} to {df['date'].max()}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load sentiment CSV: {e}")
            self._data = None
    
    def get_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get sentiment data for specified tickers and date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with sentiment data
        """
        self.validate_date_range(start_date, end_date)
        
        if self._data is None:
            print(f"[WARNING] No sentiment data available, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "ticker", "sentiment_score", 
                                         "sentiment_confidence", "news_count"])
        
        # Filter by date range and tickers
        mask = (
            (self._data["date"] >= start_date) &
            (self._data["date"] <= end_date) &
            (self._data["ticker"].isin(tickers))
        )
        
        result = self._data[mask].copy()
        
        if len(result) == 0:
            print(f"[WARNING] No sentiment data found for {tickers} in {start_date} to {end_date}")
        
        return result
    
    def get_latest_sentiment(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get the most recent sentiment for each ticker.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with latest sentiment per ticker
        """
        if self._data is None:
            return pd.DataFrame(columns=["date", "ticker", "sentiment_score",
                                         "sentiment_confidence", "news_count"])
        
        # Filter to requested tickers
        df = self._data[self._data["ticker"].isin(tickers)]
        
        # Get latest date for each ticker
        latest = df.loc[df.groupby("ticker")["date"].idxmax()].reset_index(drop=True)
        
        return latest
    
    def is_available(self) -> bool:
        """Check if sentiment data is loaded and available."""
        return self._data is not None and len(self._data) > 0
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"CsvFileProvider({os.path.basename(self.csv_path)})"
    
    @property
    def supported_tickers(self) -> List[str]:
        """Return list of tickers with data."""
        if self._data is None:
            return []
        return self._data["ticker"].unique().tolist()
    
    def get_coverage_info(self) -> dict:
        """Get detailed coverage information."""
        base_info = super().get_coverage_info()
        
        if self._data is not None:
            base_info.update({
                "start_date": self._data["date"].min(),
                "end_date": self._data["date"].max(),
                "total_records": len(self._data),
                "records_per_ticker": self._data.groupby("ticker").size().to_dict(),
            })
        
        return base_info
