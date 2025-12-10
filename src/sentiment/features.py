"""
Sentiment Feature Engineering Module

Computes derived sentiment features for the RL portfolio environment.
These features transform raw sentiment data into actionable signals.

Features computed:
- sentiment_score: Daily mean FinBERT score (-1 to 1)
- sentiment_std: Daily sentiment volatility (uncertainty indicator)
- sentiment_momentum: Short-term trend (3d SMA - 7d SMA)
- news_volume: Normalized article count (attention indicator)
- sector_sentiment: Sector peer average sentiment
- market_sentiment: SPY sentiment as market-wide proxy

Usage:
    from src.sentiment.features import compute_sentiment_features
    
    features_df = compute_sentiment_features(
        sentiment_df,
        tickers=['AAPL', 'MSFT', ...],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from src.sentiment.aggregator import (
    aggregate_daily_sentiment,
    add_rolling_sentiment,
    compute_sentiment_momentum,
    normalize_news_volume,
    compute_sector_sentiment,
    compute_market_sentiment,
)


# Default sector mapping for portfolio tickers
DEFAULT_SECTOR_MAPPING = {
    "AAPL": "tech",
    "MSFT": "tech",
    "GOOGL": "tech",
    "AMZN": "tech",
    "NVDA": "tech",
    "JPM": "finance",
    "BAC": "finance",
    "GLD": "commodity",
    "TLT": "bond",
    "SPY": "market",
}


def compute_sentiment_features(
    sentiment_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
    short_window: int = 3,
    long_window: int = 7,
    normalize_volume: bool = True,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """
    Compute all derived sentiment features from raw sentiment data.
    
    This is the main entry point for sentiment feature engineering.
    Takes raw sentiment predictions and produces features ready for
    the RL environment.
    
    Args:
        sentiment_df: DataFrame with columns:
            - date: str (YYYY-MM-DD)
            - ticker: str
            - sentiment_score: float (-1 to 1)
            - news_count: int (optional)
        tickers: List of tickers to process (None = all in data)
        sector_mapping: Dict mapping ticker -> sector (uses default if None)
        short_window: Window for short-term MA (default: 3 days)
        long_window: Window for long-term MA (default: 7 days)
        normalize_volume: Whether to normalize news volume (default: True)
        fill_missing: Whether to fill missing dates with neutral sentiment
        
    Returns:
        DataFrame with columns:
        - date: str
        - ticker: str
        - sentiment_score: float (daily mean, -1 to 1)
        - sentiment_std: float (daily volatility)
        - sentiment_momentum: float (short MA - long MA)
        - news_volume: float (normalized article count)
        - sector_sentiment: float (sector peer average)
        - market_sentiment: float (SPY sentiment proxy)
    """
    if sentiment_df.empty:
        return _create_empty_features_df()
    
    # Use default sector mapping if not provided
    if sector_mapping is None:
        sector_mapping = DEFAULT_SECTOR_MAPPING
    
    # Filter to requested tickers
    if tickers is not None:
        sentiment_df = sentiment_df[sentiment_df["ticker"].isin(tickers)].copy()
    
    if sentiment_df.empty:
        return _create_empty_features_df()
    
    # Step 1: Aggregate to daily level (handles multiple headlines per day)
    # Check if already aggregated (has sentiment_mean) or raw (has sentiment_score)
    if "sentiment_mean" not in sentiment_df.columns:
        if "sentiment_score" in sentiment_df.columns:
            daily_df = aggregate_daily_sentiment(
                sentiment_df,
                date_column="date",
                ticker_column="ticker",
                score_column="sentiment_score",
            )
        else:
            raise ValueError("DataFrame must have 'sentiment_score' or 'sentiment_mean' column")
    else:
        daily_df = sentiment_df.copy()
    
    # Rename sentiment_mean to sentiment_score for output consistency
    if "sentiment_mean" in daily_df.columns:
        daily_df = daily_df.rename(columns={"sentiment_mean": "sentiment_score"})
    
    # Step 2: Add rolling sentiment features
    daily_df = add_rolling_sentiment(
        daily_df,
        windows=[short_window, long_window],
        score_column="sentiment_score",
    )
    
    # Step 3: Compute sentiment momentum
    daily_df = compute_sentiment_momentum(
        daily_df,
        short_window=short_window,
        long_window=long_window,
        score_column="sentiment_score",
    )
    
    # Step 4: Normalize news volume
    if normalize_volume and "news_count" in daily_df.columns:
        daily_df = normalize_news_volume(
            daily_df,
            count_column="news_count",
            method="zscore",
        )
        daily_df = daily_df.rename(columns={"news_volume_norm": "news_volume"})
    elif "news_count" in daily_df.columns:
        daily_df["news_volume"] = daily_df["news_count"].astype(float)
    else:
        daily_df["news_volume"] = 0.0
    
    # Step 5: Compute sector sentiment
    daily_df = compute_sector_sentiment(daily_df, sector_mapping=sector_mapping)
    
    # Step 6: Compute market sentiment (SPY proxy)
    daily_df = compute_market_sentiment(daily_df, market_ticker="SPY")
    
    # Step 7: Fill missing values if requested
    if fill_missing:
        daily_df = _fill_missing_features(daily_df)
    
    # Select and order output columns
    output_columns = [
        "date",
        "ticker",
        "sentiment_score",
        "sentiment_std",
        "sentiment_momentum",
        "news_volume",
        "sector_sentiment",
        "market_sentiment",
    ]
    
    # Ensure all columns exist
    for col in output_columns:
        if col not in daily_df.columns:
            if col == "sentiment_std":
                daily_df[col] = 0.0
            else:
                daily_df[col] = 0.0
    
    result = daily_df[output_columns].copy()
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    return result


def _create_empty_features_df() -> pd.DataFrame:
    """Create empty DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        "date", "ticker", "sentiment_score", "sentiment_std",
        "sentiment_momentum", "news_volume", "sector_sentiment", "market_sentiment"
    ])


def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in sentiment features."""
    df = df.copy()
    
    # Fill NaN with neutral values
    fill_values = {
        "sentiment_score": 0.0,
        "sentiment_std": 0.0,
        "sentiment_momentum": 0.0,
        "news_volume": 0.0,
        "sector_sentiment": 0.0,
        "market_sentiment": 0.0,
    }
    
    for col, fill_val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)
    
    return df


def merge_sentiment_with_market_data(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "tic",
) -> pd.DataFrame:
    """
    Merge sentiment features with market OHLCV data.
    
    This function aligns sentiment features with market data,
    handling missing dates by forward-filling sentiment values.
    
    Args:
        market_df: DataFrame with market data (date, tic, OHLCV, indicators)
        sentiment_df: DataFrame with sentiment features (from compute_sentiment_features)
        date_column: Name of date column in market_df
        ticker_column: Name of ticker column in market_df
        
    Returns:
        DataFrame with market data + sentiment features merged
    """
    if sentiment_df.empty:
        # Return market data with neutral sentiment
        result = market_df.copy()
        for col in ["sentiment_score", "sentiment_std", "sentiment_momentum",
                    "news_volume", "sector_sentiment", "market_sentiment"]:
            result[col] = 0.0
        return result
    
    # Standardize column names for merge
    market_df = market_df.copy()
    sentiment_df = sentiment_df.copy()
    
    # Ensure date formats match
    market_df[date_column] = pd.to_datetime(market_df[date_column]).dt.strftime("%Y-%m-%d")
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.strftime("%Y-%m-%d")
    
    # Rename ticker column if needed
    if ticker_column != "ticker":
        sentiment_df = sentiment_df.rename(columns={"ticker": ticker_column})
    
    # Select sentiment columns for merge
    sentiment_cols = [
        "date", ticker_column, "sentiment_score", "sentiment_std",
        "sentiment_momentum", "news_volume", "sector_sentiment", "market_sentiment"
    ]
    sentiment_cols = [c for c in sentiment_cols if c in sentiment_df.columns]
    sentiment_merge = sentiment_df[sentiment_cols]
    
    # Merge on date and ticker
    result = market_df.merge(
        sentiment_merge,
        left_on=[date_column, ticker_column],
        right_on=["date", ticker_column],
        how="left",
        suffixes=("", "_sent"),
    )
    
    # Drop duplicate date column if created
    if "date_sent" in result.columns:
        result = result.drop(columns=["date_sent"])
    if "date" in result.columns and date_column != "date":
        result = result.drop(columns=["date"])
    
    # Forward fill missing sentiment (for dates without news)
    sentiment_feature_cols = [
        "sentiment_score", "sentiment_std", "sentiment_momentum",
        "news_volume", "sector_sentiment", "market_sentiment"
    ]
    
    for col in sentiment_feature_cols:
        if col in result.columns:
            # Forward fill within each ticker
            result[col] = result.groupby(ticker_column)[col].ffill()
            # Fill any remaining NaN with neutral
            result[col] = result[col].fillna(0.0)
    
    return result


def normalize_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    method: str = "zscore",
    clip_range: tuple = (-3, 3),
) -> pd.DataFrame:
    """
    Normalize sentiment features for model input.
    
    Applies normalization to ensure features are on comparable scales.
    
    Args:
        df: DataFrame with sentiment features
        feature_columns: Columns to normalize (default: all sentiment features)
        method: Normalization method ('zscore', 'minmax')
        clip_range: Range to clip z-scores (default: -3 to 3)
        
    Returns:
        DataFrame with normalized features
    """
    if feature_columns is None:
        feature_columns = [
            "sentiment_score", "sentiment_std", "sentiment_momentum",
            "news_volume", "sector_sentiment", "market_sentiment"
        ]
    
    df = df.copy()
    
    for col in feature_columns:
        if col not in df.columns:
            continue
            
        if method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
                # Clip extreme values
                df[col] = df[col].clip(clip_range[0], clip_range[1])
            else:
                df[col] = 0.0
                
        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.5
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df


def get_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for sentiment features.
    
    Useful for monitoring feature distributions and detecting anomalies.
    
    Args:
        df: DataFrame with sentiment features
        
    Returns:
        Dict with statistics for each feature
    """
    feature_columns = [
        "sentiment_score", "sentiment_std", "sentiment_momentum",
        "news_volume", "sector_sentiment", "market_sentiment"
    ]
    
    stats = {}
    for col in feature_columns:
        if col in df.columns:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
                "null_count": df[col].isna().sum(),
            }
    
    return stats
