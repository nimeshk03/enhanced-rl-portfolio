"""
Sentiment Aggregation Utilities

Functions for aggregating raw sentiment predictions to daily features.
Mirrors the aggregation logic from financial-sentiment-transformers project.

These utilities transform raw FinBERT predictions (per-headline) into
daily aggregated features suitable for the RL portfolio environment.
"""

from typing import Optional, List
import pandas as pd
import numpy as np


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    score_column: str = "sentiment_score",
    confidence_column: Optional[str] = "sentiment_confidence",
) -> pd.DataFrame:
    """
    Aggregate raw sentiment predictions to daily level per ticker.
    
    Takes multiple sentiment predictions per day (one per headline) and
    aggregates them into daily summary statistics.
    
    Args:
        df: DataFrame with raw sentiment predictions
        date_column: Name of date column
        ticker_column: Name of ticker column
        score_column: Name of sentiment score column
        confidence_column: Name of confidence column (optional)
        
    Returns:
        DataFrame with daily aggregated sentiment:
        - date, ticker: Index columns
        - sentiment_mean: Mean sentiment score
        - sentiment_std: Standard deviation of sentiment
        - sentiment_min: Minimum sentiment
        - sentiment_max: Maximum sentiment
        - news_count: Number of articles
        - positive_ratio: Fraction of positive sentiment (>0.1)
        - negative_ratio: Fraction of negative sentiment (<-0.1)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "ticker", "sentiment_mean", "sentiment_std",
            "sentiment_min", "sentiment_max", "news_count",
            "positive_ratio", "negative_ratio"
        ])
    
    # Group by date and ticker
    grouped = df.groupby([date_column, ticker_column])
    
    # Compute aggregations
    agg_df = grouped.agg(
        sentiment_mean=(score_column, "mean"),
        sentiment_std=(score_column, "std"),
        sentiment_min=(score_column, "min"),
        sentiment_max=(score_column, "max"),
        news_count=(score_column, "count"),
    ).reset_index()
    
    # Fill NaN std (when only 1 article) with 0
    agg_df["sentiment_std"] = agg_df["sentiment_std"].fillna(0)
    
    # Compute positive/negative ratios
    def compute_ratios(group):
        scores = group[score_column]
        positive = (scores > 0.1).sum() / len(scores) if len(scores) > 0 else 0
        negative = (scores < -0.1).sum() / len(scores) if len(scores) > 0 else 0
        return pd.Series({"positive_ratio": positive, "negative_ratio": negative})
    
    ratios = grouped.apply(compute_ratios, include_groups=False).reset_index()
    
    # Merge ratios
    agg_df = agg_df.merge(ratios, on=[date_column, ticker_column])
    
    # Rename columns for consistency
    agg_df = agg_df.rename(columns={
        date_column: "date",
        ticker_column: "ticker",
    })
    
    return agg_df


def add_rolling_sentiment(
    df: pd.DataFrame,
    windows: List[int] = [3, 7, 14],
    score_column: str = "sentiment_mean",
) -> pd.DataFrame:
    """
    Add rolling average sentiment features.
    
    Computes rolling averages of sentiment over different time windows.
    Useful for capturing sentiment trends and momentum.
    
    Args:
        df: DataFrame with daily sentiment (must have date, ticker, sentiment columns)
        windows: List of rolling window sizes (in days)
        score_column: Column to compute rolling averages on
        
    Returns:
        DataFrame with additional rolling sentiment columns
    """
    if df.empty or score_column not in df.columns:
        return df
    
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    
    for window in windows:
        col_name = f"sentiment_{window}d_ma"
        df[col_name] = df.groupby("ticker")[score_column].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df


def compute_sentiment_momentum(
    df: pd.DataFrame,
    short_window: int = 3,
    long_window: int = 7,
    score_column: str = "sentiment_mean",
) -> pd.DataFrame:
    """
    Compute sentiment momentum (short MA - long MA).
    
    Positive momentum indicates improving sentiment.
    Negative momentum indicates deteriorating sentiment.
    
    Args:
        df: DataFrame with daily sentiment
        short_window: Short-term moving average window
        long_window: Long-term moving average window
        score_column: Column to compute momentum on
        
    Returns:
        DataFrame with sentiment_momentum column added
    """
    if df.empty or score_column not in df.columns:
        return df
    
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    
    # Compute short and long MAs
    df["_short_ma"] = df.groupby("ticker")[score_column].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    df["_long_ma"] = df.groupby("ticker")[score_column].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )
    
    # Momentum = short MA - long MA
    df["sentiment_momentum"] = df["_short_ma"] - df["_long_ma"]
    
    # Clean up temp columns
    df = df.drop(columns=["_short_ma", "_long_ma"])
    
    return df


def normalize_news_volume(
    df: pd.DataFrame,
    count_column: str = "news_count",
    method: str = "zscore",
    window: int = 30,
) -> pd.DataFrame:
    """
    Normalize news volume to comparable scale.
    
    Raw news counts vary significantly across tickers and time periods.
    Normalization makes the feature more useful for the model.
    
    Args:
        df: DataFrame with news_count column
        count_column: Name of the count column
        method: Normalization method ('zscore', 'minmax', 'log')
        window: Rolling window for zscore normalization
        
    Returns:
        DataFrame with news_volume_norm column added
    """
    if df.empty or count_column not in df.columns:
        return df
    
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    
    if method == "zscore":
        # Rolling z-score normalization per ticker
        def rolling_zscore(x):
            rolling_mean = x.rolling(window=window, min_periods=1).mean()
            rolling_std = x.rolling(window=window, min_periods=1).std().fillna(1)
            rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
            return (x - rolling_mean) / rolling_std
        
        df["news_volume_norm"] = df.groupby("ticker")[count_column].transform(rolling_zscore)
        
    elif method == "minmax":
        # Min-max normalization per ticker
        def minmax_norm(x):
            min_val = x.min()
            max_val = x.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(x), index=x.index)
            return (x - min_val) / (max_val - min_val)
        
        df["news_volume_norm"] = df.groupby("ticker")[count_column].transform(minmax_norm)
        
    elif method == "log":
        # Log transformation (handles skewed distributions)
        df["news_volume_norm"] = np.log1p(df[count_column])
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df


def compute_sector_sentiment(
    df: pd.DataFrame,
    sector_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute sector-level sentiment as peer average.
    
    For each ticker, computes the average sentiment of other tickers
    in the same sector. Useful for relative sentiment analysis.
    
    Args:
        df: DataFrame with daily sentiment per ticker
        sector_mapping: Dict mapping ticker -> sector (optional)
        
    Returns:
        DataFrame with sector_sentiment column added
    """
    if df.empty:
        return df
    
    # Default sector mapping for our tickers
    if sector_mapping is None:
        sector_mapping = {
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
    
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_mapping).fillna("other")
    
    # Compute sector average (excluding the ticker itself)
    def sector_avg_excluding_self(group):
        result = []
        for idx, row in group.iterrows():
            others = group[(group.index != idx) & (group["date"] == row["date"])]
            if len(others) > 0:
                result.append(others["sentiment_mean"].mean())
            else:
                result.append(row["sentiment_mean"])  # Fallback to own sentiment
        return pd.Series(result, index=group.index)
    
    # Simpler approach: sector average per date
    # Handle both sentiment_mean (from aggregator) and sentiment_score (renamed)
    score_col = "sentiment_mean" if "sentiment_mean" in df.columns else "sentiment_score"
    sector_avg = df.groupby(["date", "sector"])[score_col].transform("mean")
    df["sector_sentiment"] = sector_avg
    
    # Drop temp column
    df = df.drop(columns=["sector"])
    
    return df


def compute_market_sentiment(
    df: pd.DataFrame,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Add market-wide sentiment using SPY as proxy.
    
    SPY sentiment serves as a market-wide sentiment indicator.
    
    Args:
        df: DataFrame with daily sentiment per ticker
        market_ticker: Ticker to use as market proxy
        
    Returns:
        DataFrame with market_sentiment column added
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Handle both sentiment_mean (from aggregator) and sentiment_score (renamed)
    score_col = "sentiment_mean" if "sentiment_mean" in df.columns else "sentiment_score"
    
    # Extract market sentiment
    market_df = df[df["ticker"] == market_ticker][["date", score_col]].copy()
    market_df = market_df.rename(columns={score_col: "market_sentiment"})
    
    if len(market_df) == 0:
        # No market data - use overall average
        market_avg = df.groupby("date")[score_col].mean().reset_index()
        market_avg = market_avg.rename(columns={score_col: "market_sentiment"})
        df = df.merge(market_avg, on="date", how="left")
    else:
        df = df.merge(market_df, on="date", how="left")
    
    # Fill any missing values
    df["market_sentiment"] = df["market_sentiment"].fillna(0)
    
    return df
