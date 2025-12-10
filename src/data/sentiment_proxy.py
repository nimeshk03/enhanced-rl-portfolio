"""
Sentiment Proxy Module

Generates proxy sentiment for periods with sparse or missing news data.
Uses market indicators (VIX, returns, momentum) to estimate sentiment.

Methodology:
1. VIX-based sentiment: High VIX = negative sentiment, Low VIX = positive
2. Returns momentum: Positive returns = positive sentiment
3. Sector correlation: Use sector leader returns as proxy
4. Blending: Weight real sentiment by news coverage, fill gaps with proxy

Usage:
    from src.data.sentiment_proxy import generate_complete_sentiment
    
    complete_df = generate_complete_sentiment(
        sentiment_path='data/historical_sentiment.csv',
        market_data_path='data/processed_data.csv',
        output_path='data/historical_sentiment_complete.csv',
    )
"""

import os
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Proxy Sentiment Generators
# =============================================================================

def compute_vix_sentiment(
    vix_series: pd.Series,
    vix_low: float = 15.0,
    vix_high: float = 30.0,
) -> pd.Series:
    """
    Compute sentiment proxy from VIX.
    
    VIX is inversely correlated with market sentiment:
    - Low VIX (<15): High confidence, positive sentiment
    - High VIX (>30): Fear, negative sentiment
    
    Args:
        vix_series: VIX values
        vix_low: VIX threshold for positive sentiment
        vix_high: VIX threshold for negative sentiment
        
    Returns:
        Sentiment scores (-1 to 1)
    """
    # Normalize VIX to sentiment scale
    # VIX 15 -> sentiment 0.5
    # VIX 30 -> sentiment -0.5
    # Linear interpolation between
    
    sentiment = np.where(
        vix_series <= vix_low,
        0.5,  # Low VIX = positive
        np.where(
            vix_series >= vix_high,
            -0.5,  # High VIX = negative
            # Linear interpolation
            0.5 - (vix_series - vix_low) / (vix_high - vix_low)
        )
    )
    
    return pd.Series(sentiment, index=vix_series.index)


def compute_returns_sentiment(
    returns: pd.Series,
    lookback: int = 5,
    scale: float = 10.0,
) -> pd.Series:
    """
    Compute sentiment proxy from returns momentum.
    
    Positive returns momentum = positive sentiment
    
    Args:
        returns: Daily returns
        lookback: Days for momentum calculation
        scale: Scaling factor for returns
        
    Returns:
        Sentiment scores (-1 to 1)
    """
    # Compute rolling momentum
    momentum = returns.rolling(window=lookback, min_periods=1).mean()
    
    # Fill NaN with 0 (neutral sentiment for missing data)
    momentum = momentum.fillna(0)
    
    # Scale and clip to [-1, 1]
    sentiment = (momentum * scale).clip(-1, 1)
    
    return sentiment


def compute_sector_sentiment(
    ticker: str,
    market_data: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "tic",
    close_column: str = "close",
) -> pd.DataFrame:
    """
    Compute sentiment proxy from sector leader returns.
    
    Uses SPY as market proxy and sector-specific logic.
    
    Args:
        ticker: Target ticker
        market_data: Market data with all tickers
        date_column: Date column name
        ticker_column: Ticker column name
        close_column: Close price column name
        
    Returns:
        DataFrame with date and sentiment_proxy columns
    """
    # Sector mapping
    sector_leaders = {
        "AAPL": "SPY",  # Tech -> Market
        "MSFT": "SPY",
        "GOOGL": "SPY",
        "AMZN": "SPY",
        "NVDA": "SPY",
        "JPM": "SPY",   # Financials -> Market
        "BAC": "JPM",   # Use JPM as sector leader
        "GLD": "SPY",   # Gold -> inverse correlation
        "TLT": "SPY",   # Bonds -> inverse correlation
        "SPY": "SPY",   # Market itself
    }
    
    leader = sector_leaders.get(ticker, "SPY")
    
    # Get leader returns
    leader_data = market_data[market_data[ticker_column] == leader].copy()
    leader_data = leader_data.sort_values(date_column)
    leader_data["returns"] = leader_data[close_column].pct_change()
    
    # Compute sentiment from returns
    leader_data["sentiment_proxy"] = compute_returns_sentiment(
        leader_data["returns"],
        lookback=5,
        scale=15.0,
    )
    
    # Handle inverse correlations
    if ticker in ["GLD", "TLT"]:
        leader_data["sentiment_proxy"] = -leader_data["sentiment_proxy"] * 0.5
    
    return leader_data[[date_column, "sentiment_proxy"]].rename(
        columns={date_column: "date"}
    )


# =============================================================================
# Coverage Analysis
# =============================================================================

def analyze_coverage(
    sentiment_df: pd.DataFrame,
    market_data: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """
    Analyze news coverage by date and ticker.
    
    Args:
        sentiment_df: Sentiment data with date and ticker
        market_data: Market data with all trading days
        date_column: Date column name
        ticker_column: Ticker column name
        
    Returns:
        DataFrame with coverage statistics
    """
    # Get all trading days from market data
    all_dates = market_data[date_column].unique()
    all_tickers = market_data["tic"].unique()
    
    # Create full date-ticker grid
    full_grid = pd.MultiIndex.from_product(
        [all_dates, all_tickers],
        names=[date_column, ticker_column]
    ).to_frame(index=False)
    
    # Mark which have sentiment data
    if not sentiment_df.empty:
        sentiment_dates = set(zip(
            sentiment_df[date_column].astype(str),
            sentiment_df[ticker_column]
        ))
        full_grid["has_sentiment"] = full_grid.apply(
            lambda x: (str(x[date_column]), x[ticker_column]) in sentiment_dates,
            axis=1
        )
    else:
        full_grid["has_sentiment"] = False
    
    # Compute coverage by ticker
    coverage = full_grid.groupby(ticker_column).agg({
        "has_sentiment": ["sum", "count"]
    }).reset_index()
    coverage.columns = [ticker_column, "days_with_sentiment", "total_days"]
    coverage["coverage_pct"] = coverage["days_with_sentiment"] / coverage["total_days"] * 100
    
    return coverage


def identify_sparse_periods(
    sentiment_df: pd.DataFrame,
    market_data: pd.DataFrame,
    threshold: float = 0.5,
    date_column: str = "date",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """
    Identify date-ticker combinations with sparse coverage.
    
    Args:
        sentiment_df: Sentiment data
        market_data: Market data
        threshold: Coverage threshold (0-1)
        date_column: Date column name
        ticker_column: Ticker column name
        
    Returns:
        DataFrame with sparse periods
    """
    # Get all trading days
    all_dates = sorted(market_data[date_column].unique())
    all_tickers = market_data["tic"].unique()
    
    # Create full grid
    full_grid = []
    for date in all_dates:
        for ticker in all_tickers:
            full_grid.append({date_column: date, ticker_column: ticker})
    
    full_df = pd.DataFrame(full_grid)
    
    # Merge with sentiment
    if not sentiment_df.empty:
        sentiment_df = sentiment_df.copy()
        sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column]).dt.strftime("%Y-%m-%d")
        full_df[date_column] = pd.to_datetime(full_df[date_column]).dt.strftime("%Y-%m-%d")
        
        merged = full_df.merge(
            sentiment_df[[date_column, ticker_column, "sentiment_score"]],
            on=[date_column, ticker_column],
            how="left"
        )
    else:
        merged = full_df.copy()
        merged["sentiment_score"] = np.nan
    
    # Identify missing
    merged["is_sparse"] = merged["sentiment_score"].isna()
    
    return merged


# =============================================================================
# Proxy Generation
# =============================================================================

def generate_proxy_sentiment(
    market_data: pd.DataFrame,
    tickers: List[str],
    date_column: str = "date",
    ticker_column: str = "tic",
) -> pd.DataFrame:
    """
    Generate proxy sentiment for all tickers and dates.
    
    Combines multiple proxy methods:
    1. VIX-based sentiment (40% weight)
    2. Returns momentum (30% weight)
    3. Sector correlation (30% weight)
    
    Args:
        market_data: Market data with VIX and prices
        tickers: List of tickers
        date_column: Date column name
        ticker_column: Ticker column name
        
    Returns:
        DataFrame with proxy sentiment
    """
    all_proxies = []
    
    for ticker in tickers:
        ticker_data = market_data[market_data[ticker_column] == ticker].copy()
        ticker_data = ticker_data.sort_values(date_column)
        
        if len(ticker_data) == 0:
            continue
        
        # 1. VIX-based sentiment
        if "vix" in ticker_data.columns:
            vix_sentiment = compute_vix_sentiment(ticker_data["vix"])
        else:
            vix_sentiment = pd.Series(0, index=ticker_data.index)
        
        # 2. Returns momentum
        ticker_data["returns"] = ticker_data["close"].pct_change()
        returns_sentiment = compute_returns_sentiment(
            ticker_data["returns"],
            lookback=5,
            scale=15.0,
        )
        
        # 3. Sector sentiment
        sector_df = compute_sector_sentiment(
            ticker, market_data, date_column, ticker_column
        )
        
        # Merge sector sentiment
        ticker_data = ticker_data.merge(
            sector_df,
            left_on=date_column,
            right_on="date",
            how="left"
        )
        sector_sentiment = ticker_data["sentiment_proxy"].fillna(0)
        
        # Combine with weights
        combined = (
            0.4 * vix_sentiment.values +
            0.3 * returns_sentiment.values +
            0.3 * sector_sentiment.values
        )
        
        # Clip to valid range
        combined = np.clip(combined, -1, 1)
        
        # Create output
        for i, (_, row) in enumerate(ticker_data.iterrows()):
            all_proxies.append({
                "date": row[date_column],
                "ticker": ticker,
                "sentiment_proxy": combined[i],
                "vix_component": vix_sentiment.iloc[i] if len(vix_sentiment) > i else 0,
                "returns_component": returns_sentiment.iloc[i] if len(returns_sentiment) > i else 0,
                "sector_component": sector_sentiment.iloc[i] if len(sector_sentiment) > i else 0,
            })
    
    proxy_df = pd.DataFrame(all_proxies)
    
    # Ensure date is string format
    proxy_df["date"] = pd.to_datetime(proxy_df["date"]).dt.strftime("%Y-%m-%d")
    
    return proxy_df


# =============================================================================
# Blending Real and Proxy Sentiment
# =============================================================================

def blend_sentiment(
    real_sentiment: pd.DataFrame,
    proxy_sentiment: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    real_weight: float = 0.8,
) -> pd.DataFrame:
    """
    Blend real sentiment with proxy, weighted by availability.
    
    Where real sentiment exists, use weighted blend.
    Where missing, use proxy only.
    
    Args:
        real_sentiment: Real sentiment from news
        proxy_sentiment: Proxy sentiment from market indicators
        date_column: Date column name
        ticker_column: Ticker column name
        real_weight: Weight for real sentiment when available
        
    Returns:
        Complete sentiment DataFrame
    """
    # Ensure date formats match
    proxy_sentiment = proxy_sentiment.copy()
    proxy_sentiment[date_column] = pd.to_datetime(proxy_sentiment[date_column]).dt.strftime("%Y-%m-%d")
    
    if not real_sentiment.empty:
        real_sentiment = real_sentiment.copy()
        real_sentiment[date_column] = pd.to_datetime(real_sentiment[date_column]).dt.strftime("%Y-%m-%d")
        
        # Merge
        merged = proxy_sentiment.merge(
            real_sentiment[[date_column, ticker_column, "sentiment_score", "news_count"]],
            on=[date_column, ticker_column],
            how="left",
            suffixes=("", "_real")
        )
    else:
        merged = proxy_sentiment.copy()
        merged["sentiment_score"] = np.nan
        merged["news_count"] = 0
    
    # Fill missing news_count
    merged["news_count"] = merged["news_count"].fillna(0)
    
    # Compute blended sentiment
    has_real = merged["sentiment_score"].notna()
    
    # Where real exists: blend with weight
    merged.loc[has_real, "sentiment_blended"] = (
        real_weight * merged.loc[has_real, "sentiment_score"] +
        (1 - real_weight) * merged.loc[has_real, "sentiment_proxy"]
    )
    
    # Where missing: use proxy only
    merged.loc[~has_real, "sentiment_blended"] = merged.loc[~has_real, "sentiment_proxy"]
    
    # Mark source
    merged["sentiment_source"] = np.where(has_real, "real", "proxy")
    
    # Final sentiment score
    merged["sentiment_score"] = merged["sentiment_blended"]
    
    # Select output columns
    output_cols = [
        date_column, ticker_column, "sentiment_score", "sentiment_source",
        "news_count", "sentiment_proxy", "vix_component", 
        "returns_component", "sector_component"
    ]
    
    return merged[output_cols].sort_values([date_column, ticker_column]).reset_index(drop=True)


# =============================================================================
# Main Function
# =============================================================================

def generate_complete_sentiment(
    sentiment_path: str = "./data/historical_sentiment.csv",
    market_data_path: str = "./data/processed_data.csv",
    output_path: str = "./data/historical_sentiment_complete.csv",
    real_weight: float = 0.8,
) -> pd.DataFrame:
    """
    Generate complete sentiment data for all trading days.
    
    Combines real sentiment from news with proxy sentiment from
    market indicators to achieve 100% coverage.
    
    Args:
        sentiment_path: Path to real sentiment CSV
        market_data_path: Path to market data CSV
        output_path: Path to save complete sentiment
        real_weight: Weight for real sentiment when blending
        
    Returns:
        Complete sentiment DataFrame
    """
    logger.info("Loading data...")
    
    # Load market data
    market_data = pd.read_csv(market_data_path)
    market_data["date"] = pd.to_datetime(market_data["date"]).dt.strftime("%Y-%m-%d")
    
    # Load real sentiment (may be empty or partial)
    try:
        real_sentiment = pd.read_csv(sentiment_path)
        logger.info(f"Loaded {len(real_sentiment)} real sentiment records")
    except FileNotFoundError:
        logger.warning(f"No real sentiment file found at {sentiment_path}")
        real_sentiment = pd.DataFrame()
    
    # Get tickers
    tickers = sorted(market_data["tic"].unique())
    logger.info(f"Processing {len(tickers)} tickers: {tickers}")
    
    # Analyze initial coverage
    if not real_sentiment.empty:
        coverage = analyze_coverage(real_sentiment, market_data)
        logger.info(f"Initial coverage:\n{coverage}")
    
    # Generate proxy sentiment
    logger.info("Generating proxy sentiment from market indicators...")
    proxy_sentiment = generate_proxy_sentiment(
        market_data, tickers, "date", "tic"
    )
    logger.info(f"Generated {len(proxy_sentiment)} proxy records")
    
    # Blend real and proxy
    logger.info("Blending real and proxy sentiment...")
    complete_sentiment = blend_sentiment(
        real_sentiment, proxy_sentiment,
        real_weight=real_weight,
    )
    
    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    complete_sentiment.to_csv(output_path, index=False)
    logger.info(f"Saved complete sentiment to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPLETE SENTIMENT GENERATION")
    print("=" * 60)
    print(f"Total records: {len(complete_sentiment)}")
    print(f"Date range: {complete_sentiment['date'].min()} to {complete_sentiment['date'].max()}")
    print(f"Unique dates: {complete_sentiment['date'].nunique()}")
    print(f"Tickers: {complete_sentiment['ticker'].nunique()}")
    print(f"\nSentiment source breakdown:")
    print(complete_sentiment["sentiment_source"].value_counts())
    print(f"\nReal sentiment coverage: {(complete_sentiment['sentiment_source'] == 'real').mean():.1%}")
    print(f"Proxy sentiment fill: {(complete_sentiment['sentiment_source'] == 'proxy').mean():.1%}")
    print("=" * 60)
    
    return complete_sentiment


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate complete sentiment with proxy fill")
    parser.add_argument(
        "--sentiment",
        default="./data/historical_sentiment.csv",
        help="Path to real sentiment CSV",
    )
    parser.add_argument(
        "--market",
        default="./data/processed_data.csv",
        help="Path to market data CSV",
    )
    parser.add_argument(
        "--output",
        default="./data/historical_sentiment_complete.csv",
        help="Output path for complete sentiment",
    )
    parser.add_argument(
        "--real-weight",
        type=float,
        default=0.8,
        help="Weight for real sentiment when blending (0-1)",
    )
    
    args = parser.parse_args()
    
    generate_complete_sentiment(
        sentiment_path=args.sentiment,
        market_data_path=args.market,
        output_path=args.output,
        real_weight=args.real_weight,
    )
