"""
Historical News Data Collection Module

Collects financial news headlines from multiple sources for sentiment analysis.
Designed to maximize coverage while staying within free tier limits.

Sources (prioritized):
1. Yahoo Finance - Recent news (~2 years), free, already integrated
2. GDELT - Global news archive (2015+), free, requires filtering
3. Alpha Vantage - News API with free tier (limited requests)
4. Proxy - Market-derived sentiment for sparse periods

Usage:
    from src.data.news_collector import collect_historical_news
    
    news_df = collect_historical_news(
        tickers=['AAPL', 'MSFT', 'SPY'],
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================

class NewsCollector(ABC):
    """Abstract base class for news data collectors."""
    
    @abstractmethod
    def collect(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Collect news headlines for given tickers and date range.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns:
            - date: str (YYYY-MM-DD)
            - ticker: str
            - headline: str
            - source: str
            - url: str (optional)
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this news source."""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Return rate limit information for this source."""
        pass


# =============================================================================
# Yahoo Finance Collector
# =============================================================================

class YahooFinanceCollector(NewsCollector):
    """
    Collect news from Yahoo Finance.
    
    Limitations:
    - Only ~2 years of historical data available
    - Rate limited (be respectful)
    """
    
    def __init__(self, delay_seconds: float = 1.0):
        """
        Initialize Yahoo Finance collector.
        
        Args:
            delay_seconds: Delay between requests to avoid rate limiting
        """
        self.delay_seconds = delay_seconds
        self._yfinance = None
    
    def _get_yfinance(self):
        """Lazy import yfinance."""
        if self._yfinance is None:
            try:
                import yfinance as yf
                self._yfinance = yf
            except ImportError:
                raise ImportError("yfinance required: pip install yfinance")
        return self._yfinance
    
    def collect(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Collect news from Yahoo Finance."""
        yf = self._get_yfinance()
        
        all_news = []
        # Make timezone-naive for comparison
        start_dt = pd.to_datetime(start_date).tz_localize(None)
        end_dt = pd.to_datetime(end_date).tz_localize(None)
        
        for ticker in tickers:
            logger.info(f"Collecting Yahoo Finance news for {ticker}")
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                if news:
                    for item in news:
                        # Handle new nested structure (2024+ yfinance format)
                        content = item.get("content", item)  # Fallback to item if no content
                        
                        # Try new format first (pubDate as ISO string)
                        pub_date_str = content.get("pubDate", "")
                        if pub_date_str:
                            try:
                                pub_date = pd.to_datetime(pub_date_str).tz_localize(None)
                            except:
                                pub_date = None
                        else:
                            # Fallback to old format (providerPublishTime as timestamp)
                            pub_time = item.get("providerPublishTime", 0)
                            if pub_time:
                                pub_date = datetime.fromtimestamp(pub_time)
                            else:
                                pub_date = None
                        
                        if pub_date:
                            date_str = pub_date.strftime("%Y-%m-%d")
                            
                            # Filter by date range
                            if start_dt <= pub_date <= end_dt:
                                # Get title from content or item
                                title = content.get("title", item.get("title", ""))
                                
                                # Get provider/source
                                provider = content.get("provider", {})
                                source = provider.get("displayName", item.get("publisher", "Yahoo Finance"))
                                
                                # Get URL
                                canonical = content.get("canonicalUrl", {})
                                url = canonical.get("url", item.get("link", ""))
                                
                                if title:  # Only add if we have a title
                                    all_news.append({
                                        "date": date_str,
                                        "ticker": ticker,
                                        "headline": title,
                                        "source": source,
                                        "url": url,
                                    })
                
                # Respect rate limits
                time.sleep(self.delay_seconds)
                
            except Exception as e:
                logger.warning(f"Error collecting news for {ticker}: {e}")
                continue
        
        if not all_news:
            return pd.DataFrame(columns=["date", "ticker", "headline", "source", "url"])
        
        df = pd.DataFrame(all_news)
        df = df.drop_duplicates(subset=["date", "ticker", "headline"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        
        return df
    
    @property
    def source_name(self) -> str:
        return "Yahoo Finance"
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": 60,
            "delay_seconds": self.delay_seconds,
            "historical_limit": "~2 years",
        }


# =============================================================================
# GDELT Collector
# =============================================================================

class GDELTCollector(NewsCollector):
    """
    Collect news from GDELT (Global Database of Events, Language, and Tone).
    
    GDELT provides free access to global news data from 2015+.
    Uses the GDELT DOC API for news article search.
    
    Note: Requires filtering for financial relevance.
    """
    
    def __init__(
        self,
        delay_seconds: float = 2.0,
        max_records_per_query: int = 250,
    ):
        """
        Initialize GDELT collector.
        
        Args:
            delay_seconds: Delay between API requests
            max_records_per_query: Max records per API call
        """
        self.delay_seconds = delay_seconds
        self.max_records = max_records_per_query
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def _build_query(self, ticker: str, company_name: str) -> str:
        """Build GDELT search query for a ticker."""
        # GDELT requires longer search terms - use company name + stock/shares
        return f'{company_name} stock'
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for ticker (for better search results)."""
        company_names = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google Alphabet",
            "AMZN": "Amazon",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan Chase",
            "BAC": "BofA banking",
            "GLD": "Gold prices",
            "TLT": "Treasury bonds",
            "SPY": "stock market",
        }
        return company_names.get(ticker, ticker)
    
    def collect(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Collect news from GDELT.
        
        Note: GDELT API can be slow and may have availability issues.
        This implementation includes fallback handling.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install requests")
        
        all_news = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # GDELT date format
        start_gdelt = start_dt.strftime("%Y%m%d%H%M%S")
        end_gdelt = end_dt.strftime("%Y%m%d%H%M%S")
        
        for ticker in tickers:
            logger.info(f"Collecting GDELT news for {ticker}")
            company_name = self._get_company_name(ticker)
            query = self._build_query(ticker, company_name)
            
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": self.max_records,
                "startdatetime": start_gdelt,
                "enddatetime": end_gdelt,
                "format": "json",
            }
            
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=30,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    
                    for article in articles:
                        # Parse date
                        seen_date = article.get("seendate", "")
                        if seen_date:
                            try:
                                pub_date = datetime.strptime(seen_date[:8], "%Y%m%d")
                                date_str = pub_date.strftime("%Y-%m-%d")
                            except ValueError:
                                continue
                            
                            all_news.append({
                                "date": date_str,
                                "ticker": ticker,
                                "headline": article.get("title", ""),
                                "source": article.get("domain", "GDELT"),
                                "url": article.get("url", ""),
                            })
                else:
                    logger.warning(f"GDELT API error for {ticker}: {response.status_code}")
                
                # Respect rate limits
                time.sleep(self.delay_seconds)
                
            except Exception as e:
                logger.warning(f"Error collecting GDELT news for {ticker}: {e}")
                continue
        
        if not all_news:
            return pd.DataFrame(columns=["date", "ticker", "headline", "source", "url"])
        
        df = pd.DataFrame(all_news)
        df = df.drop_duplicates(subset=["date", "ticker", "headline"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        
        return df
    
    @property
    def source_name(self) -> str:
        return "GDELT"
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": 30,
            "delay_seconds": self.delay_seconds,
            "historical_limit": "2015+",
            "max_records_per_query": self.max_records,
        }


# =============================================================================
# Market Proxy Sentiment Generator
# =============================================================================

class MarketProxySentimentGenerator:
    """
    Generate proxy sentiment from market data for periods with sparse news.
    
    Uses market indicators to derive sentiment-like signals:
    - Price momentum (returns)
    - Volatility (VIX, realized vol)
    - Volume changes
    
    This is a fallback for historical periods without news coverage.
    """
    
    def __init__(self, market_data_path: str = "./data/processed_data.csv"):
        """
        Initialize proxy generator.
        
        Args:
            market_data_path: Path to processed market data
        """
        self.market_data_path = market_data_path
    
    def generate(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Generate proxy sentiment from market data.
        
        Proxy formula:
        - sentiment_score = tanh(returns_5d * 10) * (1 - vix_normalized * 0.3)
        
        This creates a sentiment-like signal that:
        - Is positive when recent returns are positive
        - Is dampened during high volatility periods
        """
        if not os.path.exists(self.market_data_path):
            logger.warning(f"Market data not found at {self.market_data_path}")
            return pd.DataFrame(columns=[
                "date", "ticker", "headline", "source", "sentiment_proxy"
            ])
        
        # Load market data
        df = pd.read_csv(self.market_data_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        # Filter date range and tickers
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[
            (pd.to_datetime(df["date"]) >= start_dt) &
            (pd.to_datetime(df["date"]) <= end_dt) &
            (df["tic"].isin(tickers))
        ].copy()
        
        if df.empty:
            return pd.DataFrame(columns=[
                "date", "ticker", "headline", "source", "sentiment_proxy"
            ])
        
        # Calculate returns
        df = df.sort_values(["tic", "date"])
        df["returns_1d"] = df.groupby("tic")["close"].pct_change()
        df["returns_5d"] = df.groupby("tic")["close"].pct_change(5)
        
        # Normalize VIX if available
        if "vix" in df.columns:
            vix_mean = df["vix"].mean()
            vix_std = df["vix"].std()
            df["vix_norm"] = (df["vix"] - vix_mean) / (vix_std + 1e-8)
            df["vix_norm"] = df["vix_norm"].clip(-2, 2) / 4 + 0.5  # Scale to 0-1
        else:
            df["vix_norm"] = 0.5
        
        # Generate proxy sentiment
        df["sentiment_proxy"] = np.tanh(df["returns_5d"].fillna(0) * 10) * (1 - df["vix_norm"] * 0.3)
        df["sentiment_proxy"] = df["sentiment_proxy"].clip(-1, 1)
        
        # Create output format
        result = df[["date", "tic", "sentiment_proxy"]].copy()
        result = result.rename(columns={"tic": "ticker"})
        result["headline"] = "[Market Proxy]"
        result["source"] = "Market Proxy"
        
        return result[["date", "ticker", "headline", "source", "sentiment_proxy"]]


# =============================================================================
# Main Collection Function
# =============================================================================

def collect_historical_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    sources: Optional[List[str]] = None,
    output_dir: str = "./data/historical_news",
    save_intermediate: bool = True,
) -> pd.DataFrame:
    """
    Collect historical news from multiple sources.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sources: List of sources to use (default: ['yahoo', 'gdelt'])
        output_dir: Directory to save collected news
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Combined DataFrame with news from all sources
    """
    if sources is None:
        sources = ["yahoo"]  # Start with Yahoo only, GDELT can be slow
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_news = []
    
    # Collect from each source
    for source in sources:
        logger.info(f"Collecting from {source}...")
        
        if source.lower() == "yahoo":
            collector = YahooFinanceCollector()
            df = collector.collect(tickers, start_date, end_date)
            
        elif source.lower() == "gdelt":
            collector = GDELTCollector()
            df = collector.collect(tickers, start_date, end_date)
            
        else:
            logger.warning(f"Unknown source: {source}")
            continue
        
        if not df.empty:
            df["collection_source"] = source
            all_news.append(df)
            
            # Save intermediate results
            if save_intermediate:
                source_file = os.path.join(output_dir, f"news_{source}.csv")
                df.to_csv(source_file, index=False)
                logger.info(f"Saved {len(df)} headlines to {source_file}")
    
    # Combine all sources
    if not all_news:
        logger.warning("No news collected from any source")
        return pd.DataFrame(columns=[
            "date", "ticker", "headline", "source", "url", "collection_source"
        ])
    
    combined = pd.concat(all_news, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "ticker", "headline"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Save combined results
    combined_file = os.path.join(output_dir, "news_combined.csv")
    combined.to_csv(combined_file, index=False)
    logger.info(f"Saved {len(combined)} total headlines to {combined_file}")
    
    return combined


def generate_coverage_report(
    news_df: pd.DataFrame,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_path: str = "./data/historical_news_summary.csv",
) -> pd.DataFrame:
    """
    Generate a coverage report for collected news data.
    
    Args:
        news_df: DataFrame with collected news
        tickers: List of expected tickers
        start_date: Expected start date
        end_date: Expected end date
        output_path: Path to save the report
        
    Returns:
        DataFrame with coverage statistics
    """
    # Calculate trading days in range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Approximate trading days (252 per year)
    total_days = (end_dt - start_dt).days
    trading_days_approx = int(total_days * 252 / 365)
    
    # Calculate coverage per ticker
    coverage_stats = []
    
    for ticker in tickers:
        ticker_news = news_df[news_df["ticker"] == ticker] if not news_df.empty else pd.DataFrame()
        
        if ticker_news.empty:
            coverage_stats.append({
                "ticker": ticker,
                "total_headlines": 0,
                "unique_dates": 0,
                "date_coverage_pct": 0.0,
                "avg_headlines_per_day": 0.0,
                "first_date": None,
                "last_date": None,
            })
        else:
            unique_dates = ticker_news["date"].nunique()
            coverage_stats.append({
                "ticker": ticker,
                "total_headlines": len(ticker_news),
                "unique_dates": unique_dates,
                "date_coverage_pct": round(unique_dates / trading_days_approx * 100, 2),
                "avg_headlines_per_day": round(len(ticker_news) / max(unique_dates, 1), 2),
                "first_date": ticker_news["date"].min(),
                "last_date": ticker_news["date"].max(),
            })
    
    report_df = pd.DataFrame(coverage_stats)
    
    # Add summary row
    summary = {
        "ticker": "TOTAL",
        "total_headlines": report_df["total_headlines"].sum(),
        "unique_dates": news_df["date"].nunique() if not news_df.empty else 0,
        "date_coverage_pct": round(report_df["date_coverage_pct"].mean(), 2),
        "avg_headlines_per_day": round(report_df["avg_headlines_per_day"].mean(), 2),
        "first_date": news_df["date"].min() if not news_df.empty else None,
        "last_date": news_df["date"].max() if not news_df.empty else None,
    }
    report_df = pd.concat([report_df, pd.DataFrame([summary])], ignore_index=True)
    
    # Save report
    report_df.to_csv(output_path, index=False)
    logger.info(f"Coverage report saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("NEWS COVERAGE REPORT")
    print("=" * 60)
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Expected Trading Days: ~{trading_days_approx}")
    print(f"Total Headlines Collected: {summary['total_headlines']}")
    print(f"Average Coverage: {summary['date_coverage_pct']}%")
    print("=" * 60)
    print(report_df.to_string(index=False))
    print("=" * 60 + "\n")
    
    return report_df


# =============================================================================
# Sample Data Generator (for testing/development)
# =============================================================================

def generate_sample_news_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    headlines_per_day: int = 3,
    output_path: str = "./data/historical_news/sample_news.csv",
) -> pd.DataFrame:
    """
    Generate sample news data for testing and development.
    
    This creates realistic-looking sample data when real news APIs
    are unavailable or rate-limited.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        headlines_per_day: Average headlines per ticker per day
        output_path: Path to save the sample data
        
    Returns:
        DataFrame with sample news data
    """
    np.random.seed(42)
    
    # Generate business days
    dates = pd.bdate_range(start_date, end_date)
    
    # Sample headline templates
    headline_templates = {
        "AAPL": [
            "Apple stock {action} as {event}",
            "AAPL {action} on {event}",
            "Apple {event}, shares {action}",
            "Analysts {opinion} Apple amid {event}",
        ],
        "MSFT": [
            "Microsoft {action} as {event}",
            "MSFT shares {action} on {event}",
            "Microsoft {event}, stock {action}",
            "Analysts {opinion} Microsoft amid {event}",
        ],
        "GOOGL": [
            "Alphabet {action} as {event}",
            "Google parent {action} on {event}",
            "GOOGL {event}, shares {action}",
        ],
        "AMZN": [
            "Amazon {action} as {event}",
            "AMZN shares {action} on {event}",
            "Amazon {event}, stock {action}",
        ],
        "NVDA": [
            "NVIDIA {action} as {event}",
            "NVDA shares {action} on {event}",
            "NVIDIA {event}, stock {action}",
        ],
        "JPM": [
            "JPMorgan {action} as {event}",
            "JPM shares {action} on {event}",
            "JPMorgan {event}, stock {action}",
        ],
        "BAC": [
            "Bank of America {action} as {event}",
            "BAC shares {action} on {event}",
        ],
        "GLD": [
            "Gold ETF {action} as {event}",
            "GLD {action} on {event}",
        ],
        "TLT": [
            "Treasury bonds {action} as {event}",
            "TLT {action} on {event}",
        ],
        "SPY": [
            "S&P 500 {action} as {event}",
            "SPY {action} on {event}",
            "Market {action} amid {event}",
        ],
    }
    
    actions = ["rises", "falls", "surges", "drops", "gains", "declines", "rallies", "slides"]
    events = [
        "earnings beat expectations", "earnings miss", "analyst upgrade",
        "analyst downgrade", "new product launch", "market volatility",
        "Fed announcement", "economic data release", "sector rotation",
        "investor sentiment shifts", "trading volume spikes",
    ]
    opinions = ["bullish on", "bearish on", "neutral on", "upgrade", "downgrade"]
    sources = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "WSJ", "Yahoo Finance"]
    
    records = []
    for date in dates:
        for ticker in tickers:
            # Random number of headlines (0-5, weighted toward 1-3)
            n_headlines = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05])
            
            templates = headline_templates.get(ticker, ["{ticker} stock {action} on {event}"])
            
            for _ in range(n_headlines):
                template = np.random.choice(templates)
                headline = template.format(
                    ticker=ticker,
                    action=np.random.choice(actions),
                    event=np.random.choice(events),
                    opinion=np.random.choice(opinions),
                )
                
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "headline": headline,
                    "source": np.random.choice(sources),
                    "url": f"https://example.com/news/{ticker}/{date.strftime('%Y%m%d')}/{len(records)}",
                })
    
    df = pd.DataFrame(records)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} sample headlines, saved to {output_path}")
    
    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect historical news data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC", "GLD", "TLT", "SPY"],
        help="Stock tickers to collect news for",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["yahoo"],
        help="News sources to use (yahoo, gdelt)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/historical_news",
        help="Output directory for news data",
    )
    
    args = parser.parse_args()
    
    # Collect news
    news_df = collect_historical_news(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        sources=args.sources,
        output_dir=args.output_dir,
    )
    
    # Generate coverage report
    generate_coverage_report(
        news_df=news_df,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )
