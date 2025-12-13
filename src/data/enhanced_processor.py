"""
Enhanced Data Processor

Merges price data with sentiment data for RL training, handling:
1. Date alignment between price and sentiment datasets
2. Feature normalization (rolling z-score for stationarity)
3. Missing data handling and validation
4. Train/test splitting with proper temporal ordering

Usage:
    from src.data.enhanced_processor import EnhancedDataProcessor
    
    processor = EnhancedDataProcessor(
        price_path='data/processed_data.csv',
        sentiment_path='data/historical_sentiment_complete.csv'
    )
    
    # Get processed data
    df = processor.process()
    
    # Or get train/test split
    train_df, test_df = processor.get_train_test_split(
        train_end='2024-06-30',
        test_start='2024-07-01'
    )
"""

import os
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for data processing."""
    
    # Normalization settings
    normalize_features: bool = True
    normalization_window: int = 60  # Rolling window for z-score (60 trading days ~ 3 months)
    
    # Feature selection
    include_sentiment: bool = True
    sentiment_features: Optional[List[str]] = None  # None = auto-detect
    tech_indicators: Optional[List[str]] = None  # None = auto-detect
    
    # Data validation
    min_trading_days: int = 100
    max_missing_pct: float = 0.05  # Max 5% missing values allowed
    
    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class EnhancedDataProcessor:
    """
    Process and merge price and sentiment data for RL training.
    
    This processor handles:
    1. Loading and validating price and sentiment data
    2. Merging datasets with proper date/ticker alignment
    3. Feature normalization using rolling z-scores
    4. Missing value handling
    5. Train/test splitting
    
    The output is a DataFrame ready for use with EnhancedPortfolioEnv.
    """
    
    # Default columns that are not features
    BASE_COLS = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'ticker', 'day']
    
    # Known sentiment columns (non-numeric ones to exclude)
    SENTIMENT_NON_NUMERIC = ['sentiment_source']
    
    def __init__(
        self,
        price_path: str,
        sentiment_path: Optional[str] = None,
        config: Optional[ProcessorConfig] = None,
    ):
        """
        Initialize the data processor.
        
        Args:
            price_path: Path to price/technical indicator CSV
            sentiment_path: Path to sentiment data CSV (optional)
            config: Processing configuration
        """
        self.price_path = price_path
        self.sentiment_path = sentiment_path
        self.config = config or ProcessorConfig()
        
        # Data storage
        self._price_df: Optional[pd.DataFrame] = None
        self._sentiment_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self._processed_df: Optional[pd.DataFrame] = None
        
        # Feature tracking
        self._tech_indicators: List[str] = []
        self._sentiment_features: List[str] = []
        
        # Validation results
        self._validation_report: Dict[str, Any] = {}
    
    def load_data(self) -> "EnhancedDataProcessor":
        """
        Load price and sentiment data from CSV files.
        
        Returns:
            self for method chaining
        """
        logger.info(f"Loading price data from {self.price_path}")
        self._price_df = pd.read_csv(self.price_path)
        self._price_df['date'] = self._price_df['date'].astype(str)
        
        logger.info(f"  Loaded {len(self._price_df)} price records")
        logger.info(f"  Date range: {self._price_df['date'].min()} to {self._price_df['date'].max()}")
        logger.info(f"  Tickers: {self._price_df['tic'].nunique()}")
        
        # Load sentiment if provided
        if self.sentiment_path and os.path.exists(self.sentiment_path):
            logger.info(f"Loading sentiment data from {self.sentiment_path}")
            self._sentiment_df = pd.read_csv(self.sentiment_path)
            self._sentiment_df['date'] = self._sentiment_df['date'].astype(str)
            
            # Rename ticker column if needed
            if 'ticker' in self._sentiment_df.columns and 'tic' not in self._sentiment_df.columns:
                self._sentiment_df = self._sentiment_df.rename(columns={'ticker': 'tic'})
            
            logger.info(f"  Loaded {len(self._sentiment_df)} sentiment records")
        elif self.config.include_sentiment:
            logger.warning("Sentiment path not provided or file not found")
            self._sentiment_df = None
        
        return self
    
    def merge_data(self) -> "EnhancedDataProcessor":
        """
        Merge price and sentiment data on date and ticker.
        
        Returns:
            self for method chaining
        """
        if self._price_df is None:
            self.load_data()
        
        df = self._price_df.copy()
        
        # Merge sentiment if available
        if self._sentiment_df is not None and self.config.include_sentiment:
            logger.info("Merging price and sentiment data")
            
            # Filter to only numeric sentiment columns
            sentiment_cols = self._get_numeric_sentiment_columns()
            merge_cols = ['date', 'tic'] + sentiment_cols
            
            # Ensure tic column exists in sentiment
            if 'tic' in self._sentiment_df.columns:
                sentiment_subset = self._sentiment_df[merge_cols].copy()
            else:
                logger.warning("No 'tic' column in sentiment data, skipping merge")
                sentiment_subset = None
            
            if sentiment_subset is not None:
                df = df.merge(
                    sentiment_subset,
                    on=['date', 'tic'],
                    how='left'
                )
                
                # Fill missing sentiment with 0 (neutral)
                for col in sentiment_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                
                self._sentiment_features = sentiment_cols
                logger.info(f"  Merged {len(sentiment_cols)} sentiment features")
        
        # Identify technical indicators
        self._tech_indicators = self._get_tech_indicator_columns(df)
        logger.info(f"  Identified {len(self._tech_indicators)} technical indicators")
        
        self._merged_df = df
        return self
    
    def _get_numeric_sentiment_columns(self) -> List[str]:
        """Get list of numeric sentiment columns."""
        if self._sentiment_df is None:
            return []
        
        # Use configured list if provided
        if self.config.sentiment_features:
            return [c for c in self.config.sentiment_features if c in self._sentiment_df.columns]
        
        # Auto-detect numeric columns
        numeric_cols = []
        for col in self._sentiment_df.columns:
            if col in self.BASE_COLS or col in self.SENTIMENT_NON_NUMERIC:
                continue
            if pd.api.types.is_numeric_dtype(self._sentiment_df[col]):
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _get_tech_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of technical indicator columns."""
        # Use configured list if provided
        if self.config.tech_indicators:
            return [c for c in self.config.tech_indicators if c in df.columns]
        
        # Auto-detect: everything not in base cols or sentiment
        exclude_cols = set(self.BASE_COLS) | set(self._sentiment_features)
        return [c for c in df.columns if c not in exclude_cols]
    
    def normalize_features(self) -> "EnhancedDataProcessor":
        """
        Apply rolling z-score normalization to features.
        
        This helps with:
        1. Stationarity - removes trends from features
        2. Scale consistency - all features on similar scale
        3. Outlier handling - extreme values are bounded
        
        Returns:
            self for method chaining
        """
        if self._merged_df is None:
            self.merge_data()
        
        if not self.config.normalize_features:
            self._processed_df = self._merged_df.copy()
            return self
        
        logger.info(f"Normalizing features with {self.config.normalization_window}-day rolling z-score")
        
        df = self._merged_df.copy()
        window = self.config.normalization_window
        
        # Features to normalize (tech indicators + sentiment)
        features_to_normalize = self._tech_indicators + self._sentiment_features
        
        # Normalize per ticker
        normalized_dfs = []
        for ticker in df['tic'].unique():
            ticker_df = df[df['tic'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date')
            
            for col in features_to_normalize:
                if col in ticker_df.columns:
                    # Rolling z-score: (x - rolling_mean) / rolling_std
                    rolling_mean = ticker_df[col].rolling(window=window, min_periods=1).mean()
                    rolling_std = ticker_df[col].rolling(window=window, min_periods=1).std()
                    
                    # Avoid division by zero
                    rolling_std = rolling_std.replace(0, 1)
                    rolling_std = rolling_std.fillna(1)
                    
                    ticker_df[col] = (ticker_df[col] - rolling_mean) / rolling_std
                    
                    # Clip extreme values to [-3, 3] (3 sigma)
                    ticker_df[col] = ticker_df[col].clip(-3, 3)
            
            normalized_dfs.append(ticker_df)
        
        df = pd.concat(normalized_dfs, ignore_index=True)
        
        # Fill any remaining NaN with 0
        for col in features_to_normalize:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        self._processed_df = df
        logger.info("  Normalization complete")
        
        return self
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate processed data quality.
        
        Returns:
            Validation report dictionary
        """
        if self._processed_df is None:
            self.normalize_features()
        
        df = self._processed_df
        report = {}
        
        # Basic stats
        report['total_records'] = len(df)
        report['unique_dates'] = df['date'].nunique()
        report['unique_tickers'] = df['tic'].nunique()
        report['date_range'] = (df['date'].min(), df['date'].max())
        
        # Check for missing values
        missing_counts = df.isna().sum()
        total_missing = missing_counts.sum()
        missing_pct = total_missing / (len(df) * len(df.columns))
        
        report['total_missing'] = int(total_missing)
        report['missing_pct'] = float(missing_pct)
        report['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        report['infinite_values'] = inf_counts[inf_counts > 0].to_dict()
        
        # Feature statistics
        report['tech_indicators'] = self._tech_indicators
        report['sentiment_features'] = self._sentiment_features
        report['total_features'] = len(self._tech_indicators) + len(self._sentiment_features)
        
        # Validation status
        report['valid'] = (
            report['unique_dates'] >= self.config.min_trading_days and
            report['missing_pct'] <= self.config.max_missing_pct and
            len(report['infinite_values']) == 0
        )
        
        self._validation_report = report
        
        if report['valid']:
            logger.info("Data validation PASSED")
        else:
            logger.warning("Data validation FAILED")
            if report['unique_dates'] < self.config.min_trading_days:
                logger.warning(f"  Insufficient trading days: {report['unique_dates']} < {self.config.min_trading_days}")
            if report['missing_pct'] > self.config.max_missing_pct:
                logger.warning(f"  Too many missing values: {report['missing_pct']:.2%} > {self.config.max_missing_pct:.2%}")
            if report['infinite_values']:
                logger.warning(f"  Infinite values in: {list(report['infinite_values'].keys())}")
        
        return report
    
    def process(self) -> pd.DataFrame:
        """
        Run full processing pipeline.
        
        Returns:
            Processed DataFrame ready for training
        """
        self.load_data()
        self.merge_data()
        self.normalize_features()
        self.validate_data()
        
        df = self._processed_df.copy()
        
        # Apply date filters if configured
        if self.config.start_date:
            df = df[df['date'] >= self.config.start_date]
        if self.config.end_date:
            df = df[df['date'] <= self.config.end_date]
        
        # Sort by date and ticker
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        logger.info(f"Processing complete: {len(df)} records")
        
        return df
    
    def get_train_test_split(
        self,
        train_end: str,
        test_start: str,
        train_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Uses temporal split (no shuffling) to prevent look-ahead bias.
        
        Args:
            train_end: Last date for training data
            test_start: First date for testing data
            train_start: First date for training (optional)
            test_end: Last date for testing (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = self.process()
        
        # Training data
        train_mask = df['date'] <= train_end
        if train_start:
            train_mask &= df['date'] >= train_start
        train_df = df[train_mask].copy()
        
        # Testing data
        test_mask = df['date'] >= test_start
        if test_end:
            test_mask &= df['date'] <= test_end
        test_df = df[test_mask].copy()
        
        logger.info(f"Train/test split:")
        logger.info(f"  Train: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"  Test: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, test_df
    
    def prepare_for_env(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for use with EnhancedPortfolioEnv.
        
        Adds 'day' index required by the environment.
        
        Args:
            df: DataFrame to prepare (uses processed data if None)
            
        Returns:
            DataFrame with day index
        """
        if df is None:
            df = self.process()
        
        df = df.copy()
        
        # Sort by date and ticker
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Create day index
        dates = sorted(df['date'].unique())
        date_to_day = {date: i for i, date in enumerate(dates)}
        df['day'] = df['date'].map(date_to_day)
        
        # Set day as index
        df = df.set_index('day')
        
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about features in the processed data.
        
        Returns:
            Dictionary with feature information
        """
        if self._processed_df is None:
            self.process()
        
        return {
            'tech_indicators': self._tech_indicators,
            'sentiment_features': self._sentiment_features,
            'n_tech_indicators': len(self._tech_indicators),
            'n_sentiment_features': len(self._sentiment_features),
            'total_features': len(self._tech_indicators) + len(self._sentiment_features),
            'tickers': list(self._processed_df['tic'].unique()),
            'n_tickers': self._processed_df['tic'].nunique(),
        }
    
    def save_processed_data(
        self,
        output_path: str,
        include_day_index: bool = False,
    ) -> str:
        """
        Save processed data to CSV.
        
        Args:
            output_path: Path to save CSV
            include_day_index: Whether to include day index column
            
        Returns:
            Path to saved file
        """
        if include_day_index:
            df = self.prepare_for_env()
            df = df.reset_index()
        else:
            df = self.process()
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path
    
    @property
    def validation_report(self) -> Dict[str, Any]:
        """Get the validation report."""
        if not self._validation_report:
            self.validate_data()
        return self._validation_report
    
    @property
    def tech_indicators(self) -> List[str]:
        """Get list of technical indicator columns."""
        return self._tech_indicators
    
    @property
    def sentiment_features(self) -> List[str]:
        """Get list of sentiment feature columns."""
        return self._sentiment_features


def create_training_data(
    price_path: str,
    sentiment_path: str,
    train_end: str = "2024-06-30",
    test_start: str = "2024-07-01",
    normalize: bool = True,
    normalization_window: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to create training and testing data.
    
    Args:
        price_path: Path to price data CSV
        sentiment_path: Path to sentiment data CSV
        train_end: Last date for training
        test_start: First date for testing
        normalize: Whether to normalize features
        normalization_window: Window for rolling z-score
        
    Returns:
        Tuple of (train_df, test_df, feature_info)
    """
    config = ProcessorConfig(
        normalize_features=normalize,
        normalization_window=normalization_window,
    )
    
    processor = EnhancedDataProcessor(
        price_path=price_path,
        sentiment_path=sentiment_path,
        config=config,
    )
    
    train_df, test_df = processor.get_train_test_split(
        train_end=train_end,
        test_start=test_start,
    )
    
    feature_info = processor.get_feature_info()
    
    return train_df, test_df, feature_info
