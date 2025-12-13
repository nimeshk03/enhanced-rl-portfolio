"""
Tests for Enhanced Data Processor

Tests the EnhancedDataProcessor class:
- Data loading and merging
- Feature normalization
- Data validation
- Train/test splitting
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.enhanced_processor import (
    EnhancedDataProcessor,
    ProcessorConfig,
    create_training_data,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample price data with technical indicators."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    tickers = ["AAPL", "MSFT", "SPY"]
    
    records = []
    for date in dates:
        for ticker in tickers:
            base_price = {"AAPL": 150, "MSFT": 350, "SPY": 450}[ticker]
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "open": base_price + np.random.randn() * 2,
                "high": base_price + np.random.randn() * 2 + 1,
                "low": base_price + np.random.randn() * 2 - 1,
                "close": base_price + np.random.randn() * 2,
                "volume": np.random.randint(1000000, 5000000),
                "macd": np.random.randn() * 0.5,
                "rsi_30": 50 + np.random.randn() * 10,
                "cci_30": np.random.randn() * 50,
                "vix": 15 + np.random.randn() * 3,
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    tickers = ["AAPL", "MSFT", "SPY"]
    
    records = []
    for date in dates:
        for ticker in tickers:
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                "sentiment_score": np.random.uniform(-0.5, 0.5),
                "sentiment_std": np.random.uniform(0, 0.3),
                "news_count": np.random.randint(0, 10),
                "sentiment_source": "test",  # Non-numeric column
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def temp_data_files(sample_price_data, sample_sentiment_data, tmp_path):
    """Save sample data to temporary files."""
    price_path = tmp_path / "prices.csv"
    sentiment_path = tmp_path / "sentiment.csv"
    
    sample_price_data.to_csv(price_path, index=False)
    sample_sentiment_data.to_csv(sentiment_path, index=False)
    
    return str(price_path), str(sentiment_path)


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_price_data(self, temp_data_files):
        """Test loading price data."""
        price_path, _ = temp_data_files
        
        processor = EnhancedDataProcessor(price_path=price_path)
        processor.load_data()
        
        assert processor._price_df is not None
        assert len(processor._price_df) == 300  # 100 days * 3 tickers
    
    def test_load_with_sentiment(self, temp_data_files):
        """Test loading both price and sentiment data."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        processor.load_data()
        
        assert processor._price_df is not None
        assert processor._sentiment_df is not None
    
    def test_load_without_sentiment(self, temp_data_files):
        """Test loading without sentiment data."""
        price_path, _ = temp_data_files
        
        config = ProcessorConfig(include_sentiment=False)
        processor = EnhancedDataProcessor(
            price_path=price_path,
            config=config,
        )
        processor.load_data()
        
        assert processor._price_df is not None
        assert processor._sentiment_df is None


# =============================================================================
# Data Merging Tests
# =============================================================================

class TestDataMerging:
    """Tests for data merging functionality."""
    
    def test_merge_price_and_sentiment(self, temp_data_files):
        """Test merging price and sentiment data."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        processor.load_data()
        processor.merge_data()
        
        assert processor._merged_df is not None
        assert 'sentiment_score' in processor._merged_df.columns
        assert 'macd' in processor._merged_df.columns
    
    def test_non_numeric_columns_excluded(self, temp_data_files):
        """Test that non-numeric sentiment columns are excluded."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        processor.load_data()
        processor.merge_data()
        
        # sentiment_source is non-numeric and should be excluded
        assert 'sentiment_source' not in processor._sentiment_features
    
    def test_feature_detection(self, temp_data_files):
        """Test automatic feature detection."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        processor.load_data()
        processor.merge_data()
        
        # Should detect tech indicators
        assert 'macd' in processor._tech_indicators
        assert 'rsi_30' in processor._tech_indicators
        
        # Should detect sentiment features
        assert 'sentiment_score' in processor._sentiment_features


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Tests for feature normalization."""
    
    def test_rolling_zscore_normalization(self, temp_data_files):
        """Test rolling z-score normalization."""
        price_path, sentiment_path = temp_data_files
        
        config = ProcessorConfig(
            normalize_features=True,
            normalization_window=20,
        )
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
            config=config,
        )
        processor.load_data()
        processor.merge_data()
        processor.normalize_features()
        
        df = processor._processed_df
        
        # Normalized features should be roughly in [-3, 3] range
        for col in processor._tech_indicators + processor._sentiment_features:
            if col in df.columns:
                assert df[col].min() >= -3.1  # Allow small tolerance
                assert df[col].max() <= 3.1
    
    def test_no_normalization(self, temp_data_files):
        """Test processing without normalization."""
        price_path, sentiment_path = temp_data_files
        
        config = ProcessorConfig(normalize_features=False)
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
            config=config,
        )
        processor.load_data()
        processor.merge_data()
        processor.normalize_features()
        
        df = processor._processed_df
        
        # Without normalization, values should be original scale
        # RSI should be around 50
        assert df['rsi_30'].mean() > 10  # Not normalized
    
    def test_no_nan_after_normalization(self, temp_data_files):
        """Test that normalization doesn't introduce NaN values."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        df = processor.process()
        
        # Check for NaN in feature columns
        for col in processor._tech_indicators + processor._sentiment_features:
            if col in df.columns:
                assert df[col].isna().sum() == 0, f"NaN found in {col}"


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for data validation."""
    
    def test_validation_passes(self, temp_data_files):
        """Test validation passes for good data."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        df = processor.process()
        report = processor.validation_report
        
        assert report['valid'] is True
        assert report['total_records'] == 300
        assert report['unique_tickers'] == 3
    
    def test_validation_report_contents(self, temp_data_files):
        """Test validation report contains expected fields."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        processor.process()
        report = processor.validation_report
        
        assert 'total_records' in report
        assert 'unique_dates' in report
        assert 'unique_tickers' in report
        assert 'date_range' in report
        assert 'missing_pct' in report
        assert 'valid' in report


# =============================================================================
# Train/Test Split Tests
# =============================================================================

class TestTrainTestSplit:
    """Tests for train/test splitting."""
    
    def test_temporal_split(self, temp_data_files):
        """Test temporal train/test split."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        
        train_df, test_df = processor.get_train_test_split(
            train_end="2024-03-15",
            test_start="2024-03-18",
        )
        
        # Train should end before test starts
        assert train_df['date'].max() <= "2024-03-15"
        assert test_df['date'].min() >= "2024-03-18"
    
    def test_no_overlap(self, temp_data_files):
        """Test no overlap between train and test."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        
        train_df, test_df = processor.get_train_test_split(
            train_end="2024-03-15",
            test_start="2024-03-18",
        )
        
        train_dates = set(train_df['date'].unique())
        test_dates = set(test_df['date'].unique())
        
        assert len(train_dates & test_dates) == 0


# =============================================================================
# Environment Preparation Tests
# =============================================================================

class TestEnvPreparation:
    """Tests for environment preparation."""
    
    def test_prepare_for_env(self, temp_data_files):
        """Test preparing data for environment."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        
        df = processor.prepare_for_env()
        
        # Should have day as index
        assert df.index.name == 'day'
        assert df.index.min() == 0
    
    def test_day_index_sequential(self, temp_data_files):
        """Test day index is sequential."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        
        df = processor.prepare_for_env()
        
        # Each day should have all tickers
        for day in range(df.index.max() + 1):
            day_data = df.loc[day]
            assert len(day_data) == 3  # 3 tickers


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunction:
    """Tests for create_training_data factory function."""
    
    def test_create_training_data(self, temp_data_files):
        """Test create_training_data convenience function."""
        price_path, sentiment_path = temp_data_files
        
        train_df, test_df, feature_info = create_training_data(
            price_path=price_path,
            sentiment_path=sentiment_path,
            train_end="2024-03-15",
            test_start="2024-03-18",
        )
        
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert 'tech_indicators' in feature_info
        assert 'sentiment_features' in feature_info
    
    def test_feature_info(self, temp_data_files):
        """Test feature info returned by factory."""
        price_path, sentiment_path = temp_data_files
        
        _, _, feature_info = create_training_data(
            price_path=price_path,
            sentiment_path=sentiment_path,
            train_end="2024-03-15",
            test_start="2024-03-18",
        )
        
        assert feature_info['n_tickers'] == 3
        assert feature_info['n_tech_indicators'] > 0
        assert feature_info['n_sentiment_features'] > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the processor."""
    
    def test_full_pipeline(self, temp_data_files):
        """Test full processing pipeline."""
        price_path, sentiment_path = temp_data_files
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        
        df = processor.process()
        
        # Should have all records
        assert len(df) == 300
        
        # Should have no NaN
        assert df.isna().sum().sum() == 0
        
        # Should be sorted
        dates = df['date'].tolist()
        assert dates == sorted(dates)
    
    def test_save_and_reload(self, temp_data_files, tmp_path):
        """Test saving and reloading processed data."""
        price_path, sentiment_path = temp_data_files
        output_path = str(tmp_path / "processed.csv")
        
        processor = EnhancedDataProcessor(
            price_path=price_path,
            sentiment_path=sentiment_path,
        )
        
        processor.save_processed_data(output_path)
        
        # Reload and verify
        reloaded = pd.read_csv(output_path)
        assert len(reloaded) == 300


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
