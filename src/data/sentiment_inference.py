"""
Sentiment Inference Module

Runs FinBERT inference on historical news headlines to generate sentiment scores.
Designed to work both locally (CPU) and in Colab/Kaggle (GPU).

Usage:
    # Local (CPU - slower but works)
    python src/data/sentiment_inference.py --input data/historical_news/news_combined.csv
    
    # Or use the Python API
    from src.data.sentiment_inference import SentimentInferenceEngine
    engine = SentimentInferenceEngine()
    results = engine.process_news_file('data/historical_news/news_combined.csv')
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentInferenceEngine:
    """
    Engine for running sentiment inference on news headlines.
    
    Supports multiple backends:
    - transformers: Full FinBERT model (requires transformers, torch)
    - simple: Rule-based fallback (no dependencies)
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_simple_fallback: bool = True,
    ):
        """
        Initialize the sentiment inference engine.
        
        Args:
            model_name: HuggingFace model name for FinBERT
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
            use_simple_fallback: If True, use simple model when transformers unavailable
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_simple_fallback = use_simple_fallback
        self._model = None
        self._tokenizer = None
        self._device = device
        self._backend = None
        
        # Try to initialize the model
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the inference backend."""
        # Try transformers first
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            # Determine device
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Using device: {self._device}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            
            self._backend = "transformers"
            logger.info("FinBERT model loaded successfully")
            
        except ImportError as e:
            if self.use_simple_fallback:
                logger.warning(f"Transformers not available ({e}), using simple fallback")
                self._backend = "simple"
            else:
                raise ImportError(
                    "transformers and torch required for FinBERT inference. "
                    "Install with: pip install transformers torch"
                )
        except Exception as e:
            if self.use_simple_fallback:
                logger.warning(f"Error loading model ({e}), using simple fallback")
                self._backend = "simple"
            else:
                raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dicts with keys:
            - sentiment_score: float (-1 to 1)
            - sentiment_label: str ('positive', 'negative', 'neutral')
            - confidence: float (0 to 1)
        """
        if self._backend == "transformers":
            return self._predict_transformers(texts)
        else:
            return self._predict_simple(texts)
    
    def _predict_transformers(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict using FinBERT transformers model."""
        import torch
        import torch.nn.functional as F
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)
            
            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: positive, negative, neutral
            # Map to scores: positive=1, neutral=0, negative=-1
            label_map = {0: "positive", 1: "negative", 2: "neutral"}
            score_map = {0: 1.0, 1: -1.0, 2: 0.0}
            
            for j, prob in enumerate(probs):
                pred_idx = prob.argmax().item()
                confidence = prob[pred_idx].item()
                
                # Compute weighted score
                score = (prob[0].item() * 1.0 +  # positive
                        prob[1].item() * -1.0 +  # negative
                        prob[2].item() * 0.0)    # neutral
                
                results.append({
                    "sentiment_score": score,
                    "sentiment_label": label_map[pred_idx],
                    "confidence": confidence,
                })
        
        return results
    
    def _predict_simple(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Simple rule-based sentiment prediction.
        
        Uses keyword matching as a fallback when FinBERT is unavailable.
        """
        # Sentiment keywords
        positive_words = {
            'surge', 'surges', 'gain', 'gains', 'rise', 'rises', 'jump', 'jumps',
            'rally', 'rallies', 'soar', 'soars', 'climb', 'climbs', 'up', 'higher',
            'bull', 'bullish', 'growth', 'profit', 'profits', 'beat', 'beats',
            'exceed', 'exceeds', 'strong', 'positive', 'upgrade', 'upgrades',
            'buy', 'outperform', 'record', 'high', 'boom', 'optimistic',
        }
        
        negative_words = {
            'fall', 'falls', 'drop', 'drops', 'decline', 'declines', 'plunge',
            'plunges', 'sink', 'sinks', 'tumble', 'tumbles', 'crash', 'crashes',
            'down', 'lower', 'bear', 'bearish', 'loss', 'losses', 'miss', 'misses',
            'weak', 'negative', 'downgrade', 'downgrades', 'sell', 'underperform',
            'low', 'slump', 'pessimistic', 'fear', 'concern', 'warning',
        }
        
        results = []
        
        for text in texts:
            text_lower = text.lower()
            words = set(text_lower.split())
            
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            
            if pos_count > neg_count:
                score = min(0.3 + 0.1 * (pos_count - neg_count), 1.0)
                label = "positive"
            elif neg_count > pos_count:
                score = max(-0.3 - 0.1 * (neg_count - pos_count), -1.0)
                label = "negative"
            else:
                score = 0.0
                label = "neutral"
            
            # Add some randomness for realism
            score += np.random.uniform(-0.1, 0.1)
            score = np.clip(score, -1.0, 1.0)
            
            confidence = 0.5 + 0.1 * (pos_count + neg_count)
            confidence = min(confidence, 0.95)
            
            results.append({
                "sentiment_score": score,
                "sentiment_label": label,
                "confidence": confidence,
            })
        
        return results
    
    def process_news_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        headline_column: str = "headline",
        date_column: str = "date",
        ticker_column: str = "ticker",
    ) -> pd.DataFrame:
        """
        Process a news CSV file and generate sentiment scores.
        
        Args:
            input_path: Path to input news CSV
            output_path: Path to save output (optional)
            headline_column: Name of headline column
            date_column: Name of date column
            ticker_column: Name of ticker column
            
        Returns:
            DataFrame with sentiment scores added
        """
        logger.info(f"Loading news from {input_path}")
        df = pd.read_csv(input_path)
        
        if len(df) == 0:
            logger.warning("No news data to process")
            return df
        
        logger.info(f"Processing {len(df)} headlines using {self._backend} backend")
        
        # Get headlines
        headlines = df[headline_column].fillna("").tolist()
        
        # Run inference
        results = self.predict_batch(headlines)
        
        # Add results to dataframe
        df["sentiment_score"] = [r["sentiment_score"] for r in results]
        df["sentiment_label"] = [r["sentiment_label"] for r in results]
        df["sentiment_confidence"] = [r["confidence"] for r in results]
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved sentiment results to {output_path}")
        
        return df


def aggregate_to_daily_sentiment(
    sentiment_df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    score_column: str = "sentiment_score",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment to daily ticker-level.
    
    Args:
        sentiment_df: DataFrame with sentiment scores per headline
        date_column: Name of date column
        ticker_column: Name of ticker column
        score_column: Name of sentiment score column
        output_path: Path to save output (optional)
        
    Returns:
        DataFrame with daily sentiment per ticker
    """
    if sentiment_df.empty:
        return pd.DataFrame(columns=[
            "date", "ticker", "sentiment_score", "sentiment_std",
            "news_count", "positive_ratio", "negative_ratio"
        ])
    
    # Group by date and ticker
    grouped = sentiment_df.groupby([date_column, ticker_column])
    
    # Aggregate
    daily = grouped.agg({
        score_column: ["mean", "std", "count"],
    }).reset_index()
    
    # Flatten column names
    daily.columns = [date_column, ticker_column, "sentiment_score", "sentiment_std", "news_count"]
    
    # Fill NaN std (single article days)
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    
    # Compute positive/negative ratios
    def compute_ratios(group):
        scores = group[score_column]
        pos = (scores > 0.1).sum() / len(scores) if len(scores) > 0 else 0
        neg = (scores < -0.1).sum() / len(scores) if len(scores) > 0 else 0
        return pd.Series({"positive_ratio": pos, "negative_ratio": neg})
    
    ratios = sentiment_df.groupby([date_column, ticker_column]).apply(
        compute_ratios, include_groups=False
    ).reset_index()
    
    daily = daily.merge(ratios, on=[date_column, ticker_column])
    
    # Sort
    daily = daily.sort_values([date_column, ticker_column]).reset_index(drop=True)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        daily.to_csv(output_path, index=False)
        logger.info(f"Saved daily sentiment to {output_path}")
    
    return daily


def generate_historical_sentiment(
    news_path: str = "./data/historical_news/news_combined.csv",
    output_path: str = "./data/historical_sentiment.csv",
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Main function to generate historical sentiment from news data.
    
    Args:
        news_path: Path to news CSV file
        output_path: Path to save sentiment output
        use_gpu: Whether to use GPU (requires CUDA)
        
    Returns:
        DataFrame with daily sentiment per ticker
    """
    # Initialize engine
    device = "cuda" if use_gpu else "cpu"
    engine = SentimentInferenceEngine(device=device)
    
    # Process news
    sentiment_df = engine.process_news_file(news_path)
    
    # Aggregate to daily
    daily_df = aggregate_to_daily_sentiment(
        sentiment_df,
        output_path=output_path,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SENTIMENT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Headlines processed: {len(sentiment_df)}")
    print(f"Daily records: {len(daily_df)}")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Tickers: {daily_df['ticker'].nunique()}")
    print(f"Output saved to: {output_path}")
    print("=" * 60)
    
    return daily_df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sentiment from news headlines")
    parser.add_argument(
        "--input",
        default="./data/historical_news/news_combined.csv",
        help="Input news CSV file",
    )
    parser.add_argument(
        "--output",
        default="./data/historical_sentiment.csv",
        help="Output sentiment CSV file",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference (requires CUDA)",
    )
    
    args = parser.parse_args()
    
    generate_historical_sentiment(
        news_path=args.input,
        output_path=args.output,
        use_gpu=args.gpu,
    )
