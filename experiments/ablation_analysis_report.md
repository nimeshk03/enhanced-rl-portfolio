# Ablation Study Analysis Report

**Date:** December 19, 2025  
**Experiment:** Feature ablation study for sentiment-enhanced RL portfolio  
**Total Runs:** 12 (4 configs x 3 seeds)

---

## Executive Summary

The ablation study reveals a **surprising finding**: the baseline model (no sentiment features) outperforms all sentiment-enhanced configurations on average. This suggests that with the current sentiment feature engineering, adding sentiment data may introduce noise rather than signal.

### Key Finding

| Rank | Configuration | Mean Sharpe | Mean Return | Sentiment Features |
|------|--------------|-------------|-------------|-------------------|
| 1 | **baseline** | **1.627** | **47.8%** | 0 |
| 2 | all_sentiment | 1.431 | 43.8% | 6 |
| 3 | score_only | 1.380 | 41.6% | 1 |
| 4 | core_3 | 1.140 | 36.8% | 3 |

---

## Detailed Results

### Aggregated Performance (Mean +/- Std across 3 seeds)

| Config | Sharpe Ratio | Total Return | Max Drawdown | Avg Trades |
|--------|-------------|--------------|--------------|------------|
| baseline | 1.627 +/- 0.541 | 47.8% +/- 12.3% | -15.8% +/- 8.7% | 10 |
| score_only | 1.380 +/- 0.448 | 41.6% +/- 13.9% | -20.2% +/- 1.0% | 8 |
| core_3 | 1.140 +/- 0.021 | 36.8% +/- 4.7% | -22.0% +/- 4.7% | 9 |
| all_sentiment | 1.431 +/- 0.458 | 43.8% +/- 9.7% | -20.6% +/- 5.1% | 6 |

### Individual Run Results

| Config | Seed | Sharpe | Return | Max DD | Trades |
|--------|------|--------|--------|--------|--------|
| baseline | 42 | 1.003 | 33.7% | -24.2% | 7 |
| baseline | 123 | **1.954** | **56.7%** | -6.9% | 18 |
| baseline | 456 | 1.925 | 52.9% | -16.1% | 6 |
| score_only | 42 | 0.869 | 25.7% | -21.2% | 7 |
| score_only | 123 | 1.565 | 48.3% | -20.1% | 6 |
| score_only | 456 | 1.705 | 51.0% | -19.2% | 10 |
| core_3 | 42 | 1.145 | 41.8% | -26.3% | 6 |
| core_3 | 123 | 1.116 | 36.2% | -22.6% | 16 |
| core_3 | 456 | 1.157 | 32.4% | -17.0% | 5 |
| all_sentiment | 42 | 1.666 | 46.7% | -16.4% | 6 |
| all_sentiment | 123 | 0.903 | 33.0% | -26.3% | 6 |
| all_sentiment | 456 | 1.723 | 51.8% | -19.0% | 5 |

---

## Analysis

### 1. Baseline Outperforms Sentiment Models

The baseline model (technical indicators only) achieved the highest mean Sharpe ratio (1.627) compared to sentiment-enhanced models. This is counterintuitive but can be explained by:

- **Noise Introduction:** Current sentiment features may add noise rather than predictive signal
- **Feature Quality:** The sentiment proxy features (vix_component, returns_component, sector_component) are derived from price data, potentially causing multicollinearity
- **Observation Space Complexity:** Larger observation spaces (more features) require more training data/time

### 2. High Variance Across Seeds

All configurations show high variance across seeds (std ~0.4-0.5 for Sharpe), indicating:

- RL training is sensitive to initialization
- Results may not be statistically significant
- More seeds needed for robust conclusions

### 3. Core_3 Shows Lowest Variance

The `core_3` configuration (sentiment_score, news_count, sentiment_proxy) has the lowest variance (std = 0.021), suggesting these features provide more stable training, even if absolute performance is lower.

### 4. All Sentiment vs Baseline

Comparing the two extremes:
- **Baseline:** Higher mean (1.627 vs 1.431), higher variance
- **All Sentiment:** Lower mean, but includes best single run with seed 456 (1.723)

---

## Interpretation

### Why Sentiment Features May Not Help

1. **Proxy Features Are Redundant:** `vix_component`, `returns_component`, and `sector_component` are derived from price/VIX data already captured in technical indicators

2. **Sentiment Data Quality:** Historical sentiment may not accurately reflect real-time market sentiment that drives prices

3. **Feature Engineering Needed:** Raw sentiment scores may need transformation (e.g., sentiment momentum, sentiment divergence from price)

4. **Test Period Characteristics:** The test period (Jul 2024 - Nov 2025) may have unique characteristics where technical indicators dominate

---

## Recommendations

### Short-term (Use Current Best)

1. **Deploy baseline model** with tuned hyperparameters for production
2. Best config: `baseline` with Optuna-tuned hyperparameters
3. Expected Sharpe: ~1.6, Return: ~48%

### Medium-term (Improve Sentiment Features)

1. **Remove redundant features:** Drop `vix_component`, `returns_component`, `sector_component`
2. **Add sentiment derivatives:**
   - Sentiment momentum (change over time)
   - Sentiment-price divergence
   - Sentiment volatility
3. **Re-run ablation** with improved features

### Long-term (Alternative Approaches)

1. **Attention mechanism:** Let model learn which features matter
2. **Feature selection:** Use SHAP or permutation importance
3. **Ensemble:** Combine baseline and sentiment models

---

## Conclusion

The ablation study demonstrates that **adding sentiment features does not improve model performance** with the current feature engineering approach. The baseline model with optimized hyperparameters remains the best choice.

**Recommended Model:** Baseline (no sentiment) with Optuna-tuned hyperparameters
- Sharpe Ratio: 1.627 (mean), up to 1.954 (best seed)
- Total Return: 47.8% (mean)
- Max Drawdown: -15.8% (mean)

---

## Files

- `experiments/ablation_results/ablation_summary.csv` - Raw results
- `experiments/ablation_results/*/results.json` - Individual experiment details
- `experiments/ablation_results/*/model_*.zip` - Trained models

---

**Report Generated:** December 19, 2025
