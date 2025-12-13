"""
Enhanced Environment Module

Provides enhanced portfolio trading environments with sentiment integration.

Components:
- enhanced_portfolio_env.py: EnhancedPortfolioEnv with sentiment features
"""

from src.env.enhanced_portfolio_env import (
    EnhancedPortfolioEnv,
    create_enhanced_environment,
)

__all__ = [
    "EnhancedPortfolioEnv",
    "create_enhanced_environment",
]
