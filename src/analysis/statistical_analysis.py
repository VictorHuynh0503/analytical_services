"""
Statistical analysis functions
"""
import numpy as np
from scipy import stats
import pandas as pd

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns (pd.Series): Asset returns
        risk_free_rate (float): Risk-free rate (annual)
        
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate/252  # Convert annual rate to daily
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_volatility(returns: pd.Series, annualized: bool = True) -> float:
    """
    Calculate volatility
    
    Args:
        returns (pd.Series): Asset returns
        annualized (bool): Whether to annualize the volatility
        
    Returns:
        float: Volatility
    """
    vol = returns.std()
    if annualized:
        vol = vol * np.sqrt(252)  # Assuming daily returns
    return vol

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta of an asset relative to the market
    
    Args:
        asset_returns (pd.Series): Asset returns
        market_returns (pd.Series): Market returns
        
    Returns:
        float: Beta coefficient
    """
    covariance = asset_returns.cov(market_returns)
    market_variance = market_returns.var()
    return covariance / market_variance
