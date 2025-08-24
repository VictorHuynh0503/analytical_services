"""
Common technical analysis indicators
"""
import numpy as np
import pandas as pd

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        data (pd.Series): Price data
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Simple moving average
    """
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        data (pd.Series): Price data
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Exponential moving average
    """
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        data (pd.Series): Price data
        window (int): Window size for RSI calculation
        
    Returns:
        pd.Series: RSI values
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))
