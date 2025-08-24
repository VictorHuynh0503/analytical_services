"""
Module for collecting stock market data from various sources
"""
import yfinance as yf
from datetime import datetime

class StockDataCollector:
    def __init__(self):
        pass
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str = None):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
        
        Returns:
            pandas.DataFrame: Historical stock data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        ticker = yf.Ticker(symbol)
        return ticker.history(start=start_date, end=end_date)
