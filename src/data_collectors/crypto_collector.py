"""
Module for collecting cryptocurrency data from various exchanges
"""
import ccxt

class CryptoDataCollector:
    def __init__(self, exchange_name='binance'):
        """
        Initialize crypto data collector
        
        Args:
            exchange_name (str): Name of the exchange (default: 'binance')
        """
        self.exchange = getattr(ccxt, exchange_name)()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', limit: int = 100):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1m', '1h', '1d')
            limit (int): Number of candles to fetch
            
        Returns:
            list: OHLCV data
        """
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
