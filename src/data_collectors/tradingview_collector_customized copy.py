import websocket
import json
import threading
import time
import random
import string
import sqlite3
from datetime import datetime, timedelta
import logging
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import schedule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OHLCData:
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    timeframe: str

class DatabaseManager:
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlc_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                timeframe TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_time_tf ON ohlc_data(symbol, timestamp, timeframe)')
        conn.commit()
        conn.close()
    
    def insert_ohlc(self, data: OHLCData, category: str = "", subcategory: str = ""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO ohlc_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, 
                 volume, timeframe, category, subcategory)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timestamp, data.open_price, data.high_price,
                data.low_price, data.close_price, data.volume, data.timeframe,
                category, subcategory
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()
    
    def get_dataframe(self, symbol: str = None, timeframe: str = None, 
                     start_date: datetime = None, end_date: datetime = None,
                     limit: int = None) -> pd.DataFrame:
        """Get OHLC data as pandas DataFrame with flexible filtering"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM ohlc_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY symbol, timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['symbol', 'timestamp'])
        
        return df

class TimeframeManager:
    """Manages multiple timeframes for OHLC data"""
    
    TIMEFRAMES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    def __init__(self):
        self.price_cache = {}  # symbol -> {timeframe -> ohlc_data}
    
    def get_timeframe_timestamp(self, timeframe: str, current_time: datetime) -> datetime:
        """Get the appropriate timestamp for a timeframe"""
        minutes = self.TIMEFRAMES[timeframe]
        
        if minutes >= 1440:  # Daily
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif minutes >= 60:  # Hourly intervals
            hour_interval = minutes // 60
            aligned_hour = (current_time.hour // hour_interval) * hour_interval
            return current_time.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
        else:  # Minute intervals
            aligned_minute = (current_time.minute // minutes) * minutes
            return current_time.replace(minute=aligned_minute, second=0, microsecond=0)
    
    def update_ohlc(self, symbol: str, price: float, volume: float, timestamp: datetime):
        """Update OHLC data for all timeframes"""
        if symbol not in self.price_cache:
            self.price_cache[symbol] = {}
        
        for timeframe in self.TIMEFRAMES:
            tf_timestamp = self.get_timeframe_timestamp(timeframe, timestamp)
            cache_key = f"{timeframe}_{tf_timestamp}"
            
            if cache_key not in self.price_cache[symbol]:
                self.price_cache[symbol][cache_key] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'timestamp': tf_timestamp,
                    'timeframe': timeframe,
                    'tick_count': 1
                }
            else:
                data = self.price_cache[symbol][cache_key]
                data['high'] = max(data['high'], price)
                data['low'] = min(data['low'], price)
                data['close'] = price
                data['volume'] = max(data['volume'], volume)
                data['tick_count'] += 1
    
    def get_completed_candles(self, symbol: str, current_time: datetime) -> List[Dict]:
        """Get completed candles that should be saved to database"""
        if symbol not in self.price_cache:
            return []
        
        completed = []
        keys_to_remove = []
        
        for cache_key, data in self.price_cache[symbol].items():
            timeframe = data['timeframe']
            minutes = self.TIMEFRAMES[timeframe]
            candle_end_time = data['timestamp'] + timedelta(minutes=minutes)
            
            if current_time >= candle_end_time:
                completed.append(data)
                keys_to_remove.append(cache_key)
        
        # Remove completed candles from cache
        for key in keys_to_remove:
            del self.price_cache[symbol][key]
        
        return completed

class SymbolManager:
    """Simplified symbol management"""
    
    @staticmethod
    def get_symbol_sets():
        return {
            "crypto_major": {
                "symbols": ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT", "BINANCE:ADAUSDT"],
                "category": "crypto", "subcategory": "major"
            },
            "us_tech": {
                "symbols": ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL", "NASDAQ:TSLA"],
                "category": "stocks", "subcategory": "us_tech"
            },
            "forex_major": {
                "symbols": ["FX:EURUSD", "FX:GBPUSD", "FX:USDJPY", "FX:AUDUSD"],
                "category": "forex", "subcategory": "major"
            },
            "vietnam_top": {
                "symbols": ["HOSE:VIC", "HOSE:VCB", "HOSE:GAS", "HOSE:VNM"],
                "category": "stocks", "subcategory": "vietnam"
            }
        }

class TradingDataCollector:
    def __init__(self, db_manager: DatabaseManager, enabled_timeframes: List[str] = None):
        self.db_manager = db_manager
        self.timeframe_manager = TimeframeManager()
        self.enabled_timeframes = enabled_timeframes or ['1m', '5m', '15m', '1h']
        
        # Filter timeframes
        self.timeframe_manager.TIMEFRAMES = {
            tf: mins for tf, mins in self.timeframe_manager.TIMEFRAMES.items() 
            if tf in self.enabled_timeframes
        }
        
        self.ws = None
        self.session_id = self._generate_session_id()
        self.quote_session_id = f"qs_{self._generate_session_id()}"
        self.is_connected = False
        self.subscribed_symbols = {}
        
        logger.info(f"Enabled timeframes: {list(self.timeframe_manager.TIMEFRAMES.keys())}")
    
    def _generate_session_id(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    
    def _send_message(self, message):
        if self.ws and self.is_connected:
            formatted_message = f"~m~{len(message)}~m~{message}"
            self.ws.send(formatted_message)
    
    def _create_message(self, method, params=None):
        return json.dumps({"m": method, "p": params or []})
    
    def on_message(self, ws, message):
        try:
            if message.startswith('~m~'):
                parts = message.split('~m~')
                if len(parts) >= 3:
                    data = parts[2]
                    if data:
                        try:
                            parsed_data = json.loads(data)
                            self._handle_parsed_message(parsed_data)
                        except json.JSONDecodeError:
                            if data.startswith('~h~'):
                                heartbeat_id = data[3:]
                                self._send_message(f"~h~{heartbeat_id}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _handle_parsed_message(self, data):
        if isinstance(data, dict) and data.get('m') == 'qsd':
            params = data.get('p', [])
            if len(params) >= 2:
                self._handle_quote_data(params)
    
    def _handle_quote_data(self, params):
        quote_data = params[1]
        if isinstance(quote_data, dict):
            symbol = quote_data.get('n', 'Unknown')
            
            if 'v' in quote_data and symbol in self.subscribed_symbols:
                values = quote_data['v']
                if isinstance(values, dict):
                    last_price = values.get('lp')
                    volume = values.get('volume', 0)
                    
                    if last_price is not None:
                        current_time = datetime.now()
                        self.timeframe_manager.update_ohlc(symbol, last_price, volume, current_time)
                        
                        # Log real-time update
                        change = values.get('ch', 0)
                        color = "🟢" if change >= 0 else "🔴"
                        print(f"{color} {symbol}: {last_price:.4f} | Vol: {volume:.0f} | {current_time.strftime('%H:%M:%S')}")
    
    def save_completed_candles(self):
        """Save completed candles to database"""
        current_time = datetime.now()
        total_saved = 0
        
        for symbol in list(self.subscribed_symbols.keys()):
            completed_candles = self.timeframe_manager.get_completed_candles(symbol, current_time)
            
            for candle_data in completed_candles:
                symbol_info = self.subscribed_symbols[symbol]
                
                ohlc = OHLCData(
                    symbol=symbol,
                    timestamp=candle_data['timestamp'],
                    open_price=candle_data['open'],
                    high_price=candle_data['high'],
                    low_price=candle_data['low'],
                    close_price=candle_data['close'],
                    volume=candle_data['volume'],
                    timeframe=candle_data['timeframe']
                )
                
                self.db_manager.insert_ohlc(
                    ohlc, symbol_info['category'], symbol_info['subcategory']
                )
                
                logger.info(f"💾 {symbol} {candle_data['timeframe']}: "
                          f"OHLC({candle_data['open']:.4f}, {candle_data['high']:.4f}, "
                          f"{candle_data['low']:.4f}, {candle_data['close']:.4f}) "
                          f"V:{candle_data['volume']:.0f}")
                total_saved += 1
        
        if total_saved > 0:
            logger.info(f"📊 Saved {total_saved} candles to database")
    
    def get_dataframe(self, symbol: str = None, timeframe: str = None, 
                     days_back: int = None, limit: int = None) -> pd.DataFrame:
        """Convenient method to get DataFrame"""
        start_date = None
        if days_back:
            start_date = datetime.now() - timedelta(days=days_back)
        
        return self.db_manager.get_dataframe(
            symbol=symbol, timeframe=timeframe, 
            start_date=start_date, limit=limit
        )
    
    def print_summary(self):
        """Print current data summary"""
        print(f"\n{'='*80}")
        print(f"TRADING DATA SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        for timeframe in self.enabled_timeframes:
            df = self.get_dataframe(timeframe=timeframe, limit=50)
            if not df.empty:
                symbol_count = df['symbol'].nunique()
                latest_time = df['timestamp'].max()
                print(f"{timeframe:>4} | Symbols: {symbol_count:>3} | Latest: {latest_time}")
        
        print(f"{'='*80}\n")
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        self.is_connected = False
    
    def on_open(self, ws):
        logger.info("WebSocket connection opened")
        self.is_connected = True
        self._initialize_connection()
    
    def _initialize_connection(self):
        self._send_message(self._create_message("set_auth_token", ["unauthorized_user_token"]))
        self._send_message(self._create_message("quote_create_session", [self.quote_session_id]))
        
        quote_fields = [
            "ch", "chp", "lp", "volume", "bid", "ask", "high_price", 
            "low_price", "open_price", "prev_close_price"
        ]
        
        self._send_message(self._create_message("quote_set_fields", [self.quote_session_id] + quote_fields))
    
    def subscribe_symbols(self, symbol_sets: Dict):
        """Subscribe to symbol sets"""
        for set_name, set_data in symbol_sets.items():
            symbols = set_data['symbols']
            category = set_data['category']
            subcategory = set_data['subcategory']
            
            for symbol in symbols:
                self.subscribed_symbols[symbol] = {
                    'category': category,
                    'subcategory': subcategory
                }
                
                self._send_message(self._create_message("quote_add_symbols", [self.quote_session_id, symbol]))
                logger.info(f"Subscribed: {symbol} ({category}/{subcategory})")
                time.sleep(0.1)
    
    def start(self):
        """Start the data collection"""
        # Schedule candle saving
        schedule.every().minute.do(self.save_completed_candles)
        schedule.every(5).minutes.do(self.print_summary)
        
        def run_scheduler():
            while self.is_connected:
                schedule.run_pending()
                time.sleep(1)
        
        # Start scheduler
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Start WebSocket
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://data.tradingview.com/socket.io/websocket",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        return ws_thread

def main():
    # Initialize with custom timeframes
    timeframes = ['1m', '5m', '15m', '1h', '4h']  # Customize as needed
    
    db_manager = DatabaseManager()
    collector = TradingDataCollector(db_manager, timeframes)
    
    # Subscribe to symbols
    symbol_sets = SymbolManager.get_symbol_sets()
    
    # Start collection
    ws_thread = collector.start()
    time.sleep(3)  # Wait for connection
    
    collector.subscribe_symbols(symbol_sets)
    
    logger.info(f"Data collection started for {len(collector.subscribed_symbols)} symbols")
    logger.info("Press Ctrl+C to stop and view data examples")
    
    try:
        while True:
            time.sleep(10)
            
            # Example: Get latest 1-hour Bitcoin data
            btc_1h = collector.get_dataframe(symbol="BINANCE:BTCUSDT", timeframe="1h", limit=5)
            if not btc_1h.empty:
                latest = btc_1h.iloc[-1]
                print(f"📈 BTC 1H Latest: O:{latest['open_price']:.1f} H:{latest['high_price']:.1f} "
                      f"L:{latest['low_price']:.1f} C:{latest['close_price']:.1f}")
                      
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
        
        # Save final candles
        collector.save_completed_candles()
        
        # Show data examples
        print("\n📊 DATA EXAMPLES:")
        
        # 5-minute data for all symbols
        df_5m = collector.get_dataframe(timeframe="5m", limit=20)
        print(f"\n5-Minute Data (Last 20 records):")
        print(df_5m[['symbol', 'timestamp', 'close_price', 'volume', 'timeframe']].head(10))
        
        # 1-hour data for crypto
        df_crypto = db_manager.get_dataframe()
        df_crypto_1h = df_crypto[(df_crypto['category'] == 'crypto') & (df_crypto['timeframe'] == '1h')]
        print(f"\nCrypto 1H Data:")
        print(df_crypto_1h[['symbol', 'timestamp', 'close_price', 'timeframe']].head(5))
        
        collector.print_summary()

if __name__ == "__main__":
    main()