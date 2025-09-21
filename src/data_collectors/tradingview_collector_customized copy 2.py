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
import requests

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
        
        # Create table with basic structure first
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
        
        # Check if data_source column exists, if not add it
        cursor.execute("PRAGMA table_info(ohlc_data)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'data_source' not in columns:
            logger.info("Adding data_source column to existing table...")
            cursor.execute('ALTER TABLE ohlc_data ADD COLUMN data_source TEXT DEFAULT "live"')
            # Update existing records to have 'live' as data_source
            cursor.execute('UPDATE ohlc_data SET data_source = "live" WHERE data_source IS NULL')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_time_tf ON ohlc_data(symbol, timestamp, timeframe)')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_ohlc(self, data: OHLCData, category: str = "", subcategory: str = "", source: str = "live"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO ohlc_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, 
                 volume, timeframe, category, subcategory, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timestamp, data.open_price, data.high_price,
                data.low_price, data.close_price, data.volume, data.timeframe,
                category, subcategory, source
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()
    
    def insert_bulk_ohlc(self, ohlc_list: List[OHLCData], category: str = "", subcategory: str = "", source: str = "historical"):
        """Bulk insert for historical data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            data_tuples = [
                (ohlc.symbol, ohlc.timestamp, ohlc.open_price, ohlc.high_price,
                 ohlc.low_price, ohlc.close_price, ohlc.volume, ohlc.timeframe,
                 category, subcategory, source)
                for ohlc in ohlc_list
            ]
            
            cursor.executemany('''
                INSERT OR REPLACE INTO ohlc_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, 
                 volume, timeframe, category, subcategory, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_tuples)
            
            conn.commit()
            logger.info(f"Bulk inserted {len(ohlc_list)} historical records")
        except sqlite3.Error as e:
            logger.error(f"Bulk insert error: {e}")
        finally:
            conn.close()
    
    def get_dataframe(self, symbol: str = None, timeframe: str = None, 
                     start_date: datetime = None, end_date: datetime = None,
                     limit: int = None, source: str = None) -> pd.DataFrame:
        """Get OHLC data as pandas DataFrame with flexible filtering"""
        conn = sqlite3.connect(self.db_path)
        
        # First check if data_source column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(ohlc_data)")
        columns = [column[1] for column in cursor.fetchall()]
        has_data_source = 'data_source' in columns
        
        query = "SELECT * FROM ohlc_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        
        if source and has_data_source:
            query += " AND data_source = ?"
            params.append(source)
        elif source and not has_data_source:
            logger.warning(f"data_source column not found, ignoring source filter: {source}")
        
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
            
            # Add default data_source if column doesn't exist
            if not has_data_source and 'data_source' not in df.columns:
                df['data_source'] = 'live'
        
        return df
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the latest timestamp for a symbol/timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(timestamp) FROM ohlc_data 
            WHERE symbol = ? AND timeframe = ?
        ''', (symbol, timeframe))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        return None

class TimeframeManager:
    """Manages multiple timeframes for OHLC data"""
    
    TIMEFRAMES = {
        '1m': {'minutes': 1, 'tv_resolution': '1'},
        '5m': {'minutes': 5, 'tv_resolution': '5'},
        '15m': {'minutes': 15, 'tv_resolution': '15'},
        '30m': {'minutes': 30, 'tv_resolution': '30'},
        '1h': {'minutes': 60, 'tv_resolution': '60'},
        '4h': {'minutes': 240, 'tv_resolution': '240'},
        '1d': {'minutes': 1440, 'tv_resolution': '1D'},
        '1w': {'minutes': 10080, 'tv_resolution': '1W'},
        '1M': {'minutes': 43200, 'tv_resolution': '1M'}
    }
    
    def __init__(self):
        self.price_cache = {}  # symbol -> {timeframe -> ohlc_data}
    
    def get_timeframe_timestamp(self, timeframe: str, current_time: datetime) -> datetime:
        """Get the appropriate timestamp for a timeframe"""
        minutes = self.TIMEFRAMES[timeframe]['minutes']
        
        if minutes >= 43200:  # Monthly
            return current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif minutes >= 10080:  # Weekly
            days_since_monday = current_time.weekday()
            monday = current_time - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif minutes >= 1440:  # Daily
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
            minutes = self.TIMEFRAMES[timeframe]['minutes']
            candle_end_time = data['timestamp'] + timedelta(minutes=minutes)
            
            if current_time >= candle_end_time:
                completed.append(data)
                keys_to_remove.append(cache_key)
        
        # Remove completed candles from cache
        for key in keys_to_remove:
            del self.price_cache[symbol][key]
        
        return completed

class HistoricalDataFetcher:
    """Fetch historical data from TradingView"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.ws = None
        self.session_id = self._generate_session_id()
        self.chart_session_id = f"cs_{self._generate_session_id()}"
        self.pending_requests = {}  # track pending historical requests
        self.request_counter = 0
    
    def _generate_session_id(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    
    def _send_message(self, message):
        if self.ws:
            formatted_message = f"~m~{len(message)}~m~{message}"
            self.ws.send(formatted_message)
    
    def _create_message(self, method, params=None):
        return json.dumps({"m": method, "p": params or []})
    
    def fetch_historical_data(self, symbol: str, timeframe: str, bars_count: int = 5000, 
                            category: str = "", subcategory: str = "") -> bool:
        """Fetch historical data for a symbol and timeframe"""
        
        if timeframe not in TimeframeManager.TIMEFRAMES:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return False
        
        # Check if we already have recent data
        latest_timestamp = self.db_manager.get_latest_timestamp(symbol, timeframe)
        if latest_timestamp:
            time_diff = datetime.now() - latest_timestamp
            if time_diff.total_seconds() < 3600:  # Less than 1 hour old
                logger.info(f"Recent data exists for {symbol} {timeframe}, skipping historical fetch")
                return True
        
        self.request_counter += 1
        request_id = f"hist_{self.request_counter}"
        
        # Store request info
        self.pending_requests[request_id] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'category': category,
            'subcategory': subcategory,
            'timestamp': datetime.now()
        }
        
        # Connect if not connected
        if not self.ws:
            self._connect_for_historical()
            time.sleep(2)  # Wait for connection
        
        # Request historical data
        tv_resolution = TimeframeManager.TIMEFRAMES[timeframe]['tv_resolution']
        
        # Create chart session for this request
        chart_session = f"cs_{request_id}"
        
        # Initialize chart session
        self._send_message(self._create_message("chart_create_session", [chart_session, ""]))
        time.sleep(0.1)
        
        # Resolve symbol
        self._send_message(self._create_message("resolve_symbol", [
            chart_session, f"symbol_{request_id}", f"={symbol}"
        ]))
        time.sleep(0.1)
        
        # Create series
        self._send_message(self._create_message("create_series", [
            chart_session, f"s1_{request_id}", "s1", f"symbol_{request_id}", 
            tv_resolution, bars_count
        ]))
        
        logger.info(f"📈 Requested historical data: {symbol} {timeframe} ({bars_count} bars)")
        return True
    
    def _connect_for_historical(self):
        """Connect WebSocket for historical data fetching"""
        def on_message(ws, message):
            self._handle_historical_message(message)
        
        def on_error(ws, error):
            logger.error(f"Historical WS error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("Historical WebSocket closed")
        
        def on_open(ws):
            logger.info("Historical WebSocket connected")
            self._send_message(self._create_message("set_auth_token", ["unauthorized_user_token"]))
            self._send_message(self._create_message("chart_create_session", [self.chart_session_id, ""]))
        
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://data.tradingview.com/socket.io/websocket",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start in separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
    
    def _handle_historical_message(self, message):
        """Handle historical data messages"""
        try:
            if message.startswith('~m~'):
                parts = message.split('~m~')
                if len(parts) >= 3:
                    data = parts[2]
                    if data:
                        try:
                            parsed_data = json.loads(data)
                            if isinstance(parsed_data, dict):
                                method = parsed_data.get('m')
                                params = parsed_data.get('p', [])
                                
                                if method == 'timescale_update':
                                    self._process_historical_data(params)
                                elif method == 'series_completed':
                                    self._handle_series_completed(params)
                                    
                        except json.JSONDecodeError:
                            if data.startswith('~h~'):
                                heartbeat_id = data[3:]
                                self._send_message(f"~h~{heartbeat_id}")
        except Exception as e:
            logger.error(f"Error processing historical message: {e}")
    
    def _process_historical_data(self, params):
        """Process received historical OHLC data"""
        if len(params) >= 2:
            session_info = params[0]
            data_info = params[1]
            
            if isinstance(data_info, dict) and 's' in data_info:
                series_data = data_info['s']
                
                # Find matching request
                request_info = None
                for req_id, req_data in self.pending_requests.items():
                    if req_id in session_info:
                        request_info = req_data
                        break
                
                if not request_info:
                    return
                
                symbol = request_info['symbol']
                timeframe = request_info['timeframe']
                category = request_info['category']
                subcategory = request_info['subcategory']
                
                # Parse OHLC data
                ohlc_list = []
                
                for bar_data in series_data:
                    if isinstance(bar_data, dict):
                        timestamp = bar_data.get('t')  # Unix timestamp
                        open_price = bar_data.get('o')
                        high_price = bar_data.get('h')
                        low_price = bar_data.get('l')
                        close_price = bar_data.get('c')
                        volume = bar_data.get('v', 0)
                        
                        if all(v is not None for v in [timestamp, open_price, high_price, low_price, close_price]):
                            dt = datetime.fromtimestamp(timestamp)
                            
                            ohlc = OHLCData(
                                symbol=symbol,
                                timestamp=dt,
                                open_price=float(open_price),
                                high_price=float(high_price),
                                low_price=float(low_price),
                                close_price=float(close_price),
                                volume=float(volume),
                                timeframe=timeframe
                            )
                            ohlc_list.append(ohlc)
                
                if ohlc_list:
                    # Bulk insert historical data
                    self.db_manager.insert_bulk_ohlc(ohlc_list, category, subcategory, "historical")
                    logger.info(f"📊 Saved {len(ohlc_list)} historical bars for {symbol} {timeframe}")
    
    def _handle_series_completed(self, params):
        """Handle series completion"""
        logger.info("Historical data series completed")
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()

class SymbolManager:
    """Symbol management with categories"""
    
    @staticmethod
    def get_symbol_sets():
        return {
            "crypto_major": {
                "symbols": ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT", "BINANCE:ADAUSDT"],
                "category": "crypto", "subcategory": "major"
            },
            "crypto_alt": {
                "symbols": ["BINANCE:SOLUSDT", "BINANCE:DOTUSDT", "BINANCE:MATICUSDT", "BINANCE:AVAXUSDT"],
                "category": "crypto", "subcategory": "altcoins"
            },
            "us_tech": {
                "symbols": ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL", "NASDAQ:TSLA"],
                "category": "stocks", "subcategory": "us_tech"
            },
            "us_finance": {
                "symbols": ["NYSE:JPM", "NYSE:BAC", "NYSE:WFC", "NYSE:GS"],
                "category": "stocks", "subcategory": "us_finance"
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
        self.historical_fetcher = HistoricalDataFetcher(db_manager)
        self.enabled_timeframes = enabled_timeframes or ['1m', '5m', '15m', '1h', '1d']
        
        # Filter timeframes
        self.timeframe_manager.TIMEFRAMES = {
            tf: data for tf, data in self.timeframe_manager.TIMEFRAMES.items() 
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
    
    def fetch_historical_data_for_symbols(self, symbol_sets: Dict, timeframes: List[str] = None, 
                                        bars_count: int = 1000):
        """Fetch historical data for all symbols and timeframes"""
        timeframes = timeframes or self.enabled_timeframes
        
        logger.info(f"🔄 Starting historical data fetch for {len(timeframes)} timeframes...")
        
        total_requests = 0
        for set_name, set_data in symbol_sets.items():
            symbols = set_data['symbols']
            category = set_data['category']
            subcategory = set_data['subcategory']
            
            for symbol in symbols:
                for timeframe in timeframes:
                    success = self.historical_fetcher.fetch_historical_data(
                        symbol, timeframe, bars_count, category, subcategory
                    )
                    if success:
                        total_requests += 1
                    time.sleep(0.2)  # Rate limiting
        
        logger.info(f"📈 Initiated {total_requests} historical data requests")
        
        # Wait for historical data to be processed
        logger.info("⏳ Waiting for historical data processing...")
        time.sleep(10)  # Allow time for data to be received
        
        return total_requests
    
    def get_dataframe(self, symbol: str = None, timeframe: str = None, 
                     days_back: int = None, limit: int = None, 
                     source: str = None) -> pd.DataFrame:
        """Get DataFrame with optional filters"""
        start_date = None
        if days_back:
            start_date = datetime.now() - timedelta(days=days_back)
        
        return self.db_manager.get_dataframe(
            symbol=symbol, timeframe=timeframe, 
            start_date=start_date, limit=limit, source=source
        )
    
    def get_symbol_data_summary(self) -> pd.DataFrame:
        """Get summary of available data for each symbol/timeframe combination"""
        df = self.db_manager.get_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        # Check if data_source column exists
        has_data_source = 'data_source' in df.columns
        
        if has_data_source:
            summary = df.groupby(['symbol', 'timeframe', 'category']).agg({
                'timestamp': ['min', 'max', 'count'],
                'data_source': lambda x: list(set(x))
            }).reset_index()
            
            # Flatten column names
            summary.columns = ['symbol', 'timeframe', 'category', 'first_date', 'last_date', 'total_bars', 'sources']
        else:
            summary = df.groupby(['symbol', 'timeframe', 'category']).agg({
                'timestamp': ['min', 'max', 'count']
            }).reset_index()
            
            # Flatten column names
            summary.columns = ['symbol', 'timeframe', 'category', 'first_date', 'last_date', 'total_bars']
            summary['sources'] = ['live'] * len(summary)  # Default to 'live' for backward compatibility
        
        return summary
    
    def print_data_summary(self):
        """Print comprehensive data summary"""
        summary_df = self.get_symbol_data_summary()
        
        if summary_df.empty:
            print("No data available")
            return
        
        print(f"\n{'='*100}")
        print(f"TRADING DATA SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
        
        # Group by timeframe
        for timeframe in self.enabled_timeframes:
            tf_data = summary_df[summary_df['timeframe'] == timeframe]
            if not tf_data.empty:
                print(f"\n📊 {timeframe.upper()} Timeframe:")
                print(f"{'Symbol':<20} {'Category':<15} {'First Date':<12} {'Last Date':<12} {'Bars':<8} {'Sources'}")
                print("-" * 90)
                
                for _, row in tf_data.iterrows():
                    first_date = pd.to_datetime(row['first_date']).strftime('%Y-%m-%d') if row['first_date'] else 'N/A'
                    last_date = pd.to_datetime(row['last_date']).strftime('%Y-%m-%d') if row['last_date'] else 'N/A'
                    sources = ','.join(row['sources']) if row['sources'] else 'N/A'
                    
                    print(f"{row['symbol']:<20} {row['category']:<15} {first_date:<12} {last_date:<12} {row['total_bars']:<8} {sources}")
        
        print(f"\n{'='*100}")
        print(f"Total unique symbols: {summary_df['symbol'].nunique()}")
        print(f"Total timeframes: {summary_df['timeframe'].nunique()}")
        print(f"Total data points: {summary_df['total_bars'].sum():,}")
    
    def start_live_collection(self):
        """Start live data collection (simplified version of original)"""
        # This is a simplified version - you can extend with the full WebSocket implementation
        logger.info("🔴 Live data collection not implemented in this version")
        logger.info("Focus is on historical data retrieval and analysis")
    
    def close(self):
        """Clean up resources"""
        if self.historical_fetcher:
            self.historical_fetcher.close()

def main():
    """Main function demonstrating historical data fetching"""
    # Initialize system
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']  # Customize as needed
    
    db_manager = DatabaseManager()
    collector = TradingDataCollector(db_manager, timeframes)
    
    # Get symbol sets
    symbol_sets = SymbolManager.get_symbol_sets()
    
    print("🚀 Trading Data Collector - Historical Mode")
    print("=" * 50)
    
    # Fetch historical data
    print("\n1️⃣ Fetching Historical Data...")
    collector.fetch_historical_data_for_symbols(
        symbol_sets, 
        timeframes=['5m', '1h', '1d'],  # Start with these timeframes
        bars_count=500  # Get last 500 bars
    )
    
    # Wait a bit for data processing
    time.sleep(15)
    
    print("\n2️⃣ Data Summary:")
    collector.print_data_summary()
    
    print("\n3️⃣ Sample Data Analysis:")
    
    # Example 1: Bitcoin 1-hour data
    btc_1h = collector.get_dataframe(symbol="BINANCE:BTCUSDT", timeframe="1h", limit=10)
    if not btc_1h.empty:
        print(f"\n📈 Bitcoin 1H Data (Last 10 bars):")
        print(btc_1h[['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']].to_string())
    
    # Example 2: All crypto daily data
    crypto_daily = collector.get_dataframe(timeframe="1d")
    if not crypto_daily.empty:
        crypto_data = crypto_daily[crypto_daily['category'] == 'crypto']
        if not crypto_data.empty:
            print(f"\n🪙 Crypto Daily Data Summary:")
            latest_prices = crypto_data.groupby('symbol')['close_price'].last()
            print(latest_prices.head(10))
    
    # Example 3: Historical vs Live data comparison
    all_data = collector.get_dataframe()
    if not all_data.empty:
        # Check if data_source column exists before using it
        if 'data_source' in all_data.columns:
            source_summary = all_data['data_source'].value_counts()
            print(f"\n📊 Data Source Summary:")
            print(source_summary)
        else:
            print(f"\n📊 Data Source Summary:")
            print(f"All data marked as 'live' (data_source column not available)")
            print(f"Total records: {len(all_data)}")
    
    print(f"\n4️⃣ DataFrame Export Examples:")
    
    # Export specific data to CSV
    btc_all_timeframes = collector.get_dataframe(symbol="BINANCE:BTCUSDT")
    if not btc_all_timeframes.empty:
        filename = f"btc_data_{datetime.now().strftime('%Y%m%d')}.csv"
        btc_all_timeframes.to_csv(filename, index=False)
        print(f"💾 Exported BTC data to: {filename}")
    
    # Interactive mode
    print(f"\n5️⃣ Interactive Analysis:")
    print("You can now use the collector object for custom analysis:")
    print("- collector.get_dataframe(symbol='BINANCE:BTCUSDT', timeframe='1h')")
    print("- collector.get_dataframe(timeframe='1d', days_back=30)")
    print("- collector.get_symbol_data_summary()")
    
    # Keep connection open for additional requests
    print(f"\n✅ Historical data collection completed!")
    print("Database ready for analysis. Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(10)
            # You could add periodic updates here
            
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down...")
        collector.close()
        print("Goodbye! 👋")

if __name__ == "__main__":
    main()