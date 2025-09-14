import websocket
import json
import threading
import time
import random
import string
import sqlite3
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import schedule

# Configure logging
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
    timeframe: str = "1m"
    category: str = ""
    subcategory: str = ""

class DatabaseManager:
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create logs table for OHLC data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                timeframe TEXT DEFAULT '1m',
                category TEXT,
                subcategory TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        
        # Create symbols table for metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                category TEXT,
                subcategory TEXT,
                market TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_symbol_time ON logs(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_category ON logs(category, subcategory)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_ohlc_data(self, data: OHLCData):
        """Insert OHLC data into logs table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO logs 
                (symbol, timestamp, open_price, high_price, low_price, close_price, 
                 volume, timeframe, category, subcategory)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timestamp, data.open_price, data.high_price,
                data.low_price, data.close_price, data.volume, data.timeframe,
                data.category, data.subcategory
            ))
            conn.commit()
            logger.debug(f"Inserted OHLC data for {data.symbol}")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()
    
    def insert_symbol_metadata(self, symbol: str, name: str, category: str, subcategory: str, market: str):
        """Insert symbol metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO symbols (symbol, name, category, subcategory, market)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, name, category, subcategory, market))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error inserting symbol metadata: {e}")
        finally:
            conn.close()
    
    def get_latest_data(self, symbol: str, limit: int = 10):
        """Get latest OHLC data for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM logs WHERE symbol = ? 
            ORDER BY timestamp DESC LIMIT ?
        ''', (symbol, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results

class SymbolManager:
    """Manages different categories of trading symbols"""
    
    @staticmethod
    def get_vietnam_stocks():
        return {
            "top_tier": {
                "symbols": ["HOSE:VIC", "HOSE:VHM", "HOSE:VCB", "HOSE:GAS", "HOSE:MSN", 
                           "HOSE:BID", "HOSE:VNM", "HOSE:HPG", "HOSE:CTG", "HOSE:TCB", "HOSE:VN30", "HNX:VN301!"],
                "names": ["Vingroup", "Vinhomes", "Vietcombank", "Gas", "Masan Group",
                         "BIDV", "Vinamilk", "Hoa Phat", "VietinBank", "Techcombank"]
            },
            "big_cap": {
                "symbols": ["HOSE:MWG", "HOSE:PLX", "HOSE:POW", "HOSE:SAB", "HOSE:VJC",
                           "HOSE:FPT", "HOSE:MBB", "HOSE:VRE", "HOSE:ACB", "HOSE:TPB"],
                "names": ["Mobile World", "Petrolimex", "PetroVietnam Power", "Sabeco", "VietJet",
                         "FPT Corp", "Military Bank", "Vincom Retail", "ACB", "TPBank"]
            },
            "mid_cap": {
                "symbols": ["HOSE:VGC", "HOSE:VPI", "HOSE:DGC", "HOSE:GVR", "HOSE:KDH",
                           "HOSE:NVL", "HOSE:PDR", "HOSE:SSI", "HOSE:HVN", "HOSE:DXG"],
                "names": ["Viglacera", "VPI", "Hoa Sen", "GVR", "Khang Dien",
                         "No Va Land", "Phu Nhuan Jewelry", "SSI", "Vietnam Airlines", "Dat Xanh"]
            }
        }
    
    @staticmethod
    def get_us_stocks():
        return {
            "mega_cap": {
                "symbols": ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL", "NASDAQ:AMZN", "NASDAQ:NVDA",
                           "NASDAQ:TSLA", "NASDAQ:META", "NYSE:BRK.A", "NASDAQ:AVGO", "NYSE:JPM"],
                "names": ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA",
                         "Tesla", "Meta", "Berkshire Hathaway", "Broadcom", "JPMorgan Chase"]
            },
            "large_cap": {
                "symbols": ["NYSE:JNJ", "NYSE:V", "NYSE:PG", "NYSE:UNH", "NYSE:HD",
                           "NYSE:MA", "NYSE:PFE", "NYSE:BAC", "NYSE:ABBV", "NYSE:KO"],
                "names": ["Johnson & Johnson", "Visa", "Procter & Gamble", "UnitedHealth", "Home Depot",
                         "Mastercard", "Pfizer", "Bank of America", "AbbVie", "Coca-Cola"]
            },
            "mid_cap": {
                "symbols": ["NASDAQ:AMD", "NYSE:CRM", "NYSE:NFLX", "NASDAQ:ADBE", "NYSE:DIS",
                           "NYSE:PYPL", "NYSE:INTC", "NYSE:CMCSA", "NYSE:VZ", "NYSE:T"],
                "names": ["AMD", "Salesforce", "Netflix", "Adobe", "Disney",
                         "PayPal", "Intel", "Comcast", "Verizon", "AT&T"]
            }
        }
    
    @staticmethod
    def get_crypto():
        return {
            "blue_chip": {
                "symbols": ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT", 
                           "BINANCE:XRPUSDT", "BINANCE:ADAUSDT"],
                "names": ["Bitcoin", "Ethereum", "Binance Coin", "XRP", "Cardano"]
            },
            "defi": {
                "symbols": ["BINANCE:UNIUSDT", "BINANCE:AAVEUSDT", "BINANCE:COMPUSDT",
                           "BINANCE:MKRUSDT", "BINANCE:SUSHIUSDT"],
                "names": ["Uniswap", "Aave", "Compound", "Maker", "SushiSwap"]
            },
            "layer1": {
                "symbols": ["BINANCE:SOLUSDT", "BINANCE:AVAXUSDT", "BINANCE:DOTUSDT",
                           "BINANCE:MATICUSDT", "BINANCE:ATOMUSDT"],
                "names": ["Solana", "Avalanche", "Polkadot", "Polygon", "Cosmos"]
            },
            "meme": {
                "symbols": ["BINANCE:DOGEUSDT", "BINANCE:SHIBUSDT", "BINANCE:PEPEUSDT",
                           "BINANCE:FLOKIUSDT", "BINANCE:BONKUSDT"],
                "names": ["Dogecoin", "Shiba Inu", "Pepe", "Floki", "Bonk"]
            }
        }
    
    @staticmethod
    def get_forex():
        return {
            "major_pairs": {
                "symbols": ["FX:EURUSD", "FX:GBPUSD", "FX:USDJPY", "FX:USDCHF", 
                           "FX:AUDUSD", "FX:USDCAD", "FX:NZDUSD"],
                "names": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
                         "AUD/USD", "USD/CAD", "NZD/USD"]
            },
            "minor_pairs": {
                "symbols": ["FX:EURGBP", "FX:EURJPY", "FX:EURCHF", "FX:GBPJPY",
                           "FX:GBPCHF", "FX:AUDJPY", "FX:CADJPY"],
                "names": ["EUR/GBP", "EUR/JPY", "EUR/CHF", "GBP/JPY",
                         "GBP/CHF", "AUD/JPY", "CAD/JPY"]
            },
            "exotic_pairs": {
                "symbols": ["FX:USDTRY", "FX:USDZAR", "FX:USDMXN", "FX:USDBRL",
                           "FX:USDSGD", "FX:USDHKD", "FX:USDTHB"],
                "names": ["USD/TRY", "USD/ZAR", "USD/MXN", "USD/BRL",
                         "USD/SGD", "USD/HKD", "USD/THB"]
            }
        }

class TradingViewWebSocket:
    def __init__(self, db_manager: DatabaseManager):
        self.ws = None
        self.session_id = self._generate_session_id()
        self.chart_session_id = f"cs_{self._generate_session_id()}"
        self.quote_session_id = f"qs_{self._generate_session_id()}"
        self.is_connected = False
        self.subscribed_symbols = {}  # symbol -> {category, subcategory}
        self.db_manager = db_manager
        self.price_cache = {}  # Store latest prices for OHLC calculation
        
    def _generate_session_id(self):
        """Generate a random session ID"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    
    def _send_message(self, message):
        """Send message to WebSocket"""
        if self.ws and self.is_connected:
            formatted_message = f"~m~{len(message)}~m~{message}"
            self.ws.send(formatted_message)
    
    def _create_message(self, method, params=None):
        """Create a properly formatted message"""
        message = {
            "m": method,
            "p": params or []
        }
        return json.dumps(message)
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
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
        """Handle parsed JSON messages"""
        if isinstance(data, dict):
            method = data.get('m')
            params = data.get('p', [])
            
            if method == 'qsd':
                self._handle_quote_data(params)
    
    def _handle_quote_data(self, params):
        """Handle real-time quote data and create OHLC records"""
        if len(params) >= 2:
            quote_data = params[1]
            
            if isinstance(quote_data, dict):
                symbol = quote_data.get('n', 'Unknown')
                
                if 'v' in quote_data and symbol in self.subscribed_symbols:
                    values = quote_data['v']
                    if isinstance(values, dict):
                        # Extract comprehensive price data
                        last_price = values.get('lp')
                        volume = values.get('volume', 0)
                        high_price = values.get('high_price', last_price)
                        low_price = values.get('low_price', last_price)
                        open_price = values.get('open_price', last_price)
                        change = values.get('ch', 0)
                        change_percent = values.get('chp', 0)
                        
                        if last_price is not None:
                            current_time = datetime.now()
                            minute_time = current_time.replace(second=0, microsecond=0)
                            
                            # Initialize or update price cache for OHLC
                            cache_key = f"{symbol}_{minute_time}"
                            
                            if cache_key not in self.price_cache:
                                self.price_cache[cache_key] = {
                                    'open': open_price if open_price else last_price,
                                    'high': high_price if high_price else last_price,
                                    'low': low_price if low_price else last_price,
                                    'close': last_price,
                                    'volume': volume,
                                    'timestamp': minute_time,
                                    'symbol': symbol,
                                    'tick_count': 1
                                }
                            else:
                                # Update OHLC with new tick data
                                cache_data = self.price_cache[cache_key]
                                cache_data['high'] = max(cache_data['high'], last_price, high_price or 0)
                                cache_data['low'] = min(cache_data['low'], last_price, low_price or float('inf'))
                                cache_data['close'] = last_price
                                cache_data['volume'] = max(cache_data['volume'], volume)
                                cache_data['tick_count'] += 1
                            
                            # Real-time OHLC display
                            symbol_info = self.subscribed_symbols[symbol]
                            ohlc_data = self.price_cache[cache_key]
                            
                            # Format and display current OHLC data
                            self._display_ohlc_update(symbol, symbol_info, ohlc_data, 
                                                    change, change_percent, current_time)
    
    def _display_ohlc_update(self, symbol, symbol_info, ohlc_data, change, change_percent, timestamp):
        """Display formatted OHLC data for each symbol update"""
        # Color coding for price changes
        color = "🟢" if change >= 0 else "🔴"
        change_str = f"{change:+.4f}" if change != 0 else "0.0000"
        change_pct_str = f"{change_percent:+.2f}%" if change_percent != 0 else "0.00%"
        
        # Format volume with appropriate units
        volume_formatted = self._format_volume(ohlc_data['volume'])
        
        print(f"\n{color} [{timestamp.strftime('%H:%M:%S')}] {symbol}")
        print(f"Category: {symbol_info['category']} / {symbol_info['subcategory']}")
        print(f"OHLC: O:{ohlc_data['open']:.4f} | H:{ohlc_data['high']:.4f} | L:{ohlc_data['low']:.4f} | C:{ohlc_data['close']:.4f}")
        print(f"Volume: {volume_formatted} | Ticks: {ohlc_data['tick_count']}")
        print(f"Change: {change_str} ({change_pct_str})")
        print("-" * 60)
    
    def _format_volume(self, volume):
        """Format volume with appropriate units"""
        if volume >= 1_000_000_000:
            return f"{volume/1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"{volume/1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.2f}K"
        else:
            return f"{volume:.2f}"
    
    def save_ohlc_data_to_db(self):
        """Save accumulated OHLC data to database with detailed logging"""
        current_time = datetime.now()
        minute_ago = (current_time - timedelta(minutes=1)).replace(second=0, microsecond=0)
        
        keys_to_remove = []
        saved_count = 0
        
        for cache_key, data in self.price_cache.items():
            if data['timestamp'] <= minute_ago:
                symbol_info = self.subscribed_symbols.get(data['symbol'], {})
                
                ohlc_data = OHLCData(
                    symbol=data['symbol'],
                    timestamp=data['timestamp'],
                    open_price=data['open'],
                    high_price=data['high'],
                    low_price=data['low'],
                    close_price=data['close'],
                    volume=data['volume'],
                    timeframe="1m",
                    category=symbol_info.get('category', ''),
                    subcategory=symbol_info.get('subcategory', '')
                )
                
                self.db_manager.insert_ohlc_data(ohlc_data)
                
                # Log the saved OHLC data
                logger.info(f"💾 SAVED: {data['symbol']} | "
                          f"O:{data['open']:.4f} H:{data['high']:.4f} L:{data['low']:.4f} C:{data['close']:.4f} | "
                          f"Vol:{self._format_volume(data['volume'])} | "
                          f"Ticks:{data['tick_count']} | "
                          f"Time:{data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                keys_to_remove.append(cache_key)
                saved_count += 1
        
        # Clean up old cache entries
        for key in keys_to_remove:
            del self.price_cache[key]
        
        if saved_count > 0:
            logger.info(f"📊 Saved {saved_count} OHLC records to database")
    
    def get_current_ohlc_summary(self):
        """Get current OHLC summary for all tracked symbols"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        summary_data = []
        
        for cache_key, data in self.price_cache.items():
            if data['timestamp'] == current_minute:
                symbol_info = self.subscribed_symbols.get(data['symbol'], {})
                summary_data.append({
                    'symbol': data['symbol'],
                    'category': symbol_info.get('category', ''),
                    'subcategory': symbol_info.get('subcategory', ''),
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume'],
                    'tick_count': data['tick_count']
                })
        
        return summary_data
    
    def print_ohlc_summary_table(self):
        """Print a formatted table of current OHLC data"""
        summary = self.get_current_ohlc_summary()
        
        if not summary:
            logger.info("No current OHLC data available")
            return
        
        print(f"\n{'='*100}")
        print(f"{'SYMBOL':<20} {'CATEGORY':<15} {'OPEN':<10} {'HIGH':<10} {'LOW':<10} {'CLOSE':<10} {'VOLUME':<12} {'TICKS':<6}")
        print(f"{'='*100}")
        
        # Sort by category and symbol
        summary.sort(key=lambda x: (x['category'], x['symbol']))
        
        current_category = ""
        for data in summary:
            if data['category'] != current_category:
                current_category = data['category']
                print(f"\n--- {current_category} ---")
            
            volume_str = self._format_volume(data['volume'])
            print(f"{data['symbol']:<20} {data['subcategory']:<15} "
                  f"{data['open']:<10.4f} {data['high']:<10.4f} {data['low']:<10.4f} "
                  f"{data['close']:<10.4f} {volume_str:<12} {data['tick_count']:<6}")
        
        print(f"{'='*100}")
        print(f"Total symbols tracked: {len(summary)}")
    
    def start_summary_printer(self):
        """Start a thread that prints OHLC summary every 30 seconds"""
        def print_summary_periodically():
            while self.is_connected:
                time.sleep(30)
                if self.is_connected:
                    self.print_ohlc_summary_table()
        
        summary_thread = threading.Thread(target=print_summary_periodically)
        summary_thread.daemon = True
        summary_thread.start()
        logger.info("📊 Started OHLC summary printer (every 30 seconds)")
    
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
        """Initialize the TradingView WebSocket connection"""
        self._send_message(self._create_message("set_auth_token", ["unauthorized_user_token"]))
        self._send_message(self._create_message("chart_create_session", [self.chart_session_id, ""]))
        self._send_message(self._create_message("quote_create_session", [self.quote_session_id]))
        
        quote_fields = [
            "ch", "chp", "currency_code", "current_session", "description", "exchange",
            "format", "fractional", "is_tradable", "language", "local_description",
            "listed_exchange", "logoid", "lp", "minmov", "minmove2", "original_name",
            "pricescale", "pro_name", "short_name", "type", "typespecs", "update_mode",
            "volume", "bid", "ask", "high_price", "low_price", "open_price", "prev_close_price"
        ]
        
        self._send_message(self._create_message("quote_set_fields", [self.quote_session_id] + quote_fields))
    
    def subscribe_symbols_from_category(self, category_data, category_name, market_name):
        """Subscribe to symbols from a category"""
        for subcategory, data in category_data.items():
            symbols = data['symbols']
            names = data['names']
            
            for i, symbol in enumerate(symbols):
                symbol_name = names[i] if i < len(names) else symbol
                
                # Store symbol metadata
                self.subscribed_symbols[symbol] = {
                    'category': category_name,
                    'subcategory': subcategory,
                    'name': symbol_name,
                    'market': market_name
                }
                
                # Insert symbol metadata to database
                self.db_manager.insert_symbol_metadata(
                    symbol, symbol_name, category_name, subcategory, market_name
                )
                
                # Subscribe to real-time data
                self._send_message(self._create_message("quote_add_symbols", [self.quote_session_id, symbol]))
                logger.info(f"Subscribed to {symbol} ({category_name}/{subcategory})")
                time.sleep(0.1)  # Rate limiting
    
    def connect(self):
        """Connect to TradingView WebSocket"""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://data.tradingview.com/socket.io/websocket",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.ws.run_forever()

def main():
    """Main function to run the streaming system"""
    # Initialize database
    db_manager = DatabaseManager()
    
    # Initialize WebSocket client
    tv_ws = TradingViewWebSocket(db_manager)
    
    # Schedule OHLC data saving every minute
    schedule.every().minute.do(tv_ws.save_ohlc_data_to_db)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    # Start scheduler in separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Start WebSocket connection in separate thread
    ws_thread = threading.Thread(target=tv_ws.connect)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Wait for connection
    time.sleep(3)
    
    # Subscribe to different categories
    symbol_manager = SymbolManager()
    
    # Vietnam Stocks
    logger.info("Subscribing to Vietnam stocks...")
    vietnam_stocks = symbol_manager.get_vietnam_stocks()
    tv_ws.subscribe_symbols_from_category(vietnam_stocks, "Vietnam_Stock", "HOSE")
    
    time.sleep(2)
    
    # US Stocks
    logger.info("Subscribing to US stocks...")
    us_stocks = symbol_manager.get_us_stocks()
    tv_ws.subscribe_symbols_from_category(us_stocks, "US_Stock", "US")
    
    time.sleep(2)
    
    # Crypto
    logger.info("Subscribing to cryptocurrencies...")
    crypto = symbol_manager.get_crypto()
    tv_ws.subscribe_symbols_from_category(crypto, "Crypto", "Binance")
    
    time.sleep(2)
    
    # Forex
    logger.info("Subscribing to forex pairs...")
    forex = symbol_manager.get_forex()
    tv_ws.subscribe_symbols_from_category(forex, "Forex", "FX")
    
    logger.info(f"Total symbols subscribed: {len(tv_ws.subscribed_symbols)}")
    logger.info("Starting data collection... Press Ctrl+C to stop")
    
    # Start OHLC summary printer
    tv_ws.start_summary_printer()
    
    try:
        while True:
            time.sleep(1)
            
            # Optional: Print live summary every 60 seconds
            if int(time.time()) % 60 == 0:
                logger.info(f"📈 Active symbols: {len(tv_ws.subscribed_symbols)} | "
                          f"Cache size: {len(tv_ws.price_cache)} | "
                          f"Time: {datetime.now().strftime('%H:%M:%S')}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        
        # Print final summary before shutdown
        logger.info("Final OHLC Summary:")
        tv_ws.print_ohlc_summary_table()
        
        # Save any remaining OHLC data
        tv_ws.save_ohlc_data_to_db()
        
        if tv_ws.ws:
            tv_ws.ws.close()

if __name__ == "__main__":
    main()