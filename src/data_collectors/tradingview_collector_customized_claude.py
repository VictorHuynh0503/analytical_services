import websocket
import json
import threading
import time
import random
import string
import sqlite3
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import schedule
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertType(Enum):
    TARGET_NEAR = "TARGET_NEAR"
    STOPLOSS_NEAR = "STOPLOSS_NEAR"
    ENTRY_DEVIATION = "ENTRY_DEVIATION"
    TARGET_HIT = "TARGET_HIT"
    STOPLOSS_HIT = "STOPLOSS_HIT"

@dataclass
class TradingPosition:
    symbol: str
    entry_price: float
    stop_loss: float
    target_price: float
    position_size: float = 0.0
    position_type: str = "LONG"  # LONG or SHORT
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    notes: str = ""
    
    def calculate_risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio"""
        if self.position_type == "LONG":
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.target_price - self.entry_price)
        else:  # SHORT
            risk = abs(self.stop_loss - self.entry_price)
            reward = abs(self.entry_price - self.target_price)
        
        return reward / risk if risk > 0 else 0
    
    def get_position_status(self, current_price: float) -> Dict:
        """Get detailed position status"""
        if self.position_type == "LONG":
            entry_deviation = ((current_price - self.entry_price) / self.entry_price) * 100
            distance_to_target = abs(current_price - self.target_price)
            distance_to_sl = abs(current_price - self.stop_loss)
            target_hit = current_price >= self.target_price
            sl_hit = current_price <= self.stop_loss
        else:  # SHORT
            entry_deviation = ((self.entry_price - current_price) / self.entry_price) * 100
            distance_to_target = abs(current_price - self.target_price)
            distance_to_sl = abs(current_price - self.stop_loss)
            target_hit = current_price <= self.target_price
            sl_hit = current_price >= self.stop_loss
        
        return {
            'entry_deviation_pct': entry_deviation,
            'distance_to_target': distance_to_target,
            'distance_to_sl': distance_to_sl,
            'target_hit': target_hit,
            'sl_hit': sl_hit,
            'unrealized_pnl_pct': entry_deviation,
            'risk_reward_ratio': self.calculate_risk_reward_ratio()
        }

@dataclass
class Alert:
    symbol: str
    alert_type: AlertType
    message: str
    current_price: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

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

class PositionManager:
    def __init__(self, db_manager):
        self.positions: Dict[str, TradingPosition] = {}
        self.alerts: List[Alert] = []
        self.db_manager = db_manager
        self.escalation_settings = {
            'entry_deviation_threshold': 5.0,  # ±5% from entry
            'target_proximity_threshold': 2.0,  # 2% away from target
            'sl_proximity_threshold': 2.0,     # 2% away from stop loss
            'escalation_cooldown': 300,        # 5 minutes between same alerts
        }
        self.last_alert_times = {}  # Track last alert time for each symbol
        self._init_positions_table()
    
    def _init_positions_table(self):
        """Initialize positions table in database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                target_price REAL NOT NULL,
                position_size REAL DEFAULT 0,
                position_type TEXT DEFAULT 'LONG',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                notes TEXT,
                UNIQUE(symbol, entry_price, created_at)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                current_price REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_position(self, symbol: str, entry_price: float, stop_loss: float, 
                    target_price: float, position_size: float = 0, 
                    position_type: str = "LONG", notes: str = "") -> bool:
        """Add a new trading position"""
        try:
            position = TradingPosition(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                position_size=position_size,
                position_type=position_type,
                notes=notes
            )
            
            # Validate position
            if position_type == "LONG":
                if not (stop_loss < entry_price < target_price):
                    logger.error(f"Invalid LONG position setup for {symbol}: SL({stop_loss}) < Entry({entry_price}) < Target({target_price})")
                    return False
            else:  # SHORT
                if not (target_price < entry_price < stop_loss):
                    logger.error(f"Invalid SHORT position setup for {symbol}: Target({target_price}) < Entry({entry_price}) < SL({stop_loss})")
                    return False
            
            self.positions[symbol] = position
            self._save_position_to_db(position)
            
            logger.info(f"✅ Added {position_type} position for {symbol}")
            logger.info(f"   Entry: {entry_price:.4f} | SL: {stop_loss:.4f} | Target: {target_price:.4f}")
            logger.info(f"   R:R Ratio: {position.calculate_risk_reward_ratio():.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding position for {symbol}: {e}")
            return False
    
    def _save_position_to_db(self, position: TradingPosition):
        """Save position to database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (symbol, entry_price, stop_loss, target_price, position_size, 
                 position_type, created_at, is_active, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.entry_price, position.stop_loss,
                position.target_price, position.position_size, position.position_type,
                position.created_at, position.is_active, position.notes
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error saving position: {e}")
        finally:
            conn.close()
    
    def remove_position(self, symbol: str) -> bool:
        """Remove a position"""
        if symbol in self.positions:
            del self.positions[symbol]
            
            # Update database
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE positions SET is_active = 0 WHERE symbol = ?', (symbol,))
            conn.commit()
            conn.close()
            
            logger.info(f"❌ Removed position for {symbol}")
            return True
        return False
    
    def update_position_alerts(self, symbol: str, current_price: float):
        """Check and generate alerts for a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        status = position.get_position_status(current_price)
        current_time = datetime.now()
        
        # Check cooldown
        last_alert_key = f"{symbol}_last_alert"
        if last_alert_key in self.last_alert_times:
            time_since_last = (current_time - self.last_alert_times[last_alert_key]).total_seconds()
            if time_since_last < self.escalation_settings['escalation_cooldown']:
                return
        
        alerts_generated = []
        
        # Check for target/stop loss hits
        if status['target_hit']:
            alert = Alert(symbol, AlertType.TARGET_HIT, 
                         f"🎯 TARGET HIT! {symbol} reached {current_price:.4f} (Target: {position.target_price:.4f})",
                         current_price)
            alerts_generated.append(alert)
            position.is_active = False  # Deactivate position
        
        elif status['sl_hit']:
            alert = Alert(symbol, AlertType.STOPLOSS_HIT,
                         f"🛑 STOP LOSS HIT! {symbol} at {current_price:.4f} (SL: {position.stop_loss:.4f})",
                         current_price)
            alerts_generated.append(alert)
            position.is_active = False  # Deactivate position
        
        else:
            # Check proximity alerts
            target_proximity_pct = (status['distance_to_target'] / position.target_price) * 100
            sl_proximity_pct = (status['distance_to_sl'] / position.stop_loss) * 100
            
            if target_proximity_pct <= self.escalation_settings['target_proximity_threshold']:
                alert = Alert(symbol, AlertType.TARGET_NEAR,
                             f"🔥 NEAR TARGET! {symbol} at {current_price:.4f}, only {status['distance_to_target']:.4f} from target {position.target_price:.4f}",
                             current_price)
                alerts_generated.append(alert)
            
            elif sl_proximity_pct <= self.escalation_settings['sl_proximity_threshold']:
                alert = Alert(symbol, AlertType.STOPLOSS_NEAR,
                             f"⚠️ NEAR STOP LOSS! {symbol} at {current_price:.4f}, only {status['distance_to_sl']:.4f} from SL {position.stop_loss:.4f}",
                             current_price)
                alerts_generated.append(alert)
            
            # Check entry deviation
            if abs(status['entry_deviation_pct']) >= self.escalation_settings['entry_deviation_threshold']:
                direction = "above" if status['entry_deviation_pct'] > 0 else "below"
                alert = Alert(symbol, AlertType.ENTRY_DEVIATION,
                             f"📊 ENTRY DEVIATION! {symbol} at {current_price:.4f} is {abs(status['entry_deviation_pct']):.2f}% {direction} entry {position.entry_price:.4f}",
                             current_price)
                alerts_generated.append(alert)
        
        # Process alerts
        for alert in alerts_generated:
            self.alerts.append(alert)
            self._save_alert_to_db(alert)
            self._print_alert(alert, status)
            self.last_alert_times[last_alert_key] = current_time
    
    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO alerts (symbol, alert_type, message, current_price, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert.symbol, alert.alert_type.value, alert.message, 
                  alert.current_price, alert.timestamp))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving alert: {e}")
        finally:
            conn.close()
    
    def _print_alert(self, alert: Alert, status: Dict):
        """Print formatted alert with additional info"""
        print(f"\n{'='*80}")
        print(f"🚨 ALERT: {alert.alert_type.value}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Message: {alert.message}")
        if alert.symbol in self.positions:
            position = self.positions[alert.symbol]
            print(f"Position: {position.position_type} | Size: {position.position_size}")
            print(f"Unrealized P&L: {status['unrealized_pnl_pct']:+.2f}%")
            print(f"R:R Ratio: {status['risk_reward_ratio']:.2f}")
        print(f"{'='*80}\n")
    
    def get_positions_summary(self) -> List[Dict]:
        """Get summary of all positions"""
        summary = []
        for symbol, position in self.positions.items():
            if position.is_active:
                summary.append({
                    'symbol': symbol,
                    'position_type': position.position_type,
                    'entry_price': position.entry_price,
                    'stop_loss': position.stop_loss,
                    'target_price': position.target_price,
                    'position_size': position.position_size,
                    'risk_reward_ratio': position.calculate_risk_reward_ratio(),
                    'created_at': position.created_at,
                    'notes': position.notes
                })
        return summary
    
    def print_positions_table(self, current_prices: Dict[str, float] = None):
        """Print formatted table of all positions"""
        if not self.positions:
            print("No active positions")
            return
        
        print(f"\n{'='*120}")
        print(f"{'SYMBOL':<15} {'TYPE':<6} {'ENTRY':<10} {'CURRENT':<10} {'SL':<10} {'TARGET':<10} {'P&L%':<8} {'R:R':<6} {'SIZE':<10} {'STATUS':<15}")
        print(f"{'='*120}")
        
        for symbol, position in self.positions.items():
            if not position.is_active:
                continue
                
            current_price = current_prices.get(symbol, 0.0) if current_prices else 0.0
            
            if current_price > 0:
                status = position.get_position_status(current_price)
                pnl_pct = status['unrealized_pnl_pct']
                
                if status['target_hit']:
                    status_text = "🎯 TARGET HIT"
                elif status['sl_hit']:
                    status_text = "🛑 SL HIT"
                elif abs(pnl_pct) >= 5:
                    status_text = f"⚠️ {pnl_pct:+.1f}%"
                else:
                    status_text = "📈 ACTIVE"
                
                pnl_color = "🟢" if pnl_pct >= 0 else "🔴"
            else:
                current_price = 0.0
                pnl_pct = 0.0
                status_text = "❓ NO DATA"
                pnl_color = "⚪"
            
            print(f"{symbol:<15} {position.position_type:<6} "
                  f"{position.entry_price:<10.4f} {current_price:<10.4f} "
                  f"{position.stop_loss:<10.4f} {position.target_price:<10.4f} "
                  f"{pnl_color}{pnl_pct:+6.2f}% {position.calculate_risk_reward_ratio():<6.2f} "
                  f"{position.position_size:<10.2f} {status_text:<15}")
        
        print(f"{'='*120}")
        print(f"Total Active Positions: {sum(1 for p in self.positions.values() if p.is_active)}")
    
    def load_positions_from_db(self):
        """Load active positions from database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, entry_price, stop_loss, target_price, position_size, 
                   position_type, created_at, notes
            FROM positions WHERE is_active = 1
        ''')
        
        rows = cursor.fetchall()
        for row in rows:
            symbol, entry_price, stop_loss, target_price, position_size, position_type, created_at, notes = row
            
            position = TradingPosition(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                position_size=position_size,
                position_type=position_type,
                created_at=datetime.fromisoformat(created_at),
                notes=notes or ""
            )
            
            self.positions[symbol] = position
        
        conn.close()
        logger.info(f"Loaded {len(self.positions)} positions from database")

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
                           "HOSE:BID", "HOSE:VNM", "HOSE:HPG", "HOSE:CTG", "HOSE:TCB"],
                "names": ["Vingroup", "Vinhomes", "Vietcombank", "Gas", "Masan Group",
                         "BIDV", "Vinamilk", "Hoa Phat", "VietinBank", "Techcombank"]
            },
            "big_cap": {
                "symbols": ["HOSE:MWG", "HOSE:PLX", "HOSE:POW", "HOSE:SAB", "HOSE:VJC",
                           "HOSE:FPT", "HOSE:MBB", "HOSE:VRE", "HOSE:ACB", "HOSE:TPB"],
                "names": ["Mobile World", "Petrolimex", "PetroVietnam Power", "Sabeco", "VietJet",
                         "FPT Corp", "Military Bank", "Vincom Retail", "ACB", "TPBank"]
            }
        }
    
    @staticmethod
    def get_us_stocks():
        return {
            "mega_cap": {
                "symbols": ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL", "NASDAQ:AMZN", "NASDAQ:NVDA"],
                "names": ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"]
            }
        }
    
    @staticmethod
    def get_crypto():
        return {
            "blue_chip": {
                "symbols": ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT"],
                "names": ["Bitcoin", "Ethereum", "Binance Coin"]
            }
        }
    
    @staticmethod
    def get_forex():
        return {
            "major_pairs": {
                "symbols": ["FX:EURUSD", "FX:GBPUSD", "FX:USDJPY"],
                "names": ["EUR/USD", "GBP/USD", "USD/JPY"]
            }
        }

class TradingViewWebSocket:
    def __init__(self, db_manager: DatabaseManager, position_manager: PositionManager):
        self.ws = None
        self.session_id = self._generate_session_id()
        self.chart_session_id = f"cs_{self._generate_session_id()}"
        self.quote_session_id = f"qs_{self._generate_session_id()}"
        self.is_connected = False
        self.subscribed_symbols = {}  # symbol -> {category, subcategory}
        self.db_manager = db_manager
        self.position_manager = position_manager
        self.price_cache = {}  # Store latest prices for OHLC calculation
        self.current_prices = {}  # Store current prices for position tracking
        
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
                            # Update current price for position tracking
                            self.current_prices[symbol] = last_price
                            
                            # Check position alerts
                            self.position_manager.update_position_alerts(symbol, last_price)
                            
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
                            
                            # Real-time OHLC display (only for symbols with positions)
                            if symbol in self.position_manager.positions:
                                symbol_info = self.subscribed_symbols[symbol]
                                ohlc_data = self.price_cache[cache_key]
                                self._display_position_update(symbol, symbol_info, ohlc_data, 
                                                            change, change_percent, current_time, last_price)
    
    def _display_position_update(self, symbol, symbol_info, ohlc_data, change, change_percent, timestamp, current_price):
        """Display position-focused price updates"""
        if symbol not in self.position_manager.positions:
            return
            
        position = self.position_manager.positions[symbol]
        status = position.get_position_status(current_price)
        
        # Color coding for price changes and position status
        if status['target_hit']:
            color = "🎯"
            status_text = "TARGET HIT"
        elif status['sl_hit']:
            color = "🛑"
            status_text = "SL HIT"
        elif change >= 0:
            color = "🟢"
            status_text = "ACTIVE"
        else:
            color = "🔴"
            status_text = "ACTIVE"
        
        change_str = f"{change:+.4f}"
        change_pct_str = f"{change_percent:+.2f}%" if change_percent != 0 else "0.00%"
        
        # Format volume with appropriate units
        volume_formatted = self._format_volume(ohlc_data['volume'])
        
        print(f"\n{color} [{timestamp.strftime('%H:%M:%S')}] {symbol} - {status_text}")
        print(f"Position: {position.position_type} | Entry: {position.entry_price:.4f} | Current: {current_price:.4f}")
        print(f"SL: {position.stop_loss:.4f} | Target: {position.target_price:.4f}")
        print(f"P&L: {status['unrealized_pnl_pct']:+.2f}% | Distance to Target: {status['distance_to_target']:.4f}")
        print(f"Volume: {volume_formatted} | Change: {change_str} ({change_pct_str})")
        print("-" * 80)
    
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
    
    def print_positions_with_prices_table(self):
        """Print positions table with current prices and alerts"""
        self.position_manager.print_positions_table(self.current_prices)
    
    def start_position_monitor(self):
        """Start monitoring positions with periodic updates"""
        def monitor_positions():
            while self.is_connected:
                time.sleep(60)  # Check every minute
                if self.is_connected and self.position_manager.positions:
                    print(f"\n📊 Position Update - {datetime.now().strftime('%H:%M:%S')}")
                    self.print_positions_with_prices_table()
        
        monitor_thread = threading.Thread(target=monitor_positions)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("📈 Started position monitor (every 60 seconds)")
    
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

def setup_sample_positions(position_manager: PositionManager):
    """Setup sample trading positions for demonstration"""
    sample_positions = [
        # Vietnam Stocks - LONG positions
        {
            'symbol': 'HOSE:VIC',
            'entry_price': 45.50,
            'stop_loss': 43.00,
            'target_price': 50.00,
            'position_size': 1000,
            'position_type': 'LONG',
            'notes': 'Breakout from consolidation'
        },
        {
            'symbol': 'HOSE:VHM',
            'entry_price': 85.20,
            'stop_loss': 82.00,
            'target_price': 92.00,
            'position_size': 500,
            'position_type': 'LONG',
            'notes': 'Support level bounce'
        },
        # US Stocks - LONG positions
        {
            'symbol': 'NASDAQ:AAPL',
            'entry_price': 185.50,
            'stop_loss': 180.00,
            'target_price': 195.00,
            'position_size': 100,
            'position_type': 'LONG',
            'notes': 'Earnings play'
        },
        {
            'symbol': 'NASDAQ:NVDA',
            'entry_price': 450.00,
            'stop_loss': 430.00,
            'target_price': 480.00,
            'position_size': 50,
            'position_type': 'LONG',
            'notes': 'AI trend continuation'
        },
        # Crypto - Mixed positions
        {
            'symbol': 'BINANCE:BTCUSDT',
            'entry_price': 42000.00,
            'stop_loss': 40000.00,
            'target_price': 46000.00,
            'position_size': 0.5,
            'position_type': 'LONG',
            'notes': 'Bull market continuation'
        },
        # Forex - SHORT position example
        {
            'symbol': 'FX:EURUSD',
            'entry_price': 1.0950,
            'stop_loss': 1.1000,
            'target_price': 1.0850,
            'position_size': 10000,
            'position_type': 'SHORT',
            'notes': 'ECB dovish sentiment'
        }
    ]
    
    print("Setting up sample positions...")
    for pos in sample_positions:
        success = position_manager.add_position(**pos)
        if success:
            print(f"✅ Added {pos['position_type']} position for {pos['symbol']}")
    
    print(f"\nTotal positions added: {len(position_manager.positions)}")

def print_menu():
    """Print interactive menu"""
    print(f"\n{'='*60}")
    print("📊 TRADING POSITION TRACKER - MENU")
    print(f"{'='*60}")
    print("1. View All Positions")
    print("2. Add New Position")
    print("3. Remove Position")
    print("4. View Recent Alerts")
    print("5. Update Escalation Settings")
    print("6. Setup Sample Positions")
    print("7. Show Current Prices")
    print("8. Export Positions to CSV")
    print("0. Continue Monitoring")
    print(f"{'='*60}")

def handle_user_input(position_manager: PositionManager, tv_ws: TradingViewWebSocket):
    """Handle user menu interactions"""
    while True:
        try:
            print_menu()
            choice = input("Enter your choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                print("\n📊 Current Positions:")
                tv_ws.print_positions_with_prices_table()
            elif choice == '2':
                add_position_interactive(position_manager)
            elif choice == '3':
                remove_position_interactive(position_manager)
            elif choice == '4':
                show_recent_alerts(position_manager)
            elif choice == '5':
                update_escalation_settings(position_manager)
            elif choice == '6':
                setup_sample_positions(position_manager)
            elif choice == '7':
                show_current_prices(tv_ws)
            elif choice == '8':
                export_positions_to_csv(position_manager)
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in menu handler: {e}")

def add_position_interactive(position_manager: PositionManager):
    """Interactive position adding"""
    try:
        print("\n📝 Add New Position")
        symbol = input("Symbol (e.g., NASDAQ:AAPL): ").strip().upper()
        position_type = input("Position Type (LONG/SHORT) [LONG]: ").strip().upper() or "LONG"
        
        entry_price = float(input("Entry Price: "))
        stop_loss = float(input("Stop Loss Price: "))
        target_price = float(input("Target Price: "))
        position_size = float(input("Position Size [0]: ") or "0")
        notes = input("Notes (optional): ").strip()
        
        success = position_manager.add_position(
            symbol, entry_price, stop_loss, target_price, 
            position_size, position_type, notes
        )
        
        if success:
            print(f"✅ Successfully added {position_type} position for {symbol}")
        else:
            print("❌ Failed to add position. Check your inputs.")
            
    except ValueError:
        print("❌ Invalid input. Please enter valid numbers for prices and size.")
    except Exception as e:
        logger.error(f"Error adding position: {e}")

def remove_position_interactive(position_manager: PositionManager):
    """Interactive position removal"""
    if not position_manager.positions:
        print("❌ No positions to remove.")
        return
    
    print("\n📋 Active Positions:")
    for i, symbol in enumerate(position_manager.positions.keys(), 1):
        position = position_manager.positions[symbol]
        print(f"{i}. {symbol} ({position.position_type})")
    
    try:
        choice = int(input("\nEnter position number to remove: "))
        symbols = list(position_manager.positions.keys())
        
        if 1 <= choice <= len(symbols):
            symbol = symbols[choice - 1]
            success = position_manager.remove_position(symbol)
            if success:
                print(f"✅ Removed position for {symbol}")
            else:
                print(f"❌ Failed to remove position for {symbol}")
        else:
            print("❌ Invalid choice.")
    except ValueError:
        print("❌ Please enter a valid number.")

def show_recent_alerts(position_manager: PositionManager):
    """Show recent alerts"""
    if not position_manager.alerts:
        print("📭 No alerts generated yet.")
        return
    
    print(f"\n🚨 Recent Alerts (Last {min(10, len(position_manager.alerts))}):")
    for alert in position_manager.alerts[-10:]:
        status = "✅" if alert.acknowledged else "🔔"
        print(f"{status} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.alert_type.value}")
        print(f"    {alert.message}")

def update_escalation_settings(position_manager: PositionManager):
    """Update escalation settings"""
    print(f"\n⚙️ Current Escalation Settings:")
    settings = position_manager.escalation_settings
    
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    print("\nPress Enter to keep current value, or enter new value:")
    
    try:
        new_entry_threshold = input(f"Entry deviation threshold ({settings['entry_deviation_threshold']}%): ")
        if new_entry_threshold.strip():
            settings['entry_deviation_threshold'] = float(new_entry_threshold)
        
        new_target_threshold = input(f"Target proximity threshold ({settings['target_proximity_threshold']}%): ")
        if new_target_threshold.strip():
            settings['target_proximity_threshold'] = float(new_target_threshold)
        
        new_sl_threshold = input(f"Stop loss proximity threshold ({settings['sl_proximity_threshold']}%): ")
        if new_sl_threshold.strip():
            settings['sl_proximity_threshold'] = float(new_sl_threshold)
        
        new_cooldown = input(f"Escalation cooldown ({settings['escalation_cooldown']} seconds): ")
        if new_cooldown.strip():
            settings['escalation_cooldown'] = int(new_cooldown)
        
        print("✅ Settings updated successfully!")
        
    except ValueError:
        print("❌ Invalid input. Settings not changed.")

def show_current_prices(tv_ws: TradingViewWebSocket):
    """Show current prices for all tracked symbols"""
    if not tv_ws.current_prices:
        print("❌ No current price data available.")
        return
    
    print(f"\n💰 Current Prices ({len(tv_ws.current_prices)} symbols):")
    for symbol, price in sorted(tv_ws.current_prices.items()):
        has_position = "📈" if symbol in tv_ws.position_manager.positions else "📊"
        print(f"{has_position} {symbol:<20} ${price:.4f}")

def export_positions_to_csv(position_manager: PositionManager):
    """Export positions to CSV file"""
    try:
        import csv
        from datetime import datetime
        
        filename = f"positions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['symbol', 'position_type', 'entry_price', 'stop_loss', 
                         'target_price', 'position_size', 'risk_reward_ratio', 
                         'created_at', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for symbol, position in position_manager.positions.items():
                writer.writerow({
                    'symbol': symbol,
                    'position_type': position.position_type,
                    'entry_price': position.entry_price,
                    'stop_loss': position.stop_loss,
                    'target_price': position.target_price,
                    'position_size': position.position_size,
                    'risk_reward_ratio': position.calculate_risk_reward_ratio(),
                    'created_at': position.created_at.isoformat(),
                    'notes': position.notes
                })
        
        print(f"✅ Positions exported to {filename}")
        
    except ImportError:
        print("❌ CSV module not available.")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")

def main():
    """Main function to run the enhanced trading system"""
    print("🚀 Starting Enhanced TradingView Position Tracker...")
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Initialize position manager
    position_manager = PositionManager(db_manager)
    
    # Load existing positions from database
    position_manager.load_positions_from_db()
    
    # Initialize WebSocket client
    tv_ws = TradingViewWebSocket(db_manager, position_manager)
    
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
    
    # Subscribe to symbols (reduced set for demo)
    logger.info("Subscribing to Vietnam stocks...")
    vietnam_stocks = symbol_manager.get_vietnam_stocks()
    tv_ws.subscribe_symbols_from_category(vietnam_stocks, "Vietnam_Stock", "HOSE")
    
    time.sleep(2)
    
    logger.info("Subscribing to US stocks...")
    us_stocks = symbol_manager.get_us_stocks()
    tv_ws.subscribe_symbols_from_category(us_stocks, "US_Stock", "US")
    
    time.sleep(2)
    
    logger.info("Subscribing to cryptocurrencies...")
    crypto = symbol_manager.get_crypto()
    tv_ws.subscribe_symbols_from_category(crypto, "Crypto", "Binance")
    
    time.sleep(2)
    
    logger.info("Subscribing to forex pairs...")
    forex = symbol_manager.get_forex()
    tv_ws.subscribe_symbols_from_category(forex, "Forex", "FX")
    
    logger.info(f"Total symbols subscribed: {len(tv_ws.subscribed_symbols)}")
    
    # Start position monitor
    tv_ws.start_position_monitor()
    
    print(f"\n🎯 Position Tracker Ready!")
    print(f"📊 Subscribed to {len(tv_ws.subscribed_symbols)} symbols")
    print(f"📈 Tracking {len(position_manager.positions)} positions")
    
    # Show initial menu
    time.sleep(2)
    
    try:
        # Interactive menu
        menu_thread = threading.Thread(target=handle_user_input, args=(position_manager, tv_ws))
        menu_thread.daemon = True
        menu_thread.start()
        
        print("System is running... Use the menu above or press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
            # Optional: Print live summary every 60 seconds
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                logger.info(f"📈 System Status: {len(tv_ws.subscribed_symbols)} symbols | "
                          f"{len(position_manager.positions)} positions | "
                          f"{len(position_manager.alerts)} alerts | "
                          f"Time: {datetime.now().strftime('%H:%M:%S')}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        
        # Print final summary before shutdown
        if position_manager.positions:
            logger.info("Final Position Summary:")
            tv_ws.print_positions_with_prices_table()
        
        # Save any remaining OHLC data
        tv_ws.save_ohlc_data_to_db()
        
        if tv_ws.ws:
            tv_ws.ws.close()

if __name__ == "__main__":
    main()