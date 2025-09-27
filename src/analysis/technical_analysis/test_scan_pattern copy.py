import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import yfinance as yf
from datetime import datetime, timedelta
import threading
import time
import warnings
from scipy.signal import argrelextrema
import queue
import json

warnings.filterwarnings('ignore')

class RealTimeTradingSystem:
    def __init__(self, symbol="AAPL", length=14, mult=1.0, 
                 risk_reward_ratio=2.0, stop_loss_pct=0.5):
        self.symbol = symbol
        self.length = length
        self.mult = mult
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct
        
        # Trading state
        self.data_queue = queue.Queue()
        self.is_running = False
        self.current_position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Pattern detection
        self.min_pattern_bars = 5
        self.breakout_confirmation_bars = 2
        
        # Data storage
        self.historical_data = pd.DataFrame()
        self.trades = []
        self.alerts = []
        
    def fetch_5min_data(self, period="5d"):
        """Fetch 5-minute historical data"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval="5m")
            if data.empty:
                raise Exception(f"No data available for {self.symbol}")
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_pivot_points(self, data):
        """Calculate pivot highs and lows for 5-minute timeframe"""
        if len(data) < self.length * 2:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data), dtype=bool)
        
        high = data['High'].values
        low = data['Low'].values
        
        # Adjusted for 5-minute sensitivity
        lookback = max(3, self.length // 3)
        
        high_peaks = argrelextrema(high, np.greater, order=lookback)[0]
        low_peaks = argrelextrema(low, np.less, order=lookback)[0]
        
        pivot_highs = np.zeros(len(high), dtype=bool)
        pivot_lows = np.zeros(len(low), dtype=bool)
        
        pivot_highs[high_peaks] = True
        pivot_lows[low_peaks] = True
        
        return pivot_highs, pivot_lows
    
    def calculate_dynamic_trendlines(self, data):
        """Calculate dynamic trendlines optimized for 5-minute trading"""
        n_bars = len(data)
        if n_bars < self.length:
            return self._empty_results(data)
        
        # Calculate ATR for 5-minute volatility
        atr = self.calculate_atr(data, period=min(14, n_bars//4))
        slope = atr / max(1, min(14, n_bars//4)) * self.mult
        
        pivot_highs, pivot_lows = self.calculate_pivot_points(data)
        
        # Initialize arrays
        upper_trendline = np.full(n_bars, np.nan)
        lower_trendline = np.full(n_bars, np.nan)
        
        # Calculate trendlines with recent bias for 5-minute trading
        last_pivot_high_idx = None
        last_pivot_low_idx = None
        last_pivot_high_price = None
        last_pivot_low_price = None
        
        for i in range(n_bars):
            if pivot_highs[i]:
                last_pivot_high_idx = i
                last_pivot_high_price = data['High'].iloc[i]
            
            if pivot_lows[i]:
                last_pivot_low_idx = i
                last_pivot_low_price = data['Low'].iloc[i]
            
            # Calculate upper trendline
            if last_pivot_high_idx is not None and last_pivot_high_price is not None:
                bars_since_high = i - last_pivot_high_idx
                slope_val = slope.iloc[i] if not pd.isna(slope.iloc[i]) else 0
                upper_trendline[i] = last_pivot_high_price - slope_val * bars_since_high
            
            # Calculate lower trendline
            if last_pivot_low_idx is not None and last_pivot_low_price is not None:
                bars_since_low = i - last_pivot_low_idx
                slope_val = slope.iloc[i] if not pd.isna(slope.iloc[i]) else 0
                lower_trendline[i] = last_pivot_low_price + slope_val * bars_since_low
        
        return self._create_results(data, upper_trendline, lower_trendline, pivot_highs, pivot_lows, slope)
    
    def detect_trading_patterns(self, data):
        """Detect specific trading patterns for 5-minute breakouts"""
        patterns = {
            'bullish_breakout': [],
            'bearish_breakdown': [],
            'bull_flag': [],
            'bear_flag': [],
            'ascending_triangle': [],
            'descending_triangle': []
        }
        
        if len(data) < 20:
            return patterns
        
        # Recent data for pattern detection
        recent_data = data.tail(20)
        
        # Bullish breakout pattern
        if self._detect_bullish_breakout(recent_data):
            patterns['bullish_breakout'].append({
                'timestamp': recent_data.index[-1],
                'price': recent_data['Close'].iloc[-1],
                'strength': self._calculate_pattern_strength(recent_data, 'bullish')
            })
        
        # Bearish breakdown pattern
        if self._detect_bearish_breakdown(recent_data):
            patterns['bearish_breakdown'].append({
                'timestamp': recent_data.index[-1],
                'price': recent_data['Close'].iloc[-1],
                'strength': self._calculate_pattern_strength(recent_data, 'bearish')
            })
        
        # Flag patterns
        bull_flag = self._detect_bull_flag(recent_data)
        if bull_flag:
            patterns['bull_flag'].append(bull_flag)
        
        bear_flag = self._detect_bear_flag(recent_data)
        if bear_flag:
            patterns['bear_flag'].append(bear_flag)
        
        return patterns
    
    def _detect_bullish_breakout(self, data):
        """Detect bullish breakout pattern"""
        if len(data) < 10:
            return False
        
        # Check for breakout above recent resistance
        recent_highs = data['High'].tail(10)
        resistance_level = recent_highs.iloc[:-2].max()
        current_price = data['Close'].iloc[-1]
        volume_surge = data['Volume'].iloc[-1] > data['Volume'].tail(5).mean() * 1.5
        
        return (current_price > resistance_level * 1.001 and  # 0.1% breakout threshold
                volume_surge and
                data['Close'].iloc[-1] > data['Close'].iloc[-2])
    
    def _detect_bearish_breakdown(self, data):
        """Detect bearish breakdown pattern"""
        if len(data) < 10:
            return False
        
        # Check for breakdown below recent support
        recent_lows = data['Low'].tail(10)
        support_level = recent_lows.iloc[:-2].min()
        current_price = data['Close'].iloc[-1]
        volume_surge = data['Volume'].iloc[-1] > data['Volume'].tail(5).mean() * 1.5
        
        return (current_price < support_level * 0.999 and  # 0.1% breakdown threshold
                volume_surge and
                data['Close'].iloc[-1] < data['Close'].iloc[-2])
    
    def _detect_bull_flag(self, data):
        """Detect bull flag pattern"""
        if len(data) < 15:
            return None
        
        # Look for strong upward move followed by consolidation
        early_data = data.iloc[:8]
        flag_data = data.iloc[8:]
        
        # Strong initial move (>2% in 5-8 bars)
        initial_move = (early_data['High'].max() - early_data['Low'].min()) / early_data['Low'].min()
        
        # Consolidation with downward bias
        flag_slope = np.polyfit(range(len(flag_data)), flag_data['Close'], 1)[0]
        
        if initial_move > 0.015 and flag_slope < 0:  # 1.5% move and negative slope
            return {
                'timestamp': data.index[-1],
                'price': data['Close'].iloc[-1],
                'pattern_type': 'bull_flag',
                'strength': min(initial_move * 100, 5.0)
            }
        return None
    
    def _detect_bear_flag(self, data):
        """Detect bear flag pattern"""
        if len(data) < 15:
            return None
        
        # Look for strong downward move followed by consolidation
        early_data = data.iloc[:8]
        flag_data = data.iloc[8:]
        
        # Strong initial move (>2% in 5-8 bars)
        initial_move = (early_data['High'].max() - early_data['Low'].min()) / early_data['High'].max()
        
        # Consolidation with upward bias
        flag_slope = np.polyfit(range(len(flag_data)), flag_data['Close'], 1)[0]
        
        if initial_move > 0.015 and flag_slope > 0:  # 1.5% move and positive slope
            return {
                'timestamp': data.index[-1],
                'price': data['Close'].iloc[-1],
                'pattern_type': 'bear_flag',
                'strength': min(initial_move * 100, 5.0)
            }
        return None
    
    def calculate_trade_levels(self, entry_price, direction):
        """Calculate stop loss and take profit levels"""
        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_pct/100)
            take_profit = entry_price * (1 + (self.stop_loss_pct/100) * self.risk_reward_ratio)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.stop_loss_pct/100)
            take_profit = entry_price * (1 - (self.stop_loss_pct/100) * self.risk_reward_ratio)
        
        return stop_loss, take_profit
    
    def generate_trade_signal(self, data, patterns):
        """Generate trading signals based on patterns"""
        signals = []
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1]
        
        # Bullish signals
        if patterns['bullish_breakout'] or patterns['bull_flag']:
            stop_loss, take_profit = self.calculate_trade_levels(current_price, 'LONG')
            signals.append({
                'type': 'LONG',
                'timestamp': current_time,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern': 'bullish_breakout' if patterns['bullish_breakout'] else 'bull_flag',
                'confidence': patterns['bullish_breakout'][0]['strength'] if patterns['bullish_breakout'] else patterns['bull_flag'][0]['strength']
            })
        
        # Bearish signals
        if patterns['bearish_breakdown'] or patterns['bear_flag']:
            stop_loss, take_profit = self.calculate_trade_levels(current_price, 'SHORT')
            signals.append({
                'type': 'SHORT',
                'timestamp': current_time,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern': 'bearish_breakdown' if patterns['bearish_breakdown'] else 'bear_flag',
                'confidence': patterns['bearish_breakdown'][0]['strength'] if patterns['bearish_breakdown'] else patterns['bear_flag'][0]['strength']
            })
        
        return signals
    
    def create_realtime_chart(self):
        """Create real-time chart with trading signals"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                           height_ratios=[3, 1, 1])
        
        def update_chart(frame):
            if self.historical_data.empty:
                return
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Get recent data for display
            display_data = self.historical_data.tail(100)
            
            # Plot candlesticks
            self._plot_5min_candlesticks(ax1, display_data)
            
            # Calculate and plot trendlines
            results = self.calculate_dynamic_trendlines(display_data)
            self._plot_trading_trendlines(ax1, results)
            
            # Detect and highlight patterns
            patterns = self.detect_trading_patterns(display_data)
            self._highlight_trading_patterns(ax1, patterns)
            
            # Generate and display signals
            signals = self.generate_trade_signal(display_data, patterns)
            self._plot_trade_signals(ax1, signals)
            
            # Plot position and levels
            self._plot_position_levels(ax1, display_data)
            
            # Volume analysis
            self._plot_volume_analysis(ax2, display_data, patterns)
            
            # Performance metrics
            self._plot_performance_metrics(ax3)
            
            # Styling
            ax1.set_title(f'{self.symbol} - 5Min Real-Time Trading System\n'
                         f'Position: {self.current_position or "NONE"} | '
                         f'Trades: {len(self.trades)} | '
                         f'Last Update: {datetime.now().strftime("%H:%M:%S")}',
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
        # Animation for real-time updates
        ani = animation.FuncAnimation(fig, update_chart, interval=5000, cache_frame_data=False)
        
        return fig, ani
    
    def _plot_5min_candlesticks(self, ax, data):
        """Plot 5-minute candlesticks optimized for trading"""
        for i, (idx, row) in enumerate(data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            alpha = 0.8
            
            # Wicks
            ax.plot([i, i], [row['Low'], row['High']], 
                   color=color, alpha=alpha, linewidth=1)
            
            # Bodies
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height,
                           facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Set x-axis labels
        if len(data) > 0:
            step = max(1, len(data) // 10)
            ticks = range(0, len(data), step)
            labels = [data.index[i].strftime('%H:%M') for i in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45)
    
    def _plot_trading_trendlines(self, ax, results):
        """Plot trendlines optimized for trading decisions"""
        if results is None:
            return
        
        data = results
        valid_upper = ~pd.isna(data['Upper_Trendline'])
        valid_lower = ~pd.isna(data['Lower_Trendline'])
        
        if valid_upper.any():
            y_values = data.loc[valid_upper, 'Upper_Trendline']
            x_values = range(len(data))
            x_filtered = [x for x, valid in zip(x_values, valid_upper) if valid]
            ax.plot(x_filtered, y_values, color='red', linewidth=2.5, 
                   linestyle='--', alpha=0.9, label='Resistance')
        
        if valid_lower.any():
            y_values = data.loc[valid_lower, 'Lower_Trendline']
            x_values = range(len(data))
            x_filtered = [x for x, valid in zip(x_values, valid_lower) if valid]
            ax.plot(x_filtered, y_values, color='green', linewidth=2.5, 
                   linestyle='--', alpha=0.9, label='Support')
    
    def _highlight_trading_patterns(self, ax, patterns):
        """Highlight detected patterns on chart"""
        # Bullish patterns - green highlight
        for pattern in patterns['bullish_breakout'] + patterns['bull_flag']:
            ax.axvspan(len(ax.get_xlim())-5, len(ax.get_xlim()), 
                      alpha=0.2, color='green', label='Bullish Pattern')
        
        # Bearish patterns - red highlight  
        for pattern in patterns['bearish_breakdown'] + patterns['bear_flag']:
            ax.axvspan(len(ax.get_xlim())-5, len(ax.get_xlim()), 
                      alpha=0.2, color='red', label='Bearish Pattern')
    
    def _plot_trade_signals(self, ax, signals):
        """Plot trade entry signals"""
        for signal in signals:
            if signal['type'] == 'LONG':
                ax.annotate('🟢 LONG', xy=(len(ax.get_xlim())-1, signal['entry_price']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8),
                           color='white', fontweight='bold', fontsize=10)
            else:
                ax.annotate('🔴 SHORT', xy=(len(ax.get_xlim())-1, signal['entry_price']),
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                           color='white', fontweight='bold', fontsize=10)
    
    def _plot_position_levels(self, ax, data):
        """Plot current position stop loss and take profit levels"""
        if self.current_position and self.entry_price > 0:
            ax.axhline(y=self.entry_price, color='blue', linestyle='-', 
                      alpha=0.7, label=f'Entry: ${self.entry_price:.2f}')
            ax.axhline(y=self.stop_loss, color='red', linestyle=':', 
                      alpha=0.7, label=f'Stop: ${self.stop_loss:.2f}')
            ax.axhline(y=self.take_profit, color='green', linestyle=':', 
                      alpha=0.7, label=f'Target: ${self.take_profit:.2f}')
    
    def _plot_volume_analysis(self, ax, data, patterns):
        """Plot volume with pattern confirmation"""
        volume_colors = ['green' if c >= o else 'red' 
                        for c, o in zip(data['Close'], data['Open'])]
        
        bars = ax.bar(range(len(data)), data['Volume'], color=volume_colors, alpha=0.7)
        
        # Highlight volume spikes during patterns
        avg_volume = data['Volume'].tail(20).mean()
        for i, vol in enumerate(data['Volume']):
            if vol > avg_volume * 1.5:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor('black')
        
        ax.set_ylabel('Volume')
        ax.set_title('Volume Analysis')
    
    def _plot_performance_metrics(self, ax):
        """Plot real-time performance metrics"""
        if not self.trades:
            ax.text(0.5, 0.5, 'No trades yet', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Performance Metrics')
            return
        
        # Calculate metrics
        profits = [trade['profit'] for trade in self.trades if 'profit' in trade]
        win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
        total_profit = sum(profits)
        
        # Display metrics
        metrics_text = f'Total P&L: ${total_profit:.2f}\n'
        metrics_text += f'Win Rate: {win_rate:.1%}\n'
        metrics_text += f'Total Trades: {len(self.trades)}'
        
        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='center', bbox=dict(boxstyle='round', alpha=0.8))
        ax.set_title('Performance Metrics')
    
    def start_real_time_monitoring(self):
        """Start real-time data monitoring and trading"""
        self.is_running = True
        
        def data_updater():
            while self.is_running:
                try:
                    # Fetch latest data
                    new_data = self.fetch_5min_data("1d")
                    if new_data is not None and not new_data.empty:
                        self.historical_data = new_data
                        
                        # Process trading logic
                        self._process_trading_logic()
                    
                    time.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    print(f"Error in data update: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start data update thread
        self.data_thread = threading.Thread(target=data_updater, daemon=True)
        self.data_thread.start()
        
        # Create and show chart
        fig, animation = self.create_realtime_chart()
        plt.tight_layout()
        plt.show()
    
    def _process_trading_logic(self):
        """Process trading logic for position management"""
        if self.historical_data.empty:
            return
        
        current_price = self.historical_data['Close'].iloc[-1]
        
        # Check for exit conditions if in position
        if self.current_position:
            self._check_exit_conditions(current_price)
        else:
            # Look for entry opportunities
            self._check_entry_conditions()
    
    def _check_entry_conditions(self):
        """Check for trade entry conditions"""
        if len(self.historical_data) < 50:
            return
        
        recent_data = self.historical_data.tail(30)
        patterns = self.detect_trading_patterns(recent_data)
        signals = self.generate_trade_signal(recent_data, patterns)
        
        for signal in signals:
            if signal['confidence'] > 3.0:  # Minimum confidence threshold
                self._enter_trade(signal)
                break
    
    def _enter_trade(self, signal):
        """Enter a new trade"""
        self.current_position = signal['type']
        self.entry_price = signal['entry_price']
        self.stop_loss = signal['stop_loss']
        self.take_profit = signal['take_profit']
        
        trade_record = {
            'entry_time': signal['timestamp'],
            'entry_price': signal['entry_price'],
            'type': signal['type'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'pattern': signal['pattern']
        }
        
        print(f"🚀 ENTERED {signal['type']} at ${signal['entry_price']:.2f}")
        print(f"   Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"   Take Profit: ${signal['take_profit']:.2f}")
        print(f"   Pattern: {signal['pattern']}")
    
    def _check_exit_conditions(self, current_price):
        """Check for trade exit conditions"""
        if not self.current_position:
            return
        
        exit_reason = None
        profit = 0
        
        if self.current_position == 'LONG':
            if current_price <= self.stop_loss:
                exit_reason = 'Stop Loss'
                profit = self.stop_loss - self.entry_price
            elif current_price >= self.take_profit:
                exit_reason = 'Take Profit'
                profit = self.take_profit - self.entry_price
        
        elif self.current_position == 'SHORT':
            if current_price >= self.stop_loss:
                exit_reason = 'Stop Loss'
                profit = self.entry_price - self.stop_loss
            elif current_price <= self.take_profit:
                exit_reason = 'Take Profit'
                profit = self.entry_price - self.take_profit
        
        if exit_reason:
            self._exit_trade(current_price, exit_reason, profit)
    
    def _exit_trade(self, exit_price, reason, profit):
        """Exit current trade"""
        trade_record = {
            'exit_time': self.historical_data.index[-1],
            'exit_price': exit_price,
            'exit_reason': reason,
            'profit': profit,
            'position_type': self.current_position
        }
        
        self.trades.append(trade_record)
        
        print(f"🏁 EXITED {self.current_position} at ${exit_price:.2f}")
        print(f"   Reason: {reason}")
        print(f"   Profit: ${profit:.2f}")
        
        # Reset position
        self.current_position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        print("🛑 Real-time monitoring stopped")
    
    # Helper methods
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low'] 
        close = data['Close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range.rolling(window=period).mean()
    
    def _empty_results(self, data):
        """Return empty results structure"""
        return pd.DataFrame(index=data.index, columns=[
            'Upper_Trendline', 'Lower_Trendline', 'Pivot_High', 'Pivot_Low'
        ])
    
    def _create_results(self, data, upper, lower, pivot_highs, pivot_lows, slope):
        """Create results dataframe"""
        results = data.copy()
        results['Upper_Trendline'] = upper
        results['Lower_Trendline'] = lower
        results['Pivot_High'] = pivot_highs
        results['Pivot_Low'] = pivot_lows
        results['Slope'] = slope
        return results
    
    def _calculate_pattern_strength(self, data, pattern_type):
        """Calculate pattern strength score"""
        if len(data) < 5:
            return 0
        
        volume_strength = data['Volume'].iloc[-1] / data['Volume'].tail(10).mean()
        price_momentum = abs((data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100)
        
        return min(volume_strength + price_momentum, 10.0)

# Demo and Usage Functions
def demo_realtime_trading():
    """Demonstrate the real-time trading system"""
    print("🚀 Starting Real-Time 5-Minute Trading System Demo")
    print("="*60)
    
    # Initialize system
    trading_system = RealTimeTradingSystem(
        symbol="AAPL",
        length=10,           # Faster for 5-minute
        mult=1.2,           # Slightly more sensitive
        risk_reward_ratio=2.0,
        stop_loss_pct=0.5   # 0.5% stop loss for 5-minute trading
    )
    
    print(f"📊 Configured for {trading_system.symbol} 5-minute trading")
    print(f"   Risk/Reward: {trading_system.risk_reward_ratio}:1")
    print(f"   Stop Loss: {trading_system.stop_loss_pct}%")
    print(f"   Trendline Length: {trading_system.length}")
    print("\n⏰ Starting real-time monitoring...")
    print("   Updates every 30 seconds")
    print("   Press Ctrl+C to stop")
    
    try:
        trading_system.start_real_time_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping trading system...")
        trading_system.stop_monitoring()

def quick_backtest(symbol="AAPL", days=5):
    """Quick backtest on recent 5-minute data"""
    print(f"🔄 Quick Backtest: {symbol} (Last {days} days)")
    print("-" * 40)
    
    system = RealTimeTradingSystem(symbol=symbol)
    data = system.fetch_5min_data(f"{days}d")
    
    if data is None or data.empty:
        print(f"❌ No data available for {symbol}")
        return
    
    print(f"📊 Analyzing {len(data)} 5-minute bars")
    
    # Simulate trading on historical data
    simulated_trades = 0
    profitable_trades = 0
    
    for i in range(50, len(data), 10):  # Check every 10 bars
        chunk = data.iloc[max(0, i-30):i+1]
        patterns = system.detect_trading_patterns(chunk)
        signals = system.generate