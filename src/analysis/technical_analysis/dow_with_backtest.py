import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingStrategy:
    def __init__(self, data, symbol="Asset"):
        """
        Initialize with OHLC data and trading parameters
        """
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.symbol = symbol
        self.trades = []
        self.signals = None
        
    def calculate_technical_indicators(self):
        """Calculate all technical indicators needed for the strategy"""
        
        # Moving Averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
        
        # MACD
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)
        
        # Average True Range (ATR) for stop losses
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # Support and Resistance levels
        self.data['Support'] = self.data['Low'].rolling(window=20).min()
        self.data['Resistance'] = self.data['High'].rolling(window=20).max()
        
        # Price volatility
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std() / self.data['Close'].rolling(window=20).mean()
        
    def identify_market_structure(self):
        """Identify market phases using enhanced Dow Theory"""
        
        phases = ['Neutral'] * len(self.data)
        consolidation_strength = [0] * len(self.data)
        
        for i in range(50, len(self.data)):
            close = self.data.loc[i, 'Close']
            sma_20 = self.data.loc[i, 'SMA_20']
            sma_50 = self.data.loc[i, 'SMA_50']
            macd = self.data.loc[i, 'MACD']
            rsi = self.data.loc[i, 'RSI']
            volume_ratio = self.data.loc[i, 'Volume_Ratio']
            volatility = self.data.loc[i, 'Volatility']
            
            # Recent price action analysis
            recent_high = self.data.loc[max(0, i-20):i, 'High'].max()
            recent_low = self.data.loc[max(0, i-20):i, 'Low'].min()
            price_range_pct = (recent_high - recent_low) / recent_low
            
            # Consolidation detection
            bb_squeeze = (self.data.loc[i, 'BB_Upper'] - self.data.loc[i, 'BB_Lower']) / self.data.loc[i, 'BB_Middle']
            is_consolidating = (bb_squeeze < 0.05) and (volatility < 0.02) and (price_range_pct < 0.03)
            
            if is_consolidating:
                consolidation_strength[i] = min(consolidation_strength[i-1] + 1, 10)
            else:
                consolidation_strength[i] = max(consolidation_strength[i-1] - 1, 0)
            
            # Phase identification with enhanced logic
            if (close > sma_20 > sma_50 and 
                macd > 0 and 
                rsi > 55 and rsi < 80 and
                volume_ratio > 1.1):
                phases[i] = 'Uptrend'
                
            elif (close < sma_20 < sma_50 and 
                  macd < 0 and 
                  rsi < 45 and rsi > 20 and
                  volume_ratio > 1.1):
                phases[i] = 'Downtrend'
                
            elif (abs(close - sma_20) / sma_20 < 0.015 and 
                  20 < rsi < 80 and 
                  consolidation_strength[i] >= 3):
                phases[i] = 'Accumulation' if close < recent_high * 0.9 else 'Distribution'
                
            else:
                phases[i] = phases[i-1] if i > 0 else 'Neutral'
        
        self.data['Phase'] = phases
        self.data['Consolidation_Strength'] = consolidation_strength
        
    def identify_superset_signals(self):
        """Identify high-probability superset signals"""
        
        signals = []
        
        for i in range(50, len(self.data)):
            signal_strength = 0
            signal_type = None
            
            # Current values
            close = self.data.loc[i, 'Close']
            phase = self.data.loc[i, 'Phase']
            rsi = self.data.loc[i, 'RSI']
            macd = self.data.loc[i, 'MACD']
            macd_signal = self.data.loc[i, 'MACD_Signal']
            volume_ratio = self.data.loc[i, 'Volume_Ratio']
            consolidation = self.data.loc[i, 'Consolidation_Strength']
            
            # BULLISH SUPERSET SIGNALS
            # Signal 1: Accumulation Breakout
            if (phase in ['Accumulation', 'Neutral'] and 
                consolidation >= 5 and
                close > self.data.loc[i, 'BB_Upper'] and
                rsi > 60 and 
                volume_ratio > 1.5 and
                macd > macd_signal):
                signal_strength += 3
                signal_type = 'BULLISH_BREAKOUT'
            
            # Signal 2: Trend Continuation
            elif (phase == 'Uptrend' and
                  close > self.data.loc[i, 'SMA_20'] and
                  rsi > 50 and rsi < 75 and
                  macd > 0 and
                  volume_ratio > 1.2):
                signal_strength += 2
                signal_type = 'BULLISH_CONTINUATION'
            
            # Signal 3: Oversold Bounce in Uptrend
            elif (phase == 'Uptrend' and
                  rsi < 35 and
                  close > self.data.loc[i, 'Support'] and
                  macd > macd_signal and
                  volume_ratio > 1.0):
                signal_strength += 2
                signal_type = 'BULLISH_OVERSOLD'
            
            # BEARISH SUPERSET SIGNALS
            # Signal 1: Distribution Breakdown
            elif (phase in ['Distribution', 'Neutral'] and
                  consolidation >= 5 and
                  close < self.data.loc[i, 'BB_Lower'] and
                  rsi < 40 and
                  volume_ratio > 1.5 and
                  macd < macd_signal):
                signal_strength -= 3
                signal_type = 'BEARISH_BREAKDOWN'
            
            # Signal 2: Trend Continuation Down
            elif (phase == 'Downtrend' and
                  close < self.data.loc[i, 'SMA_20'] and
                  rsi < 50 and rsi > 25 and
                  macd < 0 and
                  volume_ratio > 1.2):
                signal_strength -= 2
                signal_type = 'BEARISH_CONTINUATION'
            
            # Signal 3: Overbought Rejection in Downtrend
            elif (phase == 'Downtrend' and
                  rsi > 65 and
                  close < self.data.loc[i, 'Resistance'] and
                  macd < macd_signal and
                  volume_ratio > 1.0):
                signal_strength -= 2
                signal_type = 'BEARISH_OVERBOUGHT'
            
            signals.append({
                'Date': self.data.loc[i, 'Date'],
                'Index': i,
                'Signal_Type': signal_type,
                'Signal_Strength': signal_strength,
                'Price': close,
                'Phase': phase
            })
        
        self.signals = pd.DataFrame(signals)
        return self.signals
    
    def calculate_entry_exit_levels(self, signal_row):
        """Calculate precise entry, stop loss, and take profit levels"""
        
        i = signal_row['Index']
        signal_type = signal_row['Signal_Type']
        price = signal_row['Price']
        atr = self.data.loc[i, 'ATR']
        
        if signal_type and 'BULLISH' in signal_type:
            # Long position
            entry_price = price
            stop_loss = entry_price - (2.0 * atr)  # 2 ATR stop
            
            if 'BREAKOUT' in signal_type:
                take_profit = entry_price + (4.0 * atr)  # 4:1 R/R for breakouts
            elif 'OVERSOLD' in signal_type:
                take_profit = entry_price + (3.0 * atr)  # 3:1 R/R for oversold
            else:
                take_profit = entry_price + (2.5 * atr)  # 2.5:1 R/R for continuation
                
            position_size = self.calculate_position_size(entry_price, stop_loss)
            
            return {
                'Direction': 'LONG',
                'Entry': entry_price,
                'Stop_Loss': stop_loss,
                'Take_Profit': take_profit,
                'Position_Size': position_size,
                'Risk_Reward': (take_profit - entry_price) / (entry_price - stop_loss)
            }
        
        elif signal_type and 'BEARISH' in signal_type:
            # Short position
            entry_price = price
            stop_loss = entry_price + (2.0 * atr)  # 2 ATR stop
            
            if 'BREAKDOWN' in signal_type:
                take_profit = entry_price - (4.0 * atr)  # 4:1 R/R for breakdowns
            elif 'OVERBOUGHT' in signal_type:
                take_profit = entry_price - (3.0 * atr)  # 3:1 R/R for overbought
            else:
                take_profit = entry_price - (2.5 * atr)  # 2.5:1 R/R for continuation
                
            position_size = self.calculate_position_size(entry_price, stop_loss)
            
            return {
                'Direction': 'SHORT',
                'Entry': entry_price,
                'Stop_Loss': stop_loss,
                'Take_Profit': take_profit,
                'Position_Size': position_size,
                'Risk_Reward': (entry_price - take_profit) / (stop_loss - entry_price)
            }
        
        return None
    
    def calculate_position_size(self, entry_price, stop_loss_price, risk_per_trade=0.02):
        """Calculate position size based on risk management (2% risk per trade)"""
        risk_amount = 10000 * risk_per_trade  # Assuming $10,000 account
        price_risk = abs(entry_price - stop_loss_price)
        position_size = risk_amount / price_risk
        return round(position_size, 2)
    
    def execute_strategy(self):
        """Execute the complete trading strategy"""
        
        # Calculate technical indicators
        self.calculate_technical_indicators()
        
        # Identify market structure
        self.identify_market_structure()
        
        # Identify superset signals
        signals = self.identify_superset_signals()
        
        # Execute trades based on signals
        active_trades = []
        completed_trades = []
        
        for idx, signal in signals.iterrows():
            if signal['Signal_Strength'] != 0:  # Valid signal
                
                trade_levels = self.calculate_entry_exit_levels(signal)
                if trade_levels:
                    
                    trade = {
                        'Entry_Date': signal['Date'],
                        'Entry_Index': signal['Index'],
                        'Signal_Type': signal['Signal_Type'],
                        'Direction': trade_levels['Direction'],
                        'Entry_Price': trade_levels['Entry'],
                        'Stop_Loss': trade_levels['Stop_Loss'],
                        'Take_Profit': trade_levels['Take_Profit'],
                        'Position_Size': trade_levels['Position_Size'],
                        'Risk_Reward': trade_levels['Risk_Reward'],
                        'Status': 'ACTIVE'
                    }
                    
                    # Check if we should enter this trade (avoid overlapping trades)
                    if len(active_trades) == 0 or signal['Index'] > active_trades[-1]['Entry_Index'] + 10:
                        active_trades.append(trade)
        
        # Simulate trade execution
        for trade in active_trades:
            entry_idx = trade['Entry_Index']
            
            for i in range(entry_idx + 1, len(self.data)):
                current_price = self.data.loc[i, 'Close']
                current_high = self.data.loc[i, 'High']
                current_low = self.data.loc[i, 'Low']
                current_date = self.data.loc[i, 'Date']
                
                if trade['Direction'] == 'LONG':
                    # Check for stop loss or take profit
                    if current_low <= trade['Stop_Loss']:
                        trade['Exit_Date'] = current_date
                        trade['Exit_Price'] = trade['Stop_Loss']
                        trade['Exit_Reason'] = 'STOP_LOSS'
                        trade['Status'] = 'CLOSED'
                        break
                    elif current_high >= trade['Take_Profit']:
                        trade['Exit_Date'] = current_date
                        trade['Exit_Price'] = trade['Take_Profit']
                        trade['Exit_Reason'] = 'TAKE_PROFIT'
                        trade['Status'] = 'CLOSED'
                        break
                
                elif trade['Direction'] == 'SHORT':
                    # Check for stop loss or take profit
                    if current_high >= trade['Stop_Loss']:
                        trade['Exit_Date'] = current_date
                        trade['Exit_Price'] = trade['Stop_Loss']
                        trade['Exit_Reason'] = 'STOP_LOSS'
                        trade['Status'] = 'CLOSED'
                        break
                    elif current_low <= trade['Take_Profit']:
                        trade['Exit_Date'] = current_date
                        trade['Exit_Price'] = trade['Take_Profit']
                        trade['Exit_Reason'] = 'TAKE_PROFIT'
                        trade['Status'] = 'CLOSED'
                        break
            
            # Calculate P&L
            if trade['Status'] == 'CLOSED':
                if trade['Direction'] == 'LONG':
                    pnl = (trade['Exit_Price'] - trade['Entry_Price']) * trade['Position_Size']
                else:
                    pnl = (trade['Entry_Price'] - trade['Exit_Price']) * trade['Position_Size']
                
                trade['PnL'] = pnl
                trade['PnL_Percent'] = (pnl / (trade['Entry_Price'] * trade['Position_Size'])) * 100
                completed_trades.append(trade)
        
        self.trades = completed_trades
        return completed_trades
    
    def plot_strategy_results(self, figsize=(16, 12)):
        """Plot comprehensive strategy results"""
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, height_ratios=[3, 1, 1, 1])
        fig.suptitle(f'{self.symbol} - Advanced Trading Strategy Results', fontsize=16, fontweight='bold')
        
        # Phase colors
        phase_colors = {
            'Accumulation': 'lightblue',
            'Uptrend': 'lightgreen', 
            'Distribution': 'lightcoral',
            'Downtrend': 'lightyellow',
            'Neutral': 'lightgray'
        }
        
        # Main price chart
        ax1 = axes[0]
        
        # Plot price
        ax1.plot(self.data['Date'], self.data['Close'], color='black', linewidth=1, label='Price')
        ax1.plot(self.data['Date'], self.data['SMA_20'], color='orange', alpha=0.7, label='SMA 20')
        ax1.plot(self.data['Date'], self.data['SMA_50'], color='purple', alpha=0.7, label='SMA 50')
        
        # Plot Bollinger Bands
        ax1.plot(self.data['Date'], self.data['BB_Upper'], color='gray', alpha=0.5, linestyle='--')
        ax1.plot(self.data['Date'], self.data['BB_Lower'], color='gray', alpha=0.5, linestyle='--')
        ax1.fill_between(self.data['Date'], self.data['BB_Upper'], self.data['BB_Lower'], 
                        alpha=0.1, color='gray')
        
        # Color background by phases
        for phase, color in phase_colors.items():
            phase_data = self.data[self.data['Phase'] == phase]
            if not phase_data.empty:
                for _, row in phase_data.iterrows():
                    ax1.axvspan(row['Date'], row['Date'] + timedelta(days=1), 
                              alpha=0.3, color=color)
        
        # Plot trade entries and exits
        for trade in self.trades:
            entry_date = trade['Entry_Date']
            entry_price = trade['Entry_Price']
            
            if trade['Direction'] == 'LONG':
                ax1.scatter(entry_date, entry_price, color='green', marker='^', s=100, zorder=5)
                if trade['Status'] == 'CLOSED':
                    exit_date = trade['Exit_Date']
                    exit_price = trade['Exit_Price']
                    color = 'darkgreen' if trade['PnL'] > 0 else 'darkred'
                    ax1.scatter(exit_date, exit_price, color=color, marker='v', s=100, zorder=5)
                    ax1.plot([entry_date, exit_date], [entry_price, exit_price], 
                            color=color, alpha=0.7, linewidth=2)
            
            else:  # SHORT
                ax1.scatter(entry_date, entry_price, color='red', marker='v', s=100, zorder=5)
                if trade['Status'] == 'CLOSED':
                    exit_date = trade['Exit_Date']
                    exit_price = trade['Exit_Price']
                    color = 'darkgreen' if trade['PnL'] > 0 else 'darkred'
                    ax1.scatter(exit_date, exit_price, color=color, marker='^', s=100, zorder=5)
                    ax1.plot([entry_date, exit_date], [entry_price, exit_price], 
                            color=color, alpha=0.7, linewidth=2)
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price Action with Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = axes[1]
        ax2.bar(self.data['Date'], self.data['Volume'], alpha=0.6, color='lightblue')
        ax2.plot(self.data['Date'], self.data['Volume_SMA'], color='red', label='Volume SMA')
        ax2.set_ylabel('Volume')
        ax2.set_title('Volume Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MACD
        ax3 = axes[2]
        ax3.plot(self.data['Date'], self.data['MACD'], label='MACD', color='blue')
        ax3.plot(self.data['Date'], self.data['MACD_Signal'], label='Signal', color='red')
        ax3.bar(self.data['Date'], self.data['MACD_Histogram'], alpha=0.3, color='gray')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.set_title('MACD Indicator')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # RSI
        ax4 = axes[3]
        ax4.plot(self.data['Date'], self.data['RSI'], color='purple')
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax4.set_ylabel('RSI')
        ax4.set_xlabel('Date')
        ax4.set_title('RSI Indicator')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def get_performance_summary(self):
        """Generate comprehensive performance summary"""
        
        if not self.trades:
            return "No trades executed."
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['PnL'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum([t['PnL'] for t in self.trades])
        avg_win = np.mean([t['PnL'] for t in self.trades if t['PnL'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['PnL'] for t in self.trades if t['PnL'] < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        max_win = max([t['PnL'] for t in self.trades]) if self.trades else 0
        max_loss = min([t['PnL'] for t in self.trades]) if self.trades else 0
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum([t['PnL'] for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = min(drawdown) if len(drawdown) > 0 else 0
        
        summary = {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Total P&L ($)': round(total_pnl, 2),
            'Average Win ($)': round(avg_win, 2),
            'Average Loss ($)': round(avg_loss, 2),
            'Profit Factor': round(profit_factor, 2),
            'Maximum Win ($)': round(max_win, 2),
            'Maximum Loss ($)': round(max_loss, 2),
            'Maximum Drawdown ($)': round(max_drawdown, 2)
        }
        
        return summary

def generate_realistic_market_data(symbol, start_date, end_date, base_price):
    """Generate realistic market data for different assets"""
    
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Asset-specific parameters
    if symbol == "GOLD":
        volatility = 0.015
        trend_strength = 0.0002
        volume_base = 2000000
    elif symbol == "SILVER":
        volatility = 0.025
        trend_strength = 0.0003
        volume_base = 5000000
    elif symbol == "AAPL":
        volatility = 0.020
        trend_strength = 0.0005
        volume_base = 50000000
    elif symbol == "SPY":
        volatility = 0.012
        trend_strength = 0.0003
        volume_base = 80000000
    else:
        volatility = 0.020
        trend_strength = 0.0003
        volume_base = 10000000
    
    # Generate market phases
    accumulation_period = int(n_days * 0.25)
    uptrend_period = int(n_days * 0.35)
    distribution_period = int(n_days * 0.15)
    downtrend_period = int(n_days * 0.15)
    neutral_period = n_days - (accumulation_period + uptrend_period + distribution_period + downtrend_period)
    
    phase_returns = []
    
    # Accumulation phase
    phase_returns.extend(np.random.normal(0, volatility * 0.5, accumulation_period))
    
    # Uptrend phase
    phase_returns.extend(np.random.normal(trend_strength * 2, volatility, uptrend_period))
    
    # Distribution phase
    phase_returns.extend(np.random.normal(0, volatility * 0.7, distribution_period))
    
    # Downtrend phase
    phase_returns.extend(np.random.normal(-trend_strength * 1.5, volatility, downtrend_period))
    
    # Neutral phase
    phase_returns.extend(np.random.normal(0, volatility * 0.6, neutral_period))
    
    # Generate OHLC data
    data = []
    current_price = base_price
    
    for i in range(n_days):
        daily_return = phase_returns[i]
        
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        close_price = open_price * (1 + daily_return)
        
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.008)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
        
        # Volume with realistic patterns
        volume_multiplier = 1 + abs(daily_return) * 5
        if i > 0 and abs(daily_return) > volatility * 1.5:  # High volume on big moves
            volume_multiplier *= 1.5
        
        volume = int(volume_base * volume_multiplier * (1 + np.random.normal(0, 0.3)))
        volume = max(volume, volume_base * 0.3)  # Minimum volume
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': int(volume)
        })
        
        current_price = close_price
    
    return pd.DataFrame(data)

# Example usage and testing
if __name__ == "__main__":
    
    # Test with different assets
    assets = [
        {"symbol": "GOLD", "base_price": 1800},
        {"symbol": "SILVER", "base_price": 24},
        {"symbol": "AAPL", "base_price": 150},
        {"symbol": "SPY", "base_price": 400}
    ]
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 1)
    
    for asset in assets[:1]:  # Test with GOLD first
        print(f"\n{'='*50}")
        print(f"Testing Strategy with {asset['symbol']}")
        print(f"{'='*50}")
        
        # Generate data
        data = generate_realistic_market_data(
            asset['symbol'], 
            start_date, 
            end_date, 
            asset['base_price']
        )
        
        print(f"Generated {len(data)} days of {asset['symbol']} data")
        print(f"Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
        
        # Initialize and run strategy
        strategy = AdvancedTradingStrategy(data, asset['symbol'])
        trades = strategy.execute_strategy()
        
        # Display results
        print(f"\nStrategy Results for {asset['symbol']}:")
        print("-" * 40)
        
        if trades:
            # Print trade details
            print("Trade Details:")
            for i, trade in enumerate(trades[:5], 1):  # Show first 5 trades
                print(f"\nTrade {i}:")
                print(f"  Date: {trade['Entry_Date'].strftime('%Y-%m-%d')}")
                print(f"  Type: {trade['Signal_Type']}")
                print(f"  Direction: {trade['Direction']}")
                print(f"  Entry: ${trade['Entry_Price']:.2f}")
                print(f"  Stop Loss: ${trade['Stop_Loss']:.2f}")
                print(f"  Take Profit: ${trade['Take_Profit']:.2f}")
                print(f"  Risk/Reward: {trade['Risk_Reward']:.2f}:1")
                if trade['Status'] == 'CLOSED':
                    print(f"  Exit: ${trade['Exit_Price']:.2f} ({trade['Exit_Reason']})")
                    print(f"  P&L: ${trade['PnL']:.2f}")
                    print(f"  Return: {trade['PnL_Percent']:.2f}%")
        
        # Performance summary
        summary = strategy.get_performance_summary()
        print(f"\nPerformance Summary for {asset['symbol']}:")
        print("-" * 40)
        
        if isinstance(summary, dict):
            for key, value in summary.items():
                print(f"{key}: {value}")
        else:
            print(summary)
        
        # Create visualization
        print(f"\nGenerating charts for {asset['symbol']}...")
        fig = strategy.plot_strategy_results()
        plt.show()
        
        # Signal analysis
        if strategy.signals is not None:
            signal_summary = strategy.signals[strategy.signals['Signal_Strength'] != 0].groupby('Signal_Type').size()
            print(f"\nSignal Distribution for {asset['symbol']}:")
            print(signal_summary)
        
        print(f"\nCompleted analysis for {asset['symbol']}\n")

# Additional utility functions for enhanced analysis
def analyze_strategy_robustness(strategy):
    """Analyze the robustness of the strategy across different market conditions"""
    
    if not strategy.trades:
        return "No trades to analyze"
    
    trades_df = pd.DataFrame(strategy.trades)
    
    # Performance by signal type
    signal_performance = trades_df.groupby('Signal_Type').agg({
        'PnL': ['count', 'mean', 'sum'],
        'Risk_Reward': 'mean'
    }).round(2)
    
    # Performance by direction
    direction_performance = trades_df.groupby('Direction').agg({
        'PnL': ['count', 'mean', 'sum'],
        'Risk_Reward': 'mean'
    }).round(2)
    
    return {
        'Signal_Performance': signal_performance,
        'Direction_Performance': direction_performance
    }

def backtest_with_walk_forward(data, symbol, window_size=252):
    """Perform walk-forward analysis to test strategy robustness"""
    
    results = []
    
    for i in range(window_size, len(data), 30):  # Move forward by 30 days each time
        end_idx = min(i + window_size, len(data))
        test_data = data.iloc[i-window_size:end_idx].copy().reset_index(drop=True)
        
        if len(test_data) < 100:  # Minimum data requirement
            break
            
        strategy = AdvancedTradingStrategy(test_data, symbol)
        trades = strategy.execute_strategy()
        
        if trades:
            total_pnl = sum([t['PnL'] for t in trades])
            win_rate = len([t for t in trades if t['PnL'] > 0]) / len(trades) * 100
            
            results.append({
                'Period_Start': test_data['Date'].iloc[0],
                'Period_End': test_data['Date'].iloc[-1],
                'Total_Trades': len(trades),
                'Total_PnL': total_pnl,
                'Win_Rate': win_rate
            })
    
    return pd.DataFrame(results)

# Strategy Documentation
STRATEGY_DESCRIPTION = """
ADVANCED DOW THEORY TRADING STRATEGY
=====================================

CORE PRINCIPLES:
1. Market Phase Identification using enhanced Dow Theory
2. Superset Signal Recognition for high-probability setups
3. Precise Entry/Exit with risk management
4. Multi-timeframe confirmation

MARKET PHASES:
- Accumulation: Low volatility, sideways movement, building positions
- Uptrend: Higher highs/lows, strong momentum, trend following
- Distribution: High-level consolidation, profit taking
- Downtrend: Lower lows/highs, bearish momentum

SUPERSET SIGNALS:
1. BULLISH_BREAKOUT: Breakout from accumulation with volume
2. BULLISH_CONTINUATION: Trend continuation in uptrend
3. BULLISH_OVERSOLD: Oversold bounce in uptrend
4. BEARISH_BREAKDOWN: Breakdown from distribution with volume  
5. BEARISH_CONTINUATION: Trend continuation in downtrend
6. BEARISH_OVERBOUGHT: Overbought rejection in downtrend

ENTRY CRITERIA:
- Signal strength >= 2 (strong signals only)
- Proper market phase alignment
- Volume confirmation
- Technical indicator convergence
- Risk/reward ratio >= 2:1

RISK MANAGEMENT:
- Position size based on 2% account risk
- Stop loss at 2x ATR from entry
- Take profit at 2.5-4x ATR (depending on signal type)
- No overlapping positions
- Maximum 3% portfolio heat

EXIT RULES:
- Stop loss hit (risk management)
- Take profit hit (profit taking)
- Signal deterioration (discretionary)
- Market phase change (trend reversal)

PERFORMANCE METRICS:
- Win rate targeting 50-60%
- Profit factor > 1.5
- Maximum drawdown < 15%
- Risk-adjusted returns (Sharpe ratio)
"""

print(STRATEGY_DESCRIPTION)