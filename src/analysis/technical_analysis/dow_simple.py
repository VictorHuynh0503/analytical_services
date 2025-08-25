import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DowTheoryAnalyzer:
    def __init__(self, data):
        """
        Initialize with OHLC data
        data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.phases = None
        
    def identify_swing_points(self, window=5):
        """
        Identify swing highs and lows using rolling windows
        """
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(highs) - window):
            # Check for swing high
            if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
                swing_highs.append((i, highs[i]))
            
            # Check for swing low
            if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
                swing_lows.append((i, lows[i]))
        
        return swing_highs, swing_lows
    
    def calculate_trend_strength(self, period=20):
        """
        Calculate trend strength using moving averages and momentum
        """
        self.data['SMA_20'] = self.data['Close'].rolling(window=period).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
        
        # MACD
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume trend
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
    def apply_dow_theory(self):
        """
        Apply Dow Theory to identify market phases:
        1. Accumulation - Sideways movement with low volume
        2. Uptrend - Higher highs and higher lows with increasing volume
        3. Distribution - Sideways movement at high levels with high volume
        """
        self.calculate_trend_strength()
        swing_highs, swing_lows = self.identify_swing_points()
        
        phases = ['Neutral'] * len(self.data)
        
        for i in range(50, len(self.data)):  # Start after initial MA calculation
            close = self.data.loc[i, 'Close']
            sma_20 = self.data.loc[i, 'SMA_20']
            sma_50 = self.data.loc[i, 'SMA_50']
            macd = self.data.loc[i, 'MACD']
            rsi = self.data.loc[i, 'RSI']
            volume_ratio = self.data.loc[i, 'Volume_Ratio']
            
            # Get recent price action
            recent_highs = self.data.loc[max(0, i-20):i, 'High'].max()
            recent_lows = self.data.loc[max(0, i-20):i, 'Low'].min()
            price_range = (recent_highs - recent_lows) / recent_lows
            
            # Phase identification logic
            if (close > sma_20 > sma_50 and 
                macd > 0 and 
                rsi > 50 and 
                volume_ratio > 1.0 and
                price_range > 0.05):  # Significant price movement
                phases[i] = 'Uptrend'
                
            elif (abs(close - sma_20) / sma_20 < 0.02 and  # Price near MA
                  rsi < 40 and 
                  volume_ratio < 0.8 and
                  price_range < 0.05):  # Low volatility
                phases[i] = 'Accumulation'
                
            elif (close > sma_50 and 
                  rsi > 70 and 
                  volume_ratio > 1.2 and
                  macd < 0 and
                  price_range < 0.03):  # High prices, high volume, but weakening momentum
                phases[i] = 'Distribution'
            
            else:
                # Carry forward previous phase with some momentum
                if i > 0:
                    phases[i] = phases[i-1]
        
        self.data['Phase'] = phases
        return phases
    
    def plot_analysis(self, figsize=(15, 12)):
        """
        Create comprehensive visualization of the analysis
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, height_ratios=[3, 1, 1, 1])
        fig.suptitle('Dow Theory Market Phase Analysis - BTC/USD', fontsize=16, fontweight='bold')
        
        # Color mapping for phases
        phase_colors = {
            'Accumulation': 'blue',
            'Uptrend': 'green', 
            'Distribution': 'red',
            'Neutral': 'gray'
        }
        
        # Main price chart with phases
        ax1 = axes[0]
        
        # Plot candlesticks (simplified)
        for i in range(len(self.data)):
            date = self.data.loc[i, 'Date']
            open_price = self.data.loc[i, 'Open']
            high_price = self.data.loc[i, 'High']
            low_price = self.data.loc[i, 'Low']
            close_price = self.data.loc[i, 'Close']
            phase = self.data.loc[i, 'Phase']
            
            color = 'green' if close_price >= open_price else 'red'
            ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
            ax1.plot([date, date], [open_price, close_price], color=color, linewidth=2)
        
        # Add moving averages
        ax1.plot(self.data['Date'], self.data['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
        ax1.plot(self.data['Date'], self.data['SMA_50'], label='SMA 50', color='purple', alpha=0.7)
        
        # Color background by phases
        phase_changes = self.data['Phase'] != self.data['Phase'].shift(1)
        phase_starts = self.data[phase_changes].index
        
        for i, start_idx in enumerate(phase_starts):
            end_idx = phase_starts[i + 1] if i + 1 < len(phase_starts) else len(self.data) - 1
            phase = self.data.loc[start_idx, 'Phase']
            
            if phase in phase_colors:
                ax1.axvspan(self.data.loc[start_idx, 'Date'], 
                          self.data.loc[end_idx, 'Date'],
                          alpha=0.2, color=phase_colors[phase], label=phase if i == 0 or phase != self.data.loc[phase_starts[i-1], 'Phase'] else "")
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Price Action with Market Phases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2 = axes[1]
        bars = ax2.bar(self.data['Date'], self.data['Volume'], alpha=0.6, color='lightblue')
        ax2.plot(self.data['Date'], self.data['Volume_SMA'], color='red', label='Volume SMA 20')
        ax2.set_ylabel('Volume')
        ax2.set_title('Volume Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MACD
        ax3 = axes[2]
        ax3.plot(self.data['Date'], self.data['MACD'], label='MACD', color='blue')
        ax3.plot(self.data['Date'], self.data['MACD_Signal'], label='Signal', color='red')
        ax3.bar(self.data['Date'], self.data['MACD'] - self.data['MACD_Signal'], 
               alpha=0.3, color='gray', label='Histogram')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.set_title('MACD Indicator')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # RSI
        ax4 = axes[3]
        ax4.plot(self.data['Date'], self.data['RSI'], color='purple', label='RSI')
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax4.set_ylabel('RSI')
        ax4.set_xlabel('Date')
        ax4.set_title('RSI Indicator')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def get_phase_summary(self):
        """
        Get summary of phase distribution
        """
        phase_counts = self.data['Phase'].value_counts()
        phase_percentages = (phase_counts / len(self.data) * 100).round(2)
        
        summary = pd.DataFrame({
            'Count': phase_counts,
            'Percentage': phase_percentages
        })
        
        return summary

def generate_sample_btc_data():
    """
    Generate realistic sample BTC data for testing
    """
    np.random.seed(42)  # For reproducible results
    
    # Create date range (1 year of daily data)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    n_days = len(dates)
    
    # Generate price data with realistic patterns
    base_price = 30000
    trend = np.linspace(0, 0.8, n_days)  # Overall upward trend
    
    # Add different market phases
    accumulation_phase = np.where(np.arange(n_days) < n_days * 0.2, 
                                  np.random.normal(0, 0.01, n_days), 0)
    uptrend_phase = np.where((np.arange(n_days) >= n_days * 0.2) & 
                            (np.arange(n_days) < n_days * 0.7),
                            np.random.normal(0.002, 0.02, n_days), 0)
    distribution_phase = np.where(np.arange(n_days) >= n_days * 0.7,
                                 np.random.normal(-0.001, 0.015, n_days), 0)
    
    # Combine all phases
    returns = accumulation_phase + uptrend_phase + distribution_phase
    
    # Generate OHLC data
    prices = []
    volumes = []
    current_price = base_price
    
    for i in range(n_days):
        daily_return = returns[i] + trend[i] * 0.001
        
        # Calculate OHLC
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        close_price = open_price * (1 + daily_return)
        
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        # Volume (higher during trends)
        base_volume = 50000000
        volume_multiplier = 1 + abs(daily_return) * 10  # Higher volume on big moves
        volume = int(base_volume * volume_multiplier * (1 + np.random.normal(0, 0.3)))
        
        prices.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': max(volume, 1000000)  # Minimum volume
        })
        
        current_price = close_price
    
    return pd.DataFrame(prices)

# Example usage
if __name__ == "__main__":
    # Generate sample BTC data
    print("Generating sample BTC data...")
    btc_data = generate_sample_btc_data()
    
    print(f"Generated {len(btc_data)} days of BTC data")
    print(f"Date range: {btc_data['Date'].min()} to {btc_data['Date'].max()}")
    print(f"Price range: ${btc_data['Low'].min():,.2f} - ${btc_data['High'].max():,.2f}")
    print("\nFirst 5 rows:")
    print(btc_data.head())
    
    # Initialize analyzer
    print("\nAnalyzing market phases using Dow Theory...")
    analyzer = DowTheoryAnalyzer(btc_data)
    
    # Apply Dow Theory analysis
    phases = analyzer.apply_dow_theory()
    
    # Get phase summary
    summary = analyzer.get_phase_summary()
    print("\nMarket Phase Distribution:")
    print(summary)
    
    # Create visualization
    print("\nCreating visualization...")
    fig = analyzer.plot_analysis()
    plt.show()
    
    # Additional analysis
    print("\nDetailed Phase Analysis:")
    for phase in ['Accumulation', 'Uptrend', 'Distribution']:
        phase_data = analyzer.data[analyzer.data['Phase'] == phase]
        if len(phase_data) > 0:
            avg_return = ((phase_data['Close'] / phase_data['Open'] - 1) * 100).mean()
            avg_volume = phase_data['Volume'].mean()
            print(f"{phase}:")
            print(f"  - Days: {len(phase_data)}")
            print(f"  - Avg Daily Return: {avg_return:.2f}%")
            print(f"  - Avg Volume: {avg_volume:,.0f}")
            print(f"  - Price Range: ${phase_data['Low'].min():,.2f} - ${phase_data['High'].max():,.2f}")
            print()