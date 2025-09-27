import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import yfinance as yf
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class TrendlinesIndicator:
    def __init__(self, length=14, mult=1.0, calc_method='atr', backpaint=True):
        self.length = length
        self.mult = mult
        self.calc_method = calc_method.lower()
        self.backpaint = backpaint
        
    def atr(self, data, period):
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range.rolling(window=period).mean()
    
    def stdev(self, data, period):
        """Calculate Standard Deviation"""
        return data['Close'].rolling(window=period).std()
    
    def linreg_slope(self, data, period):
        """Calculate Linear Regression based slope"""
        close = data['Close']
        n = np.arange(len(close))
        slopes = []
        
        for i in range(period, len(close)):
            y = close.iloc[i-period+1:i+1].values
            x = np.arange(len(y))
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(abs(slope))
            else:
                slopes.append(0)
        
        # Pad the beginning with NaN
        slopes = [np.nan] * period + slopes
        return pd.Series(slopes, index=close.index)
    
    def find_pivot_highs_lows(self, data):
        """Find pivot highs and lows"""
        high = data['High'].values
        low = data['Low'].values
        
        # Find local maxima and minima
        high_peaks = argrelextrema(high, np.greater, order=self.length)[0]
        low_peaks = argrelextrema(low, np.less, order=self.length)[0]
        
        # Create boolean arrays
        pivot_highs = np.zeros(len(high), dtype=bool)
        pivot_lows = np.zeros(len(low), dtype=bool)
        
        pivot_highs[high_peaks] = True
        pivot_lows[low_peaks] = True
        
        return pivot_highs, pivot_lows
    
    def calculate_slope(self, data):
        """Calculate slope based on selected method"""
        if self.calc_method == 'atr':
            slope_base = self.atr(data, self.length) / self.length
        elif self.calc_method == 'stdev':
            slope_base = self.stdev(data, self.length) / self.length
        elif self.calc_method == 'linreg':
            slope_base = self.linreg_slope(data, self.length) / 2
        else:
            raise ValueError("calc_method must be 'atr', 'stdev', or 'linreg'")
        
        return slope_base * self.mult
    
    def calculate_trendlines(self, data):
        """Main calculation function"""
        # Initialize arrays
        n_bars = len(data)
        upper = np.full(n_bars, np.nan)
        lower = np.full(n_bars, np.nan)
        slope_ph = np.full(n_bars, np.nan)
        slope_pl = np.full(n_bars, np.nan)
        
        # Find pivot points
        pivot_highs, pivot_lows = self.find_pivot_highs_lows(data)
        
        # Calculate slope
        slope = self.calculate_slope(data)
        
        # Calculate trendlines
        last_upper = np.nan
        last_lower = np.nan
        last_slope_ph = 0
        last_slope_pl = 0
        
        for i in range(n_bars):
            # Update slopes when new pivot points are found
            if pivot_highs[i]:
                last_slope_ph = slope.iloc[i] if not pd.isna(slope.iloc[i]) else last_slope_ph
                last_upper = data['High'].iloc[i]
            
            if pivot_lows[i]:
                last_slope_pl = slope.iloc[i] if not pd.isna(slope.iloc[i]) else last_slope_pl
                last_lower = data['Low'].iloc[i]
            
            # Update slope values
            slope_ph[i] = last_slope_ph
            slope_pl[i] = last_slope_pl
            
            # Calculate trendline values
            if not pd.isna(last_upper):
                if pivot_highs[i]:
                    upper[i] = last_upper
                else:
                    upper[i] = last_upper - last_slope_ph * (i - np.where(pivot_highs[:i+1])[0][-1] if np.any(pivot_highs[:i+1]) else i)
            
            if not pd.isna(last_lower):
                if pivot_lows[i]:
                    lower[i] = last_lower
                else:
                    lower[i] = last_lower + last_slope_pl * (i - np.where(pivot_lows[:i+1])[0][-1] if np.any(pivot_lows[:i+1]) else i)
        
        # Detect breakouts
        upos = np.zeros(n_bars, dtype=int)
        dnos = np.zeros(n_bars, dtype=int)
        
        for i in range(1, n_bars):
            if pivot_highs[i]:
                upos[i] = 0
            elif not pd.isna(upper[i]) and data['Close'].iloc[i] > upper[i]:
                upos[i] = 1
            else:
                upos[i] = upos[i-1]
            
            if pivot_lows[i]:
                dnos[i] = 0
            elif not pd.isna(lower[i]) and data['Close'].iloc[i] < lower[i]:
                dnos[i] = 1
            else:
                dnos[i] = dnos[i-1]
        
        # Create results DataFrame
        results = data.copy()
        results['Upper_Trendline'] = upper
        results['Lower_Trendline'] = lower
        results['Pivot_High'] = pivot_highs
        results['Pivot_Low'] = pivot_lows
        results['Upper_Break'] = (upos[1:] > upos[:-1]).tolist() + [False]
        results['Lower_Break'] = (dnos[1:] > dnos[:-1]).tolist() + [False]
        results['Slope'] = slope
        
        return results
    
    def plot_candlestick_chart(self, data, title="Trendlines with Breaks", figsize=(16, 12)):
        """Create comprehensive candlestick chart with trendlines and breakdowns"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[4, 1, 1])
        
        # Candlestick chart
        self._plot_candlesticks(ax1, data)
        
        # Plot trendlines with enhanced styling
        self._plot_trendlines(ax1, data)
        
        # Mark pivot points and breakouts
        self._plot_signals(ax1, data)
        
        # Add breakdown zones
        self._plot_breakdown_zones(ax1, data)
        
        ax1.set_title(f"{title}\nLength: {self.length} | Multiplier: {self.mult} | Method: {self.calc_method.upper()}", 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Volume subplot
        if 'Volume' in data.columns:
            self._plot_volume_with_breakouts(ax2, data)
            ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
        else:
            ax2.plot(data.index, data['Slope'], color='purple', linewidth=1.5, alpha=0.8)
            ax2.set_ylabel('Slope', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Trendline strength indicator
        self._plot_trendline_strength(ax3, data)
        ax3.set_ylabel('Trend Strength', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig, (ax1, ax2, ax3)
    
    def _plot_candlesticks(self, ax, data):
        """Plot candlestick chart"""
        up_days = data['Close'] >= data['Open']
        down_days = data['Close'] < data['Open']
        
        # Plot wicks
        ax.vlines(data.index[up_days], data.loc[up_days, 'Low'], data.loc[up_days, 'High'], 
                 colors='green', alpha=0.8, linewidth=1)
        ax.vlines(data.index[down_days], data.loc[down_days, 'Low'], data.loc[down_days, 'High'], 
                 colors='red', alpha=0.8, linewidth=1)
        
        # Plot bodies
        for i, (idx, row) in enumerate(data.iterrows()):
            if row['Close'] >= row['Open']:  # Green candle
                rect = plt.Rectangle((idx, row['Open']), 
                                   pd.Timedelta(hours=12), row['Close'] - row['Open'],
                                   facecolor='green', alpha=0.7, edgecolor='darkgreen')
            else:  # Red candle
                rect = plt.Rectangle((idx, row['Close']), 
                                   pd.Timedelta(hours=12), row['Open'] - row['Close'],
                                   facecolor='red', alpha=0.7, edgecolor='darkred')
            ax.add_patch(rect)
    
    def _plot_trendlines(self, ax, data):
        """Plot enhanced trendlines"""
        valid_upper = ~pd.isna(data['Upper_Trendline'])
        valid_lower = ~pd.isna(data['Lower_Trendline'])
        
        if valid_upper.any():
            ax.plot(data.index[valid_upper], data['Upper_Trendline'][valid_upper], 
                   color='#E74C3C', linewidth=2.5, linestyle='--', alpha=0.9, 
                   label='Resistance Trendline', zorder=5)
        
        if valid_lower.any():
            ax.plot(data.index[valid_lower], data['Lower_Trendline'][valid_lower], 
                   color='#16A085', linewidth=2.5, linestyle='--', alpha=0.9, 
                   label='Support Trendline', zorder=5)
    
    def _plot_signals(self, ax, data):
        """Plot pivot points and breakout signals"""
        # Pivot points
        pivot_highs = data[data['Pivot_High']]
        pivot_lows = data[data['Pivot_Low']]
        
        if not pivot_highs.empty:
            ax.scatter(pivot_highs.index, pivot_highs['High'], 
                      color='#E74C3C', marker='v', s=80, alpha=0.9, 
                      label='Pivot High', zorder=6, edgecolors='white', linewidth=1)
        
        if not pivot_lows.empty:
            ax.scatter(pivot_lows.index, pivot_lows['Low'], 
                      color='#16A085', marker='^', s=80, alpha=0.9, 
                      label='Pivot Low', zorder=6, edgecolors='white', linewidth=1)
        
        # Breakout markers with enhanced styling
        upper_breaks = data[data['Upper_Break']]
        lower_breaks = data[data['Lower_Break']]
        
        for idx, row in upper_breaks.iterrows():
            y_pos = row['Low'] - (row['High'] - row['Low']) * 0.02
            ax.annotate('BREAK ↑', xy=(idx, y_pos), 
                       ha='center', va='top', color='white', weight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#16A085', 
                               alpha=0.9, edgecolor='white', linewidth=1),
                       zorder=7)
        
        for idx, row in lower_breaks.iterrows():
            y_pos = row['High'] + (row['High'] - row['Low']) * 0.02
            ax.annotate('BREAK ↓', xy=(idx, y_pos), 
                       ha='center', va='bottom', color='white', weight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E74C3C', 
                               alpha=0.9, edgecolor='white', linewidth=1),
                       zorder=7)
    
    def _plot_breakdown_zones(self, ax, data):
        """Highlight breakdown/breakout zones"""
        upper_breaks = data[data['Upper_Break']]
        lower_breaks = data[data['Lower_Break']]
        
        # Highlight breakout zones with background colors
        for idx, row in upper_breaks.iterrows():
            # Find the date range for highlighting (3 days before and after)
            start_date = idx - pd.Timedelta(days=3)
            end_date = idx + pd.Timedelta(days=3)
            
            ax.axvspan(start_date, end_date, alpha=0.1, color='green', zorder=1)
        
        for idx, row in lower_breaks.iterrows():
            start_date = idx - pd.Timedelta(days=3) 
            end_date = idx + pd.Timedelta(days=3)
            
            ax.axvspan(start_date, end_date, alpha=0.1, color='red', zorder=1)
    
    def _plot_volume_with_breakouts(self, ax, data):
        """Plot volume with breakout highlighting"""
        # Base volume bars
        up_days = data['Close'] >= data['Open']
        down_days = data['Close'] < data['Open']
        
        ax.bar(data.index[up_days], data.loc[up_days, 'Volume'], 
              alpha=0.6, color='green', width=1)
        ax.bar(data.index[down_days], data.loc[down_days, 'Volume'], 
              alpha=0.6, color='red', width=1)
        
        # Highlight volume on breakout days
        upper_breaks = data[data['Upper_Break']]
        lower_breaks = data[data['Lower_Break']]
        
        if not upper_breaks.empty:
            ax.bar(upper_breaks.index, upper_breaks['Volume'], 
                  alpha=0.9, color='#16A085', width=1, 
                  label='Upward Breakout Volume')
        
        if not lower_breaks.empty:
            ax.bar(lower_breaks.index, lower_breaks['Volume'], 
                  alpha=0.9, color='#E74C3C', width=1,
                  label='Downward Breakout Volume')
        
        ax.grid(True, alpha=0.3)
        if not upper_breaks.empty or not lower_breaks.empty:
            ax.legend(loc='upper right', fontsize=9)
    
    def _plot_trendline_strength(self, ax, data):
        """Plot trendline strength indicator"""
        # Calculate distance from price to trendlines
        upper_distance = np.where(~pd.isna(data['Upper_Trendline']), 
                                 (data['Upper_Trendline'] - data['Close']) / data['Close'] * 100, 0)
        lower_distance = np.where(~pd.isna(data['Lower_Trendline']), 
                                 (data['Close'] - data['Lower_Trendline']) / data['Close'] * 100, 0)
        
        ax.fill_between(data.index, 0, upper_distance, alpha=0.3, color='red', 
                       label='Distance to Resistance')
        ax.fill_between(data.index, 0, -lower_distance, alpha=0.3, color='green', 
                       label='Distance to Support')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    def plot_chart(self, data, title="Trendlines with Breaks", figsize=(15, 10)):
        """Wrapper method for backward compatibility"""
        return self.plot_candlestick_chart(data, title, figsize)
    
    def get_detailed_breakdown_analysis(self, data):
        """Get detailed breakdown analysis with price action context"""
        upper_breaks = data[data['Upper_Break']].copy()
        lower_breaks = data[data['Lower_Break']].copy()
        
        def analyze_breakout(breaks_df, breakout_type):
            analysis = []
            for idx, row in breaks_df.iterrows():
                # Get context around breakout
                context_start = max(0, data.index.get_loc(idx) - 5)
                context_end = min(len(data), data.index.get_loc(idx) + 6)
                context_data = data.iloc[context_start:context_end]
                
                # Calculate price change after breakout
                if context_end < len(data):
                    price_change_1d = ((data.iloc[data.index.get_loc(idx) + 1]['Close'] - row['Close']) / row['Close'] * 100)
                    max_move_5d = None
                    if context_end - data.index.get_loc(idx) >= 5:
                        future_data = data.iloc[data.index.get_loc(idx):data.index.get_loc(idx) + 5]
                        if breakout_type == 'upper':
                            max_move_5d = ((future_data['High'].max() - row['Close']) / row['Close'] * 100)
                        else:
                            max_move_5d = ((row['Close'] - future_data['Low'].min()) / row['Close'] * 100)
                else:
                    price_change_1d = 0
                    max_move_5d = 0
                
                analysis.append({
                    'Date': idx,
                    'Price': row['Close'],
                    'Volume': row.get('Volume', 0),
                    'Breakout_Type': breakout_type,
                    'Price_Change_1D_%': round(price_change_1d, 2) if price_change_1d else 0,
                    'Max_Move_5D_%': round(max_move_5d, 2) if max_move_5d else 0,
                    'Trendline_Value': row['Upper_Trendline'] if breakout_type == 'upper' else row['Lower_Trendline'],
                    'Gap_%': round(((row['Close'] - (row['Upper_Trendline'] if breakout_type == 'upper' else row['Lower_Trendline'])) / row['Close'] * 100), 2)
                })
            return analysis
        
        upper_analysis = analyze_breakout(upper_breaks, 'upper')
        lower_analysis = analyze_breakout(lower_breaks, 'lower')
        
        all_breakouts = upper_analysis + lower_analysis
        all_breakouts.sort(key=lambda x: x['Date'])
        
        return {
            'All_Breakouts': pd.DataFrame(all_breakouts),
            'Upper_Breakouts': pd.DataFrame(upper_analysis),
            'Lower_Breakouts': pd.DataFrame(lower_analysis),
            'Summary': {
                'Total_Breakouts': len(all_breakouts),
                'Upper_Breakouts': len(upper_analysis),
                'Lower_Breakouts': len(lower_analysis),
                'Avg_Volume_on_Breakouts': np.mean([b['Volume'] for b in all_breakouts if b['Volume'] > 0]),
                'Success_Rate_Upper': len([b for b in upper_analysis if b['Max_Move_5D_%'] > 1]) / len(upper_analysis) if upper_analysis else 0,
                'Success_Rate_Lower': len([b for b in lower_analysis if b['Max_Move_5D_%'] > 1]) / len(lower_analysis) if lower_analysis else 0
            }
        }

# Example usage and demonstration
def demo_trendlines_indicator():
    """Demonstrate the enhanced trendlines indicator with candlestick charts"""
    print("🚀 Enhanced Trendlines with Breakdowns Demo")
    print("="*50)
    
    # Fetch sample data
    try:
        print("📊 Fetching real market data...")
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="3mo", interval="1d")
        print(f"✅ Data fetched: {len(data)} trading days for AAPL")
    except Exception as e:
        print(f"⚠️  Error fetching data: {e}")
        print("📈 Using generated sample data...")
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        np.random.seed(42)
        price = 150 + np.cumsum(np.random.randn(90) * 2)
        volatility = np.abs(np.random.randn(90) * 2)
        data = pd.DataFrame({
            'Open': price + np.random.randn(90) * 1,
            'High': price + volatility,
            'Low': price - volatility,
            'Close': price,
            'Volume': np.random.randint(10000000, 100000000, 90)
        }, index=dates)
    
    # Test different configurations
    configs = [
        {
            'length': 10, 'mult': 1.2, 'calc_method': 'atr', 
            'title': 'Short-term ATR (Sensitive)', 'color': '🟢'
        },
        {
            'length': 20, 'mult': 0.8, 'calc_method': 'stdev', 
            'title': 'Medium-term StdDev (Balanced)', 'color': '🟡'
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{config['color']} Configuration {i}: {config['title']}")
        print("-" * 45)
        
        # Initialize indicator
        indicator = TrendlinesIndicator(
            length=config['length'],
            mult=config['mult'],
            calc_method=config['calc_method']
        )
        
        # Calculate trendlines
        results = indicator.calculate_trendlines(data)
        
        # Get detailed breakdown analysis
        breakdown_analysis = indicator.get_detailed_breakdown_analysis(results)
        
        # Display summary statistics
        summary = breakdown_analysis['Summary']
        print(f"📈 Total Breakouts: {summary['Total_Breakouts']}")
        print(f"⬆️  Upper Breakouts: {summary['Upper_Breakouts']} (Success Rate: {summary['Success_Rate_Upper']:.1%})")
        print(f"⬇️  Lower Breakouts: {summary['Lower_Breakouts']} (Success Rate: {summary['Success_Rate_Lower']:.1%})")
        if summary['Avg_Volume_on_Breakouts'] > 0:
            print(f"📊 Avg Volume on Breakouts: {summary['Avg_Volume_on_Breakouts']:,.0f}")
        
        # Show breakout details
        if not breakdown_analysis['All_Breakouts'].empty:
            print(f"\n🎯 Recent Breakout Details:")
            recent_breakouts = breakdown_analysis['All_Breakouts'].tail(3)
            for _, breakout in recent_breakouts.iterrows():
                direction = "↗️ UPWARD" if breakout['Breakout_Type'] == 'upper' else "↘️ DOWNWARD"
                print(f"  {direction} on {breakout['Date'].strftime('%Y-%m-%d')}: "
                      f"${breakout['Price']:.2f} → {breakout['Price_Change_1D_%']:+.1f}% (1D), "
                      f"Max: {breakout['Max_Move_5D_%']:+.1f}% (5D)")
        
        # Create enhanced candlestick chart
        print(f"📊 Generating enhanced candlestick chart...")
        fig, axes = indicator.plot_candlestick_chart(
            results, 
            title=f"Enhanced Trendlines Analysis - {config['title']}"
        )
        
        plt.show()
        
        # Show detailed breakout table
        if not breakdown_analysis['All_Breakouts'].empty:
            print(f"\n📋 Detailed Breakout Analysis:")
            display_columns = ['Date', 'Breakout_Type', 'Price', 'Gap_%', 'Price_Change_1D_%', 'Max_Move_5D_%']
            detailed_table = breakdown_analysis['All_Breakouts'][display_columns].round(2)
            detailed_table['Date'] = detailed_table['Date'].dt.strftime('%Y-%m-%d')
            print(detailed_table.to_string(index=False))
    
    print(f"\n🎉 Analysis Complete!")
    print("💡 Tips:")
    print("   • Green zones = Upward breakout areas")
    print("   • Red zones = Downward breakdown areas") 
    print("   • Volume spikes confirm breakout strength")
    print("   • Watch for failed breakouts (quick reversals)")

def analyze_specific_stock(symbol="AAPL", period="6mo"):
    """Analyze a specific stock with the enhanced trendlines indicator"""
    print(f"🔍 Analyzing {symbol} for the last {period}")
    print("="*40)
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1d")
        
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return
            
        print(f"📊 Data loaded: {len(data)} trading days")
        
        # Use optimized settings for the analysis
        indicator = TrendlinesIndicator(length=14, mult=1.0, calc_method='atr')
        results = indicator.calculate_trendlines(data)
        breakdown_analysis = indicator.get_detailed_breakdown_analysis(results)
        
        # Create the chart
        fig, axes = indicator.plot_candlestick_chart(
            results, 
            title=f"{symbol} - Trendline Breakdown Analysis ({period})"
        )
        plt.show()
        
        # Print analysis
        summary = breakdown_analysis['Summary']
        print(f"\n📈 {symbol} Analysis Summary:")
        print(f"Total Breakouts: {summary['Total_Breakouts']}")
        print(f"Success Rate: {((summary['Success_Rate_Upper'] + summary['Success_Rate_Lower'])/2):.1%}")
        
        return breakdown_analysis
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Run the enhanced demo
    demo_trendlines_indicator()
    
    # Example: Analyze a specific stock
    # print("\n" + "="*60)
    # breakdown_data = analyze_specific_stock("TSLA", "3mo")
    
    # Example: Quick analysis function
    def quick_analysis(symbol, days=90):
        """Quick analysis function for any symbol"""
        try:
            data = yf.Ticker(symbol).history(period=f"{days}d")
            indicator = TrendlinesIndicator(length=14, mult=1.0, calc_method='atr')
            results = indicator.calculate_trendlines(data)
            fig, axes = indicator.plot_candlestick_chart(results, 
                                                       title=f"{symbol} - Quick Trendline Analysis")
            plt.show()
            return indicator.get_detailed_breakdown_analysis(results)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    # Uncomment to analyze other stocks:
    # quick_analysis("MSFT")
    # quick_analysis("GOOGL")
    # quick_analysis("NVDA")