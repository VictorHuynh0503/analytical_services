import requests
import json
import pandas as pd
import re
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

sys_path = os.getenv("sys_path")
print(sys_path)
os.chdir(sys_path)
sys.path.append(sys_path)

vn_data = os.getenv("vn_data")
df = pd.read_csv(vn_data)

df_info = pd.read_csv(os.getenv("info_vps"))
df_nganh = pd.read_csv(os.getenv("nganh"))
df_all = pd.read_csv(os.getenv("tat_ca_ma"))

def standardize_df(df):
    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df = df[df["Date"] >= date(2024, 1, 1)]
    # df["Datetime"] = pd.to_datetime(df["Datetime"])  # ensure datetime format
    df = df.rename(columns={
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume"
    })
    df = df['Ticker,Date,Open,High,Low,Close,Volume'.split(',')]
    df = df.merge(df_info[['stock_code', 'name_vn']], left_on='Ticker', right_on='stock_code', how='left')
    df = df.merge(df_nganh[['ticker', 'industry']], left_on='Ticker', right_on='ticker', how='left')
    
    return df 

df_final = standardize_df(df)

df_bank = df_final[df_final['industry'].str.contains('banks', na=False)]

from src.analysis.technical_analysis.dow_with_backtest import AdvancedTradingStrategy

for i in df_bank['Ticker'].unique():
    print(f"\nAnalyzing {i}...")
    df = df_bank[df_bank['Ticker'] == i].copy()
    print(df.shape)
    
    df =df['Date,Open,High,Low,Close,Volume'.split(',')]
    
    assets = [
        {"symbol": "VCB", "base_price": 70},
        {"symbol": "ACB", "base_price": 24},
        {"symbol": "AAPL", "base_price": 150},
        {"symbol": "SPY", "base_price": 400}
    ]
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 9, 1)
    
    for asset in assets[:1]:  # Test with GOLD first
        print(f"\n{'='*50}")
        print(f"Testing Strategy with {asset['symbol']}")
        print(f"{'='*50}")
        
        data = df
        
        # Generate data
        # data = generate_realistic_market_data(
        #     asset['symbol'], 
        #     start_date, 
        #     end_date, 
        #     asset['base_price']
        # )
        
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


  