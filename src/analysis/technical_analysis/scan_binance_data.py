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
log_data_path = os.getenv("log_data_path")

from storage import duckdb_reader as dr 

df = dr.read_from_duckdb(
db_path=f"{log_data_path}/binance_ohlc.duckdb",
query = """
SELECT * FROM binance_ohlc
WHERE timeframe = '1d' AND "open time" >= '2024-01-01' 
AND run_time = (SELECT MAX(run_time) FROM binance_ohlc)
"""
)


def standardize_df(df):
    df["Date"] = pd.to_datetime(df["open time"]).dt.date
    df = df[df["Date"] >= date(2024, 1, 1)]
    df['Volume'] = 0
    # df["Datetime"] = pd.to_datetime(df["Datetime"])  # ensure datetime format
    df = df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "Volume": "Volume",
    "ticker": "Ticker"
    })
    df = df['Ticker,Date,Open,High,Low,Close,Volume'.split(',')]

    
    return df 

df_final = standardize_df(df)


from src.analysis.technical_analysis.dow_simple import DowTheoryAnalyzer

for i in df_final['Ticker'].unique():
    print(f"\nAnalyzing {i}...")
    df = df_final[df_final['Ticker'] == i].copy()
    print(df.shape)
    
    df =df['Date,Open,High,Low,Close,Volume'.split(',')]
    
    analyzer = DowTheoryAnalyzer(df)

    # Apply Dow Theory analysis
    phases = analyzer.apply_dow_theory()

    # Get phase summary
    summary = analyzer.get_phase_summary()
    print("\nMarket Phase Distribution:")
    print(summary)

    # Create visualization
    print("\nCreating visualization...")
    fig = analyzer.plot_analysis(name=i)
    plt.show()

  