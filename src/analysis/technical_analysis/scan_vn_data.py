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


def get_top_capitalization(df, top_n=10) -> list:
    
    df['Date'] = pd.to_datetime(df['Date'])

    # Get the latest date per ticker
    latest_df = df.sort_values(['Date']).groupby('Ticker').tail(1)

    # Calculate capitalization = Volume * Close
    latest_df['capitalization'] = latest_df['Volume'] * latest_df['Close']

    # Get top 20 by capitalization
    top_list = latest_df.sort_values('capitalization', ascending=False).head(top_n)

    print(top_list[['Ticker', 'Date', 'Close', 'Volume', 'capitalization', 'name_vn']])
    
    list_of_tickers = top_list['Ticker'].tolist()
    
    print(list_of_tickers)

    return list_of_tickers


def get_top_avg_cap(df: pd.DataFrame, top_n=10) -> list:
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Get latest date
    latest_date = df['Date'].max()
    start_date = latest_date - pd.Timedelta(days=10)

    # Filter last 10 days (inclusive)
    df_window = df[(df['Date'] >= start_date) & (df['Date'] <= latest_date)]

    # Calculate capitalization
    df_window['capitalization'] = df_window['Volume'] * df_window['Close']

    # Compute average capitalization per ticker
    avg_cap = (
        df_window.groupby('Ticker')['capitalization']
        .mean()
        .reset_index()
    )

    # Get top 20
    top10 = avg_cap.sort_values('capitalization', ascending=False).head(top_n)
    
    top_list = top10.merge(df_info[['stock_code', 'name_vn']], left_on='Ticker', right_on='stock_code', how='left')
    top_list = top_list.merge(df_nganh[['ticker', 'industry']], left_on='Ticker', right_on='ticker', how='left')
            
    print(top_list[['Ticker', 'capitalization', 'name_vn']])
    
    list_of_tickers = top_list['Ticker'].tolist()
    
    print(list_of_tickers)

    return top_list

list_of_industry = ['personal__household_goods', 'chemicals', 'food__beverage',
       'financial_services', 'real_estate', 'banks', 'telecommunications',
       'insurance', 'industrial_goods__services', 'retail',
       'construction__materials', 'basic_resources', 'media',
       'health_care', 'utilities', 'travel__leisure', 'oil__gas',
       'technology', 'automobiles__parts']


df_bank = df_final[df_final['industry'].str.contains('banks', na=False)]
df_ins = df_final[df_final['industry'].str.contains('industrial_goods__services', na=False)]


top_10_each_industry = []

for i in list_of_industry:
    df_selected = df_final[df_final['industry'].str.contains(i, na=False)]
    ticker_selected = get_top_capitalization(df_selected, top_n=10)
    print(f"Top 10 for industry {i}: {ticker_selected}")
    top_10_each_industry.extend(ticker_selected)

from src.analysis.technical_analysis.dow_simple import DowTheoryAnalyzer

for i in top_10_each_industry:
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

  