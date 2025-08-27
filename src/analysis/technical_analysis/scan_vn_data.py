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

list_of_industry = ['personal__household_goods', 'chemicals', 'food__beverage',
       'financial_services', 'real_estate', 'banks', 'telecommunications',
       'insurance', 'industrial_goods__services', 'retail',
       'construction__materials', 'basic_resources', 'media',
       'health_care', 'utilities', 'travel__leisure', 'oil__gas',
       'technology', 'automobiles__parts']


df_bank = df_final[df_final['industry'].str.contains('banks', na=False)]
df_ins = df_final[df_final['industry'].str.contains('industrial_goods__services', na=False)]

df_selected = df_final[df_final['industry'].str.contains('construction__materials', na=False)]


from src.analysis.technical_analysis.dow_simple import DowTheoryAnalyzer

for i in df_selected['Ticker'].unique()[0:10]:
    print(f"\nAnalyzing {i}...")
    df = df_selected[df_selected['Ticker'] == i].copy()
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

  