import requests
import json
import pandas as pd

import os
import re
import sys

abs_path = os.path.dirname(__file__)
main_cwd = re.sub('Python-Project.*','Python-Project',abs_path)
os.chdir(main_cwd)
sys.path.append(main_cwd)

from Shared_Lib import shared_function as sf1
from Shared_Lib import pcconfig
import pandas as pd
import pygsheets
from hook.pygsheets_class import pygsheets_googlesheet
from hook.chart_plotly import chart_plotly
from hook import shared_function as sf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings 

from hook.shared_function import processing_df as pdf

pcconfig = pcconfig.Init()
json_file = pcconfig['service_account']
data_daily = pcconfig['stock_data']
folder_data = pcconfig['data_folder']

warnings.filterwarnings("ignore")
fig = go.Figure()

from hook.shared_function import processing_candlestick as pc 


if __name__ == "__main__":

    
    empty_df = pd.read_parquet(folder_data + "//binance_data")

    columns = [i.capitalize() for i in empty_df.columns]

    new_df = empty_df

    new_df.columns = columns

    new_df.set_index('Open time')
    
    for i in sorted(new_df['Ticker'].unique()):
        print(f"TIcker is  {i}")
        df_to_calculate = new_df.loc[new_df['Ticker']==i]
        
        try:
            strategy_data = pc.apply_strategies(df_to_calculate)

            # Run the backtest and calculate statistics
            trade_results = pc.backtest_strategies(strategy_data)

            # Display trade results
            print(trade_results)

            # Calculate and print backtest statistics
            # pc.calculate_statistics(trade_results)
            
            pc.plot_trading_signals(strategy_data, trade_results) 
        except:
            pass 

        # df.set_index('open time')
        
        
        # df['ma10'] = sf.ma(df['close'], 10)
        # df['ma20'] = sf.ma(df['close'], 20)
        # df['ema10'] = sf.ema(df['close'], 10)
        # df['ema20'] = sf.ema(df['close'], 20)
        # df['rsi'] = sf.rsi(df['close'])
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(df.index, df['close'], label='close', marker='o')
        # plt.plot(df.index, df['ma10'], label=f'MA 10 days)', linestyle='--', marker='o')
        # plt.plot(df.index, df['ema10'], label=f'EMA 10 days)', linestyle='--', marker='o')
        # plt.plot(df.index, df['ma20'], label=f'MA 20 days)', linestyle='--', marker='o')
        # plt.plot(df.index, df['ema20'], label=f'EMA 20 days)', linestyle='--', marker='o')
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        # plt.title(f'Close Price of {symbol}')
        # plt.legend()
        # plt.show()

        # plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        # plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        # plt.plot(df.index, df['rsi'], label='RSI', linestyle='--', marker='o')
        
        # print(df)

