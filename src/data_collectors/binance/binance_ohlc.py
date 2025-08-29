import requests
import json
import pandas as pd
import re
import sys
import os

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

sys_path = os.getenv("sys_path")
print(sys_path)
os.chdir(sys_path)
sys.path.append(sys_path)


from tqdm import tqdm
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz


hcm_tz = pytz.timezone('Asia/Ho_Chi_Minh')

# =================== SET UP START / END TIME =========================================
current_date = datetime.now(hcm_tz) 
current_date_format = current_date.strftime('%Y-%m-%d') 
current_date_timestamp = current_date.strftime('%Y-%m-%d %H:%M:%S') 

import warnings 

warnings.filterwarnings("ignore")


def get_binance_klines(symbol, interval, limit=1000):
    base_url = "https://api.binance.com/api/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def extract_ohlc_data(klines_data):
    ohlc_data = []
    for entry in klines_data:
        open_time, open_price, high_price, low_price, close_price, *_ = entry
        ohlc_data.append({
            'Open Time': pd.to_datetime(open_time, unit='ms'),
            'Open': float(open_price),
            'High': float(high_price),
            'Low': float(low_price),
            'Close': float(close_price),
        })
    return ohlc_data

def get_ticker_coin():
    url = "https://www.binance.com/fapi/v1/ticker/24hr"
    response = requests.get(url)
    print(response.status_code)
    # print(response.json())

    content = response.json()
    df = pd.DataFrame(content)
    list_symbol = df['symbol'].unique()
    return list_symbol

log_data_path = os.getenv("log_data_path")

from storage import duckdb_reader as dr 

df = dr.read_from_duckdb(
db_path=f"{log_data_path}/binance_24h.duckdb",
query = "SELECT * FROM binance_24h WHERE priceCategory IN ('Moderate Gain', 'Strong Gain')"
#'Stable','Moderate Gain', 'Strong Gain',
)

from src.data_collectors.binance.cryto_segment import generate_ticker_symbol


selected_df = generate_ticker_symbol()
selected_list = selected_df['symbol'].unique().tolist()
# selected_list = df['symbol'].unique().tolist()

full_list = get_ticker_coin()

list_symbol = list(set(selected_list) & set(full_list))

empty_df = pd.DataFrame()

i = 0
for symbol in list_symbol[0:100]:
    print(f"Symbol is {symbol}")
    i += 1
    print("This is the ticker number: ", i)
# Example usage
    # symbol = "BTCUSDT"
    interval = "1d"  # Example interval, you can choose from "1m", "5m", "1h", "1d", etc.
    try:
        klines_data = get_binance_klines(symbol, interval)
        if klines_data:
            ohlc_data = extract_ohlc_data(klines_data)

            # Display the OHLC data as a pandas DataFrame
            df = pd.DataFrame(ohlc_data)
            df.columns = [i.lower() for i in df.columns]
            df['ticker'] = symbol
            df['timeframe'] = interval
            
            empty_df = pd.concat([empty_df, df], axis=0)
    except Exception as e:
        print(f"Error for {symbol}: {e}")
        time.sleep(1)

empty_df['run_time'] = current_date_timestamp

df = empty_df

from storage import duckdb_logger as dl 

list_data = df.to_dict(orient="records")

create_path = os.getenv("log_data_path")
os.makedirs(os.path.dirname(create_path), exist_ok=True)

table_schema = dl.df_to_duckdb_schema(df)


dl.log_to_duckdb(
    db_path=f"{create_path}/binance_ohlc.duckdb",
    table_name="binance_ohlc",
    schema=table_schema,
    data=list_data,
    mode="append"
)
