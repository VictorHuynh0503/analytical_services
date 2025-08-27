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


### Bong da => Trong tran dau =>
url = "https://www.binance.com/fapi/v1/ticker/24hr"


# https://www.binance.com/fapi/v1/marketKlines?symbol=iDEFI2USDT&interval=1h&limit=24
# https://www.binance.com/bapi/asset/v1/public/asset-service/product/currency
# https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products?includeEtf=true
response = requests.get(url)
print(response.status_code)
content = response.json()

df = pd.DataFrame(content)
df['run_time'] = current_date_timestamp
df['priceChangePercent'] = df['priceChangePercent'].astype('float').apply(lambda x: x * 0.01)
df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
df['closeTime'] = pd.to_datetime(df['closeTime'], unit='ms')

def categorize_change(pct):
    if pct >= 0.05:
        return "Strong Gain"
    elif 0.02 <= pct < 0.05:
        return "Gain"
    elif 0.01 <= pct < 0.02:
        return "Moderate Gain"
    elif -0.01 < pct < 0.01:
        return "Stable"
    elif -0.02 <= pct <= -0.01:
        return "Moderate Loss"
    elif -0.05 <= pct < -0.02:
        return "Loss"
    else:
        return "Strong Loss"

# Apply to your dataframe
df["priceCategory"] = df["priceChangePercent"].apply(categorize_change)


from storage import duckdb_logger as dl 

list_data = df.to_dict(orient="records")

create_path = os.getenv("log_data_path")
os.makedirs(os.path.dirname(create_path), exist_ok=True)

table_schema = dl.df_to_duckdb_schema(df)


dl.log_to_duckdb(
    db_path=f"{create_path}/binance_24h.duckdb",
    table_name="binance_24h",
    schema=table_schema,
    data=list_data,
    mode="upsert",
    upsert_keys=["symbol", "run_time"]
)