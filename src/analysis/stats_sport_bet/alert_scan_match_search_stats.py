import requests
import json
import pandas as pd
import re
import sys
import os
import numpy as np
import sys

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

sys_path = os.getenv("sys_path")
print(sys_path)
os.chdir(sys_path)
sys.path.append(sys_path)

team_name = sys.argv[1] if len(sys.argv) > 1 else None

print(f"Team name: {team_name}")

sql =     f"""
   WITH ranked AS (
    SELECT *,
            ROW_NUMBER() OVER (PARTITION BY id ORDER BY run_time DESC) AS rn,
            now()::timestamp as current_now
    FROM "188bet_stats_first_odd"
    WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '1.5 hours'
         AND "run_time"::TIMESTAMP <= (NOW()::timestamp + INTERVAL '7 hours')
    )
    SELECT *
    FROM ranked
    WHERE 1=1
    AND rn = 1
    AND (split_part(match_name, '-', 1) LIKE '%{team_name}%' OR split_part(match_name, '-', 2) LIKE '%{team_name}%')
;
"""

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

sys_path = os.getenv("sys_path")
print(sys_path)
os.chdir(sys_path)
sys.path.append(sys_path)

from storage import duckdb_reader as dr 

df = dr.read_from_duckdb(
    db_path="log_data/188bet_stats_first_odd.duckdb",
    query = sql
)

print(f"Data fetched: {df.shape[0]} rows")    

from src.analysis.stats_sport_bet.stats_score_transition import convert_bet_odds
from src.analysis.stats_sport_bet.stats_score_transition import parse_odds_columns
from src.analysis.stats_sport_bet.stats_last_5_perf import match_stats
from src.analysis.stats_sport_bet.stats_bet_odd import extract_goal_events_with_preodds
from src.analysis.stats_sport_bet.stats_score_transition import parse_match_name
from src.analysis.stats_sport_bet.stats_first_bet_odds import get_first_bet_odds

df_parsed = df.copy()

print(f"Data fetched: {df_parsed.shape[0]} rows")

# Function to parse minute
def parse_minute(val):
    if val == "[]" or pd.isna(val):
        return np.nan  # keep track of empty
    try:
        # remove brackets and split mm:ss
        minute, _ = val.strip("[]").split(":")
        return int(minute)
    except Exception:
        return np.nan

# Create new column with parsed minute
df_parsed["minute"] = df_parsed["current_time"].apply(parse_minute)

# Filter: keep rows where minute < 90 OR is NaN (empty)
df_parsed = df_parsed[(df_parsed["minute"].isna()) | (df_parsed["minute"] < 85)]

print("Currently found match" , df_parsed.shape[0])


from hook.telegram_v2 import send_telegram_message

token="1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
chat_id = "@Victor_Trading_HL"

##### DF_UNDER
df_tele = df_parsed[['id', 'cid', 'l', 'n', 'match_name', 'score', 'match_time',
       'current_time', 'run_time', 'match_part', 'time_difference',
       'Bàn Thắng: Trên / Dưới', 'Cược Chấp', 
       'from_score', 'to_score',
       'total_for_fromscore_line', 'success_rate_fromscore',
       'matches_analyzed_home', 
       'wins_home', 'draws_home', 'goals_first_half_home', 'goals_second_half_home',
       'matches_analyzed_away',
       'wins_away', 'draws_away','goals_first_half_away', 'goals_second_half_away',
        'hh_value_first_odd', 'rate_hh_first_odd',
       'rate_ah_first_odd', 'line_value_first_odd', 'rate_over_first_odd',
       'rate_under_first_odd'   
       ]]


chunk_size = 10
df_list = [df_tele.iloc[i:i + chunk_size] for i in range(0, len(df_tele), chunk_size)]

for i in range(0, len(df_list)):
    item_tele = df_list[i]
    
    if item_tele.empty:
        print("There's nothing to alert")
    # for i in industry:
    #     print("Nganh: ", i)
    #     df_tele_f = df_tele.loc[df_tele['industry']==i]
    #     df_tele_f = df_tele_f.sort_values(by='change_price', ascending=False)
    #     df_tele_f = df_tele_f.head(5)
        pass
    else:
        send_telegram_message(item_tele, token, chat_id)