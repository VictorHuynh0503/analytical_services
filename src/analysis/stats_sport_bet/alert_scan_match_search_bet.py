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
    FROM "188bet_log"
    WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '1.5 hours'
         AND "run_time"::TIMESTAMP <= (NOW()::timestamp + INTERVAL '7 hours')
    )
    SELECT *
    FROM ranked
    WHERE rn = 1 
    AND (split_part(match_name, '-', 1) LIKE '%{team_name}%' OR split_part(match_name, '-', 2) LIKE '%{team_name}%')
;
"""

from src.analysis.stats_sport_bet.stats_score_transition import convert_bet_odds
from src.analysis.stats_sport_bet.stats_score_transition import parse_odds_columns
from src.analysis.stats_sport_bet.stats_last_5_perf import match_stats
from src.analysis.stats_sport_bet.stats_bet_odd import extract_goal_events_with_preodds
from src.analysis.stats_sport_bet.stats_score_transition import parse_match_name
from src.analysis.stats_sport_bet.stats_first_bet_odds import get_first_bet_odds


# resp = requests.post("http://165.232.188.235:8000/query/log",
#                     json={"sql": f"{sql}"})
# ##print(resp.json())

# data = resp.json()

# df = pd.DataFrame(data["rows"], columns=data["columns"])


resp = requests.post("http://165.232.188.235:8000/query/log",
                    json={"sql": f"{sql}"})
##print(resp.json())
data = resp.json()
df = pd.DataFrame(data["rows"], columns=data["columns"])
df_parsed = parse_odds_columns(df)
df_parsed['home_name'] = df_parsed['match_name'].apply(lambda x: parse_match_name(x)[0])
df_parsed['away_name'] = df_parsed['match_name'].apply(lambda x: parse_match_name(x)[1])

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

sql_stats =  f"""
SELECT * FROM "188bet_log" 
WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '8 hours'
AND "run_time"::TIMESTAMP <= (NOW()::timestamp + INTERVAL '7 hours')
AND (split_part(match_name, '-', 1) LIKE '%{team_name}%' OR split_part(match_name, '-', 2) LIKE '%{team_name}%')
"""

resp = requests.post("http://165.232.188.235:8000/query/log",
                    json={"sql": f"{sql_stats}"})
data = resp.json()
try:
    df_to_stats = pd.DataFrame(data["rows"], columns=data["columns"])
except Exception as e:
    df_to_stats = pd.DataFrame()

print(df_to_stats.shape[0])

df_first_bet = get_first_bet_odds(df_to_stats)
df_final = df_first_bet.merge(df_parsed[['id']], on='id', how='inner')

print("Currently historical match" , df_final)

from hook.telegram_v2 import send_telegram_message

token="1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
chat_id = "@vihuynh_alert"

##### DF_UNDER
df_tele = df_final[['id', 'cid', 'l', 'n', 'match_name', 'score', 'match_time',
       'current_time', 'run_time', 'match_part', 'time_difference',
       'Bàn Thắng: Trên / Dưới', 'Cược Chấp'
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