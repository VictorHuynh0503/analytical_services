import requests
import json
import pandas as pd
import re
import sys
import os

sys_path = "D:/Projects/analytical_services/"
os.chdir(sys_path)
sys.path.append(sys_path)

sql =     """
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
    WHERE rn = 1;
"""


from src.analysis.stats_sport_bet.stats_score_transition import convert_bet_odds
from src.analysis.stats_sport_bet.stats_score_transition import parse_odds_columns
from src.analysis.stats_sport_bet.stats_last_5_perf import match_stats
from src.analysis.stats_sport_bet.stats_bet_odd import extract_goal_events_with_preodds
from src.analysis.stats_sport_bet.stats_score_transition import parse_match_name

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

from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into environment

file = os.getenv("betting_stat_file")
print(file)

df_stats_hc = pd.read_excel(file, sheet_name="Handicap Stats")
df_stats_ou = pd.read_excel(file, sheet_name="OverUnder Stats")


#### Merge and filter for alerts
df_join_hc = df_parsed.merge(
    df_stats_hc,
    how="left",
    left_on=["l", "n", "score", "hh_value"],
    right_on=["country", "league", "from_score", "pre_handicap"]
    #suffixes=('', '_hc')
)

hc_condition = (
    (df_join_hc['total_for_fromscore_handicap'] >= 10) & 
    (df_join_hc['total_for_fromscore_handicap'] <= 50) &
    (df_join_hc['success_rate_frommscore'] > 0.70)
) | (
    (df_join_hc['total_for_fromscore_handicap'] >= 50) & 
    (df_join_hc['success_rate_frommscore'] >= 0.6)
)

df_alerts_hc = df_join_hc[hc_condition]
print("################### HANDICAP ALERTS ###################")
print(df_alerts_hc)


df_join_ou = df_parsed.merge(
    df_stats_ou,
    how="left",
    left_on=["l", "n", "score", "line_value"],
    right_on=["country", "league", "from_score", "pre_line"]
    #suffixes=('', '_hc')
)

ou_condition = (
    (df_join_ou['total_for_fromscore_line'] >= 10) & 
    (df_join_ou['total_for_fromscore_line'] <= 50) &
    (df_join_ou['success_rate_fromscore'] > 0.70)
) | (
    (df_join_ou['total_for_fromscore_line'] >= 50) & 
    (df_join_ou['success_rate_fromscore'] >= 0.6)
)

df_alerts_ou = df_join_ou[ou_condition]
print("################### OVER/UNDER ALERTS ###################")
print(df_alerts_ou)


sql_stats =  """
SELECT * FROM "188bet_log" 
WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '5000 hours'
AND "run_time"::TIMESTAMP <= (NOW()::timestamp - INTERVAL '7 hours')
"""

    # resp = requests.post("http://165.232.188.235:8000/query/log",
    #                     json={"sql": f"{sql}"})
    # ##print(resp.json())

    # data = resp.json()

    # df = pd.DataFrame(data["rows"], columns=data["columns"])
    
   
from storage import duckdb_reader as dr 

df_stats = dr.read_from_duckdb(
db_path="log_data/188bet_log.duckdb",
query = sql_stats
)

team_home = "Lorient"
team_away = "Rennes"

home_stats = match_stats(df_stats, team_home, last_n=5)
away_stats = match_stats(df_stats, team_away, last_n=5)

print(home_stats)
print(away_stats)

# df_join_hc = df_parsed.merge(
#     df_stats_hc,
#     how="left",
#     left_on=["l", "n", "score", "hh_value"],
#     right_on=["country", "league", "from_score", "pre_handicap"],
#     suffixes=('', '_hc')
# )


# results = []
# for team in df_parsed["team"].unique():
#     stats = match_stats(df_parsed, team, last_n=5)
#     results.append(stats)

# # Step 3: Print or process results
# for res in results:
#     print(res)