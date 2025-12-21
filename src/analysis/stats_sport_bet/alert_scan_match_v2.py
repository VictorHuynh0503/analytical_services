import requests
import json
import pandas as pd
import re
import sys
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

sys_path = os.getenv("sys_path")
print(sys_path)
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
from src.analysis.stats_sport_bet.stats_first_bet_odds import get_first_bet_odds
from src.analysis.stats_sport_bet.stats_score_records import build_score_timeline_df

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

def minute_to_range(minute):
    """
    Convert a minute value (int or float) into a 10-minute range label like '0-10', '10-20', ..., '90-100'.
    """
    if pd.isna(minute):  # handle NaN safely
        return None

    # Clamp negative or >100 values if needed
    if minute < 0:
        return "<0"
    elif minute >= 100:
        return "90-100"

    # Compute range start and end
    start = int(minute // 10) * 10
    end = start + 10
    return f"{start}-{end}"
# Create new column with parsed minute
df_parsed["minute"] = df_parsed["current_time"].apply(parse_minute)
df_parsed["minute_interval"] = df_parsed["minute"].apply(minute_to_range)

# Filter: keep rows where minute < 90 OR is NaN (empty)
df_parsed = df_parsed[(df_parsed["minute"].isna()) | (df_parsed["minute"] < 85)]



from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into environment

file = os.getenv("betting_stat_file")
print(file)

df_stats_hc = pd.read_excel(file, sheet_name="Handicap Stats")
df_stats_ou = pd.read_excel(file, sheet_name="OverUnder Stats")

file_agg = os.getenv("betting_agg_stats")
print(file_agg)

df_stats_agg = pd.read_csv(file_agg, encoding="latin1")

def group_df(df_stats_agg, group_cols = ["country", "league", "score", "hh_value"], agg_col="hh_result"):
        # 1. Count each ou_result inside each group
# 1. Sum event_count for each ou_result inside each group
    ou_counts = (
        df_stats_agg.pivot_table(
            index=group_cols,
            columns=agg_col,
            values="event_count",   # <-- use event_count instead of counting rows
            aggfunc="sum",
            fill_value=0
        )
    )
    df_stats_agg = df_stats_agg.drop(columns=["minute_interval", "line_value",  "ou_result", "hh_result", "event_count"]).drop_duplicates()
    df_stats_agg = df_stats_agg.merge(
        ou_counts,
        on=["country", "league", "score", "hh_value"],
        how="left"
    )
    df_stats_agg = df_stats_agg.fillna(0)
    return df_stats_agg

def group_df_ou(df_stats_agg, group_cols = ["country", "league", "score", "line_value"], agg_col="ou_result"):
        # 1. Count each ou_result inside each group
# 1. Sum event_count for each ou_result inside each group
    ou_counts = (
        df_stats_agg.pivot_table(
            index=group_cols,
            columns=agg_col,
            values="event_count",   # <-- use event_count instead of counting rows
            aggfunc="sum",
            fill_value=0
        )
    )
    df_stats_agg = df_stats_agg.drop(columns=["minute_interval", "hh_value",  "ou_result", "hh_result", "event_count"]).drop_duplicates()
    df_stats_agg = df_stats_agg.merge(
        ou_counts,
        on=["country", "league", "score", "line_value"],
        how="left"
    )
    df_stats_agg = df_stats_agg.fillna(0)
    return df_stats_agg

df_grouped_ou = group_df_ou(df_stats_agg)
df_grouped_hh = group_df(df_stats_agg)


# fill NA counts with 0
#### Merge and filter for alerts
# df_join_hc = df_parsed.merge(
#     df_stats_hc,
#     how="left",
#     left_on=["l", "n", "score", "hh_value"],
#     right_on=["country", "league", "from_score", "pre_handicap"]
#     #suffixes=('', '_hc')
# )

df_join_ou = df_parsed.merge(
    df_grouped_ou,
    how="left",
    left_on=["l", "n", "score", "line_value"],
    right_on=["country", "league", "score", "line_value"],
    suffixes=('', '_ou')
)

df_join_ou = df_join_ou.merge(
    df_grouped_hh,
    how="left",
    left_on=["l", "n", "score", "hh_value"],
    right_on=["country", "league", "score", "hh_value"],
    suffixes=('', '_hh')
)

ou_condition = (
    (df_join_ou['score'].isin(['1-0', '0-1', '1-1', '2-1', '1-2', '2-3', '3-2'])) &
    (df_join_ou['line_value'].isin(['1.50', '1.75', '2.50', '2.75', '3.50', '3.75', '4.50', '4.75'])) & 
    # (df_join_ou['success_rate_fromscore'] >= 0.3) &
    (df_join_ou['rate_over'].astype(float) >= 0.88)
)| (
    (df_join_ou['score'].isin(['0-0','1-0', '0-1', '1-1', '2-0', '0-2', '2-1', '1-2', '2-3', '3-2', '2-2', '1-3', '3-1', '4-1', '1-4', '3-3', '0-3', '3-0'])) 
       
    
    # (df_join_ou['line_value'].isin(['1.50', '1.75', '2.50', '2.75', '3.50', '3.75', '4.50', '4.75'])) 
    # (df_join_ou['success_rate_fromscore'] >= 0.3) &
    # (df_join_ou['rate_over'].astype(float) >= 0.88)
)

df_alerts_ou = df_join_ou[ou_condition]
print("################### OVER/UNDER ALERTS ###################")
print(df_alerts_ou)


df_stats = df_parsed

all_team = df_stats['home_name'].tolist() + df_stats['away_name'].tolist()

sql_stats =  f"""
SELECT * FROM "188bet_log" 
WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '2200 hours'
AND "run_time"::TIMESTAMP <= (NOW()::timestamp - INTERVAL '1.5 hours')
AND (split_part(match_name, '-', 1) IN {all_team} OR split_part(match_name, '-', 2) IN {all_team})
"""

resp = requests.post("http://165.232.188.235:8000/query/log",
                    json={"sql": f"{sql_stats}"})
data = resp.json()
try:
    df_to_stats = pd.DataFrame(data["rows"], columns=data["columns"])
except Exception as e:
    df_to_stats = pd.DataFrame()

data_match_stats = []
# data_extract_goal_events = []

for i in all_team:
    print(f"\nAnalyzing {i}...")
    try:
        stats = match_stats(df_to_stats, i, last_n=5)
        data_match_stats.append(stats)
    except Exception as e:
        print(f"Error processing team {i}: {e}")
        continue
    # print(f"Results is {stats}")
 

df1 = pd.DataFrame(data_match_stats)

df_alerts_ou = df_alerts_ou.merge(df1, how='left', left_on='home_name', right_on='team')
df_alerts_ou = df_alerts_ou.merge(df1, how='left', left_on='away_name', right_on='team', suffixes=("_home", "_away"))


sql_first_bet =  f"""
SELECT * FROM "188bet_log" 
WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '8 hours'
AND "run_time"::TIMESTAMP <= (NOW()::timestamp + INTERVAL '7 hours')
AND (split_part(match_name, '-', 1) IN {all_team} OR split_part(match_name, '-', 2) IN {all_team})
"""
resp = requests.post("http://165.232.188.235:8000/query/log",
                    json={"sql": f"{sql_first_bet}"})
data = resp.json()
try:
    df_to_stats = pd.DataFrame(data["rows"], columns=data["columns"])
except Exception as e:
    df_to_stats = pd.DataFrame()

print(df_to_stats.shape[0])

df_first_bet = get_first_bet_odds(df_to_stats)
df_first_bet_parsed = parse_odds_columns(df_first_bet)
    

    # Create new column with parsed minute
df_to_stats["minute"] = df_to_stats["current_time"].apply(parse_minute)

df_score_records = build_score_timeline_df(df_to_stats)

df_alerts_ou = df_alerts_ou.merge(df_first_bet_parsed[['id', 'hh_value', 'rate_hh', 'rate_ah', 'line_value', 'rate_over', 'rate_under']], on='id', how='inner', suffixes=('', '_first_odd'))
df_alerts_ou = df_alerts_ou.merge(df_score_records[['id', 'score_timeline']], on='id', how='left', suffixes=('', '_score_record'))


# Define conditions with labels
conditions = [
    ( #### Goals > 1.5 second half
        (df_alerts_ou['goals_second_half_home'] >= df_alerts_ou['matches_analyzed_home'] * 0.8) &
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3'])) &
        (df_alerts_ou['goals_second_half_away'] < df_alerts_ou['matches_analyzed_away'] * 0.6),  
        "BET Handicap HOME - Home second half goals"
    ),
    ( #### Goals > 1.5 second half
        (df_alerts_ou['goals_second_half_away'] >= df_alerts_ou['matches_analyzed_away'] * 0.8) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0'])) &
        (df_alerts_ou['goals_second_half_home'] < df_alerts_ou['matches_analyzed_home'] * 0.6),  
        "BET Handicap AWAY - Away second half goals"
    ),
    # ( #### Goals > 1.5 second half
    #     (df_alerts_ou['goals_second_half_away'] < df_alerts_ou['matches_analyzed_away'] * 0.6) &
    #     (df_alerts_ou['matches_analyzed_away'] >= 3) &
    #     (df_alerts_ou['minute'] >= 45) &
    #     (df_alerts_ou['score'].isin(['1-0', '1-1', '0-1'])) &
    #     (df_alerts_ou['goals_second_half_home'] < df_alerts_ou['matches_analyzed_home'] * 0.6),  
    #     "Bet Under"
    # ),
    ( #### Underperformance
        (df_alerts_ou['losses_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) &
        (df_alerts_ou['goals_second_half_home'] + df_alerts_ou['goals_first_half_home'] <= df_alerts_ou['matches_analyzed_home'] * 0.7) &
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['wins_away'] + df_alerts_ou['draws_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0']))      
        ,
        "BET Handicap AWAY - Home underperform"
    ),
    (
        (df_alerts_ou['losses_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
        (df_alerts_ou['goals_second_half_away'] + df_alerts_ou['goals_first_half_away'] <= df_alerts_ou['matches_analyzed_away'] * 0.7) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['wins_home'] + df_alerts_ou['draws_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) &
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3']))
        ,
        "BET Handicap HOME - Away underperform"
    ),
    ( #### Home + Away win over 80%
        (df_alerts_ou['wins_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) & 
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['losses_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3']))        
        ,
        "BET Handicap HOME - Home win over 80%"
    ),
    (
        (df_alerts_ou['wins_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['losses_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) & 
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0']))         
        ,
        "BET Handicap AWAY - Away win over 80%"
    ),
    ( #### Home win + draw over 80%
        (df_alerts_ou['wins_home'] + df_alerts_ou['draws_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) &
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['losses_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3']))           
        ,
        "BET Handicap HOME - Home win+draw over 80%"
    ),
    (
        (df_alerts_ou['wins_away'] + df_alerts_ou['draws_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['losses_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) & 
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['minute'] >= 45) &
        (df_alerts_ou['score'].isin(['1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0']))         
        ,
        "BET Handicap AWAY - Away win+draw over 80%"
    ),
    ( #### Goals > 1.5 second half
        (df_alerts_ou['goals_second_half_home'] >= df_alerts_ou['matches_analyzed_home'] * 1.5) &
        (df_alerts_ou['matches_analyzed_home'] >= 3) &
        (df_alerts_ou['losses_away'] >= df_alerts_ou['matches_analyzed_away'] - 2) &
        (df_alerts_ou['minute'] >= 65) 
        ,  
        "Home second half avg goals > 1.5"
    ),
    (
        (df_alerts_ou['goals_second_half_away'] >= df_alerts_ou['matches_analyzed_away'] * 1.5) &
        (df_alerts_ou['matches_analyzed_away'] >= 3) &
        (df_alerts_ou['losses_home'] >= df_alerts_ou['matches_analyzed_home'] - 2) &
        (df_alerts_ou['minute'] >= 65), 
        "Away second half avg goals > 1.5"
    ),
    
    ( #### Goals > 1.5 second half
        (df_alerts_ou['line_value'].isin([3, 3.25, 3.75, 4])) &
        (df_alerts_ou['line_value_first_odd'].isin([2.25, 3.25])) &
        (
            (df_alerts_ou['score'].isin(['1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0'])
            & df_alerts_ou['hh_value_first_odd'].isin([-0.5, -0.75, -1, -1.25])
            ) |
            (df_alerts_ou['score'].isin(['0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3'])          
            & df_alerts_ou['hh_value_first_odd'].isin([0.5, 0.75, 1, 1.25])
            ) 
        ) &
        (df_alerts_ou['minute'] >= 25),  
        "BET Under - Handicap Over attain target"
    ),
    (
        (df_alerts_ou['line_value'].isin([1.75, 2, 2.25])) &
        (df_alerts_ou['line_value_first_odd'].isin([2.25, 2.75, 3.25])) &
        (
            (df_alerts_ou['score'].isin(['1-0', '1-1'])
            & df_alerts_ou['hh_value_first_odd'].isin([-0.5, -0.75, -1, -1.25])
            ) |
            (df_alerts_ou['score'].isin(['0-1', '1-1'])          
            & df_alerts_ou['hh_value_first_odd'].isin([0.5, 0.75, 1, 1.25])
            ) 
        )   &
        (df_alerts_ou['minute'] >= 60),  
        "BET Under - Until Second Half Time has goal"
    ),
    (
        (df_alerts_ou['line_value_first_odd'].isin([2, 2.25, 2.5, 2.75, 3, 3.25, 3.5])) &
        (
            (df_alerts_ou['score'].isin(['0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3'])
            & df_alerts_ou['hh_value_first_odd'].isin([0.25, 0, -0.25, -0.5, -1, -1.25]) 
            & df_alerts_ou['rate_hh_first_odd'].astype(float) > df_alerts_ou['rate_ah_first_odd'].astype(float)
            ) |
            (df_alerts_ou['score'].isin(['1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0'])          
            & df_alerts_ou['hh_value_first_odd'].isin([-0.25, 0, 0.25, 0.5, 1, 1.25])
            & df_alerts_ou['rate_hh_first_odd'].astype(float) < df_alerts_ou['rate_ah_first_odd'].astype(float)
            ) 
        )   &
        (df_alerts_ou['minute'] >= 50),  
        "BET Team Find Draw Match"
    ), 
    (
        (df_alerts_ou['line_value_first_odd'].isin([2, 2.25, 2.5, 2.75, 3, 3.25, 3.5])) &
        (
            (df_alerts_ou['score'].isin(['0-0', '0-1', '1-1', '0-2', '1-2', '2-3', '2-2', '1-3', '1-4', '3-3', '0-3'])
            & df_alerts_ou['hh_value_first_odd'].isin([0.25, 0, -0.25, -0.5, -1, -1.25, -1.5]) 
            & df_alerts_ou['rate_hh_first_odd'].astype(float) > df_alerts_ou['rate_ah_first_odd'].astype(float) * 1.05
            ) |
            (df_alerts_ou['score'].isin(['0-0', '1-0', '1-1', '2-0', '2-1', '3-2', '2-2', '3-1', '4-1', '3-3', '3-0'])          
            & df_alerts_ou['hh_value_first_odd'].isin([-0.25, 0, 0.25, 0.5, 1, 1.25, 1.5])
            & df_alerts_ou['rate_hh_first_odd'].astype(float) < df_alerts_ou['rate_ah_first_odd'].astype(float) * 0.95
            ) 
        )   &
        (df_alerts_ou['minute'] >= 50),  
        "BET Team With Higher Rate First Odd"
    ),
    (    
        (df_alerts_ou['minute'] >= 5) & 
        (df_alerts_ou['minute'] <=85) &
        (
            (df_alerts_ou['score'].isin(['0-1', '0-2', '1-2', '2-3', '1-3', '1-4', '0-3', '0-4', '2-4', '0-5']))  |  (df_alerts_ou['score'].isin(['1-0', '2-0', '2-1', '3-2', '3-1', '4-1', '3-0','4-0', '4-2', '5-0']))          
        ),
        "ALERT - AFTER 10 MINS - NOT DRAW"
    ),
    (    
        (df_alerts_ou['minute'] >=60) & 
        (df_alerts_ou['minute'] <=85) &
        (df_alerts_ou['score'].isin(['1-1', '0-0', '2-2', '3-3', '4-4', '5-5'])),         
        "ALERT - AFTER 60 MINS - DRAW"
    ),
    (    
        (df_alerts_ou['minute'] >= 5) & 
        (df_alerts_ou['minute'] <=85) &
        (
            (df_alerts_ou['score'].isin(['0-2', '1-3', '2-4'])) & df_alerts_ou['hh_value'].isin([0.25, 0, -0.25]) |  

            (df_alerts_ou['score'].isin(['2-0', '3-1', '4-2'])) & df_alerts_ou['hh_value'].isin([0.25, 0, -0.25])
        ),
        "ALERT - AFTER 10 MINS - TWO GOAL DIFFERENCE"
    ),
    (    
        (df_alerts_ou['minute'] >= 5) & 
        (df_alerts_ou['minute'] <=85) &
        (
            (df_alerts_ou['score'].isin(['0-3', '1-4', '2-5'])) & df_alerts_ou['hh_value'].isin([-0.25]) |  

            (df_alerts_ou['score'].isin(['3-0', '4-1', '5-2'])) & df_alerts_ou['hh_value'].isin([0.25])
        ),
        "ALERT - AFTER 10 MINS - THREE GOAL DIFFERENCE"
    )
    
    # ( #### Goals > 1.5 first half
    #     (df_alerts_ou['goals_first_half_home'] >= df_alerts_ou['matches_analyzed_home'] * 1.3) &
    #     (df_alerts_ou['matches_analyzed_home'] >= 3) &
    #     (df_alerts_ou['losses_away'] >= df_alerts_ou['matches_analyzed_away'] - 2),  
    #     "Home first half avg goals > 1.3"
    # ),
    # (
    #     (df_alerts_ou['goals_first_half_away'] >= df_alerts_ou['matches_analyzed_away'] * 1.3) &
    #     (df_alerts_ou['matches_analyzed_away'] >= 3) &
    #     (df_alerts_ou['losses_home'] >= df_alerts_ou['matches_analyzed_home'] - 2),
    #     "Away first half avg goals > 1.3"
    # )
    # (
    #     (df_alerts_ou['goals_second_half_away'] >= df_alerts_ou['matches_analyzed_away'] * 1.5) &
    #     (df_alerts_ou['matches_analyzed_away'] >= 3) &
    #     (df_alerts_ou['hh_value'].isin(['-0.5', '0.5', '-0.75', '0.75'])) &
    #     (df_alerts_ou['line_value'].isin(['1.50', '1.75', '3.50', '3.75', '4.50', '4.75'])),
    #     "Bet Over Away"
    # ),
    # (
    #     (df_alerts_ou['goals_second_half_home'] >= df_alerts_ou['matches_analyzed_home'] * 1.5) &
    #     (df_alerts_ou['matches_analyzed_home'] >= 3) &
    #     (df_alerts_ou['hh_value'].isin(['-0.5', '0.5', '-0.75', '0.75'])) &
    #     (df_alerts_ou['line_value'].isin(['1.50', '1.75', '3.50', '3.75', '4.50', '4.75'])),
    #     "Bet Over Home"
    # ),
]

# Start with empty comment column
df_alerts_ou["comment"] = np.nan

# Apply conditions one by one and add the comment
for cond, label in conditions:
    df_alerts_ou.loc[cond, "comment"] = label

# Finally, filter only rows that match at least one condition
df_alerts_ou = df_alerts_ou[df_alerts_ou["comment"].notna()]

# stats_condition = ( #### Home win over 80%
#     (df_alerts_ou['wins_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) & 
#     (df_alerts_ou['matches_analyzed_home'] >=3) 
# ) | (
#     (df_alerts_ou['wins_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
#     (df_alerts_ou['matches_analyzed_away'] >=3) 
# ) | ( #### Home win + draw over 80%
#     (df_alerts_ou['wins_home'] + df_alerts_ou['draws_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) &
#     (df_alerts_ou['matches_analyzed_home'] >=3) 
# ) | (
#     (df_alerts_ou['wins_away'] + df_alerts_ou['draws_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
#     (df_alerts_ou['matches_analyzed_away'] >=3) 
# ) | ( #### Number of goals exceed 1.5 in each match for last 5 matches
#     (df_alerts_ou['goals_second_half_home'] + df_alerts_ou['goals_first_half_home'] >= df_alerts_ou['matches_analyzed_home'] * 1.5) &
#     (df_alerts_ou['matches_analyzed_home'] >=3) 
# ) | (
#     (df_alerts_ou['goals_second_half_away'] + df_alerts_ou['goals_first_half_away'] >= df_alerts_ou['matches_analyzed_away'] * 1.5) &
#     (df_alerts_ou['matches_analyzed_away'] >=3) 
# ) | ( #### Number of match under perform
#     (df_alerts_ou['losses_home'] >= df_alerts_ou['matches_analyzed_home'] - 1) &
#     (df_alerts_ou['goals_second_half_home'] + df_alerts_ou['goals_first_half_home'] <= df_alerts_ou['matches_analyzed_home'] * 0.7) &
#     (df_alerts_ou['matches_analyzed_home'] >=3) 
# ) | (
#     (df_alerts_ou['losses_away'] >= df_alerts_ou['matches_analyzed_away'] - 1) &
#     (df_alerts_ou['goals_second_half_away'] + df_alerts_ou['goals_first_half_away'] <= df_alerts_ou['matches_analyzed_away'] * 0.7) &
#     (df_alerts_ou['matches_analyzed_away'] >=3) 
# )

# df_alerts_ou = df_alerts_ou[stats_condition]

# sql_realtime = """
#    WITH ranked AS (
#     SELECT *,
#             ROW_NUMBER() OVER (PARTITION BY id ORDER BY run_time DESC) AS rn,
#             now()::timestamp as current_now
#     FROM "188bet_log"
#     WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '1.5 hours'
#          AND "run_time"::TIMESTAMP <= (NOW()::timestamp + INTERVAL '7 hours')
#     )
#     SELECT *
#     FROM ranked;
# """

# resp = requests.post("http://165.232.188.235:8000/query/log",
#                     json={"sql": f"{sql_realtime}"})
# data = resp.json()
# try:
#     df_realtime = pd.DataFrame(data["rows"], columns=data["columns"])
# except Exception as e:
#     df_realtime = pd.DataFrame()

# realtime_match = pd.DataFrame()

# for i in df_stats['match_name'].tolist():
#     # print(i)
#     extract_goals = extract_goal_events_with_preodds(df_realtime, i)
#     realtime_match = pd.concat([extract_goals, realtime_match], axis=0, ignore_index=True)


# all_match_within_signals = df_alerts_hc['match_name'].tolist() + df_alerts_ou['match_name'].tolist()

# df_to_inform_realtime = realtime_match[realtime_match['match_name'].isin(all_match_within_signals)]
# df_to_inform_realtime = df_to_inform_realtime.sort_values(by=['match_name', 'goal_time'], ascending=[0, 1])



from hook.telegram_v2 import send_telegram_message

token="1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
chat_id = "@vihuynh_alert"


##### DF_UNDER
df_tele = df_alerts_ou[['id', 'cid', 'l', 'n', 'match_name', 'score', 'match_time',
       'current_time', 'run_time', 'match_part', 'time_difference',
       'Bàn Thắng: Trên / Dưới', 'Cược Chấp',
       'matches_analyzed_home', 
       'wins_home', 'draws_home', 'goals_first_half_home', 'goals_second_half_home',
       'matches_analyzed_away',
       'wins_away', 'draws_away','goals_first_half_away', 'goals_second_half_away',
       'comment', 
       'hh_value_first_odd', 'rate_hh_first_odd', 'rate_ah_first_odd',
       'line_value_first_odd', 'score_timeline', 
       'draw', 'under_lose', 'under_lose_half', 'under_win', 'under_win_half', 
       'draw_hh', 'home_lose', 'home_lose_half', 'home_win', 'home_win_half'
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