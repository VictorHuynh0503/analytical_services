import pandas as pd
import numpy as np

def evaluate_over_expectation(bet_odds_goals: float, current_score: tuple[int, int]) -> str:
    total_goals = sum(current_score)
    
    if total_goals > bet_odds_goals:
        return f"Over {bet_odds_goals} has already won, as total goals ({total_goals}) exceeded the line."
    elif total_goals == bet_odds_goals:
        return f"The total goals ({total_goals}) match the bet line {bet_odds_goals}, so Over/Under is very close."
    else:
        remaining_goals_needed = bet_odds_goals - total_goals
        if bet_odds_goals % 1 == 0.75 and remaining_goals_needed == 1:
            return "Over wins half if one more goal is scored."
        elif bet_odds_goals % 1 == 0.25 and remaining_goals_needed == 1:
            return "Over loses half if one more goal is scored."
        return f"For Over {bet_odds_goals} to win, at least {remaining_goals_needed} more goal(s) are needed."

def evaluate_over_under_result(bet_odds_goals: float, current_score: tuple[int, int]) -> str:
    total_goals = sum(current_score)
    
    if total_goals > bet_odds_goals:
        return "Over wins."
    elif total_goals < bet_odds_goals:
        return "Under wins."
    else:
        if bet_odds_goals % 1 == 0.25:
            return "Over loses half, Under wins half."
        elif bet_odds_goals % 1 == 0.75:
            return "Over wins half, Under loses half."
        else:
            return "Over/Under Draw"


def evaluate_handicap(bet_odds_handicap: float, current_score: tuple[int, int]) -> str:
    home_score, away_score = current_score
    
    # Handling quarter handicaps (e.g., -1.25, +1.25)
    if bet_odds_handicap % 1 == 0.25 or bet_odds_handicap % 1 == 0.75:

        adjusted_home_score = home_score + bet_odds_handicap
        
        delta = adjusted_home_score - away_score
            
        if delta >= 0.5:
            return f"Home wins"
        elif delta > 0 and delta < 0.5:
            return f"Home win half"
        elif delta == 0:
            return f"Draw"
        elif delta > -0.5 and delta < 0:
            return f"Home lose half"
        elif delta <= -0.5:
            return f"Home loses"
        else:
            pass
    
    else:
        adjusted_home_score = home_score + bet_odds_handicap
    
        if adjusted_home_score > away_score:
            return f"Home wins"
        elif adjusted_home_score == away_score:
            return f"Draw"
        elif adjusted_home_score < away_score:
            return f"Home loses"
        else:
            pass


def process_over_under(df):
    
    df[['home_score', 'away_score']] = df['score'].str.split('-', expand=True).astype(int)

    
    df['over_under_result'] = df.apply(lambda row: evaluate_over_under_result(row['hh_first'], (row['home_score'], row['away_score'])), axis=1)
    # df['handicap_result'] = df.apply(lambda row: evaluate_handicap(row['ah_first'], (row['home_score'], row['away_score'])), axis=1)
    
    return df

def process_handicap(df):
    
    df[['home_score', 'away_score']] = df['score'].str.split('-', expand=True).astype(int)
       
    # df['over_under_result'] = df.apply(lambda row: evaluate_over_under_result(row['hh_first'], (row['home_score'], row['away_score'])), axis=1)
    df['handicap_result'] = df.apply(lambda row: evaluate_handicap(row['hh_first'], (row['home_score'], row['away_score'])), axis=1)
    
    return df

if __name__ == "__main__":  
    import requests
    import json
    import pandas as pd
    import re
    import sys
    import os

    sql =     """
        SELECT * FROM "188bet_log" 
        WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '5000 hours'
        AND "run_time"::TIMESTAMP <= (NOW()::timestamp - INTERVAL '7 hours')
        """

    # resp = requests.post("http://165.232.188.235:8000/query/log",
    #                     json={"sql": f"{sql}"})
    # ##print(resp.json())

    # data = resp.json()

    # df = pd.DataFrame(data["rows"], columns=data["columns"])
    
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env into environment

    sys_path = os.getenv("sys_path")
    print(sys_path)
    os.chdir(sys_path)
    sys.path.append(sys_path)
    
    from storage import duckdb_reader as dr 
    from src.analysis.stats_sport_bet.stats_score_transition import parse_odds_columns

    df = dr.read_from_duckdb(
        db_path="log_data/188bet_log.duckdb",
        query = sql
    )
    
    df_parsed = parse_odds_columns(df)
    
    df_under_over = process_over_under(df_parsed.copy())
    df_handicap = process_handicap(df_parsed.copy())
    
   ## df.rename(columns={'l': "country", "n": "league"}, inplace=True)