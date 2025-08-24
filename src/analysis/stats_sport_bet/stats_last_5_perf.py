import pandas as pd
import numpy as np

def match_stats(df: pd.DataFrame, team: str, last_n: int = 5):
    """
    Compute recent stats for a given team.
    
    Parameters
    ----------
    df : pd.DataFrame
        Snapshot dataframe with columns at least:
        ['match_name', 'score', 'match_time', 'match_part', 'run_time']
    team : str
        Team name to analyze (home or away).
    last_n : int, optional
        Number of recent matches to include, default=5.
    
    Returns
    -------
    dict
        Stats summary with wins, losses, goals by half.
    """

    # --- Step 1: filter matches containing team ---
    team_matches = df[df["match_name"].str.contains(team, case=False, na=False)].copy()

    # --- Step 2: get final snapshot of each match ---
    # Assume run_time increases with time, so take last snapshot
    final_scores = (
        team_matches.sort_values("run_time")
        .groupby("match_name")
        .tail(1)
    )

    # --- Step 3: take only last N matches chronologically ---
    last_matches = final_scores.sort_values("match_time").tail(last_n)

    wins, losses, draws = 0, 0, 0
    goals_first_half, goals_second_half = 0, 0

    for _, row in last_matches.iterrows():
        score = row["score"]  # e.g. "2-1"
        try:
            home_goals, away_goals = map(int, score.split("-"))
        except:
            continue  # skip if score invalid
        
        # determine home/away team from match_name
        home_team, away_team = row["match_name"].split("-")
        
        if team.lower() == home_team.strip().lower():
            team_goals, opp_goals = home_goals, away_goals
        else:
            team_goals, opp_goals = away_goals, home_goals
        
        # --- Outcome ---
        if team_goals > opp_goals:
            wins += 1
        elif team_goals < opp_goals:
            losses += 1
        else:
            draws += 1

        # --- Goal timing info ---
        # We use snapshots (rows for the same match) to infer goal times
        match_snaps = team_matches[team_matches["match_name"] == row["match_name"]]
        match_snaps = match_snaps.sort_values("run_time")
        
        prev_home, prev_away = 0, 0
        for _, snap in match_snaps.iterrows():
            try:
                h, a = map(int, snap["score"].split("-"))
            except:
                continue
            if h != prev_home or a != prev_away:
                # goal happened!
                goal_time = float(snap["time_difference"])  # minutes
                if goal_time <= 45:
                    if team.lower() == home_team.strip().lower() and h > prev_home:
                        goals_first_half += 1
                    elif team.lower() == away_team.strip().lower() and a > prev_away:
                        goals_first_half += 1
                else:
                    if team.lower() == home_team.strip().lower() and h > prev_home:
                        goals_second_half += 1
                    elif team.lower() == away_team.strip().lower() and a > prev_away:
                        goals_second_half += 1
            prev_home, prev_away = h, a

    return {
        "team": team,
        "matches_analyzed": len(last_matches),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "goals_first_half": goals_first_half,
        "goals_second_half": goals_second_half,
        "last_matches": last_matches
    }


if __name__ == "__main__":  
    import requests
    import json
    import pandas as pd
    import re
    import sys
    import os

    sql =     """
        SELECT * FROM "188bet_log" 
        WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '3000 hours'
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

    df = dr.read_from_duckdb(
        db_path="log_data/188bet_log.duckdb",
        query = sql
    )
    
   ## df.rename(columns={'l': "country", "n": "league"}, inplace=True)
    
    team = "Red Bull Salzburg"
    stats = match_stats(df, team, last_n=5)
    print(stats)