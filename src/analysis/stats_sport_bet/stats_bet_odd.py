import pandas as pd

def extract_goal_events_with_preodds(df: pd.DataFrame, match_name: str):
    """
    Extract all goal events for a given match, capturing the odds
    just before the goal happened.
    
    Parameters
    ----------
    df : pd.DataFrame
        Snapshot dataframe with columns:
        ['match_name', 'score', 'time_difference',
         'Bàn Thắng: Trên / Dưới', 'Cược Chấp', 'run_time']
    match_name : str
        The match to analyze.
    
    Returns
    -------
    pd.DataFrame
        One row per goal event with timing + odds before goal.
    """
    # Filter and sort
    match_snaps = df[df["match_name"] == match_name].copy()
    match_snaps = match_snaps.sort_values("run_time").reset_index(drop=True)

    events = []
    prev_home, prev_away = 0, 0

    for i, snap in match_snaps.iterrows():
        try:
            home_goals, away_goals = map(int, snap["score"].split("-"))
        except:
            continue

        # Detect goal event
        if home_goals != prev_home or away_goals != prev_away:
            # Take the previous snapshot (if available)
            if i > 0:
                pre_snap = match_snaps.iloc[i-1]
                events.append({
                    "match_name": match_name,
                    "new_score": snap["score"],  # score after the goal
                    "goal_time": snap["time_difference"],
                    "new_over_under": snap["Bàn Thắng: Trên / Dưới"],
                    "new_handicap": snap["Cược Chấp"],
                    # when score updated
                    "pre_score": pre_snap["score"],
                    "pre_time": pre_snap["time_difference"],  # latest before goal
                    "pre_over_under": pre_snap["Bàn Thắng: Trên / Dưới"],
                    "pre_handicap": pre_snap["Cược Chấp"],
                })

        prev_home, prev_away = home_goals, away_goals

    return pd.DataFrame(events)


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
    
    sys_path = "D:/Projects/analytical_services/"
    os.chdir(sys_path)
    sys.path.append(sys_path)
    
    from storage import duckdb_reader as dr 

    df = dr.read_from_duckdb(
        db_path="log_data/188bet_log.duckdb",
        query = sql
    )
    
   ## df.rename(columns={'l': "country", "n": "league"}, inplace=True)
    
    team = "Red Bull Salzburg"
    stats = extract_goal_events_with_preodds(df, "TSV Hartberg-Red Bull Salzburg")
    print(stats)
