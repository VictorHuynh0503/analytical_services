import pandas as pd
import numpy as np

def build_score_timeline_df(df, id_col="id", score_col="score", minute_col="minute"):
    """
    Builds score change timeline per match.

    Returns a DataFrame:
        id | score_timeline
        -------------------------
        1  | ["0-0 | 5", "0-1 | 24", "1-1 | 30"]
        2  | ["0-0 | 10", "1-0 | 48", "1-1 | 77"]
    """

    records = []

    for match_id, group in df.groupby(id_col):
        group = group.sort_values(by=minute_col)

        timeline = []
        last_score = None

        for _, row in group.iterrows():
            score = row[score_col]
            minute = row[minute_col]

            if score != last_score:
                timeline.append(f"{score} | {minute}")
                last_score = score

        records.append({id_col: match_id, "score_timeline": timeline})

    return pd.DataFrame(records)

def build_second_half_goal_df(df, id_col="id", score_col="score", minute_col="minute"):
    """
    Detects score changes after minute 45 (second half goals).

    Returns a DataFrame:
        id | second_half_goals
        -----------------------------------------
        1  | ["0-1 | 50", "0-2 | 65"]
        2  | ["1-1 | 70"]
    """

    records = []

    for match_id, group in df.groupby(id_col):
        group = group.sort_values(by=minute_col)

        last_score = None
        second_half_events = []

        for _, row in group.iterrows():
            score = row[score_col]
            minute = row[minute_col]

            # Detect score change
            if score != last_score:
                # Only consider second half
                if minute > 45 and last_score is not None:
                    second_half_events.append(f"{score} | {minute}")

                last_score = score

        records.append({
            id_col: match_id,
            "second_half_goals": second_half_events
        })

    return pd.DataFrame(records)

if __name__ == "__main__":    
    import requests
    import json
    import pandas as pd
    import re
    import sys
    import os

    sql =     """
        SELECT * FROM "188bet_log" 
        WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '1024 hours'
        AND "run_time"::TIMESTAMP <= (NOW()::timestamp - INTERVAL '1000 hours')
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
    df["minute"] = df["current_time"].apply(parse_minute)
   ## df.rename(columns={'l': "country", "n": "league"}, inplace=True)
    
    stats = build_score_timeline_df(df)
    print(stats)
