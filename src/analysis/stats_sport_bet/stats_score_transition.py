import pandas as pd

# --- Parsing helpers ---
def parse_match_name(match_name: str):
    """Parse the score into home and away goals."""
    home_name, away_name = map(str, match_name.split('-'))
    return home_name, home_name

def convert_bet_odds(odds: str) -> float:
    """
    Convert fractional betting odds (e.g., '-1/1.5' → -1.25, '+0/0.5' → 0.25).
    """
    try:
        if '/' in odds:
            left, right = odds.split('/')
            left, right = float(left), float(right)

            if odds.startswith('-0/'):
                return round(-right / 2, 2)
            elif odds.startswith('0/') or odds.startswith('+0/'):
                return round(right / 2, 2)

            if left < 0 or right < 0:
                return round(-(abs(left) + abs(right)) / 2, 2)
            else:
                return round((left + right) / 2, 2)

        return float(odds)
    except Exception:
        return None


def parse_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'Cược Chấp' and 'Bàn Thắng: Trên / Dưới' into numeric values.
    """
    # --- Handicap parsing ---
    pattern_handicap = (
        r'(?P<rate_hh>-?[\d\.]+)-(?P<rate_ah>-?[\d\.]+) '
        r'(?P<hh>[+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?) \| '
        r'(?P<ah>[+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)'
    )
    df_hc = df["Cược Chấp"].astype(str).str.extract(pattern_handicap)
    df = pd.concat([df, df_hc], axis=1)
    df["hh_value"] = df["hh"].apply(convert_bet_odds)
    df["ah_value"] = df["ah"].apply(convert_bet_odds)

    # --- Over/Under parsing ---
    pattern_ou = (
        r'(?P<rate_over>-?[\d\.]+)-(?P<rate_under>-?[\d\.]+) '
        r'(?P<line>[+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)'
    )
    df_ou = df["Bàn Thắng: Trên / Dưới"].astype(str).str.extract(pattern_ou)
    df = pd.concat([df, df_ou], axis=1)
    df["line_value"] = df["line"].apply(convert_bet_odds)

    return df


# --- Betting stats function using parsed odds ---

def betting_stats_by_league(df: pd.DataFrame):
    """
    Analyze betting odds before goals, grouped by country & league.
    Uses parsed numeric values from Cược Chấp and Bàn Thắng: Trên / Dưới.
    """
    # Parse odds columns first
    df = parse_odds_columns(df)

    all_events = []

    for match in df["match_name"].unique():
        match_snaps = df[df["match_name"] == match].sort_values("run_time").reset_index(drop=True)
        prev_home, prev_away = 0, 0

        for i, snap in match_snaps.iterrows():
            try:
                home_goals, away_goals = map(int, snap["score"].split("-"))
            except:
                continue

            if home_goals != prev_home or away_goals != prev_away:  # goal detected
                if i > 0:
                    pre_snap = match_snaps.iloc[i-1]
                    all_events.append({
                        "country": snap["l"],
                        "league": snap["n"],
                        "from_score": f"{prev_home}-{prev_away}",
                        "to_score": snap["score"],
                        "pre_handicap": pre_snap["hh_value"],   # numeric handicap (home side)
                        "pre_ah": pre_snap["ah_value"],         # numeric handicap (away side)
                        "pre_line": pre_snap["line_value"],     # numeric O/U line
                    })

            prev_home, prev_away = home_goals, away_goals

    events_df = pd.DataFrame(all_events)
    if events_df.empty:
        return pd.DataFrame()

    # --- Aggregate stats ---
    stats = (
        events_df.groupby(["country", "league", "pre_handicap", "from_score", "to_score"])
        .size()
        .reset_index(name="count")
    )
    stats["total_for_handicap"] = stats.groupby(["country", "league", "pre_handicap"])["count"].transform("sum")
    stats["success_rate"] = stats["count"] / stats["total_for_handicap"]

    # --- Aggregate Handicap stats ---
    handicap_stats = (
        events_df.groupby(["country", "league", "pre_handicap", "from_score", "to_score"])
        .size()
        .reset_index(name="count")
    )
    handicap_stats["total_for_handicap"] = handicap_stats.groupby(
        ["country", "league", "pre_handicap"]
    )["count"].transform("sum")
    handicap_stats["success_rate"] = handicap_stats["count"] / handicap_stats["total_for_handicap"]
    
    handicap_stats["total_for_fromscore_handicap"] = handicap_stats.groupby(
        ["country", "league", "pre_handicap", "from_score"]
    )["count"].transform("sum")
    handicap_stats["success_rate_fromscore"] = (
        handicap_stats["count"] / handicap_stats["total_for_fromscore_handicap"]
    )
    
    # --- Aggregate Over/Under stats ---
    ou_stats = (
        events_df.groupby(["country", "league", "pre_line", "from_score", "to_score"])
        .size()
        .reset_index(name="count")
    )
    ou_stats["total_for_line"] = ou_stats.groupby(
        ["country", "league", "pre_line"]
    )["count"].transform("sum")
    ou_stats["success_rate"] = ou_stats["count"] / ou_stats["total_for_line"]
    
    ou_stats["total_for_fromscore_line"] = ou_stats.groupby(
        ["country", "league", "pre_line", "from_score"]
    )["count"].transform("sum")
    ou_stats["success_rate_fromscore"] = ou_stats["count"] / ou_stats["total_for_fromscore_line"]    


    return handicap_stats.sort_values(
        ["country", "league", "from_score", "success_rate"], ascending=[True, True, True, False]
    ), ou_stats.sort_values(
        ["country", "league", "from_score", "success_rate"], ascending=[True, True, True, False]
    )


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

    df = dr.read_from_duckdb(
        db_path="log_data/188bet_log.duckdb",
        query = sql
    )
    
   ## df.rename(columns={'l': "country", "n": "league"}, inplace=True)
    
    team = "Red Bull Salzburg"
    stats = betting_stats_by_league(df)
    print(stats)
    
    
    import pandas as pd

    # Example: you already have handicap_stats, ou_stats
    handicap_stats, ou_stats = stats

    # Save to Excel with 2 sheets
    with pd.ExcelWriter("betting_stats.xlsx", engine="openpyxl") as writer:
        handicap_stats.to_excel(writer, sheet_name="Handicap Stats", index=False)
        ou_stats.to_excel(writer, sheet_name="OverUnder Stats", index=False)

    print("✅ Excel file saved as betting_stats.xlsx")