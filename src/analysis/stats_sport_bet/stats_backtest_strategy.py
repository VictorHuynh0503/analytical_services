import pandas as pd
import numpy as np
# --- Parsing helpers ---
def parse_match_name(match_name: str):
    """Parse the match name into home and away teams."""
    try:
        # keep your old logic
        home_name, away_name = map(str.strip, match_name.split('-'))
    except ValueError:
        # fallback: only split on the last '-'
        home_name, away_name = map(str.strip, match_name.rsplit('-', 1))
    return home_name, away_name

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
        WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '10000 hours'
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
    
    from src.analysis.stats_sport_bet.stats_score_transition import convert_bet_odds
    from src.analysis.stats_sport_bet.stats_score_transition import parse_odds_columns
    from src.analysis.stats_sport_bet.stats_last_5_perf import match_stats
    from src.analysis.stats_sport_bet.stats_bet_odd import extract_goal_events_with_preodds
    from src.analysis.stats_sport_bet.stats_score_transition import parse_match_name
    from src.analysis.stats_sport_bet.stats_first_bet_odds import get_first_bet_odds

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
    
    df_exmple = df_parsed.copy()
    

    
    df1 = df_exmple[df_exmple['id']=='10019635']
    df1 = df1[['id', 'cid', 'l', 'n', 'match_name', 'score', 'match_time',
       'current_time', 'run_time', 'rate_hh', 'rate_ah', 'hh', 'ah',
       'hh_value', 'ah_value', 'rate_over', 'rate_under', 'line', 'line_value',
       'minute', 'minute_interval']]
    
    import pandas as pd
    import numpy as np

    def backtest_strategy(df: pd.DataFrame):
        df = df.copy()

        # --- Ensure correct types ---
        df['hh_value'] = pd.to_numeric(df['hh_value'], errors='coerce')
        df['line_value'] = pd.to_numeric(df['line_value'], errors='coerce')

        # --- Extract numeric goals ---
        df[['home_goals', 'away_goals']] = df['score'].str.split('-', expand=True).astype(int)

        # --- Compute final match score per match ---
        final_scores = (
            df.groupby('match_name')
            .agg(final_home=('home_goals', 'last'),
                final_away=('away_goals', 'last'))
            .reset_index()
        )
        final_scores['final_total'] = final_scores['final_home'] + final_scores['final_away']
        final_scores['goal_diff'] = final_scores['final_home'] - final_scores['final_away']

        # --- Merge back to main DF ---
        df = df.merge(final_scores, on='match_name', how='left')

        # --- Compute bet outcomes ---
        def handicap_result(row):
            """
            Asian handicap bet on HOME team.
            """
            if pd.isna(row['hh_value']):
                return np.nan

            # Goal difference from home POV
            diff = row['final_home'] - row['final_away']
            handicap = row['hh_value']

            # Compute outcome
            result = diff - handicap

            # Rules for Asian Handicap
            if result > 0.25:
                return 1.0    # Win
            elif result == 0.25:
                return 0.5    # Half win
            elif result == 0:
                return 0.0    # Push
            elif result == -0.25:
                return -0.5   # Half loss
            else:
                return -1.0   # Loss

        def over_result(row):
            """
            Over/Under bet on total goals being above line_value.
            """
            if pd.isna(row['line_value']):
                return np.nan

            total_goals = row['final_home'] + row['final_away']
            diff = total_goals - row['line_value']

            if diff > 0.25:
                return 1.0    # Over wins
            elif diff == 0.25:
                return 0.5
            elif diff == 0:
                return 0.0    # Push
            elif diff == -0.25:
                return -0.5
            else:
                return -1.0   # Over loses

        df['home_bet_result'] = df.apply(handicap_result, axis=1)
        df['over_bet_result'] = df.apply(over_result, axis=1)

        # --- Aggregate by minute interval ---
        summary = (
            df.groupby('minute_interval')
            .agg(
                count=('match_name', 'count'),
                avg_home_result=('home_bet_result', 'mean'),
                avg_over_result=('over_bet_result', 'mean')
            )
            .reset_index()
            .sort_values('minute_interval')
        )

        # Add interpretation
        summary['home_winrate_%'] = (summary['avg_home_result'] > 0).astype(int)
        summary['over_winrate_%'] = (summary['avg_over_result'] > 0).astype(int)

        return df, summary


    import pandas as pd
    import numpy as np

    def backtest_betting_strategy(df: pd.DataFrame):
        """
        Backtest handicap and over/under betting strategies.
        
        For each snapshot:
        - Handicap bet (Home): Check if home team + hh_value > away team at final score
        - Over/Under bet: Check if total goals > line_value at final score
        
        Returns DataFrames with bet results and profitability analysis.
        """
        
        # Ensure numeric columns
        numeric_cols = ['hh_value', 'ah_value', 'line_value', 'rate_hh', 'rate_ah', 'rate_over', 'rate_under']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = []
        
        # Group by match to get final scores
        for match_id in df['id'].unique():
            match_data = df[df['id'] == match_id].sort_values('run_time').reset_index(drop=True)
            
            if match_data.empty:
                continue
                
            # Get final score from the last row
            final_score = match_data.iloc[-1]['score']
            try:
                final_home, final_away = map(int, final_score.split('-'))
            except:
                continue
            
            # Analyze each betting snapshot
            for idx, row in match_data.iterrows():
                try:
                    current_home, current_away = map(int, row['score'].split('-'))
                except:
                    continue
                
                # Skip if this is already the final score
                if current_home == final_home and current_away == final_away:
                    continue
                
                # --- HANDICAP BET (Home team with handicap) ---
                if pd.notna(row['hh_value']) and pd.notna(row['rate_hh']):
                    # Apply handicap to home team
                    home_with_handicap = final_home + row['hh_value']
                    
                    # Determine result
                    if home_with_handicap > final_away:
                        hh_result = 'WIN'
                        hh_profit = row['rate_hh'] - 1  # Net profit (odds - stake)
                    elif home_with_handicap == final_away:
                        hh_result = 'PUSH'
                        hh_profit = 0
                    else:
                        hh_result = 'LOSS'
                        hh_profit = -1  # Lost stake
                else:
                    hh_result = None
                    hh_profit = None
                
                # --- HANDICAP BET (Away team with handicap) ---
                if pd.notna(row['ah_value']) and pd.notna(row['rate_ah']):
                    away_with_handicap = final_away + row['ah_value']
                    
                    if away_with_handicap > final_home:
                        ah_result = 'WIN'
                        ah_profit = row['rate_ah'] - 1
                    elif away_with_handicap == final_home:
                        ah_result = 'PUSH'
                        ah_profit = 0
                    else:
                        ah_result = 'LOSS'
                        ah_profit = -1
                else:
                    ah_result = None
                    ah_profit = None
                
                # --- OVER/UNDER BET ---
                if pd.notna(row['line_value']):
                    total_goals = final_home + final_away
                    
                    # OVER bet
                    if pd.notna(row['rate_over']):
                        if total_goals > row['line_value']:
                            over_result = 'WIN'
                            over_profit = row['rate_over'] - 1
                        elif total_goals == row['line_value']:
                            over_result = 'PUSH'
                            over_profit = 0
                        else:
                            over_result = 'LOSS'
                            over_profit = -1
                    else:
                        over_result = None
                        over_profit = None
                    
                    # UNDER bet
                    if pd.notna(row['rate_under']):
                        if total_goals < row['line_value']:
                            under_result = 'WIN'
                            under_profit = row['rate_under'] - 1
                        elif total_goals == row['line_value']:
                            under_result = 'PUSH'
                            under_profit = 0
                        else:
                            under_result = 'LOSS'
                            under_profit = -1
                    else:
                        under_result = None
                        under_profit = None
                else:
                    over_result = under_result = None
                    over_profit = under_profit = None
                
                # Store results
                results.append({
                    'match_id': row['id'],
                    'country': row['l'],
                    'league': row['n'],
                    'match_name': row['match_name'],
                    'bet_time': row['run_time'],
                    'current_score': row['score'],
                    'final_score': final_score,
                    'minute': row['minute'],
                    'minute_interval': row['minute_interval'],
                    
                    # Handicap Home
                    'hh_value': row['hh_value'],
                    'rate_hh': row['rate_hh'],
                    'hh_result': hh_result,
                    'hh_profit': hh_profit,
                    
                    # Handicap Away
                    'ah_value': row['ah_value'],
                    'rate_ah': row['rate_ah'],
                    'ah_result': ah_result,
                    'ah_profit': ah_profit,
                    
                    # Over/Under
                    'line_value': row['line_value'],
                    'rate_over': row['rate_over'],
                    'over_result': over_result,
                    'over_profit': over_profit,
                    'rate_under': row['rate_under'],
                    'under_result': under_result,
                    'under_profit': under_profit,
                })
        
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            return results_df, pd.DataFrame(), pd.DataFrame()
        
        # --- SUMMARY STATISTICS ---
        
        # Handicap Home Summary
        hh_summary = results_df[results_df['hh_result'].notna()].groupby(['country', 'league', 'minute_interval']).agg({
            'hh_result': 'count',
            'hh_profit': ['sum', 'mean']
        }).reset_index()
        hh_summary.columns = ['country', 'league', 'minute_interval', 'total_bets', 'total_profit', 'avg_profit']
        hh_summary['win_rate'] = results_df[results_df['hh_result'].notna()].groupby(
            ['country', 'league', 'minute_interval']
        )['hh_result'].apply(lambda x: (x == 'WIN').sum() / len(x)).values
        hh_summary['roi'] = (hh_summary['total_profit'] / hh_summary['total_bets']) * 100
        hh_summary['bet_type'] = 'Handicap Home'
        
        # Over Summary
        over_summary = results_df[results_df['over_result'].notna()].groupby(['country', 'league', 'minute_interval']).agg({
            'over_result': 'count',
            'over_profit': ['sum', 'mean']
        }).reset_index()
        over_summary.columns = ['country', 'league', 'minute_interval', 'total_bets', 'total_profit', 'avg_profit']
        over_summary['win_rate'] = results_df[results_df['over_result'].notna()].groupby(
            ['country', 'league', 'minute_interval']
        )['over_result'].apply(lambda x: (x == 'WIN').sum() / len(x)).values
        over_summary['roi'] = (over_summary['total_profit'] / over_summary['total_bets']) * 100
        over_summary['bet_type'] = 'Over'
        
        # Combine summaries
        summary = pd.concat([hh_summary, over_summary], ignore_index=True)
        summary = summary.sort_values(['roi'], ascending=False)
        
        return results_df, summary, results_df


    # --- USAGE EXAMPLE ---
    # results_df, summary, detailed = backtest_betting_strategy(df)

    # View top performing strategies
    # print(summary.head(20))

    # View detailed bet results
    # print(results_df[results_df['hh_result'] == 'WIN'].head())

    # Filter profitable strategies
    # profitable = summary[summary['roi'] > 0]
    # print(profitable)