import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
import requests
import json
import re
import sys
import os
warnings.filterwarnings('ignore')

class BettingPatternAnalyzer:
    """
    Analyzes betting patterns to find leagues/countries with >70% winning rate
    Adapted for actual betting data format
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with match data
        
        Expected columns:
        - country, cup (league), match_name
        - score (final score like "0-0", "1-1", etc.)
        - minute (current minute in format like "[19:10]")
        - minute_interval (like "10-20")
        - hh_value (home handicap odds), ah_value (away handicap odds)
        - rate_over, rate_under (over/under odds)
        - line, line_value (over/under line values)
        """
        self.df = df.copy()
        self.prepare_data()
        self.results = {}
        
    def prepare_data(self):
        """Prepare and clean the data"""
        # Parse score into home and away goals
        if 'score' in self.df.columns:
            score_split = self.df['score'].str.split('-', expand=True)
            self.df['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
            self.df['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')
        
        # Parse minute from format like "[19:10]" to numeric
        if 'current_time' in self.df.columns:
            self.df['minute_numeric'] = self.df['current_time'].str.extract(r'\[(\d+):').astype(float)
        
        # Parse minute_interval to get start minute
        if 'minute_interval' in self.df.columns:
            self.df['interval_start'] = self.df['minute_interval'].str.split('-', expand=True)[0].astype(float)
        
        # Convert odds to numeric
        numeric_cols = ['hh_value', 'ah_value', 'rate_over', 'rate_under', 
                       'line', 'line_value', 'hh', 'ah']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create match identifier
        self.df['id'] = self.df['id'].astype(str) + '_' + self.df['cid'].astype(str)
        
    def simulate_betting_strategies(self, 
                                   minute_threshold: float = None,
                                   interval_filter: str = None) -> pd.DataFrame:
        """
        Simulate different betting strategies
        
        Parameters:
        minute_threshold: Only analyze data after this minute (e.g., 20 for after 20')
        interval_filter: Filter by minute_interval (e.g., "20-30", "30-40")
        
        Returns:
        DataFrame with win rates by country/league and strategy
        """
        
        # Filter data
        betting_data = self.df.copy()
        
        if minute_threshold is not None:
            betting_data = betting_data[betting_data['interval_start'] >= minute_threshold]
        
        if interval_filter is not None:
            betting_data = betting_data[betting_data['minute_interval'] == interval_filter]
        
        strategies = {
            'home_win': self._bet_home_win,
            'away_win': self._bet_away_win,
            'draw': self._bet_draw,
            'over_goals': self._bet_over,
            'under_goals': self._bet_under,
            'home_handicap': self._bet_home_handicap,
            'away_handicap': self._bet_away_handicap,
            'high_rate_over': self._bet_high_rate_over,
            'high_rate_under': self._bet_high_rate_under,
        }
        
        results_list = []
        
        # Group by country and cup (league)
        for (country, cup), group in betting_data.groupby(['l', 'n',]):
            total_bets = len(group)
            
            if total_bets < 10:  # Minimum sample size
                continue
            
            for strategy_name, strategy_func in strategies.items():
                try:
                    wins, eligible_bets = strategy_func(group)
                    
                    if eligible_bets == 0:
                        continue
                        
                    win_rate = (wins / eligible_bets * 100)
                    
                    results_list.append({
                        'country': country,
                        'league': cup,
                        'strategy': strategy_name,
                        'minute_filter': interval_filter if interval_filter else 'all',
                        'eligible_bets': eligible_bets,
                        'wins': wins,
                        'losses': eligible_bets - wins,
                        'win_rate': round(win_rate, 2),
                        'passes_70_threshold': win_rate >= 70,
                        'passes_75_threshold': win_rate >= 75,
                        'passes_80_threshold': win_rate >= 80,
                    })
                except Exception as e:
                    continue
  
        results_df = pd.DataFrame(results_list)
        key = interval_filter if interval_filter else 'all'
        self.results[key] = results_df
        
        return results_df
    
    def _bet_home_win(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on home team to win"""
        valid = group.dropna(subset=['home_goals', 'away_goals'])
        wins = (valid['home_goals'] > valid['away_goals']).sum()
        return wins, len(valid)
    
    def _bet_away_win(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on away team to win"""
        valid = group.dropna(subset=['home_goals', 'away_goals'])
        wins = (valid['away_goals'] > valid['home_goals']).sum()
        return wins, len(valid)
    
    def _bet_draw(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on draw"""
        valid = group.dropna(subset=['home_goals', 'away_goals'])
        wins = (valid['home_goals'] == valid['away_goals']).sum()
        return wins, len(valid)
    
    def _bet_over(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on over (total goals > line)"""
        valid = group.dropna(subset=['home_goals', 'away_goals', 'line'])
        if len(valid) == 0:
            return 0, 0
        total_goals = valid['home_goals'] + valid['away_goals']
        wins = (total_goals > valid['line']).sum()
        return wins, len(valid)
    
    def _bet_under(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on under (total goals < line)"""
        valid = group.dropna(subset=['home_goals', 'away_goals', 'line'])
        if len(valid) == 0:
            return 0, 0
        total_goals = valid['home_goals'] + valid['away_goals']
        wins = (total_goals < valid['line']).sum()
        return wins, len(valid)
    
    def _bet_home_handicap(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on home team with handicap when hh is negative (favorite)"""
        valid = group.dropna(subset=['home_goals', 'away_goals', 'hh'])
        # Only bet when home is favorite (negative handicap)
        valid = valid[valid['hh'] < 0]
        if len(valid) == 0:
            return 0, 0
        
        adjusted_home = valid['home_goals'] + valid['hh']
        wins = (adjusted_home > valid['away_goals']).sum()
        return wins, len(valid)
    
    def _bet_away_handicap(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on away team with handicap when ah is positive (favorite)"""
        valid = group.dropna(subset=['home_goals', 'away_goals', 'ah'])
        # Only bet when away gets advantage (positive handicap)
        valid = valid[valid['ah'] > 0]
        if len(valid) == 0:
            return 0, 0
        
        adjusted_away = valid['away_goals'] + valid['ah']
        wins = (adjusted_away > valid['home_goals']).sum()
        return wins, len(valid)
    
    def _bet_high_rate_over(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on over when rate_over is high (>2.0)"""
        valid = group.dropna(subset=['home_goals', 'away_goals', 'line', 'rate_over'])
        valid = valid[valid['rate_over'] > 2.0]
        if len(valid) == 0:
            return 0, 0
        
        total_goals = valid['home_goals'] + valid['away_goals']
        wins = (total_goals > valid['line']).sum()
        return wins, len(valid)
    
    def _bet_high_rate_under(self, group: pd.DataFrame) -> Tuple[int, int]:
        """Bet on under when rate_under is high (>2.0)"""
        valid = group.dropna(subset=['home_goals', 'away_goals', 'line', 'rate_under'])
        valid = valid[valid['rate_under'] > 2.0]
        if len(valid) == 0:
            return 0, 0
        
        total_goals = valid['home_goals'] + valid['away_goals']
        wins = (total_goals < valid['line']).sum()
        return wins, len(valid)
    
    def find_profitable_patterns(self, 
                                min_win_rate: float = 70, 
                                min_bets: int = 15) -> pd.DataFrame:
        """
        Find patterns with win rate >= threshold
        
        Parameters:
        min_win_rate: Minimum win rate percentage (default 70%)
        min_bets: Minimum number of eligible bets for validity
        
        Returns:
        DataFrame of profitable patterns sorted by win rate
        """
        all_results = pd.concat(self.results.values(), ignore_index=True)
        
        profitable = all_results[
            (all_results['win_rate'] >= min_win_rate) & 
            (all_results['eligible_bets'] >= min_bets)
        ].sort_values(['win_rate', 'eligible_bets'], ascending=[False, False])
        
        return profitable
    
    def analyze_all_intervals(self) -> pd.DataFrame:
        """Analyze all minute intervals"""
        # Get unique intervals
        intervals = sorted(self.df['minute_interval'].dropna().unique())
        
        all_results = []
        for interval in intervals:
            print(f"Analyzing interval: {interval}")
            result = self.simulate_betting_strategies(interval_filter=interval)
            all_results.append(result)
        
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics"""
        all_results = pd.concat(self.results.values(), ignore_index=True)
        
        summary = all_results.groupby('strategy').agg({
            'win_rate': ['mean', 'max', 'min', 'std'],
            'eligible_bets': 'sum',
            'wins': 'sum'
        }).round(2)
        
        return summary


# Main execution
def main(df: pd.DataFrame):
    """
    Main function to analyze betting patterns
    
    Parameters:
    df: Your DataFrame with betting data
    """
    
    print("="*100)
    print("BETTING PATTERN ANALYSIS - FINDING 70%+ WIN RATE PATTERNS")
    print("="*100)
    
    # Initialize analyzer
    analyzer = BettingPatternAnalyzer(df)
    
    print(f"\nTotal matches: {df['id'].nunique()}")
    print(f"Total records: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Leagues: {df['cup'].nunique()}")
    
    # Analyze all data
    print("\n" + "-"*100)
    print("ANALYZING ALL TIME INTERVALS")
    print("-"*100)
    all_results = analyzer.analyze_all_intervals()
    
    # Find profitable patterns
    print("\n" + "="*100)
    print("PATTERNS WITH 70%+ WIN RATE")
    print("="*100)
    
    profitable_70 = analyzer.find_profitable_patterns(min_win_rate=70, min_bets=15)
    
    if len(profitable_70) > 0:
        print(f"\n✓ Found {len(profitable_70)} patterns with 70%+ win rate:\n")
        print(profitable_70[['country', 'league', 'strategy', 'minute_filter', 
                            'eligible_bets', 'wins', 'win_rate']].to_string(index=False))
    else:
        print("\n✗ No patterns found with 70%+ win rate and minimum 15 bets.")
        
    # Show 75% and 80% if available
    profitable_75 = analyzer.find_profitable_patterns(min_win_rate=75, min_bets=10)
    if len(profitable_75) > 0:
        print(f"\n" + "="*100)
        print(f"PATTERNS WITH 75%+ WIN RATE (minimum 10 bets)")
        print("="*100)
        print(profitable_75[['country', 'league', 'strategy', 'minute_filter', 
                            'eligible_bets', 'wins', 'win_rate']].to_string(index=False))
    
    # Show top patterns
    print("\n" + "="*100)
    print("TOP 20 PATTERNS BY WIN RATE (minimum 10 bets)")
    print("="*100)
    top_patterns = all_results[all_results['eligible_bets'] >= 10].nlargest(20, 'win_rate')
    if len(top_patterns) > 0:
        print(top_patterns[['country', 'league', 'strategy', 'minute_filter', 
                           'eligible_bets', 'wins', 'win_rate']].to_string(index=False))
    
    # Strategy summary
    print("\n" + "="*100)
    print("STRATEGY SUMMARY STATISTICS")
    print("="*100)
    summary = analyzer.get_summary_stats()
    print(summary)
    
    return analyzer, all_results, profitable_70


# How to use with your data:
if __name__ == "__main__":
    # Example: Load your CSV file
    # df = pd.read_csv('your_betting_data.csv')
    # analyzer, all_results, profitable = main(df)
    
    print("\nTo use this code:")
    print("1. Load your data: df = pd.read_csv('your_file.csv')")
    print("2. Run analysis: analyzer, all_results, profitable = main(df)")
    print("3. Explore results: profitable[profitable['win_rate'] >= 70]")
    pass

    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env into environment

    sys_path = os.getenv("sys_path")
    print(sys_path)
    os.chdir(sys_path)
    sys.path.append(sys_path)
    
    sql =     """
    SELECT * FROM "188bet_log" 
    WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '10000 hours'
    AND "run_time"::TIMESTAMP <= (NOW()::timestamp - INTERVAL '7 hours')
    """
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
    
    analyzer, all_results, profitable = main(df1)