import requests
import pandas as pd
import os
import sys
import re
import pytz

from datetime import datetime
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def parse_score(score):
    """Parse the score into home and away goals."""
    home_goals, away_goals = map(int, score.split('-'))
    return home_goals, away_goals

def get_first_bet_odds(df: pd.DataFrame):

    # Load the Parquet file
    data = df.sort_values(by=['match_name', 'run_time'])  # Ensure data is ordered by match and run time

    timezone = pytz.timezone('Asia/Bangkok')  # UTC+7 is typically the timezone for Bangkok
    timezone_utc = pytz.UTC
    now = datetime.now(timezone)
    # data['match_time'] = pd.to_datetime(data['match_time'], errors='coerce')

    # # now = pd.Timestamp(now)
    
    # # data['match_time'] = pd.to_datetime(data['match_time'])
    # data['match_time'] = data['match_time'].dt.tz_localize('UTC').dt.tz_convert(timezone_utc)
    # data = data[data['match_time'] >= now - timedelta(minutes=120)]   
    data = data[data['l'] != 'efootball_sports']
    
    # Group data by match_name
    matches = data.groupby('match_name')

    changes = pd.DataFrame()

    for match_name, match_data in matches:
        match_data = match_data.reset_index(drop=True)
        
        # Get the latest two rows for each match
        latest_data = match_data.head(1)
        print(latest_data[['l', 'n', 'match_name', 'score', 'time_difference',  'Bàn Thắng: Trên / Dưới', 'Cược Chấp', 'run_time']].to_string())

        changes = pd.concat([changes,latest_data], axis = 0)
            
    return changes




