"""
Streamlit Data Loader
Handles loading of local parquet files and data freshness checks
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional

class StreamlitDataLoader:
    """Data loader for Streamlit app using local parquet files"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def load_team_data(self) -> pd.DataFrame:
        """Load team information."""
        # Team abbreviation mapping for ESPN logos
        team_logo_mapping = {
            'ARI': 'ari', 'ATL': 'atl', 'BAL': 'bal', 'BUF': 'buf', 'CAR': 'car', 
            'CHI': 'chi', 'CIN': 'cin', 'CLE': 'cle', 'DAL': 'dal', 'DEN': 'den',
            'DET': 'det', 'GB': 'gb', 'HOU': 'hou', 'IND': 'ind', 'JAX': 'jax', 
            'KC': 'kc', 'LV': 'lv', 'LAC': 'lac', 'LAR': 'lar', 'MIA': 'mia',
            'MIN': 'min', 'NE': 'ne', 'NO': 'no', 'NYG': 'nyg', 'NYJ': 'nyj', 
            'PHI': 'phi', 'PIT': 'pit', 'SF': 'sf', 'SEA': 'sea', 'TB': 'tb',
            'TEN': 'ten', 'WAS': 'was'
        }
        
        # Create team data with proper logo URLs
        team_data = []
        for team_abbr, logo_code in team_logo_mapping.items():
            # Use fallback logo if ESPN URL fails
            logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{logo_code}.png"
            fallback_url = f"https://static.www.nfl.com/league/api/clubs/logos/{team_abbr}.svg"
            
            team_data.append({
                'team_abbr': team_abbr,
                'team_name': team_abbr,
                'team_logo_espn': logo_url,
                'team_logo_fallback': fallback_url
            })
        
        return pd.DataFrame(team_data)
    
    def load_schedule_data(self, years: List[int]) -> pd.DataFrame:
        """Load schedule data for given years."""
        schedule_files = []
        
        for year in years:
            file_path = os.path.join(self.data_dir, f"schedule_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                schedule_files.append(df)
        
        if schedule_files:
            schedule_df = pd.concat(schedule_files, ignore_index=True)
        else:
            # Create a sample schedule for Week 1 if no file exists
            sample_games = [
                {'week': 1, 'away_team': 'DAL', 'home_team': 'PHI', 'spread_line': -3.0, 'total_line': 47.5, 'home_moneyline': -150, 'away_moneyline': 130},
                {'week': 1, 'away_team': 'KC', 'home_team': 'LAC', 'spread_line': -2.5, 'total_line': 52.5, 'home_moneyline': -120, 'away_moneyline': 100},
                {'week': 1, 'away_team': 'TB', 'home_team': 'ATL', 'spread_line': -1.0, 'total_line': 44.5, 'home_moneyline': -110, 'away_moneyline': -110},
                {'week': 1, 'away_team': 'CIN', 'home_team': 'CLE', 'spread_line': 2.5, 'total_line': 40.5, 'home_moneyline': 120, 'away_moneyline': -140},
                {'week': 1, 'away_team': 'MIA', 'home_team': 'IND', 'spread_line': -1.5, 'total_line': 45.0, 'home_moneyline': -105, 'away_moneyline': -115},
                {'week': 1, 'away_team': 'BUF', 'home_team': 'NYJ', 'spread_line': -6.5, 'total_line': 42.0, 'home_moneyline': 260, 'away_moneyline': -320},
                {'week': 1, 'away_team': 'HOU', 'home_team': 'LAR', 'spread_line': -3.0, 'total_line': 43.5, 'home_moneyline': -140, 'away_moneyline': 120},
                {'week': 1, 'away_team': 'MIN', 'home_team': 'CHI', 'spread_line': -1.0, 'total_line': 41.5, 'home_moneyline': -110, 'away_moneyline': -110},
                {'week': 1, 'away_team': 'NE', 'home_team': 'PIT', 'spread_line': 3.0, 'total_line': 37.5, 'home_moneyline': -140, 'away_moneyline': 120},
                {'week': 1, 'away_team': 'CAR', 'home_team': 'WAS', 'spread_line': 1.5, 'total_line': 43.0, 'home_moneyline': -105, 'away_moneyline': -115},
                {'week': 1, 'away_team': 'LV', 'home_team': 'DEN', 'spread_line': 2.5, 'total_line': 41.0, 'home_moneyline': -130, 'away_moneyline': 110},
                {'week': 1, 'away_team': 'TEN', 'home_team': 'NO', 'spread_line': 2.0, 'total_line': 42.5, 'home_moneyline': -115, 'away_moneyline': -105},
                {'week': 1, 'away_team': 'SF', 'home_team': 'DET', 'spread_line': -1.5, 'total_line': 51.5, 'home_moneyline': 105, 'away_moneyline': -125},
                {'week': 1, 'away_team': 'ARI', 'home_team': 'LAR', 'spread_line': 3.5, 'total_line': 49.0, 'home_moneyline': -160, 'away_moneyline': 140},
                {'week': 1, 'away_team': 'NYG', 'home_team': 'GB', 'spread_line': 3.0, 'total_line': 48.5, 'home_moneyline': -140, 'away_moneyline': 120},
                {'week': 1, 'away_team': 'SEA', 'home_team': 'TB', 'spread_line': 2.5, 'total_line': 46.5, 'home_moneyline': -130, 'away_moneyline': 110}
            ]
            schedule_df = pd.DataFrame(sample_games)
        
        return schedule_df

def check_data_freshness() -> Dict[str, datetime]:
    """Check when data files were last updated."""
    data_dir = "data"
    file_status = {}
    
    files_to_check = [
        "pbp_2022.parquet",
        "pbp_2023.parquet", 
        "pbp_2024.parquet",
        "schedule_2025.parquet"
    ]
    
    for file_name in files_to_check:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            file_status[file_name] = mod_time
        else:
            file_status[file_name] = "Missing"
    
    return file_status
