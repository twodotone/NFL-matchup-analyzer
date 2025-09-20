"""
Streamlit-compatible Simple NFL Model
Uses local parquet files instead of nfl_data_py downloads
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StreamlitSimpleNFLModel:
    """
    Streamlit-compatible version of SimpleNFLModel using local parquet files
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.pbp_data = None
        
    def load_data_from_parquet(self, years: list) -> None:
        """Load play-by-play data from parquet files with memory optimization."""
        import os
        
        dfs = []
        loaded_years = []
        
        for year in years:
            file_path = os.path.join(self.data_dir, f"pbp_{year}.parquet")
            if os.path.exists(file_path):
                # Load only essential columns to reduce memory
                try:
                    df = pd.read_parquet(file_path, columns=[
                        'season', 'week', 'posteam', 'defteam', 'play_type', 'epa'
                    ])
                    # Filter to only pass/run plays with valid EPA
                    df = df[
                        (df['play_type'].isin(['pass', 'run'])) & 
                        (~df['epa'].isna())
                    ].copy()
                    
                    if len(df) > 0:  # Only add if there's actual data
                        dfs.append(df)
                        loaded_years.append(year)
                        print(f"Loaded {year}: {len(df):,} plays")
                    else:
                        print(f"Skipped {year}: file exists but no valid plays")
                        
                except Exception as e:
                    print(f"Error loading {year}: {e}")
                    # Try fallback to full load
                    try:
                        df = pd.read_parquet(file_path)
                        if len(df) > 0:
                            dfs.append(df)
                            loaded_years.append(year)
                            print(f"Loaded {year} (fallback): {len(df):,} plays")
                    except Exception as e2:
                        print(f"Failed to load {year} completely: {e2}")
            else:
                print(f"File not found: {file_path}")
        
        if dfs:
            self.pbp_data = pd.concat(dfs, ignore_index=True)
            print(f"Total plays loaded from years {loaded_years}: {len(self.pbp_data):,}")
        else:
            raise FileNotFoundError("No play-by-play data files found or all files were empty")
    
    def calculate_team_epa_stats(self, team: str, pbp_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate EPA statistics for a team."""
        # Get plays where team is on offense
        offensive_plays = pbp_df[
            (pbp_df['posteam'] == team) & 
            (pbp_df['play_type'].isin(['pass', 'run'])) &
            (~pbp_df['epa'].isna())
        ]
        
        # Get plays where team is on defense
        defensive_plays = pbp_df[
            (pbp_df['defteam'] == team) & 
            (pbp_df['play_type'].isin(['pass', 'run'])) &
            (~pbp_df['epa'].isna())
        ]
        
        # Check for minimum data requirements
        min_plays = 10  # Require at least 10 plays for meaningful stats
        
        if len(offensive_plays) < min_plays:
            print(f"Warning: {team} has only {len(offensive_plays)} offensive plays (minimum {min_plays})")
            if len(offensive_plays) == 0:
                return {}
        
        if len(defensive_plays) < min_plays:
            print(f"Warning: {team} has only {len(defensive_plays)} defensive plays (minimum {min_plays})")
            if len(defensive_plays) == 0:
                return {}
        
        try:
            off_epa = offensive_plays['epa'].mean()
            def_epa = defensive_plays['epa'].mean()
            net_epa = off_epa - def_epa
            
            # Validate EPA values are reasonable
            if abs(off_epa) > 1.0 or abs(def_epa) > 1.0:
                print(f"Warning: {team} has extreme EPA values - Off: {off_epa:.3f}, Def: {def_epa:.3f}")
            
            return {
                'off_epa_per_play': off_epa,
                'def_epa_per_play': def_epa,
                'net_epa_per_play': net_epa,
                'offensive_plays': len(offensive_plays),
                'defensive_plays': len(defensive_plays)
            }
        except Exception as e:
            print(f"Error calculating EPA for {team}: {e}")
            return {}
    
    def predict_spread(self, home_team: str, away_team: str, week: int, year: int) -> Tuple[float, Dict]:
        """Predict the spread for a game."""
        if self.pbp_data is None:
            raise ValueError("No play-by-play data loaded")
        
        # Filter data up to prediction point
        pbp_for_prediction = self.pbp_data[
            (self.pbp_data['season'] < year) | 
            ((self.pbp_data['season'] == year) & (self.pbp_data['week'] < week))
        ].copy()
        
        if pbp_for_prediction.empty:
            raise ValueError("No historical data available for prediction")
        
        # Calculate team stats
        home_stats = self.calculate_team_epa_stats(home_team, pbp_for_prediction)
        away_stats = self.calculate_team_epa_stats(away_team, pbp_for_prediction)
        
        if not home_stats or not away_stats:
            raise ValueError("Unable to calculate team statistics")
        
        # Calculate EPA advantage
        home_net_epa = home_stats['net_epa_per_play']
        away_net_epa = away_stats['net_epa_per_play']
        epa_advantage = home_net_epa - away_net_epa
        
        # Convert to points (NFL average ~25 points per EPA)
        points_per_epa = 25
        
        # Home field advantage
        home_field_advantage = 2.5
        
        # Calculate spread (negative = home favored)
        predicted_spread = -(epa_advantage * points_per_epa) - home_field_advantage
        
        # Model details for transparency
        model_details = {
            'home_stats': home_stats,
            'away_stats': away_stats,
            'epa_advantage': epa_advantage,
            'home_field_advantage': home_field_advantage,
            'predicted_spread': predicted_spread
        }
        
        return round(predicted_spread, 1), model_details
    
    def predict_total(self, home_team: str, away_team: str, week: int, year: int) -> Tuple[float, Dict]:
        """Predict the total points for a game."""
        if self.pbp_data is None:
            raise ValueError("No play-by-play data loaded")
        
        # Filter data up to prediction point
        pbp_for_prediction = self.pbp_data[
            (self.pbp_data['season'] < year) | 
            ((self.pbp_data['season'] == year) & (self.pbp_data['week'] < week))
        ].copy()
        
        if pbp_for_prediction.empty:
            raise ValueError("No historical data available for prediction")
        
        # Calculate team stats
        home_stats = self.calculate_team_epa_stats(home_team, pbp_for_prediction)
        away_stats = self.calculate_team_epa_stats(away_team, pbp_for_prediction)
        
        if not home_stats or not away_stats:
            raise ValueError("Unable to calculate team statistics")
        
        # Base scoring expectation (NFL average)
        league_avg_points_per_game = 23
        baseline_epa = 0.02
        
        # Calculate expected points for each team
        home_epa = home_stats['off_epa_per_play']
        away_epa = away_stats['off_epa_per_play']
        home_def_epa = home_stats['def_epa_per_play']
        away_def_epa = away_stats['def_epa_per_play']
        
        # Expected points = base + offensive advantage - defensive resistance
        home_expected = league_avg_points_per_game + (home_epa - baseline_epa) * 30 - (away_def_epa - baseline_epa) * 30
        away_expected = league_avg_points_per_game + (away_epa - baseline_epa) * 30 - (home_def_epa - baseline_epa) * 30
        
        # Ensure reasonable bounds
        home_expected = max(10, min(50, home_expected))
        away_expected = max(10, min(50, away_expected))
        
        total_predicted = home_expected + away_expected
        
        # Total details
        total_details = {
            'home_expected_points': home_expected,
            'away_expected_points': away_expected,
            'total_predicted': total_predicted,
            'home_off_epa': home_epa,
            'away_off_epa': away_epa,
            'home_def_epa': home_def_epa,
            'away_def_epa': away_def_epa
        }
        
        return round(total_predicted, 1), total_details
