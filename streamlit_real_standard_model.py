"""
Streamlit-compatible Real Standard Model
Uses the actual tiered historical stats and stable matchup line logic from app.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class StreamlitRealStandardModel:
    """
    Streamlit-compatible version of the real Standard Model using tiered historical stats
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.pbp_data = None
        self.current_week = None
        self.current_year = None
        
    def load_standard_data(self, current_year: int, current_week: int) -> None:
        """Load data for the real standard model."""
        import os
        
        print(f"Loading real standard model data for Week {current_week} of {current_year} season")
        
        self.current_week = current_week
        self.current_year = current_year
        
        # Load play-by-play data from parquet files (using all available years)
        dfs = []
        years_to_load = [2022, 2023, 2024]  # Load all historical data
        
        # Check if 2025 data exists and add it dynamically (same as Simple Model)
        if os.path.exists(os.path.join(self.data_dir, "pbp_2025.parquet")):
            try:
                # Test if 2025 file has actual data
                import pandas as pd
                df_2025 = pd.read_parquet(os.path.join(self.data_dir, "pbp_2025.parquet"))
                if len(df_2025) > 0:
                    years_to_load.append(2025)
                    print(f"✅ Standard Model: Loading with 2025 data ({len(df_2025)} plays)")
                else:
                    print("ℹ️ Standard Model: 2025 data file exists but is empty - using historical data only")
            except Exception as e:
                print(f"⚠️ Standard Model: 2025 data file exists but couldn't be loaded: {str(e)}")
        else:
            print("ℹ️ Standard Model: No 2025 data available yet - using historical data only")
        
        for year in years_to_load:
            file_path = os.path.join(self.data_dir, f"pbp_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                if len(df) > 0:
                    dfs.append(df)
                    print(f"Loaded {year}: {len(df):,} plays")
                else:
                    print(f"Warning: {year} data is empty")
            else:
                print(f"Warning: {file_path} not found")
        
        if dfs:
            self.pbp_data = pd.concat(dfs, ignore_index=True)
            print(f"✅ Successfully loaded real standard model data")
            print(f"Total plays: {len(self.pbp_data):,}")
        else:
            raise FileNotFoundError("No play-by-play data files found")
    
    def predict_spread_standard(self, home_team: str, away_team: str, week: int, year: int) -> Tuple[float, Dict]:
        """Predict spread using the real Standard Model logic from app.py."""
        if self.pbp_data is None:
            raise ValueError("No play-by-play data loaded")
        
        # Import the real functions from stats_calculator
        from stats_calculator import calculate_tiered_historical_stats, generate_stable_matchup_line
        
        # Filter data up to prediction point (same as original app.py)
        pbp_data_for_stats = self.pbp_data[
            (self.pbp_data['season'] < year) | 
            ((self.pbp_data['season'] == year) & (self.pbp_data['week'] < week))
        ].copy()
        
        if pbp_data_for_stats.empty:
            raise ValueError("No historical data available for prediction")
        
        # Use the real tiered historical stats system (same parameters as app.py)
        recent_games_window = 8
        recent_form_weight = 0.30
        
        # Calculate stats with tiered historical weighting
        away_stats_w = calculate_tiered_historical_stats(
            away_team, 
            pbp_data_for_stats, 
            year,
            recent_games_window, 
            recent_form_weight
        )
        
        home_stats_w = calculate_tiered_historical_stats(
            home_team, 
            pbp_data_for_stats, 
            year,
            recent_games_window, 
            recent_form_weight
        )
        
        # Prepare game info for HFA calculation (same as app.py)
        game_info = {'current_season': year}
        
        # For 2025 or later seasons, load older seasons for HFA calculation
        if year >= 2025:
            import os
            hfa_years = []
            for i in range(2, 4):  # Look back to seasons 2-3 years ago
                old_year = year - i
                old_file_path = os.path.join(self.data_dir, f"pbp_{old_year}.parquet")
                if os.path.exists(old_file_path):
                    hfa_years.append(old_year)
            
            if hfa_years:
                game_info['historical_seasons'] = hfa_years
        
        # Generate the stable matchup line (same as app.py)
        model_result, model_weights, hfa_value, hfa_components = generate_stable_matchup_line(
            home_stats_w, away_stats_w, return_weights=True, 
            pbp_df=pbp_data_for_stats, home_team=home_team, away_team=away_team, 
            game_info=game_info
        )
        
        # Convert to spread (same as app.py)
        model_home_spread = -model_result
        
        # Model details for transparency
        model_details = {
            'home_stats': home_stats_w,
            'away_stats': away_stats_w,
            'model_weights': model_weights,
            'hfa_value': hfa_value,
            'hfa_components': hfa_components,
            'model_result_raw': model_result,
            'predicted_spread': model_home_spread,
            'recent_games_window': recent_games_window,
            'recent_form_weight': recent_form_weight,
            'week': week,
            'year': year,
            'method': 'Real Standard Model with Tiered Historical Stats'
        }
        
        return round(model_home_spread, 1), model_details
