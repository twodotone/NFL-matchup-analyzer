import streamlit as st
import pandas as pd
import os
import traceback

# Global variable to cache data and prevent reloading
_cached_data = {}
_cache_timestamp = {}

def load_rolling_data(current_year):
    """
    Loads play-by-play data for the given year and up to 3 previous years,
    then combines them into a single DataFrame.
    """
    cache_key = f"rolling_data_{current_year}"
    
    # Check if we have cached data that's still fresh (5 minutes)
    import time
    current_time = time.time()
    if (cache_key in _cached_data and 
        cache_key in _cache_timestamp and 
        current_time - _cache_timestamp[cache_key] < 300):  # 5 minutes
        return _cached_data[cache_key]
    
    try:
        pbp_dfs = []
        years_loaded = []
        
        # Check data directory exists (no UI messages to prevent reruns)
        data_dir = "data"
        if not os.path.exists(data_dir):
            return pd.DataFrame()
        
        # Load current year data
        current_file_path = os.path.join("data", f"pbp_{current_year}.parquet")
        
        if os.path.exists(current_file_path):
            try:
                pbp_current = pd.read_parquet(current_file_path)
                pbp_dfs.append(pbp_current)
                years_loaded.append(current_year)
            except Exception:
                pass  # Silent fail to prevent rerun loops
        
        # Load data from up to 3 previous years
        for i in range(1, 4):  # Look back up to 3 years
            previous_year = current_year - i
            previous_file_path = os.path.join("data", f"pbp_{previous_year}.parquet")
            if os.path.exists(previous_file_path):
                try:
                    pbp_previous = pd.read_parquet(previous_file_path)
                    pbp_dfs.append(pbp_previous)
                    years_loaded.append(previous_year)
                except Exception:
                    pass  # Silent fail to prevent rerun loops

        if not pbp_dfs:
            return pd.DataFrame()

        # Combine the dataframes
        combined_df = pd.concat(pbp_dfs, ignore_index=True)
        
        # Quick validation
        if combined_df.empty:
            return pd.DataFrame()
        
        # Cache the result
        _cached_data[cache_key] = combined_df
        _cache_timestamp[cache_key] = current_time
        
        return combined_df
        
    except Exception:
        return pd.DataFrame()  # Silent fail to prevent rerun loops

def load_full_season_pbp(year):
    """
    Loads the full play-by-play dataset for a given year from a local parquet file.
    """
    cache_key = f"season_data_{year}"
    
    # Check cache
    import time
    current_time = time.time()
    if (cache_key in _cached_data and 
        cache_key in _cache_timestamp and 
        current_time - _cache_timestamp[cache_key] < 300):
        return _cached_data[cache_key]
    
    file_path = os.path.join("data", f"pbp_{year}.parquet")
    
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        pbp_df = pd.read_parquet(file_path)
        
        # Cache the result
        _cached_data[cache_key] = pbp_df
        _cache_timestamp[cache_key] = current_time
        
        return pbp_df
    except Exception:
        return pd.DataFrame()