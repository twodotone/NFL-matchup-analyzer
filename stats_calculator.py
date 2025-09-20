import pandas as pd
import streamlit as st
import sys

# Check if we're running in Streamlit or as a standalone script
try:
    # This will raise an exception if we're not in a Streamlit context
    st.runtime.get_instance()
    IN_STREAMLIT = True
except:
    IN_STREAMLIT = False

# Create a decorator that uses st.cache_data when in Streamlit, otherwise does nothing
def cache_wrapper(func):
    if IN_STREAMLIT:
        return st.cache_data(func)
    else:
        return func

@cache_wrapper
def get_last_n_games_pbp(full_pbp_df, team_abbr, n_games):
    """
    Extracts the play-by-play data for the last N regular season games for a given team,
    correctly handling data that spans multiple seasons.
    """
    team_games = full_pbp_df[((full_pbp_df['home_team'] == team_abbr) | (full_pbp_df['away_team'] == team_abbr)) & (full_pbp_df['season_type'] == 'REG')]
    if team_games.empty:
        return pd.DataFrame()
    
    # Sort by season and week to get the true chronological order of games
    unique_games = team_games[['game_id', 'season', 'week']].drop_duplicates().sort_values(
        by=['season', 'week'], ascending=[False, False]
    )
    
    last_n_game_ids = unique_games['game_id'].head(n_games).tolist()
    return full_pbp_df[full_pbp_df['game_id'].isin(last_n_game_ids)]

#@st.cache_data
@cache_wrapper
def calculate_explosive_play_rates(_pbp_df):
    """
    Calculates the rate of explosive plays (runs >= 10 yards, passes >= 20 yards).
    """
    pbp_df = _pbp_df.copy()
    explosive_runs = pbp_df[(pbp_df['play_type'] == 'run') & (pbp_df['yards_gained'] >= 10)]
    explosive_passes = pbp_df[(pbp_df['play_type'] == 'pass') & (pbp_df['yards_gained'] >= 20)]
    total_runs = pbp_df[pbp_df['play_type'] == 'run'].shape[0]
    total_passes = pbp_df[pbp_df['play_type'] == 'pass'].shape[0]
    total_plays = total_runs + total_passes
    total_explosive_plays = len(explosive_runs) + len(explosive_passes)
    return total_explosive_plays / total_plays if total_plays > 0 else 0

#@st.cache_data
@cache_wrapper
def calculate_granular_epa_stats(_pbp_df, team_abbr, use_sos_adjustment=True):
    """
    Calculates opponent-adjusted EPA stats for Offense and Defense.
    Can be toggled to calculate raw (unadjusted) EPA.
    """
    if _pbp_df.empty:
        return {}

    pbp_reg = _pbp_df[_pbp_df['season_type'] == 'REG'].copy()
    if pbp_reg.empty:
        return {}
    
    stats = {}
    
    # --- Offense and Defense Calculations ---
    pbp_off_def = pbp_reg[(pbp_reg['play_type'] == 'pass') | (pbp_reg['play_type'] == 'run')].copy()
    if not pbp_off_def.empty:
        # Calculate league-wide baselines for SOS adjustment
        lg_off_rush_epa = pbp_off_def[pbp_off_def['play_type'] == 'run'].groupby('posteam')['epa'].mean().rename('lg_off_rush_epa')
        lg_off_pass_epa = pbp_off_def[pbp_off_def['play_type'] == 'pass'].groupby('posteam')['epa'].mean().rename('lg_off_pass_epa')
        lg_def_rush_epa = pbp_off_def[pbp_off_def['play_type'] == 'run'].groupby('defteam')['epa'].mean().rename('lg_def_rush_epa')
        lg_def_pass_epa = pbp_off_def[pbp_off_def['play_type'] == 'pass'].groupby('defteam')['epa'].mean().rename('lg_def_pass_epa')

        off_plays = pbp_off_def[pbp_off_def['posteam'] == team_abbr]
        def_plays = pbp_off_def[pbp_off_def['defteam'] == team_abbr]

        if not off_plays.empty:
            total_plays = len(off_plays)
            rush_plays = off_plays[off_plays['play_type'] == 'run']
            pass_plays = off_plays[off_plays['play_type'] == 'pass']
            stats['Rush_Pct'] = len(rush_plays) / total_plays if total_plays > 0 else 0
            stats['Pass_Pct'] = len(pass_plays) / total_plays if total_plays > 0 else 0
            
            if not rush_plays.empty:
                if use_sos_adjustment:
                    # Join opponent defensive averages to each play
                    rush_plays_adj = rush_plays.merge(lg_def_rush_epa, left_on='defteam', right_index=True, how='left')
                    # Subtract the opponent's average from each play's EPA
                    stats['Off_Rush_EPA'] = (rush_plays_adj['epa'] - rush_plays_adj['lg_def_rush_epa']).mean()
                else:
                    stats['Off_Rush_EPA'] = rush_plays['epa'].mean()

            if not pass_plays.empty:
                if use_sos_adjustment:
                    pass_plays_adj = pass_plays.merge(lg_def_pass_epa, left_on='defteam', right_index=True, how='left')
                    stats['Off_Pass_EPA'] = (pass_plays_adj['epa'] - pass_plays_adj['lg_def_pass_epa']).mean()
                else:
                    stats['Off_Pass_EPA'] = pass_plays['epa'].mean()

            stats['Off_Explosive_Rate'] = calculate_explosive_play_rates(off_plays)

        if not def_plays.empty:
            rush_plays_faced = def_plays[def_plays['play_type'] == 'run']
            pass_plays_faced = def_plays[def_plays['play_type'] == 'pass']

            if not rush_plays_faced.empty:
                if use_sos_adjustment:
                    rush_plays_faced_adj = rush_plays_faced.merge(lg_off_rush_epa, left_on='posteam', right_index=True, how='left')
                    stats['Def_Rush_EPA'] = (rush_plays_faced_adj['epa'] - rush_plays_faced_adj['lg_off_rush_epa']).mean()
                else:
                    stats['Def_Rush_EPA'] = rush_plays_faced['epa'].mean()

            if not pass_plays_faced.empty:
                if use_sos_adjustment:
                    pass_plays_faced_adj = pass_plays_faced.merge(lg_off_pass_epa, left_on='posteam', right_index=True, how='left')
                    stats['Def_Pass_EPA'] = (pass_plays_faced_adj['epa'] - pass_plays_faced_adj['lg_off_pass_epa']).mean()
                else:
                    stats['Def_Pass_EPA'] = pass_plays_faced['epa'].mean()
                
            stats['Def_Explosive_Rate'] = calculate_explosive_play_rates(def_plays)
            
    # --- Pace of Play Calculation ---
    team_plays = pbp_reg[(pbp_reg['posteam'] == team_abbr) & ((pbp_reg['play_type'] == 'pass') | (pbp_reg['play_type'] == 'run'))]
    if not team_plays.empty:
        # Calculate plays per game
        games_played = team_plays['game_id'].nunique()
        total_plays = len(team_plays)
        stats['plays_per_game'] = total_plays / games_played if games_played > 0 else 65 # Default if no games

    return stats

def calculate_weighted_stats(stats_std, stats_recent, full_season_weight, recent_form_weight):
    """
    Calculates a weighted average of full-season and recent stats.
    """
    all_keys = set(stats_std.keys()) | set(stats_recent.keys())
    stats_w = {}
    for key in all_keys:
        # Ensure 'plays_per_game' uses the full season data, not recency-weighted
        if key == 'plays_per_game':
            stats_w[key] = stats_std.get(key, 65)
        else:
            stats_w[key] = (stats_std.get(key, 0) * full_season_weight) + (stats_recent.get(key, 0) * recent_form_weight)
    return stats_w

def calculate_tiered_historical_stats(team_abbr, pbp_df, current_year, recent_games_window=8, recent_form_weight=0.3):
    """
    Calculates team stats using a tiered weighting system that:
    1. Weights different years of historical data with declining importance
    2. Applies recency weighting to the most recent games
    
    Parameters:
    -----------
    team_abbr : str
        The team abbreviation to calculate stats for
    pbp_df : DataFrame
        The combined play-by-play data across multiple years
    current_year : int
        The current year being analyzed
    recent_games_window : int, optional
        Number of recent games to apply recency weighting to
    recent_form_weight : float, optional
        Weight to apply to recent games (between 0 and 1)
        
    Returns:
    --------
    dict
        Dictionary of weighted team statistics
    """
    if pbp_df.empty:
        return {}
    
    # Filter for regular season only
    reg_games = pbp_df[pbp_df['season_type'] == 'REG'].copy()
    if reg_games.empty:
        return {}
    
    # Get available seasons
    available_seasons = sorted(reg_games['season'].unique(), reverse=True)
    
    # Setup year weights (declining importance)
    year_weights = {}
    base_weights = [0.5, 0.3, 0.2]  # Current year (50%), Previous year (30%), Two years ago (20%)
    
    # Assign weights to available years
    for i, year in enumerate(available_seasons):
        if i < len(base_weights):
            year_weights[year] = base_weights[i]
        else:
            # Any additional years get minimal weight
            year_weights[year] = 0.0
    
    # Normalize weights if some years are missing
    if year_weights:
        total_weight = sum(year_weights.values())
        if total_weight > 0:
            for year in year_weights:
                year_weights[year] /= total_weight
    
    # Calculate full stats for each year
    year_stats = {}
    for year in available_seasons:
        if year_weights.get(year, 0) > 0:
            year_data = reg_games[reg_games['season'] == year]
            year_stats[year] = calculate_granular_epa_stats(year_data, team_abbr)
    
    # Combine yearly stats with appropriate weights
    combined_stats = {}
    if year_stats:
        all_keys = set()
        for year_stat in year_stats.values():
            all_keys.update(year_stat.keys())
        
        # Weight and combine stats from different years
        for key in all_keys:
            if key == 'plays_per_game':
                # For plays per game, use the most recent year's value
                most_recent_year = available_seasons[0] if available_seasons else current_year
                combined_stats[key] = year_stats.get(most_recent_year, {}).get(key, 65)
            else:
                # Weight other stats by year
                weighted_sum = 0
                for year, weight in year_weights.items():
                    if year in year_stats:
                        weighted_sum += year_stats[year].get(key, 0) * weight
                combined_stats[key] = weighted_sum
    
    # Apply recency weighting if we have current year data
    if current_year in available_seasons:
        current_year_data = reg_games[reg_games['season'] == current_year]
        recent_games_pbp = get_last_n_games_pbp(current_year_data, team_abbr, recent_games_window)
        
        if not recent_games_pbp.empty:
            recent_stats = calculate_granular_epa_stats(recent_games_pbp, team_abbr)
            
            # Standard weight is what remains after recent form weight
            standard_weight = 1 - recent_form_weight
            
            # Combine with recency weighting
            for key in set(combined_stats.keys()) | set(recent_stats.keys()):
                if key == 'plays_per_game':
                    # Don't apply recency weighting to plays per game
                    continue
                else:
                    combined_stats[key] = ((combined_stats.get(key, 0) * standard_weight) + 
                                         (recent_stats.get(key, 0) * recent_form_weight))
    
    return combined_stats

def calculate_matchup_specific_weights(home_stats, away_stats):
    """
    Calculate weights that reflect the specific matchup dynamics between teams.
    This approach weights offense and defense based on relative team strengths.
    """
    # Default EPA values if keys are missing
    default_epa = 0.0
    
    # Get offensive and defensive EPA values
    home_off_epa = max(min(home_stats.get('Off_Pass_EPA', default_epa), 0.5), -0.5)
    home_def_epa = max(min(home_stats.get('Def_Pass_EPA', default_epa), 0.5), -0.5)
    away_off_epa = max(min(away_stats.get('Off_Pass_EPA', default_epa), 0.5), -0.5)
    away_def_epa = max(min(away_stats.get('Def_Pass_EPA', default_epa), 0.5), -0.5)
    
    # Normalize to 0-1 range (approximating percentiles)
    # For offense, higher is better; for defense, lower (more negative) is better
    home_off_strength = (home_off_epa + 0.5) / 1.0
    home_def_strength = 1.0 - ((home_def_epa + 0.5) / 1.0)  # Inverse for defense
    away_off_strength = (away_off_epa + 0.5) / 1.0
    away_def_strength = 1.0 - ((away_def_epa + 0.5) / 1.0)  # Inverse for defense
    
    # Calculate relative strength of each unit in the matchup
    home_off_vs_away_def = (home_off_strength + away_def_strength) / 2
    away_off_vs_home_def = (away_off_strength + home_def_strength) / 2
    
    # Calculate weights (allowing 55-75% range for offense)
    home_off_weight = 0.55 + (0.2 * home_off_vs_away_def)
    home_def_weight = 1.0 - home_off_weight
    away_off_weight = 0.55 + (0.2 * away_off_vs_home_def)
    away_def_weight = 1.0 - away_off_weight
    
    return {
        'home_off_weight': home_off_weight,
        'home_def_weight': home_def_weight,
        'away_off_weight': away_off_weight,
        'away_def_weight': away_def_weight
    }

def generate_stable_matchup_line(home_stats, away_stats, return_weights=False, home_field_advantage=None, 
                            pbp_df=None, home_team=None, away_team=None, game_info=None):
    """
    Generates a predicted line based on the granular EPA matchup engine with
    dynamic offense/defense weighting based on team strengths.
    HFA can be added in three ways:
    1. Explicitly passed as home_field_advantage parameter (backward compatibility)
    2. Using pbp_df, home_team, and away_team to calculate dynamic HFA
    3. Using game_info to provide additional context for HFA calculation
    """
    # Calculate dynamic weights based on matchup
    weights = calculate_matchup_specific_weights(home_stats, away_stats)
    
    # Use team average passing EPA
    home_pass_offense_epa = home_stats.get('Off_Pass_EPA', 0)
    away_pass_offense_epa = away_stats.get('Off_Pass_EPA', 0)

    # Apply weights to offensive and defensive contributions
    home_rush_outcome = (home_stats.get('Off_Rush_EPA', 0) * weights['home_off_weight'] + 
                         away_stats.get('Def_Rush_EPA', 0) * weights['away_def_weight'])
    
    home_pass_outcome = (home_pass_offense_epa * weights['home_off_weight'] + 
                         away_stats.get('Def_Pass_EPA', 0) * weights['away_def_weight'])
    
    home_exp_outcome_per_play = (home_rush_outcome * home_stats.get('Rush_Pct', 0.5)) + \
                                (home_pass_outcome * home_stats.get('Pass_Pct', 0.5))

    away_rush_outcome = (away_stats.get('Off_Rush_EPA', 0) * weights['away_off_weight'] + 
                         home_stats.get('Def_Rush_EPA', 0) * weights['home_def_weight'])
    
    away_pass_outcome = (away_pass_offense_epa * weights['away_off_weight'] + 
                         home_stats.get('Def_Pass_EPA', 0) * weights['home_def_weight'])
    
    away_exp_outcome_per_play = (away_rush_outcome * away_stats.get('Rush_Pct', 0.5)) + \
                                (away_pass_outcome * away_stats.get('Pass_Pct', 0.5))

    # Dynamic Pace of Play
    expected_plays = (home_stats.get('plays_per_game', 65) + away_stats.get('plays_per_game', 65)) / 2

    net_adv_per_play = home_exp_outcome_per_play - away_exp_outcome_per_play
    neutral_margin_off_def = net_adv_per_play * expected_plays
    
    # Calculate HFA using the most appropriate method available
    hfa_value = 0
    hfa_components = None
    
    # Option 1: Explicitly provided HFA value (highest priority, for backward compatibility)
    if home_field_advantage is not None:
        hfa_value = home_field_advantage
    # Option 2: Calculate dynamic HFA if we have all the required inputs
    elif pbp_df is not None and home_team is not None and away_team is not None:
        # Dynamically import to avoid circular import
        try:
            from dynamic_hfa import calculate_dynamic_hfa
            
            # For backtesting, limit HFA range to improve performance
            if game_info and 'hfa_range' in game_info:
                hfa_range = game_info['hfa_range']
                if hfa_range == 'low':
                    hfa_value = 0.5  # Very low HFA
                elif hfa_range == 'medium':
                    hfa_value = 1.5  # Medium HFA
                elif hfa_range == 'high':
                    hfa_value = 2.5  # Higher HFA
                else:
                    # Calculate dynamic HFA with components for analysis
                    hfa_value, hfa_components = calculate_dynamic_hfa(pbp_df, home_team, away_team, game_info, return_components=True)
            else:
                # Calculate dynamic HFA with components for analysis
                hfa_value, hfa_components = calculate_dynamic_hfa(pbp_df, home_team, away_team, game_info, return_components=True)
        except ImportError:
            # Fallback if dynamic_hfa module is not available
            from hfa_data import HFA_DATA
            hfa_value = HFA_DATA.get(home_team, 2.0)  # Default to standard HFA
    
    # Add HFA to the neutral field projection
    final_margin = neutral_margin_off_def + hfa_value

    # The final margin IS the spread. A positive margin means the home team is favored.
    if return_weights:
        return final_margin, weights, hfa_value, hfa_components
    else:
        return final_margin