import pandas as pd
import os
from datetime import datetime, timezone

def get_game_line_movement(home_team, away_team, week, year):
    """
    Get line movement data for a specific game using multiple methods:
    1. Captured early lines (for future weeks)
    2. Live comparison (local vs fresh nfl_data_py)
    """
    
    # Method 1: Try captured early lines first
    captured_movement = load_captured_line_movement(home_team, away_team, week, year)
    if captured_movement['found']:
        return captured_movement
    
    # Method 2: Try live comparison (local vs fresh data)
    try:
        from live_line_movement import get_live_line_movement
        live_movement = get_live_line_movement(home_team, away_team, week, year)
        if live_movement['found']:
            # Convert to standard format
            from datetime import datetime
            return {
                'found': True,
                'early_spread': live_movement['opening_spread'],
                'current_spread': live_movement['current_spread'],
                'spread_movement': live_movement['spread_movement'],
                'early_total': live_movement['opening_total'],
                'current_total': live_movement['current_total'],
                'total_movement': live_movement['total_movement'],
                'capture_date': live_movement['opening_time'],
                'days_elapsed': (datetime.now() - live_movement['opening_time']).days,
                'data_source': 'live_tracking',
                'movement_type': 'live_comparison'
            }
        else:
            return {'found': False, 'reason': f'Live tracking failed: {live_movement.get("reason", "Unknown")}'} 
    except ImportError as e:
        return {'found': False, 'reason': f'Live tracking import failed: {str(e)}'}
    except Exception as e:
        return {'found': False, 'reason': f'Live tracking error: {str(e)}'}

    return {'found': False, 'reason': 'No line movement data available'}

def load_captured_line_movement(home_team, away_team, week, year):
    """
    Load historical line movement data for a specific week from captured data
    Returns both early lines and current lines for comparison
    """
    
    try:
        # Load saved early lines for this week
        early_lines_file = f"data/lookahead_lines/week_{week}_early_lines_{year}.parquet"
        
        if not os.path.exists(early_lines_file):
            return {'found': False, 'reason': f'No early lines saved for Week {week}'}
        
        early_lines_df = pd.read_parquet(early_lines_file)
        
        # Load current schedule to get latest lines
        current_schedule_file = f"data/schedule_{year}.parquet"
        if not os.path.exists(current_schedule_file):
            return {'found': False, 'reason': 'Current schedule data not available'}
        
        current_schedule_df = pd.read_parquet(current_schedule_file)
        current_week_games = current_schedule_df[current_schedule_df['week'] == week]
        
        # Find this specific game in early lines
        early_game = early_lines_df[
            (early_lines_df['home_team'] == home_team) &
            (early_lines_df['away_team'] == away_team)
        ]
        
        if early_game.empty:
            return {'found': False, 'reason': 'Game not found in captured early lines'}
        
        early_game = early_game.iloc[0]
        
        # Find corresponding current game
        current_game = current_week_games[
            (current_week_games['home_team'] == home_team) &
            (current_week_games['away_team'] == away_team)
        ]
        
        if current_game.empty:
            return {'found': False, 'reason': 'Game not found in current schedule'}
        
        current_game = current_game.iloc[0]
        
        # Calculate movement
        spread_movement = current_game['spread_line'] - early_game['spread_line']
        total_movement = current_game['total_line'] - early_game['total_line'] if pd.notna(current_game['total_line']) and pd.notna(early_game['total_line']) else 0
        
        return {
            'found': True,
            'early_spread': early_game['spread_line'],
            'current_spread': current_game['spread_line'],
            'spread_movement': spread_movement,
            'early_total': early_game['total_line'],
            'current_total': current_game['total_line'],
            'total_movement': total_movement,
            'capture_date': early_game['capture_date'],
            'days_elapsed': (datetime.now(timezone.utc) - pd.to_datetime(early_game['capture_date'])).days,
            'data_source': 'captured_lines',
            'movement_type': 'early_vs_current'
        }
        
    except Exception as e:
        return {'found': False, 'reason': f'Error loading captured line movement: {str(e)}'}

def load_historical_line_movement(week, year):
    """
    Load historical line movement data for a specific week
    Returns both early lines and current lines for comparison
    """
    
    try:
        # Load saved early lines for this week
        early_lines_file = f"data/lookahead_lines/week_{week}_early_lines_{year}.parquet"
        
        if not os.path.exists(early_lines_file):
            return {'found': False, 'reason': f'No early lines saved for Week {week}'}
        
        early_lines_df = pd.read_parquet(early_lines_file)
        
        # Load current schedule to get latest lines
        current_schedule_file = f"data/schedule_{year}.parquet"
        if not os.path.exists(current_schedule_file):
            return {'found': False, 'reason': 'Current schedule data not available'}
        
        current_schedule_df = pd.read_parquet(current_schedule_file)
        current_week_games = current_schedule_df[current_schedule_df['week'] == week]
        
        # Merge to compare early vs current lines
        movement_data = []
        
        for _, early_game in early_lines_df.iterrows():
            # Find corresponding current game
            current_game = current_week_games[
                (current_week_games['home_team'] == early_game['home_team']) &
                (current_week_games['away_team'] == early_game['away_team'])
            ]
            
            if not current_game.empty:
                current = current_game.iloc[0]
                
                # Calculate movement
                spread_movement = current['spread_line'] - early_game['spread_line']
                total_movement = current['total_line'] - early_game['total_line'] if pd.notna(current['total_line']) and pd.notna(early_game['total_line']) else 0
                
                movement_data.append({
                    'away_team': early_game['away_team'],
                    'home_team': early_game['home_team'],
                    'early_spread': early_game['spread_line'],
                    'current_spread': current['spread_line'],
                    'spread_movement': spread_movement,
                    'early_total': early_game['total_line'],
                    'current_total': current['total_line'],
                    'total_movement': total_movement,
                    'capture_date': early_game['capture_date'],
                    'days_elapsed': (datetime.now(timezone.utc) - pd.to_datetime(early_game['capture_date'])).days
                })
        
        return {
            'found': True,
            'movement_data': pd.DataFrame(movement_data),
            'total_games': len(movement_data),
            'capture_date': early_lines_df.iloc[0]['capture_date'] if len(early_lines_df) > 0 else None
        }
        
    except Exception as e:
        return {'found': False, 'reason': f'Error loading line movement data: {str(e)}'}

def get_game_line_movement(home_team, away_team, week, year):
    """
    Get line movement data for a specific game
    """
    
    movement_data = load_historical_line_movement(week, year)
    
    if not movement_data['found']:
        return movement_data
    
    # Find this specific game
    game_movement = movement_data['movement_data'][
        (movement_data['movement_data']['home_team'] == home_team) &
        (movement_data['movement_data']['away_team'] == away_team)
    ]
    
    if game_movement.empty:
        return {'found': False, 'reason': 'Game not found in movement data'}
    
    game_data = game_movement.iloc[0]
    
    return {
        'found': True,
        'early_spread': game_data['early_spread'],
        'current_spread': game_data['current_spread'],
        'spread_movement': game_data['spread_movement'],
        'early_total': game_data['early_total'],
        'current_total': game_data['current_total'],
        'total_movement': game_data['total_movement'],
        'capture_date': game_data['capture_date'],
        'days_elapsed': game_data['days_elapsed']
    }

def analyze_week_line_movement(week, year):
    """
    Analyze overall line movement patterns for a week using best available method
    """
    
    # Try captured data first
    movement_data = load_historical_line_movement(week, year)
    
    if movement_data['found']:
        df = movement_data['movement_data']
        
        # Calculate movement statistics
        significant_spread_moves = df[abs(df['spread_movement']) >= 1.0]
        significant_total_moves = df[abs(df['total_movement']) >= 2.0]
        
        analysis = {
            'found': True,
            'total_games': len(df),
            'significant_spread_moves': len(significant_spread_moves),
            'significant_total_moves': len(significant_total_moves),
            'avg_spread_movement': df['spread_movement'].abs().mean(),
            'avg_total_movement': df['total_movement'].abs().mean(),
            'max_spread_movement': df['spread_movement'].abs().max(),
            'max_total_movement': df['total_movement'].abs().max(),
            'games_moving_toward_home': len(df[df['spread_movement'] < 0]),
            'games_moving_toward_away': len(df[df['spread_movement'] > 0]),
            'capture_date': movement_data['capture_date'],
            'biggest_movers': df.nlargest(3, 'spread_movement', keep='all')[['away_team', 'home_team', 'spread_movement']].to_dict('records')
        }
        
        return analysis
    
    # Try live comparison method
    try:
        from live_line_movement import analyze_live_week_movement
        live_analysis = analyze_live_week_movement(week, year)
        if live_analysis['found']:
            return {
                'found': True,
                'total_games': live_analysis['total_games'],
                'significant_spread_moves': live_analysis['significant_spread_moves'],
                'significant_total_moves': live_analysis['significant_total_moves'],
                'avg_spread_movement': live_analysis['avg_spread_movement'],
                'avg_total_movement': live_analysis['avg_total_movement'],
                'max_spread_movement': live_analysis['max_spread_movement'],
                'max_total_movement': live_analysis['max_total_movement'],
                'games_moving_toward_home': len(live_analysis['movements_data'][live_analysis['movements_data']['spread_movement'] < 0]),
                'games_moving_toward_away': len(live_analysis['movements_data'][live_analysis['movements_data']['spread_movement'] > 0]),
                'biggest_movers': live_analysis['biggest_movers'],
                'data_source': 'live_tracking'
            }
    except ImportError:
        pass
    
    return {'found': False, 'reason': 'No line movement analysis available'}

if __name__ == "__main__":
    # Test the functions
    print("🧪 Testing line movement functions...")
    
    # Test Week 4 analysis (should work since we just captured it)
    week4_analysis = analyze_week_line_movement(4, 2025)
    print(f"\nWeek 4 Analysis: {week4_analysis}")
    
    # Test specific game movement
    buf_game_movement = get_game_line_movement('BUF', 'NO', 4, 2025)
    print(f"\nBUF vs NO movement: {buf_game_movement}")
    
    print("\n✅ Line movement functions ready for integration!")