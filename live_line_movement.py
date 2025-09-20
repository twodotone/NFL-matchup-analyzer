import pandas as pd
import nfl_data_py as nfl
import os
from datetime import datetime

def get_live_line_movement(home_team, away_team, week, year):
    """
    Get real-time line movement by comparing local (opening) vs live (current) data
    """
    try:
        # Load local data (opening lines)
        local_schedule_file = f"data/schedule_{year}.parquet"
        if not os.path.exists(local_schedule_file):
            return {'found': False, 'reason': 'Local schedule data not available'}
        
        local_schedule = pd.read_parquet(local_schedule_file)
        
        # Find the game in local data
        local_game = local_schedule[
            (local_schedule['week'] == week) &
            (local_schedule['home_team'] == home_team) &
            (local_schedule['away_team'] == away_team)
        ]
        
        if local_game.empty:
            return {'found': False, 'reason': 'Game not found in local data'}
        
        local_game = local_game.iloc[0]
        
        # Get fresh data from nfl_data_py (current lines)
        fresh_schedule = nfl.import_schedules([year])
        
        # Find the game in fresh data
        fresh_game = fresh_schedule[
            (fresh_schedule['week'] == week) &
            (fresh_schedule['home_team'] == home_team) &
            (fresh_schedule['away_team'] == away_team)
        ]
        
        if fresh_game.empty:
            return {'found': False, 'reason': 'Game not found in live data'}
        
        fresh_game = fresh_game.iloc[0]
        
        # Calculate movement
        opening_spread = local_game.get('spread_line')
        current_spread = fresh_game.get('spread_line')
        opening_total = local_game.get('total_line')
        current_total = fresh_game.get('total_line')
        
        # Calculate movement (current - opening)
        spread_movement = current_spread - opening_spread if pd.notna(current_spread) and pd.notna(opening_spread) else 0
        total_movement = current_total - opening_total if pd.notna(current_total) and pd.notna(opening_total) else 0
        
        # Get local file timestamp for "opening" time
        local_file_time = os.path.getmtime(local_schedule_file)
        opening_time = datetime.fromtimestamp(local_file_time)
        
        return {
            'found': True,
            'opening_spread': opening_spread,
            'current_spread': current_spread,
            'spread_movement': spread_movement,
            'opening_total': opening_total,
            'current_total': current_total,
            'total_movement': total_movement,
            'opening_time': opening_time,
            'data_source': 'live_comparison',
            'movement_type': 'live_tracking'
        }
        
    except Exception as e:
        return {'found': False, 'reason': f'Error in live line tracking: {str(e)}'}

def analyze_live_week_movement(week, year):
    """
    Analyze line movement for all games in a week using live data comparison
    """
    try:
        # Load local data
        local_schedule = pd.read_parquet(f"data/schedule_{year}.parquet")
        week_games = local_schedule[local_schedule['week'] == week]
        
        # Get fresh data
        fresh_schedule = nfl.import_schedules([year])
        fresh_week = fresh_schedule[fresh_schedule['week'] == week]
        
        movements = []
        
        for _, local_game in week_games.iterrows():
            # Find matching fresh game
            fresh_game = fresh_week[
                (fresh_week['home_team'] == local_game['home_team']) &
                (fresh_week['away_team'] == local_game['away_team'])
            ]
            
            if not fresh_game.empty:
                fresh_game = fresh_game.iloc[0]
                
                # Calculate movement
                spread_movement = fresh_game['spread_line'] - local_game['spread_line']
                total_movement = fresh_game['total_line'] - local_game['total_line']
                
                movements.append({
                    'away_team': local_game['away_team'],
                    'home_team': local_game['home_team'],
                    'opening_spread': local_game['spread_line'],
                    'current_spread': fresh_game['spread_line'],
                    'spread_movement': spread_movement,
                    'opening_total': local_game['total_line'],
                    'current_total': fresh_game['total_line'],
                    'total_movement': total_movement
                })
        
        if not movements:
            return {'found': False, 'reason': 'No movement data available'}
        
        movements_df = pd.DataFrame(movements)
        
        # Calculate statistics
        significant_spread_moves = movements_df[abs(movements_df['spread_movement']) >= 1.0]
        significant_total_moves = movements_df[abs(movements_df['total_movement']) >= 2.0]
        
        return {
            'found': True,
            'total_games': len(movements_df),
            'movements_data': movements_df,
            'significant_spread_moves': len(significant_spread_moves),
            'significant_total_moves': len(significant_total_moves),
            'avg_spread_movement': movements_df['spread_movement'].abs().mean(),
            'avg_total_movement': movements_df['total_movement'].abs().mean(),
            'max_spread_movement': movements_df['spread_movement'].abs().max(),
            'max_total_movement': movements_df['total_movement'].abs().max(),
            'biggest_movers': movements_df.nlargest(3, 'spread_movement', keep='all')[['away_team', 'home_team', 'spread_movement']].to_dict('records')
        }
        
    except Exception as e:
        return {'found': False, 'reason': f'Error analyzing week movement: {str(e)}'}

if __name__ == "__main__":
    import os
    
    # Test the live line movement tracking
    print("🧪 Testing Live Line Movement Tracking")
    print("=" * 50)
    
    # Test Week 3 ATL @ CAR (we know this has movement)
    movement_data = get_live_line_movement('CAR', 'ATL', 3, 2025)
    print(f"ATL @ CAR movement: {movement_data}")
    
    # Test Week 3 analysis
    week_analysis = analyze_live_week_movement(3, 2025)
    print(f"\nWeek 3 analysis: {week_analysis['found']}")
    if week_analysis['found']:
        print(f"Games with movement: {week_analysis['total_games']}")
        print(f"Significant moves: {week_analysis['significant_spread_moves']}")
        print(f"Average movement: {week_analysis['avg_spread_movement']:.2f} pts")
        
        if week_analysis['biggest_movers']:
            print("Biggest movers:")
            for mover in week_analysis['biggest_movers']:
                print(f"  {mover['away_team']} @ {mover['home_team']}: {mover['spread_movement']:+.1f} pts")
    
    print("\n✅ Live line movement tracking ready for integration!")