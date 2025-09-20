"""
Player Impact Analysis Module
Calculates individual player EPA impact and injury report analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st

class PlayerImpactAnalyzer:
    """Analyzes individual player impact using EPA and usage metrics"""
    
    def __init__(self, pbp_data: pd.DataFrame):
        self.pbp_data = pbp_data
        self.current_season = pbp_data['season'].max()
        
    def calculate_qb_impact(self, player_name: str, team: str, weeks_back: int = 18) -> Dict:
        """Calculate QB impact metrics"""
        
        # Filter recent games for this QB
        qb_plays = self.pbp_data[
            (self.pbp_data['passer_player_name'] == player_name) &
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season)
        ].copy()
        
        if len(qb_plays) == 0:
            return {}
            
        # Calculate passing metrics
        pass_epa_per_play = qb_plays['qb_epa'].mean()
        total_dropbacks = len(qb_plays)
        
        # Rushing plays by this QB
        rush_plays = self.pbp_data[
            (self.pbp_data['rusher_player_name'] == player_name) &
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season)
        ]
        
        rush_epa_per_play = rush_plays['epa'].mean() if len(rush_plays) > 0 else 0
        
        # Team's backup QB performance (for comparison)
        team_other_qbs = self.pbp_data[
            (self.pbp_data['passer_player_name'] != player_name) &
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season)
        ]
        
        backup_epa = team_other_qbs['qb_epa'].mean() if len(team_other_qbs) > 0 else 0
        
        return {
            'player_name': player_name,
            'position': 'QB',
            'team': team,
            'pass_epa_per_play': round(pass_epa_per_play, 3),
            'rush_epa_per_play': round(rush_epa_per_play, 3),
            'total_dropbacks': total_dropbacks,
            'backup_epa_diff': round(pass_epa_per_play - backup_epa, 3),
            'impact_score': round((pass_epa_per_play - backup_epa) * total_dropbacks, 1)
        }
    
    def calculate_skill_position_impact(self, player_name: str, team: str, position: str, weeks_back: int = 8) -> Dict:
        """Calculate RB/WR/TE impact metrics"""
        
        # Receiving plays
        receiving_plays = self.pbp_data[
            (self.pbp_data['receiver_player_name'] == player_name) &
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['week'] >= (self.pbp_data['week'].max() - weeks_back))
        ].copy()
        
        # Rushing plays (for RBs)
        rushing_plays = self.pbp_data[
            (self.pbp_data['rusher_player_name'] == player_name) &
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['week'] >= (self.pbp_data['week'].max() - weeks_back))
        ].copy()
        
        # Calculate metrics
        receiving_epa = receiving_plays['epa'].mean() if len(receiving_plays) > 0 else 0
        rushing_epa = rushing_plays['epa'].mean() if len(rushing_plays) > 0 else 0
        
        targets = len(receiving_plays)
        carries = len(rushing_plays)
        
        # Target/carry share
        team_targets = len(self.pbp_data[
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['pass'] == 1)
        ])
        
        team_carries = len(self.pbp_data[
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['rush'] == 1)
        ])
        
        target_share = (targets / team_targets * 100) if team_targets > 0 else 0
        carry_share = (carries / team_carries * 100) if team_carries > 0 else 0
        
        # Combined impact score
        total_epa = (receiving_epa * targets) + (rushing_epa * carries)
        usage_factor = (targets + carries) / 100  # Normalize usage
        
        return {
            'player_name': player_name,
            'position': position,
            'team': team,
            'receiving_epa_per_target': round(receiving_epa, 3),
            'rushing_epa_per_carry': round(rushing_epa, 3),
            'targets': targets,
            'carries': carries,
            'target_share': round(target_share, 1),
            'carry_share': round(carry_share, 1),
            'total_touches': targets + carries,
            'impact_score': round(total_epa * usage_factor, 1)
        }
    
    def calculate_defensive_impact(self, player_name: str, team: str, weeks_back: int = 8) -> Dict:
        """Calculate defensive player impact (tackles, sacks, etc.)"""
        
        # Tackles
        tackle_plays = self.pbp_data[
            ((self.pbp_data['solo_tackle_1_player_name'] == player_name) |
             (self.pbp_data['solo_tackle_2_player_name'] == player_name) |
             (self.pbp_data['tackle_with_assist_1_player_name'] == player_name) |
             (self.pbp_data['tackle_with_assist_2_player_name'] == player_name)) &
            (self.pbp_data['defteam'] == team) &
            (self.pbp_data['season'] == self.current_season)
        ]
        
        # Sacks
        sack_plays = self.pbp_data[
            ((self.pbp_data['sack_player_id'].notna()) & 
             (self.pbp_data['sack_player_name'] == player_name)) &
            (self.pbp_data['defteam'] == team) &
            (self.pbp_data['season'] == self.current_season)
        ]
        
        # Interceptions
        int_plays = self.pbp_data[
            (self.pbp_data['interception_player_name'] == player_name) &
            (self.pbp_data['defteam'] == team) &
            (self.pbp_data['season'] == self.current_season)
        ]
        
        # EPA allowed when involved in play (tackles)
        epa_on_tackles = tackle_plays['epa'].mean() if len(tackle_plays) > 0 else 0
        
        return {
            'player_name': player_name,
            'position': 'DEF',
            'team': team,
            'tackles': len(tackle_plays),
            'sacks': len(sack_plays),
            'interceptions': len(int_plays),
            'epa_per_tackle': round(epa_on_tackles, 3),
            'impact_score': round(len(sack_plays) * -2.0 + len(int_plays) * -2.5 + len(tackle_plays) * -0.1, 1)
        }
    
    def get_team_key_players(self, team: str, min_touches: int = 10) -> List[Dict]:
        """Get key players for a team based on usage"""
        
        players = []
        
        # Get QBs
        qb_data = self.pbp_data[
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['passer_player_name'].notna())
        ]['passer_player_name'].value_counts()
        
        for qb_name in qb_data.head(2).index:  # Top 2 QBs
            if qb_data[qb_name] >= min_touches:
                qb_impact = self.calculate_qb_impact(qb_name, team, weeks_back=18)  # Full season
                if qb_impact:
                    players.append(qb_impact)
        
        # Get skill position players
        # Receivers
        rec_data = self.pbp_data[
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['receiver_player_name'].notna())
        ]['receiver_player_name'].value_counts()
        
        for player_name in rec_data.head(15).index:  # Top 15 receivers
            if rec_data[player_name] >= 5:  # Lower threshold for receivers
                impact = self.calculate_skill_position_impact(player_name, team, 'WR/TE', weeks_back=18)
                if impact and impact['total_touches'] >= 5:
                    players.append(impact)
        
        # Rushers
        rush_data = self.pbp_data[
            (self.pbp_data['posteam'] == team) &
            (self.pbp_data['season'] == self.current_season) &
            (self.pbp_data['rusher_player_name'].notna())
        ]['rusher_player_name'].value_counts()
        
        for player_name in rush_data.head(10).index:  # Top 10 rushers
            if rush_data[player_name] >= 5:  # Lower threshold for rushers
                impact = self.calculate_skill_position_impact(player_name, team, 'RB', weeks_back=18)
                if impact and impact['total_touches'] >= 5:
                    players.append(impact)
        
        return sorted(players, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)
    
    def format_injury_impact_display(self, team: str, injured_players: List[str] = None) -> pd.DataFrame:
        """Format player impact data for display"""
        
        key_players = self.get_team_key_players(team)
        
        display_data = []
        for player in key_players[:15]:  # Top 15 players
            is_injured = injured_players and player['player_name'] in injured_players
            
            # Create display row
            row = {
                'Player': player['player_name'],
                'Pos': player['position'],
                'Impact Score': player['impact_score'],
                'Status': '🚑 INJURED' if is_injured else '✅ Healthy'
            }
            
            # Add position-specific metrics
            if player['position'] == 'QB':
                row['Pass EPA/Play'] = player.get('pass_epa_per_play', 0)
                row['vs Backup'] = f"+{player.get('backup_epa_diff', 0)}"
            elif player['position'] in ['RB', 'WR/TE']:
                row['Touches'] = player.get('total_touches', 0)
                row['EPA/Touch'] = round((player.get('receiving_epa_per_target', 0) + player.get('rushing_epa_per_carry', 0)) / 2, 3)
            
            display_data.append(row)
        
        return pd.DataFrame(display_data)

def get_mock_injury_report(team: str) -> List[str]:
    """Mock injury report - replace with real data source"""
    # This would connect to real injury report APIs
    mock_injuries = {
        'BUF': ['Josh Allen', 'Stefon Diggs'],
        'KC': ['Travis Kelce'],
        'PHI': ['A.J. Brown', 'Lane Johnson'],
        'SF': ['Christian McCaffrey', 'Deebo Samuel'],
        'DAL': ['Dak Prescott'],
        # Add more as needed
    }
    return mock_injuries.get(team, [])