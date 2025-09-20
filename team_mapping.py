"""
Team Mapping Module for NFL Prediction Model

This module provides mapping functions to handle team relocations and name changes.
"""

# Define a mapping from current team abbreviations to historical abbreviations
TEAM_RELOCATION_MAP = {
    # Current team abbreviation -> List of historical abbreviations
    "LAR": ["STL"],  # LA Rams were previously St. Louis Rams
    "LV": ["OAK"],   # Las Vegas Raiders were previously Oakland Raiders
    "LAC": ["SD"]    # LA Chargers were previously San Diego Chargers
}

# Define the reverse mapping for lookups
HISTORICAL_TO_CURRENT_MAP = {}
for current, historical_list in TEAM_RELOCATION_MAP.items():
    for historical in historical_list:
        HISTORICAL_TO_CURRENT_MAP[historical] = current

def get_current_team_abbr(team_abbr):
    """
    Convert a historical team abbreviation to its current abbreviation.
    
    Parameters:
    -----------
    team_abbr : str
        Team abbreviation (could be current or historical)
        
    Returns:
    --------
    str
        Current team abbreviation
    """
    return HISTORICAL_TO_CURRENT_MAP.get(team_abbr, team_abbr)

def get_all_team_abbrs(team_abbr):
    """
    Get all historical abbreviations for a team, including current.
    
    Parameters:
    -----------
    team_abbr : str
        Current team abbreviation
        
    Returns:
    --------
    list
        List of all abbreviations for this team (current + historical)
    """
    # If we're given a historical abbreviation, convert to current first
    current_abbr = get_current_team_abbr(team_abbr)
    
    # Get all historical abbreviations and add the current one
    historical_abbrs = TEAM_RELOCATION_MAP.get(current_abbr, [])
    return [current_abbr] + historical_abbrs
