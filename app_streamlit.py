# app_streamlit.py - Production Streamlit App

import streamlit as st
import pandas as pd
import os
import sys
import traceback

# Set page config first
st.set_page_config(page_title="NFL Matchup Analyzer", layout="wide")

# Add comprehensive error handling for deployment
try:
    from streamlit_data_loader import StreamlitDataLoader, check_data_freshness
    from streamlit_simple_model import StreamlitSimpleNFLModel
    from streamlit_real_standard_model import StreamlitRealStandardModel
    from data_loader import load_rolling_data
except Exception as e:
    st.error(f"❌ Critical import error: {e}")
    st.code(traceback.format_exc())
    st.stop()
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

# Add global error handling for deployment
def handle_deployment_error(error, context=""):
    """Handle errors gracefully for deployment"""
    error_msg = f"Deployment Error in {context}: {str(error)}"
    st.error(error_msg)
    st.write("**Error Details:**")
    st.code(traceback.format_exc())
    st.write("**Troubleshooting Steps:**")
    st.write("1. Try refreshing the page")
    st.write("2. Select a different week/year combination")
    st.write("3. Contact support if the issue persists")
    return None

# --- Page & Sidebar Configuration ---
st.title('🏈 NFL Matchup Analyzer - Production')

# Check if we're in deployment environment
IS_DEPLOYMENT = os.getenv('RAILWAY_ENVIRONMENT') is not None or \
                os.getenv('STREAMLIT_SHARING', 'false').lower() == 'true' or \
                'streamlit.app' in os.getenv('HOSTNAME', '') or \
                os.getenv('STREAMLIT_CLOUD', 'false').lower() == 'true'

if IS_DEPLOYMENT:
    if os.getenv('RAILWAY_ENVIRONMENT'):
        st.sidebar.info("🚂 Running on Railway")
    else:
        st.sidebar.info("🌟 Running on Streamlit Cloud")
    # Reduce memory usage for deployment
    os.environ['PYTHONHASHSEED'] = '0'  # Make Python hash-stable

# --- Data Freshness Check ---
with st.sidebar:
    st.header('📊 Data Status')
    data_freshness = check_data_freshness()
    
    for file_name, last_updated in data_freshness.items():
        if last_updated == "Missing":
            st.error(f"❌ {file_name}: Missing")
        else:
            st.success(f"✅ {file_name}")
            st.caption(f"Updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")

# --- Settings ---
st.sidebar.header('⚙️ Settings')
CURRENT_YEAR = st.sidebar.selectbox('Year', [2025, 2024, 2023, 2022], index=0)
CURRENT_WEEK = st.sidebar.number_input('Week', min_value=1, max_value=18, value=1, step=1)

st.sidebar.header('🔧 Model Settings')
show_standard_model = st.sidebar.checkbox('Show Standard Model', value=True, 
                                         help="Shows complex SOS-adjusted model")
show_simple_model = st.sidebar.checkbox('Show Simple Model', value=True, 
                                       help="Shows transparent EPA-based model")

# Season trend analysis toggle
show_season_trends = st.sidebar.checkbox('Show 2025 Season Trends', value=True,
                                        help="Compare current season performance vs historical baseline")
current_season_weight = st.sidebar.slider('Current Season Weight', 
                                         min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                         help="0.0 = Historical only, 1.0 = 2025 only") if show_season_trends else 0.0

# --- Data Loading ---
@st.cache_data(max_entries=1, ttl=3600)  # Limit cache for deployment
def load_data():
    """Load all necessary data with caching"""
    loader = StreamlitDataLoader()
    
    try:
        # Load team data
        team_desc = loader.load_team_data()
        
        # Load schedule data
        schedule_data = loader.load_schedule_data([CURRENT_YEAR])
        
        return team_desc, schedule_data
        
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

@st.cache_data(max_entries=1, ttl=3600)  # Limit cache and add TTL for deployment
def load_simple_model():
    """Load and cache the simple model with safe year loading"""
    simple_model = StreamlitSimpleNFLModel(data_dir="data")
    
    # Optimize for deployment - use fewer years if on deployment
    if IS_DEPLOYMENT:
        # Use only recent years for deployment to save memory
        years_to_load = [2023, 2024]
        st.info("🌐 Deployment mode: Using optimized data (2023-2024)")
    else:
        # Try to load years safely, starting with guaranteed years
        years_to_load = [2022, 2023, 2024]
    
    # Check if 2025 data exists before adding it
    import os
    if os.path.exists(os.path.join("data", "pbp_2025.parquet")):
        try:
            # Test if 2025 file has actual data
            import pandas as pd
            df_2025 = pd.read_parquet(os.path.join("data", "pbp_2025.parquet"))
            if len(df_2025) > 0:
                years_to_load.append(2025)
                st.success(f"✅ Loading with 2025 data ({len(df_2025)} plays)")
            else:
                st.info("ℹ️ 2025 data file exists but is empty - using historical data only")
        except Exception as e:
            st.warning(f"⚠️ 2025 data file exists but couldn't be loaded: {str(e)}")
    else:
        st.info("ℹ️ No 2025 data available yet - using historical data only")
    
    simple_model.load_data_from_parquet(years_to_load)
    return simple_model

@st.cache_data(max_entries=1, ttl=1800)  # Shorter TTL for standard model
def load_standard_model(current_year, current_week):
    """Load and cache the real standard model"""
    standard_model = StreamlitRealStandardModel(data_dir="data")
    standard_model.load_standard_data(current_year, current_week)
    return standard_model

try:
    team_desc, schedule_data = load_data()
except Exception as e:
    handle_deployment_error(e, "Data Loading")
    st.stop()

# --- Main App Tabs ---
main_tab1, main_tab2 = st.tabs(["🏈 Game Analysis", "🏆 Power Rankings"])

with main_tab1:
    # --- Main Page: Matchup Selection ---
    try:
        st.header(f'Week {CURRENT_WEEK} Matchups for the {CURRENT_YEAR} Season')
        week_schedule = schedule_data[schedule_data['week'] == CURRENT_WEEK].copy()

        if week_schedule.empty:
            st.warning(f"No schedule found for Week {CURRENT_WEEK} of the {CURRENT_YEAR} season.")
            st.info("Available weeks in schedule:")
            st.write(sorted(schedule_data['week'].unique()))
            st.stop()
    except Exception as e:
        handle_deployment_error(e, "Schedule Loading")
        st.stop()

    week_schedule['game_description'] = week_schedule['away_team'] + ' @ ' + week_schedule['home_team']
    selected_game_str = st.selectbox('Select a Game:', week_schedule['game_description'].tolist())

    if selected_game_str:
        with st.spinner("Loading game analysis..."):
            game_details = week_schedule[week_schedule['game_description'] == selected_game_str].iloc[0]
            away_abbr, home_abbr = selected_game_str.split(' @ ')
            
            # Display Betting Odds Banner
            st.subheader("🎰 Betting Odds & Game Info")
        
            col1, col2, col3, col4, col5 = st.columns(5)
        
            # Get team logos with error handling
            try:
                away_logo = team_desc.loc[team_desc['team_abbr'] == away_abbr, 'team_logo_espn'].values[0] if len(team_desc.loc[team_desc['team_abbr'] == away_abbr]) > 0 else ""
            except:
                away_logo = ""
            
            try:
                home_logo = team_desc.loc[team_desc['team_abbr'] == home_abbr, 'team_logo_espn'].values[0] if len(team_desc.loc[team_desc['team_abbr'] == home_abbr]) > 0 else ""
            except:
                home_logo = ""
        
            # Determine spread display based on betting convention
            spread_magnitude = abs(game_details.get('spread_line', 0))
            home_moneyline = game_details.get('home_moneyline', 0)
        
            if home_moneyline < 0:  # Home team favored
                home_spread_vegas = -spread_magnitude
                away_spread_vegas = spread_magnitude
            elif home_moneyline > 0:  # Away team favored
                home_spread_vegas = spread_magnitude
                away_spread_vegas = -spread_magnitude
            else:
                home_spread_vegas = game_details.get('spread_line', 0)
                away_spread_vegas = -home_spread_vegas
        
            total_line = game_details.get('total_line', 0)
            away_moneyline = game_details.get('away_moneyline', 0)
        
            # Format moneylines
            away_ml_str = f"+{int(away_moneyline)}" if away_moneyline > 0 else f"{int(away_moneyline)}"
            home_ml_str = f"+{int(home_moneyline)}" if home_moneyline > 0 else f"{int(home_moneyline)}"
        
            # Display logos with error handling
            try:
                if away_logo:
                    col1.image(away_logo, width=70)
                else:
                    col1.write(f"**{away_abbr}**")
            except:
                col1.write(f"**{away_abbr}**")
            
            col1.markdown(f"<p style='text-align: center; margin: 0; font-weight: bold; color: #1f77b4;'>{away_ml_str}</p>", unsafe_allow_html=True)
            col2.metric("Away Spread", f"{away_spread_vegas:+.1f}")
            col3.metric("Over/Under", f"{total_line:.1f}")
            col4.metric("Home Spread", f"{home_spread_vegas:+.1f}")
        
            try:
                if home_logo:
                    col5.image(home_logo, width=70)
                else:
                    col5.write(f"**{home_abbr}**")
            except:
                col5.write(f"**{home_abbr}**")
            
            col5.markdown(f"<p style='text-align: center; margin: 0; font-weight: bold; color: #1f77b4;'>{home_ml_str}</p>", unsafe_allow_html=True)

            # --- Line Movement Tracking ---
            # Import line movement functions with fallback
            def get_live_line_movement_for_app(home_team, away_team, week, year):
                """Get live line movement for the app"""
                try:
                    from live_line_movement import get_live_line_movement
                    return get_live_line_movement(home_team, away_team, week, year)
                except:
                    return {'found': False, 'reason': 'Live tracking not available'}
        
            def get_app_line_movement(home_team, away_team, week, year):
                """Get line movement using multiple methods for the app"""
            
                # Try captured lines first
                try:
                    from line_movement_analyzer import load_captured_line_movement
                    captured = load_captured_line_movement(home_team, away_team, week, year)
                    if captured['found']:
                        return captured
                except:
                    pass
            
                # Try live tracking
                live_result = get_live_line_movement_for_app(home_team, away_team, week, year)
                if live_result['found']:
                    from datetime import datetime
                    return {
                        'found': True,
                        'early_spread': live_result['opening_spread'],
                        'current_spread': live_result['current_spread'],
                        'spread_movement': live_result['spread_movement'],
                        'early_total': live_result['opening_total'],
                        'current_total': live_result['current_total'],
                        'total_movement': live_result['total_movement'],
                        'capture_date': live_result['opening_time'],
                        'days_elapsed': (datetime.now() - live_result['opening_time']).days,
                        'data_source': 'live_tracking'
                    }
            
                return {'found': False, 'reason': 'No line movement data available'}
        
            # Get line movement data for this game
            try:
                line_movement_data = get_app_line_movement(home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR)
            
                if line_movement_data['found']:
                    # Show opening line vs current line
                    st.divider()
                    st.subheader("📈 Line Movement")
                
                    line_col1, line_col2, line_col3 = st.columns(3)
                
                    with line_col1:
                        opening_spread_display = f"{home_abbr} {-line_movement_data['early_spread']:+.1f}"
                        st.metric("Opening Line", opening_spread_display,
                                 help=f"Line when first posted {line_movement_data['days_elapsed']} days ago")
                
                    with line_col2:
                        current_spread_display = f"{home_abbr} {home_spread_vegas:+.1f}"
                        movement_value = line_movement_data['spread_movement']
                    
                        if abs(movement_value) >= 0.5:
                            movement_delta = f"{movement_value:+.1f} pts"
                            if abs(movement_value) >= 2.0:
                                movement_delta += " 🔥"
                            elif abs(movement_value) >= 1.0:
                                movement_delta += " 📈"
                        else:
                            movement_delta = "Stable ✅"
                    
                        st.metric("Current Line", current_spread_display,
                                 delta=movement_delta,
                                 help="Current Vegas line with movement from opening")
                
                    with line_col3:
                        # Show movement direction and magnitude
                        movement_mag = abs(movement_value)
                        if movement_mag >= 0.5:
                            direction = "toward" if movement_value > 0 else "against"
                            team = away_abbr if movement_value > 0 else home_abbr
                            movement_desc = f"Moving {direction} {team}"
                        
                            if movement_mag >= 2.0:
                                movement_icon = "🔥 Major"
                            elif movement_mag >= 1.0:
                                movement_icon = "📈 Significant" 
                            else:
                                movement_icon = "📊 Minor"
                            
                            st.metric("Movement", movement_icon,
                                     delta=movement_desc,
                                     help=f"{movement_mag:.1f} point movement since opening")
                        else:
                            st.metric("Movement", "✅ Stable",
                                     help="Line has remained stable since opening")
                
                    # Show capture info
                    capture_date = pd.to_datetime(line_movement_data['capture_date']).strftime('%m/%d %I:%M %p')
                    st.caption(f"📅 Opening line captured: {capture_date} ({line_movement_data['days_elapsed']} days ago)")
            
            except ImportError:
                # Fallback - show note about line movement tracking
                st.divider()
                st.info("📊 **Line Movement Tracking**: Starting Week 4, opening vs closing line movement will be displayed here!")

            # --- Team Performance Dashboard ---
            st.divider()
            st.subheader("📈 Recent Team Performance")
        
            def get_team_recent_stats(team_abbr, week, year, pbp_data, include_trends=False):
                """Get team's recent performance stats with optional trend analysis"""
                # Get team's games from current season (weeks 1 to current-1)
                current_season_games = pbp_data[
                    (pbp_data['season'] == year) & 
                    (pbp_data['week'] < week) &
                    ((pbp_data['posteam'] == team_abbr) | (pbp_data['defteam'] == team_abbr))
                ]
            
                if len(current_season_games) == 0:
                    return None
                
                # Calculate current season stats
                current_off_plays = current_season_games[current_season_games['posteam'] == team_abbr]
                current_avg_epa_off = current_off_plays['epa'].mean() if len(current_off_plays) > 0 else 0
            
                current_def_plays = current_season_games[current_season_games['defteam'] == team_abbr]
                current_avg_epa_def = current_def_plays['epa'].mean() if len(current_def_plays) > 0 else 0
            
                games_played = len(current_season_games.groupby(['game_id']))
            
                base_stats = {
                    'games_played': games_played,
                    'epa_offense': current_avg_epa_off,
                    'epa_defense': -current_avg_epa_def,  # Flip sign so positive is good defense (for display)
                    'epa_defense_raw': current_avg_epa_def,  # Keep raw values for matchup calculations
                    'total_plays': len(current_off_plays) + len(current_def_plays)
                }
            
                if not include_trends or year != 2025:
                    return base_stats
                
                # Add historical baseline for trend analysis (2022-2024)
                historical_games = pbp_data[
                    (pbp_data['season'].isin([2022, 2023, 2024])) &
                    ((pbp_data['posteam'] == team_abbr) | (pbp_data['defteam'] == team_abbr))
                ]
            
                if len(historical_games) > 0:
                    hist_off_plays = historical_games[historical_games['posteam'] == team_abbr]
                    hist_avg_epa_off = hist_off_plays['epa'].mean() if len(hist_off_plays) > 0 else 0
                
                    hist_def_plays = historical_games[historical_games['defteam'] == team_abbr]
                    hist_avg_epa_def = hist_def_plays['epa'].mean() if len(hist_def_plays) > 0 else 0
                
                    # Calculate trends
                    off_trend = current_avg_epa_off - hist_avg_epa_off
                    def_trend = (-current_avg_epa_def) - (-hist_avg_epa_def)  # Both flipped for consistency
                
                    base_stats.update({
                        'historical_offense': hist_avg_epa_off,
                        'historical_defense': -hist_avg_epa_def,
                        'offense_trend': off_trend,
                        'defense_trend': def_trend,
                        'has_trends': True
                    })
                else:
                    base_stats['has_trends'] = False
                
                return base_stats
        
            try:
                # Get PBP data for analysis with comprehensive error handling
                if 'pbp_data' not in st.session_state:
                    with st.spinner("Loading NFL data..."):
                        pbp_data = load_rolling_data(CURRENT_YEAR)
                        if not pbp_data.empty:
                            st.session_state.pbp_data = pbp_data
                        else:
                            st.error("❌ Unable to load NFL data. Please try refreshing the page or selecting a different year.")
                            st.stop()
                else:
                    pbp_data = st.session_state.pbp_data
            
                col1, col2 = st.columns(2)
            
                with col1:
                    st.markdown(f"**{home_abbr} (Home)**")
                    home_stats = get_team_recent_stats(home_abbr, CURRENT_WEEK, CURRENT_YEAR, pbp_data, include_trends=show_season_trends)
                    if home_stats and home_stats['games_played'] > 0:
                    
                        perf_col1, perf_col2 = st.columns(2)
                        with perf_col1:
                            epa_off_color = "🟢" if home_stats['epa_offense'] > 0.05 else "🔴" if home_stats['epa_offense'] < -0.05 else "🟡"
                        
                            # Show trend if available
                            delta_text = None
                            if show_season_trends and home_stats.get('has_trends', False):
                                trend = home_stats['offense_trend']
                                if abs(trend) >= 0.05:
                                    trend_icon = "📈" if trend > 0 else "📉"
                                    delta_text = f"{trend:+.3f} vs 2022-24 {trend_icon}"
                                else:
                                    delta_text = "Stable vs 2022-24 📊"
                        
                            st.metric("Offense EPA/Play", f"{home_stats['epa_offense']:+.3f}", 
                                     delta=delta_text,
                                     help="Expected Points Added per offensive play")
                            st.caption(f"{epa_off_color} Offensive efficiency")
                    
                        with perf_col2:
                            epa_def_color = "🟢" if home_stats['epa_defense'] > 0.05 else "🔴" if home_stats['epa_defense'] < -0.05 else "🟡"
                        
                            # Show trend if available
                            delta_text = None
                            if show_season_trends and home_stats.get('has_trends', False):
                                trend = home_stats['defense_trend'] 
                                if abs(trend) >= 0.05:
                                    trend_icon = "📈" if trend > 0 else "📉"
                                    delta_text = f"{trend:+.3f} vs 2022-24 {trend_icon}"
                                else:
                                    delta_text = "Stable vs 2022-24 📊"
                        
                            st.metric("Defense EPA/Play", f"{home_stats['epa_defense']:+.3f}",
                                     delta=delta_text,
                                     help="Expected Points Added per defensive play (positive = good defense)")
                            st.caption(f"{epa_def_color} Defensive efficiency")
                    
                        st.caption(f"📊 Based on {home_stats['games_played']} games ({home_stats['total_plays']} plays)")
                    else:
                        st.info("📋 No recent performance data available")
            
                with col2:
                    st.markdown(f"**{away_abbr} (Away)**")
                    away_stats = get_team_recent_stats(away_abbr, CURRENT_WEEK, CURRENT_YEAR, pbp_data, include_trends=show_season_trends)
                    if away_stats and away_stats['games_played'] > 0:
                    
                        perf_col1, perf_col2 = st.columns(2)
                        with perf_col1:
                            epa_off_color = "🟢" if away_stats['epa_offense'] > 0.05 else "🔴" if away_stats['epa_offense'] < -0.05 else "🟡"
                        
                            # Show trend if available
                            delta_text = None
                            if show_season_trends and away_stats.get('has_trends', False):
                                trend = away_stats['offense_trend']
                                if abs(trend) >= 0.05:
                                    trend_icon = "📈" if trend > 0 else "📉"
                                    delta_text = f"{trend:+.3f} vs 2022-24 {trend_icon}"
                                else:
                                    delta_text = "Stable vs 2022-24 📊"
                        
                            st.metric("Offense EPA/Play", f"{away_stats['epa_offense']:+.3f}",
                                     delta=delta_text,
                                     help="Expected Points Added per offensive play")
                            st.caption(f"{epa_off_color} Offensive efficiency")
                    
                        with perf_col2:
                            epa_def_color = "🟢" if away_stats['epa_defense'] > 0.05 else "🔴" if away_stats['epa_defense'] < -0.05 else "🟡"
                        
                            # Show trend if available
                            delta_text = None
                            if show_season_trends and away_stats.get('has_trends', False):
                                trend = away_stats['defense_trend']
                                if abs(trend) >= 0.05:
                                    trend_icon = "📈" if trend > 0 else "📉"
                                    delta_text = f"{trend:+.3f} vs 2022-24 {trend_icon}"
                                else:
                                    delta_text = "Stable vs 2022-24 📊"
                        
                            st.metric("Defense EPA/Play", f"{away_stats['epa_defense']:+.3f}",
                                     delta=delta_text,
                                     help="Expected Points Added per defensive play (positive = good defense)")
                            st.caption(f"{epa_def_color} Defensive efficiency")
                    
                        st.caption(f"📊 Based on {away_stats['games_played']} games ({away_stats['total_plays']} plays)")
                    else:
                        st.info("📋 No recent performance data available")
            
                # Add performance matchup insight
                if (home_stats and home_stats['games_played'] > 0 and 
                    away_stats and away_stats['games_played'] > 0):
                
                    st.markdown("**🎯 Performance Matchup**")
                
                    # Calculate advantages using raw defensive EPA (negative = good defense)
                    # Offensive advantage: Home offense vs Away defense (raw)
                    off_advantage = home_stats['epa_offense'] - away_stats['epa_defense_raw']
                    # Defensive advantage: Home defense vs Away offense 
                    def_advantage = away_stats['epa_offense'] - home_stats['epa_defense_raw']
                
                    insight_col1, insight_col2 = st.columns(2)
                    with insight_col1:
                        if off_advantage > 0.1:
                            st.success(f"🏠 {home_abbr} has offensive advantage (+{off_advantage:.3f} EPA)")
                        elif off_advantage < -0.1:
                            st.warning(f"✈️ {away_abbr} has offensive advantage ({off_advantage:.3f} EPA)")
                        else:
                            st.info(f"⚖️ Even offensive matchup ({off_advantage:+.3f} EPA)")
                
                    with insight_col2:
                        if def_advantage > 0.1:
                            st.success(f"🏠 {home_abbr} has defensive advantage (+{def_advantage:.3f} EPA)")
                        elif def_advantage < -0.1:
                            st.warning(f"✈️ {away_abbr} has defensive advantage ({def_advantage:.3f} EPA)")
                        else:
                            st.info(f"⚖️ Even defensive matchup ({def_advantage:+.3f} EPA)")
                        
            except Exception as e:
                st.warning("📊 Performance dashboard temporarily unavailable")

            # --- Lookahead Lines (Early Posted Lines for Next Week) ---
            @st.cache_data(ttl=3600)
            def get_next_week_lookahead_lines(current_week, current_year):
                """Get early posted lines for next week's games"""
                try:
                    next_week = current_week + 1
                    schedule_file = f"data/schedule_{current_year}.parquet"
                
                    if os.path.exists(schedule_file):
                        schedule_df = pd.read_parquet(schedule_file)
                    
                        # Get next week's games with posted lines
                        next_week_games = schedule_df[
                            (schedule_df['week'] == next_week) & 
                            (schedule_df['spread_line'].notna())
                        ].copy()
                    
                        return next_week_games
                
                    return pd.DataFrame()
                
                except Exception as e:
                    st.warning(f"Error loading next week lookahead lines: {e}")
                    return pd.DataFrame()

            @st.cache_data(ttl=3600)
            def get_team_next_week_game(team_abbr, current_week, current_year):
                """Get a specific team's next week game and early line"""
                try:
                    next_week_games = get_next_week_lookahead_lines(current_week, current_year)
                
                    if not next_week_games.empty:
                        # Find this team's next game
                        team_next_game = next_week_games[
                            (next_week_games['home_team'] == team_abbr) | 
                            (next_week_games['away_team'] == team_abbr)
                        ]
                    
                        if not team_next_game.empty:
                            game = team_next_game.iloc[0]
                        
                            # Determine if team is home or away
                            is_home = game['home_team'] == team_abbr
                            opponent = game['away_team'] if is_home else game['home_team']
                        
                            # Get spread from team's perspective
                            raw_spread = game['spread_line']
                            if is_home:
                                # Home team - flip sign (positive spread_line = home underdog)
                                team_spread = -raw_spread
                            else:
                                # Away team - use spread directly
                                team_spread = raw_spread
                        
                            return {
                                'found': True,
                                'opponent': opponent,
                                'is_home': is_home,
                                'spread': team_spread,
                                'total': game.get('total_line'),
                                'week': game['week'],
                                'venue': 'vs' if is_home else '@'
                            }
                
                    return {'found': False}
                
                except Exception as e:
                    return {'found': False}

            st.header("🤖 Model Predictions")
        
            # Initialize model prediction variables
            simple_model_spread = None
            simple_model_details = None
            simple_model_total = None
            simple_total_details = None
            model_home_spread = None
            dynamic_model_details = None
            total_edge = 0
            total_pick = "N/A"
        
            # --- Season-Weighted Model Blending ---
            def create_blended_model_data(pbp_data, current_season_weight):
                """Create blended dataset based on current season weight with proper scaling"""
                historical_data = pbp_data[pbp_data['season'].isin([2022, 2023, 2024])]
                current_data = pbp_data[pbp_data['season'] == 2025]
            
                if len(current_data) == 0:
                    return historical_data
                if len(historical_data) == 0:
                    return current_data
                
                if current_season_weight == 0.0:
                    return historical_data
                elif current_season_weight == 1.0:
                    return current_data
                else:
                    # Create smooth blending by controlling the relative sample sizes
                    # Target: achieve the desired weight ratio in the final dataset
                
                    # Decide on a reasonable total sample size (use historical as baseline)
                    target_total_size = len(historical_data)
                
                    # Calculate how many plays we want from each era
                    target_current_plays = int(target_total_size * current_season_weight)
                    target_historical_plays = int(target_total_size * (1 - current_season_weight))
                
                    # Handle current season data - might need to replicate if we want more weight
                    if target_current_plays > len(current_data):
                        # Need to replicate current season data
                        replication_factor = target_current_plays // len(current_data)
                        remainder = target_current_plays % len(current_data)
                    
                        # Replicate full datasets
                        current_replicated = pd.concat([current_data] * replication_factor, ignore_index=True)
                    
                        # Add partial sample for remainder
                        if remainder > 0:
                            current_partial = current_data.sample(n=remainder, random_state=42)
                            current_replicated = pd.concat([current_replicated, current_partial], ignore_index=True)
                    
                        current_sample = current_replicated
                    else:
                        # Sample from current season data
                        current_sample = current_data.sample(n=target_current_plays, random_state=42)
                
                    # Sample historical data
                    if target_historical_plays > len(historical_data):
                        historical_sample = historical_data  # Use all available
                    else:
                        historical_sample = historical_data.sample(n=target_historical_plays, random_state=42)
                
                    # Combine the samples
                    blended_data = pd.concat([historical_sample, current_sample], ignore_index=True)
                    return blended_data

            # Load Simple Model
            if show_simple_model:
                # Show current model configuration
                if show_season_trends and current_season_weight != 0.5:
                    if current_season_weight == 0.0:
                        st.info("🕰️ **Model Mode**: Historical baseline only (2022-2024 data)")
                    elif current_season_weight == 1.0:
                        st.warning("🆕 **Model Mode**: Current season only (2025 data) - ⚠️ Limited sample size may cause extreme predictions")
                    else:
                        st.info(f"⚖️ **Model Mode**: Blended ({current_season_weight:.0%} current season, {(1-current_season_weight):.0%} historical)")
                    
                        if current_season_weight >= 0.8:
                            st.caption("⚠️ **High current season weight**: Predictions may be more volatile due to smaller 2025 sample size")
            
                with st.spinner("Loading Simple Model..."):
                    try:
                        simple_model = load_simple_model()
                    
                        # Apply season weighting to model data if enabled
                        if show_season_trends and current_season_weight != 0.5:
                            original_data = simple_model.pbp_data.copy()
                            blended_data = create_blended_model_data(original_data, current_season_weight)
                            simple_model.pbp_data = blended_data
                        
                            # Show what we're doing
                            orig_2025 = len(original_data[original_data['season'] == 2025])
                            orig_hist = len(original_data[original_data['season'].isin([2022, 2023, 2024])])
                            new_2025 = len(blended_data[blended_data['season'] == 2025])
                            new_hist = len(blended_data[blended_data['season'].isin([2022, 2023, 2024])])
                        
                            actual_weight = new_2025 / (new_2025 + new_hist) if (new_2025 + new_hist) > 0 else 0
                        
                            st.caption(f"📊 Data: {new_2025:,} current season plays, {new_hist:,} historical plays (actual weight: {actual_weight:.0%})")
                        
                            if current_season_weight == 1.0:
                                st.caption(f"⚠️ Using only {orig_2025:,} plays from 2025 vs {orig_2025 + orig_hist:,} normally - expect more volatile predictions")
                        
                            # Force model to recalculate team stats with new data
                            if hasattr(simple_model, '_team_stats_cache'):
                                simple_model._team_stats_cache = {}
                            if hasattr(simple_model, 'team_stats'):
                                simple_model.team_stats = None
                    
                        # Validate teams exist in data
                        if simple_model.pbp_data is not None:
                            available_teams = set(simple_model.pbp_data['posteam'].dropna().unique()) | set(simple_model.pbp_data['defteam'].dropna().unique())
                        
                            if home_abbr not in available_teams:
                                st.error(f"Team {home_abbr} not found in play-by-play data")
                                st.write(f"Available teams: {sorted(available_teams)}")
                                show_simple_model = False
                            elif away_abbr not in available_teams:
                                st.error(f"Team {away_abbr} not found in play-by-play data")
                                st.write(f"Available teams: {sorted(available_teams)}")
                                show_simple_model = False
                            else:
                                # Generate predictions with team validation
                                try:
                                    simple_model_spread, simple_model_details = simple_model.predict_spread(
                                        home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                                    )
                                    st.success(f"✅ Simple Model: {home_abbr} {simple_model_spread:+.1f}")
                                
                                    # Generate total prediction
                                    try:
                                        simple_model_total, simple_total_details = simple_model.predict_total(
                                            home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                                        )
                                    except Exception as e:
                                        st.warning(f"Total prediction failed for {home_abbr} vs {away_abbr}: {e}")
                                    
                                except Exception as e:
                                    st.error(f"Prediction failed for {home_abbr} vs {away_abbr}: {e}")
                                    st.write(f"Error details: {str(e)}")
                                    show_simple_model = False
                        else:
                            st.error("No play-by-play data loaded in model")
                            show_simple_model = False
                        
                    except Exception as e:
                        st.error(f"Simple model failed to load: {e}")
                        st.exception(e)
                        show_simple_model = False

            # Load Standard Model (Real Standard Model with tiered historical stats)
            if show_standard_model:
                with st.spinner("Loading Standard Model..."):
                    try:
                        # Initialize the Real Standard Model
                        standard_model = load_standard_model(CURRENT_YEAR, CURRENT_WEEK)
                    
                        # Generate predictions
                        model_home_spread, dynamic_model_details = standard_model.predict_spread_standard(
                            home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                        )
                    
                    except Exception as e:
                        st.warning(f"Standard model failed, using simplified approach: {e}")
                        model_home_spread = simple_model_spread + 0.5 if simple_model_spread else 0
                        dynamic_model_details = None

            # --- Display Results ---
            if show_simple_model and simple_model_spread is not None:
                st.subheader("📊 Model Analysis")
            
                # Create tabs for organized display
                tab1, tab2 = st.tabs(["📈 Predictions", "🔍 Model Details"])
            
                with tab1:
                    # Main metrics
                    if show_standard_model:
                        col1, col2, col3, col4, col5 = st.columns(5)
                    else:
                        col1, col2, col3 = st.columns(3)
                
                    with col1:
                        # Show current Vegas line and opening line if available
                        st.metric("Vegas Line", f"{home_abbr} {home_spread_vegas:+.1f}")
                    
                        # Try to get opening line for this game
                        try:
                            from line_movement_analyzer import get_game_line_movement
                            opening_data = get_game_line_movement(home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR)
                        
                            if opening_data['found']:
                                opening_display = f"{home_abbr} {-opening_data['early_spread']:+.1f}"
                                movement = opening_data['spread_movement']
                            
                                if abs(movement) >= 0.5:
                                    delta_text = f"{movement:+.1f} pts"
                                    if abs(movement) >= 1.0:
                                        delta_text += " 📈"
                                else:
                                    delta_text = "Stable"
                            
                                st.metric("Opening Line", opening_display, 
                                         delta=delta_text,
                                         help=f"Line when first posted {opening_data['days_elapsed']} days ago")
                            else:
                                st.caption("📋 Opening line: Not available for this week")
                        except:
                            st.caption("📋 Opening line: Will be available starting Week 4")
                    
                        st.metric("Vegas Total", f"{total_line:.1f}")
                
                    with col2:
                        if show_standard_model and model_home_spread is not None:
                            standard_edge = home_spread_vegas - model_home_spread
                            standard_pick = home_abbr if standard_edge > 0 else away_abbr
                        
                            if abs(standard_edge) >= 3:
                                edge_text = f"**{standard_pick}** 🔥"
                            elif abs(standard_edge) >= 1:
                                edge_text = f"**{standard_pick}** 💡"
                            else:
                                edge_text = f"{standard_pick}"
                        
                            st.metric("Standard Model", f"{home_abbr} {model_home_spread:+.1f}", 
                                     delta=f"Edge: {edge_text}")
                
                    with col3:
                        simple_edge = home_spread_vegas - simple_model_spread
                        simple_pick = home_abbr if simple_edge > 0 else away_abbr
                    
                        if abs(simple_edge) >= 3:
                            edge_text = f"**{simple_pick}** 🔥"
                        elif abs(simple_edge) >= 1:
                            edge_text = f"**{simple_pick}** 💡"
                        else:
                            edge_text = f"{simple_pick}"
                    
                        st.metric("Simple Model", f"{home_abbr} {simple_model_spread:+.1f}",
                                 delta=f"Edge: {edge_text}")
                
                    if show_simple_model and len(st.columns(5)) > 3:
                        with st.columns(5)[3]:
                            if simple_model_total is not None:
                                total_edge = simple_model_total - total_line
                                total_pick = "OVER" if total_edge > 0 else "UNDER"
                            
                                if abs(total_edge) >= 3:
                                    total_text = f"**{total_pick}** 🔥"
                                elif abs(total_edge) >= 1:
                                    total_text = f"**{total_pick}** 💡"
                                else:
                                    total_text = f"{total_pick}"
                                
                                st.metric("Model Total", f"{simple_model_total:.1f}",
                                         delta=f"Edge: {total_text}")
                            else:
                                st.metric("Model Total", "N/A")
                
                    # Recommendations
                    st.divider()
                
                    # Spread recommendation
                    if abs(simple_edge) >= 2:
                        if abs(simple_edge) >= 4:
                            rec_color = "#ff4757"
                            rec_icon = "🔥🔥"
                            rec_strength = "STRONG EDGE"
                        else:
                            rec_color = "#ffa502"
                            rec_icon = "🔥"
                            rec_strength = "MODERATE EDGE"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {rec_color}; padding: 20px; border-radius: 15px; margin: 15px 0; border: 3px solid #2f3542;">
                                <h2 style="text-align: center; margin: 0; color: white;">{rec_icon} {rec_strength} {rec_icon}</h2>
                                <h1 style="text-align: center; margin: 10px 0; color: white; font-size: 2.5em;">TAKE {simple_pick}</h1>
                                <p style="text-align: center; margin: 5px 0; color: white; font-size: 1.2em;">
                                    Model edge: {abs(simple_edge):.1f} points
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                    # Total recommendation
                    if simple_model_total is not None and abs(total_edge) >= 2:
                        if abs(total_edge) >= 4:
                            total_rec_color = "#ff4757"
                            total_rec_icon = "🔥🔥"
                            total_rec_strength = "STRONG TOTAL EDGE"
                        else:
                            total_rec_color = "#ffa502"
                            total_rec_icon = "🔥"
                            total_rec_strength = "MODERATE TOTAL EDGE"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {total_rec_color}; padding: 15px; border-radius: 15px; margin: 10px 0; border: 2px solid #2f3542;">
                                <h3 style="text-align: center; margin: 0; color: white;">{total_rec_icon} {total_rec_strength} {total_rec_icon}</h3>
                                <h2 style="text-align: center; margin: 10px 0; color: white; font-size: 2em;">TAKE THE {total_pick}</h2>
                                <p style="text-align: center; margin: 5px 0; color: white; font-size: 1.1em;">
                                    Model: {simple_model_total:.1f} • Vegas: {total_line:.1f} • Edge: {abs(total_edge):.1f} pts
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            
                with tab2:
                    # Model details
                    if show_simple_model and simple_model_details:
                        st.subheader("⚡ Simple Model Details")
                        st.write(f"**Method:** Pure EPA Analysis")
                        st.write(f"**Home Spread:** {simple_model_spread:+.1f}")
                        st.write(f"**Edge vs Vegas:** {abs(simple_edge):.1f} pts on {simple_pick}")
                        st.write(f"**{home_abbr} EPA:** {simple_model_details['home_stats']['net_epa_per_play']:.3f}")
                        st.write(f"**{away_abbr} EPA:** {simple_model_details['away_stats']['net_epa_per_play']:.3f}")
                        st.write(f"**EPA Advantage:** {simple_model_details['epa_advantage']:.3f}")
                        st.write(f"**HFA Used:** {simple_model_details['home_field_advantage']:.1f} points")
                    
                        if simple_model_total is not None and simple_total_details is not None:
                            st.write("---")
                            st.write("**Total Points Prediction:**")
                            st.write(f"**Model Total:** {simple_model_total:.1f}")
                            st.write(f"**Vegas Total:** {total_line:.1f}")
                            st.write(f"**Edge:** {abs(total_edge):.1f} pts ({total_pick})")
                            st.write(f"**{home_abbr} Expected:** {simple_total_details['home_expected_points']:.1f}")
                            st.write(f"**{away_abbr} Expected:** {simple_total_details['away_expected_points']:.1f}")
                
                    if show_standard_model and dynamic_model_details:
                        st.write("---")
                        st.subheader("🎯 Standard Model Details")
                        st.write(f"**Method:** {dynamic_model_details.get('method', 'Tiered Historical Stats')}")
                        st.write(f"**Home Spread:** {model_home_spread:+.1f}")
                        if 'recent_games_window' in dynamic_model_details:
                            st.write(f"**Recent Games Window:** {dynamic_model_details['recent_games_window']}")
                            st.write(f"**Recent Form Weight:** {dynamic_model_details['recent_form_weight']:.1%}")
                        if 'hfa_value' in dynamic_model_details:
                            st.write(f"**Home Field Advantage:** {dynamic_model_details['hfa_value']:.1f} points")
                        if 'model_weights' in dynamic_model_details:
                            st.write(f"**Model Components:** EPA, Recent Form, SOS, HFA")
                        st.write(f"**Raw Model Result:** {dynamic_model_details.get('model_result_raw', 'N/A'):.1f}")
                        st.write(f"**Final Prediction:** {dynamic_model_details.get('predicted_spread', model_home_spread):+.1f}")

            else:
                st.warning("No model predictions available. Please check data files.")

            # --- Betting Value Calculator ---
            if show_simple_model and simple_model_spread is not None:
                st.divider()
                st.subheader("💰 Betting Value Assessment")
            
                # Calculate all edges
                simple_edge = home_spread_vegas - simple_model_spread if simple_model_spread else 0
                standard_edge = home_spread_vegas - model_home_spread if model_home_spread else 0
                total_edge = simple_model_total - total_line if simple_model_total and total_line else 0
            
                value_col1, value_col2, value_col3 = st.columns(3)
            
                with value_col1:
                    st.markdown("**🎯 Spread Value**")
                
                    if abs(simple_edge) >= 2:
                        confidence = "High 🔥" if abs(simple_edge) >= 3 else "Medium 💡"
                        best_bet = home_abbr if simple_edge > 0 else away_abbr
                        edge_points = abs(simple_edge)
                    
                        st.success(f"**{best_bet}** {confidence}")
                        st.metric("Model Edge", f"{edge_points:.1f} points",
                                 help="How many points better than Vegas line")
                    
                        # Simple value assessment
                        if edge_points >= 3:
                            st.caption("🔥 Strong betting value detected")
                        else:
                            st.caption("💡 Moderate betting value")
                        
                    elif abs(simple_edge) >= 1:
                        best_bet = home_abbr if simple_edge > 0 else away_abbr
                        st.info(f"**{best_bet}** - Slight Edge")
                        st.metric("Model Edge", f"{abs(simple_edge):.1f} points")
                        st.caption("⚡ Minor betting value")
                    else:
                        st.warning("No Clear Value")
                        st.metric("Model Edge", f"{abs(simple_edge):.1f} points")
                        st.caption("😐 Line appears efficient")
            
                with value_col2:
                    if show_standard_model and model_home_spread is not None:
                        st.markdown("**🎲 Standard Model Value**")
                    
                        if abs(standard_edge) >= 2:
                            confidence = "High 🔥" if abs(standard_edge) >= 3 else "Medium 💡"
                            best_bet = home_abbr if standard_edge > 0 else away_abbr
                            edge_points = abs(standard_edge)
                        
                            st.success(f"**{best_bet}** {confidence}")
                            st.metric("Advanced Edge", f"{edge_points:.1f} points")
                        
                            if edge_points >= 3:
                                st.caption("🔥 Strong advanced value")
                            else:
                                st.caption("💡 Moderate advanced value")
                            
                        elif abs(standard_edge) >= 1:
                            best_bet = home_abbr if standard_edge > 0 else away_abbr
                            st.info(f"**{best_bet}** - Slight Edge")
                            st.metric("Advanced Edge", f"{abs(standard_edge):.1f} points")
                            st.caption("⚡ Minor advanced value")
                        else:
                            st.warning("No Clear Value")
                            st.metric("Advanced Edge", f"{abs(standard_edge):.1f} points")
                            st.caption("😐 Line appears efficient")
                    else:
                        st.markdown("**🎲 Standard Model**")
                        st.info("Enable Standard Model for advanced value analysis")
            
                with value_col3:
                    if simple_model_total and total_line:
                        st.markdown("**📊 Total Value**")
                    
                        if abs(total_edge) >= 2:
                            confidence = "High 🔥" if abs(total_edge) >= 3 else "Medium 💡"
                            best_total = "OVER" if total_edge > 0 else "UNDER"
                            edge_points = abs(total_edge)
                        
                            st.success(f"**{best_total}** {confidence}")
                            st.metric("Total Edge", f"{edge_points:.1f} points")
                        
                            if edge_points >= 3:
                                st.caption("🔥 Strong total value")
                            else:
                                st.caption("💡 Moderate total value")
                            
                        elif abs(total_edge) >= 1:
                            best_total = "OVER" if total_edge > 0 else "UNDER"
                            st.info(f"**{best_total}** - Slight Edge")
                            st.metric("Total Edge", f"{abs(total_edge):.1f} points")
                            st.caption("⚡ Minor total value")
                        else:
                            st.warning("No Clear Value")
                            st.metric("Total Edge", f"{abs(total_edge):.1f} points")
                            st.caption("😐 Total appears efficient")
                    else:
                        st.markdown("**📊 Total Value**")
                        st.info("Total analysis unavailable")
            
                # Overall recommendation
                st.markdown("**🎯 Overall Assessment**")
            
                # Count strong edges
                strong_edges = []
                if abs(simple_edge) >= 2:
                    strong_edges.append(f"Spread: {home_abbr if simple_edge > 0 else away_abbr}")
                if show_standard_model and model_home_spread and abs(standard_edge) >= 2:
                    strong_edges.append(f"Advanced: {home_abbr if standard_edge > 0 else away_abbr}")
                if simple_model_total and total_line and abs(total_edge) >= 2:
                    strong_edges.append(f"Total: {'OVER' if total_edge > 0 else 'UNDER'}")
            
                if len(strong_edges) >= 2:
                    st.success(f"🔥 **Multiple Value Opportunities**: {', '.join(strong_edges)}")
                elif len(strong_edges) == 1:
                    st.info(f"💡 **Single Value Play**: {strong_edges[0]}")
                else:
                    st.warning("😐 **No Strong Value Detected** - Consider waiting for better spots")
            
                # Add disclaimers
                st.caption("⚠️ **Disclaimer**: This analysis is for entertainment purposes only. Always gamble responsibly and within your means.")

# Power Rankings Tab
with main_tab2:
    st.header("🏆 NFL Power Rankings")
    st.caption(f"Based on EPA performance through Week {CURRENT_WEEK-1}, {CURRENT_YEAR}")
    
    def get_all_team_stats(pbp_data, current_year, current_week, include_trends=False):
        """Get stats for all teams"""
        team_stats = {}
        
        # Get all unique teams
        all_teams = set(pbp_data['posteam'].dropna().unique()) | set(pbp_data['defteam'].dropna().unique())
        all_teams = {team for team in all_teams if pd.notna(team) and team != ''}
        
        for team in all_teams:
            # Get team's games from current season (weeks 1 to current-1)
            current_season_games = pbp_data[
                (pbp_data['season'] == current_year) & 
                (pbp_data['week'] < current_week) &
                ((pbp_data['posteam'] == team) | (pbp_data['defteam'] == team))
            ]
            
            if len(current_season_games) == 0:
                continue
                
            # Calculate current season stats
            current_off_plays = current_season_games[current_season_games['posteam'] == team]
            current_avg_epa_off = current_off_plays['epa'].mean() if len(current_off_plays) > 0 else 0
            
            current_def_plays = current_season_games[current_season_games['defteam'] == team]
            current_avg_epa_def = current_def_plays['epa'].mean() if len(current_def_plays) > 0 else 0
            
            games_played = len(current_season_games.groupby(['game_id']))
            
            stats = {
                'team': team,
                'games_played': games_played,
                'epa_offense': current_avg_epa_off,
                'epa_defense_raw': current_avg_epa_def,
                'epa_defense_display': -current_avg_epa_def,  # Flipped for display
                'total_plays': len(current_off_plays) + len(current_def_plays),
                'net_epa': current_avg_epa_off - current_avg_epa_def  # Combined metric
            }
            
            # Add historical trends if requested
            if include_trends and current_year == 2025:
                historical_games = pbp_data[
                    (pbp_data['season'].isin([2022, 2023, 2024])) &
                    ((pbp_data['posteam'] == team) | (pbp_data['defteam'] == team))
                ]
                
                if len(historical_games) > 0:
                    hist_off_plays = historical_games[historical_games['posteam'] == team]
                    hist_avg_epa_off = hist_off_plays['epa'].mean() if len(hist_off_plays) > 0 else 0
                    
                    hist_def_plays = historical_games[historical_games['defteam'] == team]
                    hist_avg_epa_def = hist_def_plays['epa'].mean() if len(hist_def_plays) > 0 else 0
                    
                    stats.update({
                        'offense_trend': current_avg_epa_off - hist_avg_epa_off,
                        'defense_trend': (-current_avg_epa_def) - (-hist_avg_epa_def),
                        'has_trends': True
                    })
                else:
                    stats['has_trends'] = False
            else:
                stats['has_trends'] = False
                
            team_stats[team] = stats
            
        return team_stats
    
    try:
        # Get PBP data for rankings with session state caching
        cache_key = f"rankings_data_{CURRENT_YEAR}_{CURRENT_WEEK}"
        if cache_key not in st.session_state:
            with st.spinner("Loading data for Power Rankings..."):
                pbp_data = load_rolling_data(CURRENT_YEAR)
            
            if pbp_data.empty:
                st.error("❌ Cannot load Power Rankings - no data available.")
                st.stop()
                
            st.session_state[cache_key] = pbp_data
        else:
            pbp_data = st.session_state[cache_key]
            
        team_stats = get_all_team_stats(pbp_data, CURRENT_YEAR, CURRENT_WEEK, include_trends=show_season_trends)
        
        if not team_stats:
            st.warning("No team data available for current week range")
        else:
            # Create three ranking sections
            rank_tab1, rank_tab2, rank_tab3 = st.tabs(["🚀 Offensive Leaders", "🛡️ Defensive Leaders", "⚡ Overall Power Rankings"])
            
            # Helper function to display rankings in compact table format
            def display_team_ranking_compact(teams_data, metric_key, title, is_descending=True, help_text=""):
                st.subheader(title)
                if help_text:
                    st.caption(help_text)
                
                # Sort teams by metric
                sorted_teams = sorted(teams_data.items(), 
                                    key=lambda x: x[1][metric_key], 
                                    reverse=is_descending)
                
                # Create compact table data
                table_data = []
                for rank, (team, stats) in enumerate(sorted_teams, 1):
                    if stats['games_played'] == 0:
                        continue
                        
                    # Medal emoji for top 3
                    rank_display = "🥇" if rank == 1 else "�" if rank == 2 else "�" if rank == 3 else str(rank)
                    
                    # Get metric value
                    value = stats[metric_key]
                    
                    # Format trend if available
                    trend_text = ""
                    if show_season_trends and stats.get('has_trends', False):
                        if 'offense' in metric_key and 'offense_trend' in stats:
                            trend = stats['offense_trend']
                            trend_icon = "📈" if trend > 0.02 else "📉" if trend < -0.02 else "━"
                            trend_text = f"{trend:+.3f} {trend_icon}"
                        elif 'defense' in metric_key and 'defense_trend' in stats:
                            trend = stats['defense_trend']
                            trend_icon = "📈" if trend > 0.02 else "📉" if trend < -0.02 else "━"
                            trend_text = f"{trend:+.3f} {trend_icon}"
                        elif metric_key == 'net_epa':
                            off_trend = stats.get('offense_trend', 0)
                            def_trend = stats.get('defense_trend', 0)
                            net_trend = off_trend + def_trend
                            trend_icon = "📈" if net_trend > 0.02 else "📉" if net_trend < -0.02 else "━"
                            trend_text = f"{net_trend:+.3f} {trend_icon}"
                    
                    if not trend_text:
                        trend_text = f"{stats['games_played']}G"
                    
                    table_data.append({
                        "Rank": rank_display,
                        "Team": team,
                        "EPA": f"{value:+.3f}",
                        "Trend/Games": trend_text
                    })
                
                # Display as dataframe for compact view
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True, height=min(400, len(table_data) * 35 + 38))
                else:
                    st.info("No team data available")
            
            # Offensive Leaders Tab
            with rank_tab1:
                display_team_ranking_compact(
                    team_stats, 
                    'epa_offense', 
                    "🚀 Best Offensive Teams",
                    help_text="Teams ranked by offensive EPA per play • Trend vs 2022-24 baseline"
                )
            
            # Defensive Leaders Tab  
            with rank_tab2:
                display_team_ranking_compact(
                    team_stats, 
                    'epa_defense_display', 
                    "🛡️ Best Defensive Teams", 
                    help_text="Teams ranked by defensive EPA per play • Positive = better defense • Trend vs 2022-24"
                )
            
            # Overall Power Rankings Tab
            with rank_tab3:
                display_team_ranking_compact(
                    team_stats, 
                    'net_epa', 
                    "⚡ Overall Power Rankings",
                    help_text="Net EPA = Offensive EPA - Defensive EPA (raw) • Higher = better overall team"
                )
                
                # Add summary stats
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                # Get top teams
                overall_rankings = sorted(team_stats.items(), key=lambda x: x[1]['net_epa'], reverse=True)
                
                with col1:
                    if overall_rankings:
                        top_team, top_stats = overall_rankings[0]
                        st.metric("🥇 Best Overall", top_team, f"{top_stats['net_epa']:+.3f} Net EPA")
                
                with col2:
                    # Best offense
                    off_rankings = sorted(team_stats.items(), key=lambda x: x[1]['epa_offense'], reverse=True)
                    if off_rankings:
                        best_off_team, best_off_stats = off_rankings[0]
                        st.metric("🚀 Best Offense", best_off_team, f"{best_off_stats['epa_offense']:+.3f} EPA")
                
                with col3:
                    # Best defense
                    def_rankings = sorted(team_stats.items(), key=lambda x: x[1]['epa_defense_display'], reverse=True)
                    if def_rankings:
                        best_def_team, best_def_stats = def_rankings[0]
                        st.metric("🛡️ Best Defense", best_def_team, f"{best_def_stats['epa_defense_display']:+.3f} EPA")
                
    except Exception as e:
        st.error(f"Error loading power rankings: {str(e)}")
        st.caption("Please try refreshing or selecting a different week.")

# --- Footer ---
st.divider()
st.markdown("### 📈 About the Models")
st.markdown("""
- **Simple Model**: Pure EPA-based predictions using team offensive/defensive efficiency
- **Standard Model**: Complex model with strength-of-schedule adjustments and dynamic factors
- **Total Predictions**: Expected points based on offensive vs defensive EPA matchups
- **Edge Analysis**: Difference between our model and Vegas lines to identify betting opportunities
""")

st.caption("Data updated daily. Model predictions for entertainment purposes only.")
