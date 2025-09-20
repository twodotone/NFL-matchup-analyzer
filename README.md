# NFL Matchup Analyzer

A comprehensive NFL game analysis tool with EPA-based predictions, line movement tracking, and power rankings.

## Features

- **🏈 Game Analysis**: EPA-based predictions for spreads and totals
- **📈 Line Movement**: Track opening vs current line movements
- **🏆 Power Rankings**: Offensive, defensive, and overall team rankings
- **📊 Season Trends**: Compare current season vs historical performance
- **🎯 Betting Value**: Identify potential value opportunities

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

### Deployment Options

#### Railway (Recommended)
1. Fork this repository
2. Connect to Railway
3. Deploy automatically

#### Render
1. Fork this repository  
2. Connect to Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0`

#### Heroku
1. Add `Procfile`: `web: streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0`
2. Deploy via Git

## Project Structure

```
📁 nfl-analyzer/
├── 📄 app_streamlit.py              # Main Streamlit app
├── 📄 data_loader.py                # Data loading functionality  
├── 📄 stats_calculator.py           # EPA calculations
├── 📄 streamlit_simple_model.py     # Simple model implementation
├── 📄 streamlit_real_standard_model.py # Advanced model
├── 📄 line_movement_analyzer.py     # Line movement tracking
├── 📄 requirements.txt              # Dependencies
├── 📁 data/                         # Game data files
└── 📁 .streamlit/                   # Streamlit configuration
```

## Models

- **Simple Model**: Pure EPA-based predictions using team offensive/defensive efficiency
- **Standard Model**: Complex model with strength-of-schedule adjustments and dynamic factors

## Data Sources

- Play-by-play data from nfl_data_py
- Betting lines and odds
- Team performance metrics

## License

MIT License - see LICENSE file for details