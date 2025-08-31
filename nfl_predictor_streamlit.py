import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from openai import OpenAI

# -----------------------------
# OpenAI Client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Load Game Data
# -----------------------------
@st.cache_data
def load_game_data(start_year=1999, end_year=2024):
    years = list(range(start_year, end_year + 1))
    games = nfl.import_schedules(years)
    df = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].dropna()
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['point_diff'] = df['home_score'] - df['away_score']
    return df

# -----------------------------
# Load Player Data
# -----------------------------
@st.cache_data
def load_player_data(start_year=2022, end_year=2024):
    years = list(range(start_year, end_year + 1))
    stats = nfl.import_pbp_data(years, downcast=True, cache=False)

    # PASSING
    pass_stats = stats[stats['passer_player_id'].notna()]
    pass_stats = pass_stats.groupby(['season','week','posteam','passer_player_id','passer_player_name']) \
                           .agg({'passing_yards':'sum','pass_touchdown':'sum'}).reset_index()

    # RUSHING
    rush_stats = stats[stats['rusher_player_id'].notna()]
    rush_stats = rush_stats.groupby(['season','week','posteam','rusher_player_id','rusher_player_name']) \
                           .agg({'rushing_yards':'sum','rush_touchdown':'sum'}).reset_index()

    # RECEIVING
    rec_stats = stats[stats['receiver_player_id'].notna()]
    rec_stats = rec_stats.groupby(['season','week','posteam','receiver_player_id','receiver_player_name']) \
                         .agg({'receiving_yards':'sum','touchdown':'sum'}).reset_index()

    # Merge
    player_stats = pass_stats.merge(rush_stats, left_on=['season','week','posteam','passer_player_id','passer_player_name'],
                                    right_on=['season','week','posteam','rusher_player_id','rusher_player_name'], how='outer')
    player_stats = player_stats.merge(rec_stats, left_on=['season','week','posteam','passer_player_id','passer_player_name'],
                                      right_on=['season','week','posteam','receiver_player_id','receiver_player_name'], how='outer')

    player_stats[['passing_yards','pass_touchdown','rushing_yards','rush_touchdown','receiving_yards','touchdown']] = \
        player_stats[['passing_yards','pass_touchdown','rushing_yards','rush_touchdown','receiving_yards','touchdown']].fillna(0)

    return player_stats

# -----------------------------
# Prepare Team Features
# -----------------------------
def prepare_team_features(df):
    teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
    team_map = {team: i for i, team in enumerate(teams)}
    df['home_team_id'] = df['home_team'].map(team_map)
    df['away_team_id'] = df['away_team'].map(team_map)
    X = df[['home_team_id','away_team_id','season','week']]
    y_class = df['home_win']
    y_reg_scores = df[['home_score','away_score']]
    y_reg_spread = df['point_diff']
    return X, y_class, y_reg_scores, y_reg_spread, team_map

# -----------------------------
# Train Models
# -----------------------------
@st.cache_resource
def train_team_models(X, y_class, y_reg_scores, y_reg_spread):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_class, test_size=0.2, stratify=y_class, random_state=42)
    clf.fit(X_tr, y_tr)

    reg_scores = RandomForestRegressor(n_estimators=200, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg_scores, test_size=0.2, random_state=42)
    reg_scores.fit(X_tr, y_tr)

    reg_spread = RandomForestRegressor(n_estimators=200, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg_spread, test_size=0.2, random_state=42)
    reg_spread.fit(X_tr, y_tr)

    return clf, reg_scores, reg_spread

@st.cache_resource
def train_player_model(player_stats):
    feature_cols = ['season','week']
    targets = ['passing_yards','rushing_yards','receiving_yards']
    y = player_stats[targets].fillna(0)
    X = player_stats[feature_cols].fillna(0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {}
    for col in targets:
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, y_tr[col])
        models[col] = reg
    return models

# -----------------------------
# Game & Player Prediction
# -----------------------------
def predict_game(home_team, away_team, clf, reg_scores, reg_spread, team_map, season=2024, week=1):
    home_id = team_map[home_team]
    away_id = team_map[away_team]
    X_new = pd.DataFrame([{'home_team_id': home_id,'away_team_id': away_id,'season': season,'week': week}])
    pred_class = clf.predict(X_new)[0]
    prob = clf.predict_proba(X_new)[0]
    winner = home_team if pred_class==1 else away_team
    pred_scores = reg_scores.predict(X_new)[0]
    home_score, away_score = int(pred_scores[0]), int(pred_scores[1])
    pred_spread = reg_spread.predict(X_new)[0]

    return (
        f"Prediction for {home_team} vs {away_team}:\n"
        f"- Likely winner: {winner} (Home: {prob[1]*100:.1f}% / Away: {prob[0]*100:.1f}%)\n"
        f"- Predicted score: {home_team} {home_score} - {away_score} {away_team}\n"
        f"- Expected spread: {pred_spread:.1f} points"
    )

def predict_player(player_name, reg_player, season=2024, week=1):
    X_new = pd.DataFrame([{'season':season,'week':week}])
    pred = {col: reg.predict(X_new)[0] for col, reg in reg_player.items()}
    return (
        f"Player {player_name} is projected this game to:\n"
        f"- Throw for ~{pred['passing_yards']:.1f} yards\n"
        f"- Rush for ~{pred['rushing_yards']:.1f} yards\n"
        f"- Receive ~{pred['receiving_yards']:.1f} yards"
    )

# -----------------------------
# LLM Query
# -----------------------------
def query_nfl_llm(user_question, game_info="", player_info="", play_info=""):
    context = f"{game_info}\n{player_info}\n{play_info}"
    prompt = (
        f"You are an NFL assistant. The information below contains predictions based on historical game and player data.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {user_question}\n"
        f"Note: These are forecasts; consider any factors like recent injuries, trades, retirements, or other news that could affect the outcome, if known. "
        f"Answer clearly and in simple terms, explaining that the predictions are based on past performance."
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# -----------------------------
# Streamlit App
# -----------------------------
st.title("NFL Predictor 📊🏈")
st.write("This app predicts NFL game outcomes and player performance.")

# Load & Train Models
games = load_game_data()
X, y_class, y_reg_scores, y_reg_spread, team_map = prepare_team_features(games)
clf, reg_scores, reg_spread = train_team_models(X, y_class, y_reg_scores, y_reg_spread)
player_stats = load_player_data()
reg_player = train_player_model(player_stats)

# User Inputs
home_team = st.selectbox("Select Home Team", options=list(team_map.keys()))
away_team = st.selectbox("Select Away Team", options=list(team_map.keys()))
player_name = st.text_input("Enter Player Name (optional)", "")
week = st.number_input("Week", min_value=1, max_value=18, value=1)
season = st.number_input("Season", min_value=1999, max_value=2030, value=2025)
events = st.text_area("Enter recent events or news (optional)", "")

# Predictions
if st.button("Predict Game & Player"):
    game_info = predict_game(home_team, away_team, clf, reg_scores, reg_spread, team_map, season, week)

    # If player name is provided, predict player performance
    if player_name.strip():
        player_info = predict_player(player_name, reg_player, season, week)
    else:
        player_info = f"No specific player provided. Predictions focus on team-level outcomes."

    # Include user events in context for LLM
    if events.strip():
        context_info = f"{game_info}\n{player_info}\nRecent events: {events}"
    else:
        context_info = f"{game_info}\n{player_info}"

    play_info = "Next play prediction (PASS/RUSH) can be implemented separately."

    st.subheader("Predictions")
    st.write(game_info)
    if player_name.strip():
        st.write(player_info)

    user_question = f"Who will likely win the {home_team} vs {away_team} game"
    if player_name.strip():
        user_question += f" and what should I expect from {player_name}?"
    else:
        user_question += "?"

    llm_answer = query_nfl_llm(user_question, game_info=context_info, player_info="", play_info=play_info)
    st.subheader("LLM Explanation")
    st.write(llm_answer)
