# nfl_predictor_streamlit.py

import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from openai import OpenAI

# Initialize OpenAI client using Streamlit secrets
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

    pass_stats = stats[stats['passer_player_id'].notna()].groupby(
        ['season','week','posteam','passer_player_id','passer_player_name']
    ).agg({'passing_yards':'sum','pass_touchdown':'sum'}).reset_index()

    rush_stats = stats[stats['rusher_player_id'].notna()].groupby(
        ['season','week','posteam','rusher_player_id','rusher_player_name']
    ).agg({'rushing_yards':'sum','rush_touchdown':'sum'}).reset_index()

    rec_stats = stats[stats['receiver_player_id'].notna()].groupby(
        ['season','week','posteam','receiver_player_id','receiver_player_name']
    ).agg({'receiving_yards':'sum','touchdown':'sum'}).reset_index()

    player_stats = pass_stats.merge(
        rush_stats,
        left_on=['season','week','posteam','passer_player_id','passer_player_name'],
        right_on=['season','week','posteam','rusher_player_id','rusher_player_name'],
        how='outer'
    )
    player_stats = player_stats.merge(
        rec_stats,
        left_on=['season','week','posteam','passer_player_id','passer_player_name'],
        right_on=['season','week','posteam','receiver_player_id','receiver_player_name'],
        how='outer'
    )

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
    return X, y_class, y_reg_scores, y_reg_spread, team_map, teams

# -----------------------------
# Train Team Models
# -----------------------------
@st.cache_resource
def train_team_models(X, y_class, y_reg_scores, y_reg_spread):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y_class)

    reg_scores = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_scores.fit(X, y_reg_scores)

    reg_spread = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_spread.fit(X, y_reg_spread)

    return clf, reg_scores, reg_spread

# -----------------------------
# Train Player Model
# -----------------------------
@st.cache_resource
def train_player_model(player_stats):
    feature_cols = ['season','week']
    targets = ['passing_yards','rushing_yards','receiving_yards']
    y = player_stats[targets].fillna(0)
    X = player_stats[feature_cols].fillna(0)
    models = {}
    for col in targets:
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X, y[col])
        models[col] = reg
    return models

# -----------------------------
# Game Prediction
# -----------------------------
def predict_game(home_team, away_team, season, week, clf, reg_scores, reg_spread, team_map):
    home_id = team_map[home_team]
    away_id = team_map[away_team]
    X_new = pd.DataFrame([{'home_team_id': home_id,'away_team_id': away_id,'season': season,'week': week}])
    pred_class = clf.predict(X_new)[0]
    prob = clf.predict_proba(X_new)[0]
    winner = home_team if pred_class==1 else away_team
    pred_scores = reg_scores.predict(X_new)[0]
    home_score, away_score = int(pred_scores[0]), int(pred_scores[1])
    pred_spread = reg_spread.predict(X_new)[0]

    return winner, prob, home_score, away_score, pred_spread

# -----------------------------
# Player Prediction
# -----------------------------
def predict_player(player_name, reg_player, season, week):
    X_new = pd.DataFrame([{'season':season,'week':week}])
    pred = {col: reg.predict(X_new)[0] for col, reg in reg_player.items()}
    return pred

# -----------------------------
# LLM Query
# -----------------------------
def query_nfl_llm(user_question, game_info="", player_info="", play_info=""):
    context = f"{game_info}\n{player_info}\n{play_info}"
    prompt = (
        f"You are an NFL assistant. Answer clearly in simple terms using the info below.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {user_question}"
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role":"user", "content": prompt}]
    )
    return response.choices[0].message.content

# -----------------------------
# Streamlit App
# -----------------------------
st.title("NFL Game & Player Predictor")
st.markdown("Predict game outcomes, player stats, and ask natural-language questions using AI!")

# Load data
games = load_game_data()
player_stats = load_player_data()
X, y_class, y_reg_scores, y_reg_spread, team_map, teams = prepare_team_features(games)

st.sidebar.header("Select Game")
home_team = st.sidebar.selectbox("Home Team", teams)
away_team = st.sidebar.selectbox("Away Team", [t for t in teams if t != home_team])
season = st.sidebar.number_input("Season", min_value=1999, max_value=2024, value=2024)
week = st.sidebar.number_input("Week", min_value=1, max_value=18, value=1)

# Train models
clf, reg_scores, reg_spread = train_team_models(X, y_class, y_reg_scores, y_reg_spread)
reg_player = train_player_model(player_stats)

# Predictions
winner, prob, home_score, away_score, spread = predict_game(home_team, away_team, season, week, clf, reg_scores, reg_spread, team_map)
st.subheader("Game Prediction")
st.write(f"**Winner:** {winner} (Home: {prob[1]*100:.1f}% / Away: {prob[0]*100:.1f}%)")
st.write(f"**Predicted Score:** {home_team} {home_score} - {away_score} {away_team}")
st.write(f"**Expected Spread:** {spread:.1f} points")

# Player selection
player_name = st.text_input("Player Name (for stats prediction)", value="Dak Prescott")
player_pred = predict_player(player_name, reg_player, season, week)
st.subheader(f"{player_name} Prediction")
st.write(f"- Passing Yards: ~{player_pred['passing_yards']:.1f}")
st.write(f"- Rushing Yards: ~{player_pred['rushing_yards']:.1f}")
st.write(f"- Receiving Yards: ~{player_pred['receiving_yards']:.1f}")

# LLM question
user_question = st.text_input("Ask a question about this game or player")
if st.button("Ask AI"):
    game_info = f"{home_team} vs {away_team}: Predicted winner {winner}, score {home_score}-{away_score}, spread {spread:.1f}"
    player_info = f"{player_name} projected stats: Pass ~{player_pred['passing_yards']:.1f}, Rush ~{player_pred['rushing_yards']:.1f}, Rec ~{player_pred['receiving_yards']:.1f}"
    play_info = "Next play predictions: PASS/RUSH (can be implemented later)"
    answer = query_nfl_llm(user_question, game_info, player_info, play_info)
    st.subheader("AI Answer")
    st.write(answer)
