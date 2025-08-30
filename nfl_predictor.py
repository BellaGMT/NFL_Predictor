import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# -----------------------------
# Team Aliases for user input
# -----------------------------
TEAM_ALIASES = {
    'ARI': ['arizona cardinals', 'cardinals', 'ari'],
    'ATL': ['atlanta falcons', 'falcons', 'atl'],
    'BAL': ['baltimore ravens', 'ravens', 'bal'],
    'BUF': ['buffalo bills', 'bills', 'buf'],
    'CAR': ['carolina panthers', 'panthers', 'car'],
    'CHI': ['chicago bears', 'bears', 'chi'],
    'CIN': ['cincinnati bengals', 'bengals', 'cin'],
    'CLE': ['cleveland browns', 'browns', 'cle'],
    'DAL': ['dallas cowboys', 'cowboys', 'dal'],
    'DEN': ['denver broncos', 'broncos', 'den'],
    'DET': ['detroit lions', 'lions', 'det'],
    'GB': ['green bay packers', 'packers', 'gb'],
    'HOU': ['houston texans', 'texans', 'hou'],
    'IND': ['indianapolis colts', 'colts', 'ind'],
    'JAX': ['jacksonville jaguars', 'jaguars', 'jax'],
    'KC': ['kansas city chiefs', 'chiefs', 'kc'],
    'LV': ['las vegas raiders', 'raiders', 'lv'],
    'LAC': ['los angeles chargers', 'chargers', 'lac'],
    'LAR': ['los angeles rams', 'rams', 'lar'],
    'MIA': ['miami dolphins', 'dolphins', 'mia'],
    'MIN': ['minnesota vikings', 'vikings', 'min'],
    'NE': ['new england patriots', 'patriots', 'ne'],
    'NO': ['new orleans saints', 'saints', 'no'],
    'NYG': ['new york giants', 'giants', 'nyg'],
    'NYJ': ['new york jets', 'jets', 'nyj'],
    'PHI': ['philadelphia eagles', 'eagles', 'phi'],
    'PIT': ['pittsburgh steelers', 'steelers', 'pit'],
    'SEA': ['seattle seahawks', 'seahawks', 'sea'],
    'SF': ['san francisco 49ers', '49ers', 'sf'],
    'TB': ['tampa bay buccaneers', 'buccaneers', 'tb'],
    'TEN': ['tennessee titans', 'titans', 'ten'],
    'WAS': ['washington commanders', 'commanders', 'was']
}

def get_team_code(user_input):
    user_input = user_input.lower().strip()
    for code, aliases in TEAM_ALIASES.items():
        if user_input in [alias.lower() for alias in aliases]:
            return code
    raise ValueError(f"Team '{user_input}' not recognized. Please try again.")

# -----------------------------
# Load Game Data
# -----------------------------
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
# Train Team Models
# -----------------------------
def train_team_models(X, y_class, y_reg_scores, y_reg_spread):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_class, test_size=0.2, stratify=y_class, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    print(f"[Classifier] Win/Loss Accuracy: {accuracy_score(y_te, clf.predict(X_te)):.2f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg_scores, test_size=0.2, random_state=42)
    reg_scores = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_scores.fit(X_tr, y_tr)
    print(f"[Regressor] Score MAE: {mean_absolute_error(y_te, reg_scores.predict(X_te)):.2f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg_spread, test_size=0.2, random_state=42)
    reg_spread = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_spread.fit(X_tr, y_tr)
    print(f"[Regressor] Spread MAE: {mean_absolute_error(y_te, reg_spread.predict(X_te)):.2f}")

    return clf, reg_scores, reg_spread

# -----------------------------
# Train Player Model
# -----------------------------
def train_player_model(player_stats):
    feature_cols = ['season','week']  # Can add more features as needed
    targets = ['passing_yards','rushing_yards','receiving_yards']
    y = player_stats[targets].fillna(0)
    X = player_stats[feature_cols].fillna(0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    for col in targets:
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, y_tr[col])
        models[col] = reg
        y_pred = reg.predict(X_te)
        mae = np.mean(np.abs(y_pred - y_te[col]))
        print(f"[Player Regressor] {col} MAE: {mae:.2f}")
    return models

# -----------------------------
# Explain Game Prediction
# -----------------------------
def explain_game_prediction(home_team, away_team, clf, reg_scores, reg_spread, team_map, season=2024, week=1):
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

# -----------------------------
# Explain Player Prediction
# -----------------------------
def explain_player_prediction(player_name, reg_player, season=2024, week=1):
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
        f"You are an NFL assistant. Answer the user's question using the info below.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {user_question}\n"
        f"Answer clearly and in simple terms."
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role":"user", "content": prompt}]
    )
    return response.choices[0].message.content

# -----------------------------
# Main Function
# -----------------------------
def main():
    print("Loading NFL game data...")
    games = load_game_data()
    X, y_class, y_reg_scores, y_reg_spread, team_map = prepare_team_features(games)

    print("Training team models...")
    clf, reg_scores, reg_spread = train_team_models(X, y_class, y_reg_scores, y_reg_spread)

    print("Loading player stats...")
    player_stats = load_player_data()

    print("Training player models...")
    reg_player = train_player_model(player_stats)

    # User-friendly input
    home_input = "Cowboys"
    away_input = "49ers"
    home_team = get_team_code(home_input)
    away_team = get_team_code(away_input)

    game_info = explain_game_prediction(home_team, away_team, clf, reg_scores, reg_spread, team_map)
    player_info = explain_player_prediction("Dak Prescott", reg_player)
    play_info = "Next play prediction: PASS or RUSH can be implemented separately."

    print("\n--- Game & Player Predictions ---")
    print(game_info)
    print(player_info)

    # LLM example
    user_question = f"Who will likely win the {home_input} vs {away_input} game and what should I expect from Dak Prescott?"
    answer = query_nfl_llm(user_question, game_info, player_info, play_info)
    print("\n--- LLM Answer ---")
    print(answer)

if __name__ == "__main__":
    main()
