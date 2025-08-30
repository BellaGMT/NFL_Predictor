# NFL Predictor (Stats + LLM Explanations)

This starter lets you predict **win/loss**, **final score**, **point spread**, and **QB passing yards** (if data is available), then uses an **LLM** to generate clear, human-readable explanations.

**You do not need to code** to use it—just fill the CSVs (or replace them with your own), set your OpenAI API key, and run.

---

## Quick Start

1) Install dependencies:
```bash
pip install pandas numpy scikit-learn openai matplotlib
```

2) Set your OpenAI API key (skip if you use `--no-llm`):
- macOS/Linux:
```bash
export OPENAI_API_KEY="sk-..."
```
- Windows (Powershell):
```powershell
setx OPENAI_API_KEY "sk-..."
```

3) Run with the sample data:
```bash
python nfl_predictor.py   --historical data/historical_games_sample.csv   --upcoming data/upcoming_games_sample.csv   --out predictions.md
```

4) Open the generated **predictions.md** to see the result.

---

## CSV Schemas

### `historical_games.csv` (train data)
Each row = a past NFL game.

Required columns (case-sensitive):
- `date` (YYYY-MM-DD)
- `home_team`, `away_team`
- `home_score`, `away_score`

Optional but recommended **numeric** columns (the script will auto-use whatever numeric columns exist):
- Team strength: `home_elo`, `away_elo`
- Offense: `home_points_pg`, `away_points_pg`, `home_yards_per_play`, `away_yards_per_play`
- Defense: `home_points_allowed_pg`, `away_points_allowed_pg`
- Situational: `home_rest_days`, `away_rest_days`, `home_injured_starters`, `away_injured_starters`
- Market: `spread_close` (closing spread)
- Player stats: `home_qb_passing_yds`, `away_qb_passing_yds` (these help train QB yard models)

The script auto-generates targets:
- `home_win` = 1 if `home_score > away_score`, else 0
- `spread_actual` = `home_score - away_score`

### `upcoming_games.csv` (predict data)
Each row = a future NFL game you want to predict.

Required:
- `date` (YYYY-MM-DD)
- `home_team`, `away_team`

Optional numeric columns (any columns that also exist in historical will be used as features):
- Same flavor as above: `home_elo`, `away_elo`, points per game, yards/play, injuries, rest days, etc.

Optional player context (for nicer explanations, not required):
- `home_qb_name`, `away_qb_name`

> Tip: If you don't have many features, predictions still run using what you provide.

---

## What It Predicts

- **Win/Loss** (classification)
- **Final score** for each team (regression)
- **Point spread** = (home_score_pred − away_score_pred)
- **QB passing yards** (if your historical has `*_qb_passing_yds`)

Then the **LLM** produces a short explanation per game using the data + predictions.

---

## Notes

- This is a baseline. Accuracy improves with richer, cleaner features and more historical data.
- You can wire in data from sources like **nflfastR**, **Pro Football Reference**, or **ESPN APIs** by exporting to CSV with the columns above.
- Use `--no-llm` to skip explanations when testing.

---

## Example Command (no LLM)
```bash
python nfl_predictor.py --historical data/historical_games_sample.csv --upcoming data/upcoming_games_sample.csv --out predictions.md --no-llm
```

Enjoy!
