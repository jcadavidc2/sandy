"""Builder for notebooks/football_explore.ipynb.

Run: .venv/bin/python notebooks/football_explore.py
Then executes inline so outputs are embedded.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Sandy Football — Data Exploration

Explores the `football` schema we just backfilled from API-Football
(senior men's national-team competitions, 2019–2026).

Tables:
- **football.matches** — one row per fixture (goals, competition, weight, status)
- **football.teams** — national teams (name, country, fifa_code)
- **football.match_stats** — per-team match stats (corners/cards/possession) — *trickling in*
- **football.team_ratings** — Dixon-Coles attack/defense snapshots — *populated by `fit_and_persist`*
- **football.match_predictions** — predictions + reconciled outcomes — *populated by the predictor*

Run on the EC2 with the DB env vars set.""")

md("## 1. Connect")
code("""import pandas as pd
from sqlalchemy import text
from sandy.config import load_config
from sandy.db import create_engine

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
cfg = load_config()
engine = create_engine(cfg)

def q(sql, **params):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)

print("Connected to", cfg.database.name)""")

md("## 2. Overview — row counts per table")
code("""q('''
SELECT 'matches' AS table, count(*) AS rows FROM football.matches
UNION ALL SELECT 'teams', count(*) FROM football.teams
UNION ALL SELECT 'match_stats', count(*) FROM football.match_stats
UNION ALL SELECT 'team_ratings', count(*) FROM football.team_ratings
UNION ALL SELECT 'match_predictions', count(*) FROM football.match_predictions
ORDER BY rows DESC
''')""")

md("## 3. Matches by calendar year\nNote how qualifier 'season' buckets carry matches into 2025–2026 — that's our recent form.")
code("""q('''
SELECT EXTRACT(YEAR FROM match_date)::int AS year,
       count(*) AS matches,
       count(*) FILTER (WHERE status IN ('FT','AET','PEN')) AS finished,
       round(avg(home_goals + away_goals) FILTER (WHERE status IN ('FT','AET','PEN'))::numeric, 2) AS avg_total_goals
FROM football.matches GROUP BY year ORDER BY year
''')""")

md("## 4. Matches by competition (importance weight)")
code("""q('''
SELECT competition, league_id, competition_weight AS weight, count(*) AS matches,
       min(match_date) AS first, max(match_date) AS last
FROM football.matches GROUP BY competition, league_id, competition_weight
ORDER BY competition_weight DESC, matches DESC
''')""")

md("## 5. Sample: most recent finished matches")
code("""q('''
SELECT m.match_date, m.competition,
       th.name AS home, m.home_goals, m.away_goals, ta.name AS away, m.status
FROM football.matches m
JOIN football.teams th ON th.team_id = m.home_team_id
JOIN football.teams ta ON ta.team_id = m.away_team_id
WHERE m.status IN ('FT','AET','PEN')
ORDER BY m.match_date DESC LIMIT 15
''')""")

md("## 6. WC2026 qualifiers played in 2025–2026 (recent competitive form)")
code("""q('''
SELECT m.match_date, m.competition,
       th.name AS home, m.home_goals, m.away_goals, ta.name AS away
FROM football.matches m
JOIN football.teams th ON th.team_id = m.home_team_id
JOIN football.teams ta ON ta.team_id = m.away_team_id
WHERE m.competition LIKE 'World Cup - Qualification%' AND m.match_date >= '2025-01-01'
ORDER BY m.match_date DESC LIMIT 15
''')""")

md("## 7. Teams sample")
code("""q('''
SELECT team_id, name, fifa_code, country
FROM football.teams WHERE country IS NOT NULL ORDER BY name LIMIT 20
''')""")

md("## 8. Goal-scoring leaders (avg goals for / against per match, min 20 matches)")
code("""q('''
WITH per_team AS (
  SELECT home_team_id AS team_id, home_goals AS gf, away_goals AS ga FROM football.matches WHERE status IN ('FT','AET','PEN')
  UNION ALL
  SELECT away_team_id, away_goals, home_goals FROM football.matches WHERE status IN ('FT','AET','PEN')
)
SELECT t.name, count(*) AS matches,
       round(avg(p.gf)::numeric,2) AS gf_per_game,
       round(avg(p.ga)::numeric,2) AS ga_per_game,
       round((avg(p.gf)-avg(p.ga))::numeric,2) AS goal_diff
FROM per_team p JOIN football.teams t ON t.team_id = p.team_id
GROUP BY t.name HAVING count(*) >= 20
ORDER BY goal_diff DESC LIMIT 15
''')""")

md("## 9. match_stats (corners/cards/possession)\nTrickling in within the API daily cap — likely empty or partial for now.")
code("""print('statted matches:', q("SELECT count(DISTINCT fixture_id) AS n FROM football.match_stats").iloc[0]['n'])
q('''
SELECT s.fixture_id, t.name AS team, s.is_home, s.possession, s.shots_total,
       s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards, s.xg
FROM football.match_stats s JOIN football.teams t ON t.team_id = s.team_id
LIMIT 10
''')""")

md("## 10. team_ratings & predictions\nPopulated once `sandy.football.ratings.fit_and_persist()` and the predictor run (F3).")
code("""print('team_ratings rows:', q("SELECT count(*) AS n FROM football.team_ratings").iloc[0]['n'])
print('predictions rows:', q("SELECT count(*) AS n FROM football.match_predictions").iloc[0]['n'])""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = "/home/ec2-user/sandy/notebooks/football_explore.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
