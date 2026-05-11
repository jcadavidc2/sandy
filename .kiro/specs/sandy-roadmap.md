# Sandy — Product Roadmap

> **This is a living document, not a spec.** It captures the long-term vision for Sandy so no ideas get lost, and so each phase's spec can be designed with future phases in mind. Requirements here are NOT contractual. They will change. Detailed, EARS-style requirements live in each phase's own spec under `.kiro/specs/sandy-phase{N}/requirements.md`.

## What Sandy is

Sandy is a personal MLB game-day assistant. The end-state experience:

- You're watching a Mariners game. It's inning 2.
- You open Telegram and ask Sandy "will the M's get on base in inning 3?"
- Sandy thinks, consults specialist agents (live game state, historical model, weather, social chatter, odds movement), and replies with a probability plus a short narrative.
- Sandy can also push you proactively: "Inning 3 coming up, Mariners look spicy — want the prediction?"
- Over time Sandy tracks its own calls and learns what it's good at.

Mode: **vibes modeling**. Sandy is a fun personal project, not a wagering tool. Correctness over perfection. Not trying to beat Vegas.

## Core architectural commitments (must hold across all phases)

These are the things Phase 1's design must NOT close doors on:

1. **Model is a library, not a CLI-only artifact.** The `predict(...)` function must be importable from Python code, so future agents can call it directly.
2. **Data store is normalized and extensible.** Adding weather / social / odds tables later should be additive, no disruptive migrations.
3. **Config via env vars + TOML.** No hard-coded paths or hostnames. Moving between dev / EC2 / Telegram should need no code changes.
4. **Structured logs everywhere.** JSON logs so any future orchestrator or observability layer can parse them.
5. **Pure functions where possible.** Feature builders, label generators, predictors should be side-effect free so agents can wrap them however they want.
6. **Everything runs on a single t3.small for now.** No cloud services, no Kubernetes, no serverless until the project actually needs them.

## Phased plan

Each phase ships a thing that's useful on its own. No "big bang" reveals.

### Phase 1 — Offline foundation (currently scoped in `sandy-phase1/`)

- Postgres on EC2 (docker-compose)
- Ingestion from MLB Stats API: 3 seasons backfill + daily incremental
- Label generation: `reached_base` per (game, team, inning)
- Feature engineering: pitcher stats, lineup spots due up, recent team form
- LightGBM binary classifier, serialized to disk
- `sandy predict / ingest / train` CLI
- Property-based tests for idempotency, monotonicity, probability range, round-trip

Done-criteria: on the EC2, the operator can run `sandy predict --team SEA --opp LAD --inning 3 --starter "Walker Buehler"` and get a sensible JSON response.

### Phase 1.5 — Additional prediction targets

Immediately after Phase 1 ships. Uses the same data already in the DB — no new ingestion needed. Each target gets its own label generator, model artifact, and predict function.

- **Game winner:** P(home team wins). Binary classification. One row per game. Same features as reaches-base plus score differential context. Label: `home_team_wins = home_score > away_score`.
- **Total runs:** Expected runs per game or per inning. Regression (`objective: regression`). Label: actual runs scored.
- **Player props (hits, Ks, HRs):** Per (game, player) predictions. Needs player-centric feature builder (batter season stats, matchup history). Medium effort.
- CLI extension: `sandy predict --target game_winner --team SEA --opp LAD`
- Architecture: `--target` flag selects which model artifact to load and which predict function to call.

Done-criteria: `sandy predict --target game_winner --team SEA --opp LAD` returns a win probability.

### Phase 2 — Live game state + player-level features

- Poller service that watches active MLB games via the live feed
- In-memory (or Redis) cache of current state: inning, score, batter, pitcher, pitch count
- Python API: `get_current_game_state(team_code) -> dict`
- CLI extension: `sandy live --team SEA` prints current state
- Prediction is still CLI: `sandy predict --live SEA` uses live state as inputs

**Player-level feature enhancements (deferred from Phase 1):**
- `lineup_spot1_season_obp`, `lineup_spot2_season_obp`, `lineup_spot3_season_obp` — OBP of each specific batter due up, computed from `raw.plays` for the current season before game_date. Data already exists in DB; deferred because prediction-time requires knowing the actual lineup (available from live feed in Phase 2).
- `lineup_avg_recent_obp` — average OBP of the 3 due-up batters over trailing 15 games.
- `starter_vs_team_era` — this starter's ERA specifically against this batting team historically.
- When adding these: bump `FEATURE_SCHEMA_VERSION` to 3, update `inning_features` DDL with new columns, retrain model.

Done-criteria: you can run `sandy predict --live SEA --inning 3` during a game and get a prediction conditioned on what's actually happening right now.

### Phase 3 — Single chat agent on Telegram

- OpenCLAW bot already connected to Telegram (bot token restricted to user's Telegram ID)
- Single agent with MCP tools: `get_game`, `get_live_state`, `predict_hit_probability`
- Natural-language interface: "will mariners hit in inning 3?" → tool calls → answer
- Conversation memory (short-term, last N turns)

Done-criteria: ask a question in Telegram, get a useful answer about the current Mariners game.

### Phase 4 — Multi-agent with orchestrator + specialists

- Break the single agent into: Orchestrator, Live Game Agent, Model Agent, Context Agent
- Orchestrator decides who to call, in what order, how to merge
- Clarifying questions: "do you mean any baserunner or just hits?"
- Long-term memory of predictions + outcomes (separate table)

Done-criteria: orchestrator demonstrably routes intelligently (e.g., doesn't call Model Agent if the user asks "what's the score")

### Phase 5 — External context

- Context Agent: weather (NWS free API), umpire tendencies, ballpark factors, lineup confirmation
- Social Agent: Reddit game threads + a few beat-reporter RSS feeds, LLM summarizes themes
- These do NOT change the model features. They enrich the narrative layer.
- Orchestrator blends model output with contextual cues into a final answer

Done-criteria: Sandy produces answers that reference things like "wind blowing out to right" or "r/Mariners noting Julio looks off tonight"

### Phase 6 — Odds movement signal

- Integrate The Odds API free tier for pre-game lines
- Track how moneyline / total moved from open to first pitch
- Odds Agent exposes `get_market_mood(game_id)` to orchestrator
- Use as a sanity check / interesting commentary, not a feature in the model

Done-criteria: Sandy can say "market moved toward Dodgers today but my model is slightly higher on Mariners than the line implies"

### Phase 7 — Proactive push + self-evaluation

- Sandy watches your flagged games and pushes inning-break prompts unprompted
- Logs every prediction it makes alongside the actual outcome
- Weekly self-report: "I was right on 17 of 25 calls last week, best at inning 1, worst in extras"

Done-criteria: Sandy opens a Telegram message to you unprompted with a useful in-game update.

### Phase 8+ — Growth (unordered, grab-bag)

- More props: strikeouts, hits, HRs, first-to-score, total bases
- Multiple games at once (leaderboard view)
- Player-level props
- Voice input via Telegram voice messages
- Other leagues: NPB, KBO, fantasy integration
- Model ensemble (LightGBM + XGBoost + small neural net)
- Migrate storage off-EC2 when it outgrows the instance (probably RDS Postgres)
- GitHub repo + CI (currently local-only on EC2)

## Things we deliberately said NO to (for now)

- **Scoring/runs prediction.** Deferred; "reaches base" is the Phase 1 target.
- **Postseason games.** Excluded from training data (different statistical distribution).
- **X / Twitter as a source.** API is locked behind a $100+/mo tier. Reddit covers similar ground for free.
- **OddsJam, Action Network paid feeds.** Free tiers cover our vibes use case.
- **Web UI / dashboard as primary interface.** Telegram is the primary UX. Dashboard is debug-only.
- **Multi-user support.** Sandy is for one user (you).
- **Any paid inference APIs for the model.** LightGBM is local and free.
- **Building a custom PM/developer agent from scratch.** Kiro handles that at dev-time.
- **Kubernetes, serverless, microservices.** Overkill for a single-user app on a t3.small.

## How phases map to specs

- Phase 1 → `.kiro/specs/sandy-phase1/` (current)
- Phase 2 → `.kiro/specs/sandy-phase2/` (created when Phase 1 ships)
- ... and so on

Each phase's spec is written right before the phase begins, informed by what we learned building the previous one. The roadmap is updated freely between phases as ideas evolve.

## Open questions / things to decide later

- Model retraining cadence (weekly? daily? triggered by drift?)
- Where to store long-term conversation memory (Postgres vs. something purpose-built)
- Whether to ever expose Sandy to friends/family, and what that means for security
- Observability stack (probably just logs + a daily summary email to start)
- Whether to eventually open-source Sandy

## Over/Under σ (residual standard deviation) — RESOLVED May 2026

**History:** Originally used hardcoded σ=2.8. Accuracy at high probability thresholds started declining (May 7-8, 2026).

**Data analysis (May 8, 2026):**
- Overall actual game total σ = 4.46 (much higher than 2.8)
- By pitcher quality:
  - Both good starters (ERA < 3.5): σ = 3.08
  - Mixed starters (ERA 3.5-5.0): σ = 2.96
  - Both bad starters (ERA > 5.0): σ = 3.97
- Coors Field: σ = 5.28
- Residual σ (predicted vs actual): 3.27

**Solution implemented (May 9, 2026): Matchup-specific volatility model**
- LightGBM regression model trained on |actual_total - predicted_total| per game
- Uses the same 10 game features as the runs model (starter ERA/WHIP, trailing RPG, OBP, ballpark, is_home)
- Predicts a per-game σ (clamped to [1.0, 8.0]) instead of using a single hardcoded value
- Fallback: σ=3.3 (data-driven average) when model artifact not available
- Training metrics: MAE=2.14, mean_residual=3.5
- Observed range in production: σ ≈ 3.39–3.58 (varies by matchup quality)
- Model retrained nightly alongside runs/game_winner models
- DB columns added: `home_expected_runs`, `away_expected_runs`, `sigma_used`
- Code: `sandy/over_under/volatility.py`

**Monitoring:** If calibration shows 6.5 accuracy < 70% for 3+ consecutive days, investigate whether the volatility model needs hyperparameter tuning or additional features.

## Over/Under Meta-Model (Correctness Predictor) — May 2026

**Purpose:** A binary classifier that predicts P(our O5.5 prediction will be correct) for each game, using the prediction context as features. Complements the v1 threshold-based calibration with a learned model that captures feature interactions.

**Model:** LightGBM binary classifier, heavily regularized (num_leaves=8, min_data_in_leaf=15, max 50 rounds, early stopping at 20).

**Input features (9):**
1. `p_over_5_5` — our predicted probability
2. `sigma_used` — matchup-specific volatility
3. `home_starter_era` / `away_starter_era`
4. `home_trailing15_rpg` / `away_trailing15_rpg`
5. `ballpark_id`
6. `pitcher_fallback` (0/1)
7. `total_expected_runs` (home + away expected)

**Label:** `actual_over_5_5` (boolean, populated at 1 AM reconciliation)

**Training data:** All reconciled games from `derived.over_under_outcomes` where `actual_over_5_5` is not null and `sigma_used` is not null. Minimum 50 games required to train.

**Pipeline integration:**
- Nightly (1 AM PST): Retrained in Step 5 alongside runs/game_winner/volatility
- Morning (7 AM PST): Each prediction scored with P(correct), shown in Telegram

**Telegram output:** Additional section at the bottom of morning predictions:
```
🤖 Meta-model picks (P(correct) from 9 features):
  CLE vs LAA  O5.5=84.6%  P(correct)=91%  σ=3.44 🥇
  TOR vs TB   O5.5=78.5%  P(correct)=87%  σ=3.43 🥈
  HOU vs SEA  O5.5=81.1%  P(correct)=84%  σ=3.49 🥉
  BAL vs NYY  O5.5=87.6%  P(correct)=72%  σ=3.57
  ...
```
All games shown sorted by P(correct), top 3 get medals.

**Current state (May 11, 2026):** Model trained on 64 games. Only 1 boosting round (data too small for more). Produces near-uniform P(correct) ≈ 78% for all games. Only feature used so far: `away_starter_era`. Expected to differentiate meaningfully at ~200+ games (2-3 weeks).

**Code:** `sandy/over_under/meta_model.py`

**Does NOT change:** v1 calibration, σ analysis, σ edge, game list, top 3 picks (high prob + low σ). All existing messages remain exactly as they were.

## Over/Under Daily Pipeline — Current State (May 2026)

**Nightly pipeline (1 AM PST / 08:00 UTC):**
1. Ingest new games (yesterday's final scores)
2. Build labels (reached_base, game_winner, runs)
3. Build game features (incremental)
4. Reconcile over/under predictions vs actuals
5. Retrain ALL models (runs, game_winner, volatility, meta)
6. Calibrate (v1 thresholds + σ analysis)
7. Predict today's games + send Telegram

**Morning predictions (7 AM PST / 14:00 UTC):**
- Run predictions for today's games
- Score with meta-model
- Send Telegram with: trust signal, σ edge, all games (sorted by O5.5 desc), σ range, top 3 (high prob + low σ), meta-model picks

**Weekly (1:30 AM PST Monday):**
- Deeper 4-week calibration analysis

**Telegram messages received daily:**
1. ~1:02 AM: "Data updated: X new games ingested..."
2. ~1:02 AM: Reconciliation results (yesterday's ✅/❌)
3. ~1:03 AM: "Models retrained: runs, game_winner, volatility, meta"
4. ~1:03 AM: Calibration report (v1 thresholds + σ analysis)
5. ~1:04 AM: Today's predictions (full report with meta-model picks)
6. 7:00 AM: Morning predictions (same format, fresh schedule data)

**Cron schedule:**
```
0 8 * * *   nightly_pipeline.sh   (1 AM PST)
0 14 * * *  over_under_morning.sh (7 AM PST)
30 8 * * 1  over_under_weekly.sh  (1:30 AM PST Monday)
```

**Infrastructure:** EC2 t3.small, Elastic IP 3.150.16.103, PostgreSQL in docker-compose, models on disk at `/home/ec2-user/sandy/models/`.
