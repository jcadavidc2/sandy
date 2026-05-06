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
