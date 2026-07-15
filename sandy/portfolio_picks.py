"""🎯 Portafolio B "Picks del Día" — PAPER-money portfolio over the ✅ picks.

The user's explicit A/B experiment, running FOREVER beside the 🎰 value
portfolio (sandy/portfolio.py). Same optimizer machinery, opposite thesis:

  PORTFOLIO A (🎰 sandy/portfolio.py)          PORTFOLIO B (🎯 this module)
  ------------------------------------         ------------------------------------
  bets only odds.value_log picks               bets the day's ✅ ACCURACY picks
  (positive market edge required)              even when NO positive edge exists
  stakes with SHRUNK probabilities             stakes with OUR RAW calibrated
  p_bet = 0.7·ours + 0.3·market                probabilities — "bet our beliefs"
  (prudent, market-deferent arm)               (confident, model-trusting arm)
  no-value day → $0 staked                     ALWAYS-BET floor (see below)
  odds.portfolio_log / odds.bankroll           odds.portfolio_picks_log /
                                               odds.bankroll_picks (own $100,000)

The two bankroll curves ARE the experiment: if raw model beliefs beat prudent
market deference over months, B's curve says so. Expect losing stretches — B
deliberately bets picks the market prices against us.

Rules (decided with the user — mirror sandy/portfolio.py unless stated):
  * CANDIDATES  — every ✅ meta-approved pick of the day across ALL leagues
    that has a matched cuota (TheOddsAPI). Picks without an odds feed (btts,
    corners, NHL 1X) are excluded by necessity — no price, no bet.
    MAX ONE candidate PER GAME: per game we pre-select the pick with the
    highest expected value under OUR probability (p·cuota − 1). Intra-game
    legs are heavily correlated; one leg per match keeps parlays honest.
  * PROBABILITIES — OUR calibrated side probability RAW, no market shrink
    (p_bet = prob). This is the point of the experiment; portfolio A already
    covers the prudent blend.
  * SIZING      — identical machinery to A (imported, not copied): singles +
    cross-game parlays ≤ 7 legs, Monte-Carlo E[log wealth], greedy $500
    coordinate ascent, Agresivo ½-Kelly default, budget = 30% of B's OWN
    bankroll, ≤ 30%-of-budget exposure per game.
  * ALWAYS-BET (the user's core thesis) — if the Kelly optimizer allocates $0
    (every candidate negative-EV under our probs), we FORCE a minimum
    deployment: MIN_DEPLOY_FRACTION (10%) of the day's budget, floored to
    $500 steps (min one $500 ticket), placed step-by-step on the ticket mix
    with the highest expected log outcome — the least-bad best combination.
    Days with NO candidates at all genuinely stake $0.
  * SETTLING    — legs are NOT value_log rows (most have no positive edge),
    so each leg re-grades straight from the prediction tables via
    sandy.betmeta._correct — the same source odds.reconcile_value_log uses.
    Grading rule is A's settle_ticket (imported): all win → stake·cuota, any
    lose → 0, unreconciled beyond VOID_AFTER_DAYS → void → refund.
  * STORAGE     — odds.portfolio_picks_log / odds.bankroll_picks, same shapes
    as A's tables but fully separate; A's rows are never touched.

CLI: python -m sandy.portfolio_picks build|settle|report
(logs 'portfolio picks build COMPLETE' / 'portfolio picks settle COMPLETE'
markers into odds.log via the daily cron scripts).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import text

from sandy.betmeta import SPECS, _correct, market_threshold, score_candidate
from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.odds import DISPLAY_TZ, market_to_api, odds_index, pick_side
# Shared portfolio math — imported from A, never copy-pasted. B only adds its
# own candidate pool, the always-bet floor and its own tables.
from sandy.portfolio import (
    BUDGET_FRACTION,
    DEFAULT_RISK,
    GAME_CAP_FRACTION,
    INITIAL_BANK,
    MAX_TICKETS_PER_DAY,
    N_SIMS,
    RISKS,
    STEP,
    VOID_AFTER_DAYS,
    _dist_summary,
    _league_title,
    _market_label,
    _optimize,
    _pick_label,
    default_budget,
    enumerate_tickets,
    floor500,
    settle_ticket,
)

logger = logging.getLogger(__name__)

# ALWAYS-BET floor: when the Kelly optimizer would stake $0 (no candidate has
# positive EV under our raw probabilities), portfolio B still deploys this
# fraction of the day's budget (floored to $500 steps, min one $500 ticket)
# on the least-bad best ticket mix — the experiment's core thesis is that our
# models deserve to be bet EVERY day they produce ✅ picks.
MIN_DEPLOY_FRACTION = 0.10

# B prices with RAW model probabilities (no market shrink — its thesis), so parlay
# probability = Π p_i compounds the tails' optimism. Calibration audit 2026-07-14
# over all settled tickets: ≤4 legs predicted-vs-real is honest (even conservative),
# but 5+ legs believed 26.6% and hit 9.1% (1/11, ROI −50%). Cap where calibration
# breaks. Portfolio A keeps MAX_PARLAY_LEGS=7 — its 70/30 shrink showed NO
# overconfidence at any size (it is the control for this experiment).
B_MAX_PARLAY_LEGS = 4


# ------------------------------------------------------------------- schema --
def ensure_tables(engine) -> None:
    """Run the idempotent picks-portfolio migration (CREATE ... IF NOT EXISTS)."""
    sql = (Path(__file__).parent / "migrations" / "add_portfolio_picks_tables.sql").read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))


# --------------------------------------------------------------- candidates --
def best_per_game(cands: list[dict]) -> list[dict]:
    """MAX ONE candidate PER GAME — selected EXACTLY like the 🏁 Picks del Día
    finals: the pick with the highest 🤖 meta score wins the game (tie-break by
    raw EV). So B's pool is literally the Picks del Día list, restricted to the
    picks that have a matched cuota. Pure + unit-tested. Result sorted by
    descending ev (the parlay pool takes the head)."""
    def _key(c):
        return (c.get("meta") if c.get("meta") is not None else -1.0, c["ev"])
    best: dict[tuple, dict] = {}
    for c in cands:
        g = c["game"]
        if g not in best or _key(c) > _key(best[g]):
            best[g] = c
    return sorted(best.values(), key=lambda c: -c["ev"])


def candidates_for_picks(day: date, engine, cfg: Config | None = None) -> list[dict]:
    """The day's ✅ accuracy picks with a matched cuota → one per game.

    Scan is the same as odds.log_value_picks (pending games, meta-approved
    picks, matched TheOddsAPI odds) WITHOUT the positive-edge filter: any ✅
    pick with a price is a candidate, even when the market disagrees with us.
    Probabilities stay RAW (p_bet = prob — no market shrink, by design)."""
    cfg = cfg or load_config()
    from sandy.portfolio import started_games
    _started = started_games(engine, day)
    out: list[dict] = []
    game_best: dict[tuple, tuple] = {}   # (league,home,away) -> (best 🤖, market)
    for league in SPECS:
        idx = odds_index(league, day, day, engine)
        if not idx:
            continue
        spec = SPECS[league]
        extra = f" AND {spec['where']}" if spec.get("where") else ""
        bt = "TRUE" if spec.get("no_backtest_col") else "NOT is_backtest"
        with engine.begin() as conn:
            rows = conn.execute(text(f"""
                SELECT * FROM {spec['table']}
                WHERE match_date = :d AND outcome_filled_at_utc IS NULL AND {bt}{extra}
                ORDER BY match_date, id"""), {"d": day}).fetchall()
        # Doubleheaders: two rows, same (date, teams) — odds keyed by (date, teams)
        # can't be split per game, so those games are NOT candidates (a wrong-game
        # price is worse than no bet). Same rule as A's log_value_picks.
        from collections import Counter
        _dh = {k for k, n in Counter(
            ((dict(x._mapping).get("home_team") or "").strip(),
             (dict(x._mapping).get("away_team") or "").strip()) for x in rows).items() if n > 1}
        for r in rows:
            rd = dict(r._mapping)
            home, away = (rd["home_team"] or "").strip(), (rd["away_team"] or "").strip()
            if (league, home, away) in _started:
                continue  # game already kicked off — not biddable anymore
            if (home, away) in _dh:
                continue  # doubleheader — odds ambiguous, skip both games
            for market, (pcol, kind, line) in spec["markets"].items():
                p = rd.get(pcol)
                if p is None:
                    continue  # no prediction for this market
                p = float(p)
                prob = p if p >= 0.5 else 1 - p   # the pick's own side prob (NOT 🤖)
                mp = score_candidate(league, cfg, rd, market, p)
                thr = market_threshold(league, cfg, market)
                if mp is None or thr is None or mp < thr:
                    continue  # only ✅ meta-approved picks — B's whole pool
                # Track the game's OVERALL best 🤖 among ALL ✅ picks (priced or
                # not): used for the "⚠️sust." mark when the true Picks del Día
                # pick has no cuota and B bets the best PRICED one instead.
                gk = (league, home, away)
                if gk not in game_best or mp > game_best[gk][0]:
                    game_best[gk] = (mp, market)
                mapping = market_to_api(league, market)
                if mapping is None:
                    continue  # no odds feed for this market (corners/BTTS/NHL-1X)
                api_market, pt = mapping
                side = pick_side(kind, p)
                hit = idx.get((day, home, away, api_market, pt, side))
                if not hit:
                    continue  # no matched price for this pick → can't be bet
                cuota, novig, _n = hit
                cuota = float(cuota)
                ev = prob * cuota - 1.0           # raw-model EV per $1 (may be < 0!)
                fp = rd.get("first_pitch_utc")
                out.append({
                    "date": str(day), "game": (league, home, away),
                    "liga": league, "liga_titulo": _league_title(league),
                    "partido": f"{home} vs {away}", "home": home, "away": away,
                    "hora": fp.astimezone(DISPLAY_TZ).strftime("%I:%M %p").lstrip("0") if fp is not None else None,
                    "market": market, "side": side,
                    "line": None if pt is None else float(pt),
                    "mercado": _market_label(league, market),
                    "pick": _pick_label(league, market, side, line, home, away),
                    "cuota": round(cuota, 2),
                    "prob": round(prob, 4),        # raw calibrated side prob
                    "meta": None if mp is None else round(float(mp), 4),  # 🤖 (selection key)
                    "p_bet": round(prob, 4),       # == prob: NO shrink (B's thesis)
                    "mercado_pct": None if novig is None else round(float(novig), 4),
                    "edge": None if novig is None else round(prob - float(novig), 4),
                    "ev": round(ev, 4),
                })
    # ⚠️sust. — the selected candidate is NOT the game's overall best-🤖 pick
    # (the exact Picks del Día pick has no cuota); mark it in every label.
    for c in out:
        bm = game_best.get(c["game"])
        c["sustituto"] = bool(bm and bm[1] != c["market"])
        if c["sustituto"]:
            c["pick"] = f"{c['pick']} ⚠️sust."
    return best_per_game(out)


# ---------------------------------------------------------------- optimizer --
def _forced_deploy(cands: list[dict], tickets: list[dict], forced_budget: float,
                   cap: float, bank: float, risk: str, n_sims: int, seed: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """ALWAYS-BET allocator: spend `forced_budget` in $500 steps, each step on
    the ticket with the HIGHEST expected log wealth (even when every option
    lowers it — least-bad best). Same MC construction as portfolio._optimize
    (deterministic seed, per-game exposure cap) minus the must-improve gate."""
    rng = np.random.default_rng(seed)
    U = rng.random((n_sims, len(cands)))
    wins = U < np.array([c["prob"] for c in cands])  # raw probs — B's thesis
    V = max(bank * RISKS[risk], 1.0)
    pay = [t["cuota"] * wins[:, list(t["legs_idx"])].all(axis=1) - 1.0
           for t in tickets]
    stakes = np.zeros(len(tickets))
    pnl = np.zeros(n_sims)
    expo: dict[tuple, float] = {}
    spent = 0.0
    n_open = 0
    while spent + STEP <= forced_budget + 1e-9:
        best_i, best_u = None, -np.inf
        for i, t in enumerate(tickets):
            if stakes[i] == 0.0 and n_open >= MAX_TICKETS_PER_DAY:
                continue  # ticket cap (same as the shared optimizer)
            if any(expo.get(g, 0.0) + STEP > cap + 1e-9 for g in t["games"]):
                continue  # per-game exposure cap still applies
            u = float(np.log(np.maximum(V + pnl + STEP * pay[i], 1e-9)).mean())
            if u > best_u:
                best_i, best_u = i, u
        if best_i is None:
            break
        if stakes[best_i] == 0.0:
            n_open += 1
        stakes[best_i] += STEP
        pnl += STEP * pay[best_i]
        for g in tickets[best_i]["games"]:
            expo[g] = expo.get(g, 0.0) + STEP
        spent += STEP
    return stakes, pnl


def allocate_day(cands: list[dict], tickets: list[dict], budget: float,
                 bank: float, risk: str, n_sims: int, seed: int
                 ) -> tuple[np.ndarray, np.ndarray, bool]:
    """Kelly allocation with the ALWAYS-BET floor. Returns (stakes, pnl, forced).

    First the shared greedy optimizer (with p_bet == prob the 'prudente' and
    'modelo' P&L distributions coincide — B has ONE probability regime). If it
    stakes $0 while candidates exist and the budget allows a minimum bet, the
    forced deployment kicks in (MIN_DEPLOY_FRACTION of the budget)."""
    if not cands or not tickets or budget < STEP:
        return np.zeros(len(tickets)), np.zeros(1), False
    stakes, pnl, _pnl_raw = _optimize(cands, tickets, budget, bank, risk, n_sims, seed)
    if stakes.sum() > 0:
        return stakes, pnl, False
    forced_budget = max(floor500(MIN_DEPLOY_FRACTION * budget), STEP)
    stakes, pnl = _forced_deploy(cands, tickets, forced_budget,
                                 GAME_CAP_FRACTION * budget, bank, risk, n_sims, seed)
    return stakes, pnl, bool(stakes.sum() > 0)


# ----------------------------------------------------------------- bankroll --
def available_bank(engine=None, before: date | None = None) -> float:
    """Portfolio B's own cash: chain end_bank over odds.bankroll_picks; open
    days lock their stakes. `before=day` excludes that day's own row (budget
    basis for building/what-if rebuilding day X)."""
    engine = engine or create_engine(load_config())
    ensure_tables(engine)
    bank = INITIAL_BANK
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT start_bank, staked, returned, end_bank FROM odds.bankroll_picks "
            "WHERE (CAST(:b AS DATE) IS NULL OR date < :b) ORDER BY date"
        ), {"b": before}).fetchall()
    for r in rows:
        bank = float(r.end_bank) if r.end_bank is not None else float(r.start_bank) - float(r.staked)
    return bank


# -------------------------------------------------------------------- build --
def build_portfolio(day: date | None = None, budget: float | None = None,
                    risk: str = DEFAULT_RISK, persist: bool = True,
                    force: bool = False, n_sims: int = N_SIMS,
                    cfg: Config | None = None) -> dict:
    """Build (and optionally persist) the day's 🎯 picks portfolio.

    persist=False → pure what-if recompute for the dashboard (deterministic
    seed, so default budget+risk reproduces the persisted portfolio)."""
    cfg = cfg or load_config()
    day = day or datetime.now(DISPLAY_TZ).date()
    engine = create_engine(cfg)
    ensure_tables(engine)
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {list(RISKS)}")

    if persist and not force:
        with engine.begin() as conn:
            built = conn.execute(text(
                "SELECT 1 FROM odds.bankroll_picks WHERE date = :d UNION ALL "
                "SELECT 1 FROM odds.portfolio_picks_log WHERE date = :d LIMIT 1"
            ), {"d": day}).fetchone()
        if built:
            logger.info("portfolio picks %s: already built — skipping (idempotent)", day)
            return {"date": str(day), "skipped": "already_built"}

    bank = available_bank(engine, before=day)  # B's OWN bank, excludes the day
    budget = default_budget(bank) if budget is None else floor500(min(budget, bank))
    cands = candidates_for_picks(day, engine, cfg)
    result: dict = {"date": str(day), "bank": round(bank, 2), "budget": budget,
                    "risk": risk, "n_candidates": len(cands), "tickets": [],
                    "staked": 0.0, "forzado": False, "summary": None, "persisted": False}

    if cands and budget >= STEP:
        tickets = enumerate_tickets(cands, max_legs=B_MAX_PARLAY_LEGS)
        seed = int(day.strftime("%Y%m%d"))
        stakes, pnl, forced = allocate_day(cands, tickets, budget, bank, risk, n_sims, seed)
        result["forzado"] = forced
        out = []
        for t, s in zip(tickets, stakes):
            if s <= 0:
                continue
            legs = [{k: v for k, v in cands[i].items() if k not in ("game", "p_bet")}
                    for i in t["legs_idx"]]
            out.append({
                "legs": legs, "n_legs": len(legs),
                "cuota": round(t["cuota"], 4),
                "prob": round(t["prob"], 4),      # raw-model ticket prob (Π legs)
                "stake": float(s),
                "ev": round(float(s) * t["ev"], 2),
            })
        out.sort(key=lambda x: (-x["stake"], -x["ev"]))
        for i, t in enumerate(out, start=1):
            t["ticket_id"] = i
        staked = float(sum(t["stake"] for t in out))
        result["tickets"] = out
        result["staked"] = staked
        if staked > 0:
            result["summary"] = _dist_summary(pnl)
    if not result["tickets"]:
        result["motivo"] = ("sin picks ✅ con cuota hoy" if not cands
                            else "presupuesto menor a la apuesta mínima")
        logger.info("portfolio picks %s: $0 staked (%s)", day, result["motivo"])

    if persist:
        with engine.begin() as conn:
            if force:
                _clear_day(conn, day)
            for t in result["tickets"]:
                conn.execute(text("""
                    INSERT INTO odds.portfolio_picks_log
                        (date, ticket_id, legs, ticket_cuota, ticket_prob, stake)
                    VALUES (:d, :tid, :legs, :cuota, :prob, :stake)
                """), {"d": day, "tid": t["ticket_id"], "legs": json.dumps(t["legs"]),
                       "cuota": t["cuota"], "prob": t["prob"], "stake": t["stake"]})
            if result["staked"] > 0:
                conn.execute(text("""
                    INSERT INTO odds.bankroll_picks (date, start_bank, staked)
                    VALUES (:d, :b, :s)
                """), {"d": day, "b": bank, "s": result["staked"]})
            else:  # $0 day (no candidates): logged AND settled on the spot
                conn.execute(text("""
                    INSERT INTO odds.bankroll_picks
                        (date, start_bank, staked, returned, end_bank, settled_at)
                    VALUES (:d, :b, 0, 0, :b, now())
                """), {"d": day, "b": bank})
        result["persisted"] = True
    return result


def _clear_day(conn, day: date) -> None:
    """--force rebuild: wipe the day's B rows, refusing if anything settled OR in play
    (an in-flight leg is a frozen bet — same guard as portfolio A's _clear_day)."""
    settled = conn.execute(text(
        "SELECT count(*) FROM odds.portfolio_picks_log WHERE date = :d AND status != 'open'"
    ), {"d": day}).scalar()
    if settled:
        raise RuntimeError(f"{day}: picks tickets already settled — refusing to rebuild")
    from sandy.portfolio import started_games
    rows = conn.execute(text(
        "SELECT legs FROM odds.portfolio_picks_log WHERE date = :d"), {"d": day}).fetchall()
    started = started_games(conn, day)
    for r in rows:
        legs = r.legs if isinstance(r.legs, list) else json.loads(r.legs)
        for l in legs:
            if (l.get("liga"), (l.get("home") or "").strip(), (l.get("away") or "").strip()) in started:
                raise RuntimeError(
                    f"{day}: ticket leg {l.get('partido')} already kicked off — refusing to "
                    "rebuild (in-flight bets are frozen; void the specific leg instead)")
    conn.execute(text("DELETE FROM odds.portfolio_picks_log WHERE date = :d"), {"d": day})
    conn.execute(text(
        "DELETE FROM odds.bankroll_picks WHERE date = :d AND (staked = 0 OR settled_at IS NULL)"
    ), {"d": day})


# ------------------------------------------------------------------- settle --
def _leg_result(conn, leg: dict, today: date) -> str:
    """Grade one leg straight from the prediction tables (B legs are not in
    value_log). Same source + _correct convention as odds.reconcile_value_log:
    the logged side is encoded as an extreme p so _correct scores OUR side."""
    spec = SPECS.get(leg["liga"])
    if spec:
        extra = f" AND {spec['where']}" if spec.get("where") else ""
        # Doubleheader guard: two prediction rows for the same (date, teams) means we
        # can't know WHICH game this leg belongs to — ungradeable → void (stake back).
        n_games = conn.execute(text(f"""
            SELECT COUNT(*) FROM {spec['table']}
            WHERE match_date = :d AND btrim(home_team) = :h AND btrim(away_team) = :a{extra}
        """), {"d": leg["date"], "h": leg["home"], "a": leg["away"]}).scalar()
        if n_games and n_games > 1:
            return "void"
        g = conn.execute(text(f"""
            SELECT * FROM {spec['table']}
            WHERE match_date = :d AND btrim(home_team) = :h AND btrim(away_team) = :a
              AND outcome_filled_at_utc IS NOT NULL{extra}
            ORDER BY id LIMIT 1
        """), {"d": leg["date"], "h": leg["home"], "a": leg["away"]}).fetchone()
        if g is not None:
            _pcol, kind, _line = spec["markets"][leg["market"]]
            p_side = 0.99 if leg["side"] in ("over", "home", "home_or_draw") else 0.01
            won = _correct(dict(g._mapping), kind,
                           None if leg.get("line") is None else float(leg["line"]), p_side)
            if won is not None:
                return "win" if won else "lose"
    if date.fromisoformat(leg["date"]) <= today - timedelta(days=VOID_AFTER_DAYS):
        return "void"  # never reconciled → postponed/cancelled
    return "pending"


def settle_portfolio(cfg: Config | None = None, today: date | None = None) -> dict:
    """Grade every open B ticket (shared settle_ticket rule), then close the
    bankroll_picks rows of days with no tickets left open:
    end_bank = start_bank − staked + Σ returned."""
    cfg = cfg or load_config()
    today = today or datetime.now(DISPLAY_TZ).date()
    engine = create_engine(cfg)
    ensure_tables(engine)
    graded, still_open = {"won": 0, "lost": 0, "void": 0}, 0
    with engine.begin() as conn:
        open_rows = conn.execute(text(
            "SELECT * FROM odds.portfolio_picks_log WHERE status = 'open' "
            "ORDER BY date, ticket_id")).fetchall()
        for t in open_rows:
            legs = t.legs if isinstance(t.legs, list) else json.loads(t.legs)
            results = [_leg_result(conn, leg, today) for leg in legs]
            status, returned = settle_ticket(results, float(t.stake), float(t.ticket_cuota))
            if status == "open":
                still_open += 1
                continue
            conn.execute(text("""
                UPDATE odds.portfolio_picks_log
                SET status = :st, returned = :ret, settled_at = now()
                WHERE id = :id
            """), {"st": status, "ret": returned, "id": t.id})
            graded[status] += 1
            logger.info("picks ticket %s #%d: %s (stake %.0f → %.0f)",
                        t.date, t.ticket_id, status, t.stake, returned or 0.0)
        closed_days = []
        for (d,) in conn.execute(text(
                "SELECT date FROM odds.bankroll_picks WHERE end_bank IS NULL ORDER BY date")):
            pending = conn.execute(text(
                "SELECT count(*) FROM odds.portfolio_picks_log "
                "WHERE date = :d AND status = 'open'"), {"d": d}).scalar()
            if pending:
                continue
            conn.execute(text("""
                UPDATE odds.bankroll_picks b
                SET returned = r.tot,
                    end_bank = b.start_bank - b.staked + r.tot,
                    settled_at = now()
                FROM (SELECT COALESCE(SUM(returned), 0) AS tot
                      FROM odds.portfolio_picks_log WHERE date = :d) r
                WHERE b.date = :d
            """), {"d": d})
            closed_days.append(str(d))
    rep = {"graded": graded, "still_open": still_open, "bankroll_days_closed": closed_days,
           "bank": available_bank(engine)}
    logger.info("portfolio picks settle: %s", rep)
    return rep


# --------------------------------------------------------------- dashboards --
def bankroll_frame(cfg: Config | None = None):
    """odds.bankroll_picks ledger for the 🎯 page (date-ordered)."""
    import pandas as pd
    engine = create_engine(cfg or load_config())
    ensure_tables(engine)
    with engine.begin() as conn:
        return pd.read_sql(text("""
            SELECT date, start_bank, staked, returned, end_bank, settled_at
            FROM odds.bankroll_picks ORDER BY date
        """), conn)


def tickets_frame(cfg: Config | None = None):
    """All persisted B tickets, newest first, legs pre-formatted for display."""
    import pandas as pd
    engine = create_engine(cfg or load_config())
    ensure_tables(engine)
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT date, ticket_id, legs, ticket_cuota, ticket_prob, stake,
                   status, returned
            FROM odds.portfolio_picks_log ORDER BY date DESC, ticket_id
        """), conn)
    if df.empty:
        return df

    def _legs(x):
        return x if isinstance(x, list) else json.loads(x)

    df["tiquete"] = df["legs"].map(lambda x: " + ".join(
        f"{l['liga_titulo']} {l['partido']}{' · ' + l['hora'] if l.get('hora') else ''}: {l['pick']} @{l['cuota']}"
        for l in _legs(x)))
    df["tipo"] = df["legs"].map(
        lambda x: "Individual" if len(_legs(x)) == 1 else f"Combinada x{len(_legs(x))}")
    return df.drop(columns=["legs"])


# ---------------------------------------------------------------------- cli --
def _print_sheet(rep: dict) -> None:
    print(f"\n🎯 PORTAFOLIO PICKS {rep['date']} — banca ${rep.get('bank', 0):,.0f} · "
          f"presupuesto ${rep.get('budget', 0):,.0f} · riesgo {rep.get('risk')}")
    if rep.get("skipped"):
        print(f"  (omitido: {rep['skipped']})")
        return
    if not rep["tickets"]:
        print(f"  🙅 $0 apostado ({rep.get('motivo')})")
        return
    if rep.get("forzado"):
        print("  ⚠️ DESPLIEGUE FORZADO: ningún candidato tiene EV positivo bajo nuestras "
              f"probabilidades — se apuesta el {MIN_DEPLOY_FRACTION:.0%} del presupuesto "
              "en la mejor combinación (regla siempre-apostar del experimento).")
    for t in rep["tickets"]:
        kind = "Individual" if t["n_legs"] == 1 else f"Combinada x{t['n_legs']}"
        legs_txt = " + ".join(
            f"{l['liga_titulo']} {l['partido']}{' · ' + l['hora'] if l.get('hora') else ''}: {l['pick']} @{l['cuota']}"
            for l in t["legs"])
        print(f"  Apuesta {t['ticket_id']}: {kind} — ${t['stake']:,.0f} en {legs_txt}")
        print(f"      cuota total {t['cuota']:.2f} · prob modelo {t['prob']:.1%} "
              f"· EV modelo ${t['ev']:+,.0f}")
    s = rep["summary"] or {}
    print(f"  Σ apostado ${rep['staked']:,.0f}")
    print(f"  SEGÚN NUESTROS MODELOS (prob cruda, sin recorte): esperado "
          f"${s.get('expected_profit', 0):+,.0f} · P(día verde) {s.get('p_green', 0):.1%} · "
          f"peor 5% ${s.get('p5', 0):+,.0f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Sandy 🎯 Portafolio B — paper portfolio over the day's ✅ picks")
    ap.add_argument("cmd", choices=["build", "settle", "report"],
                    help="build = optimize+persist today; settle = grade open tickets "
                         "+ close bankroll days; report = ledger summary")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today America/Los_Angeles)")
    ap.add_argument("--budget", type=float, help="override the 30%%-of-bank default")
    ap.add_argument("--risk", default=DEFAULT_RISK, choices=list(RISKS))
    ap.add_argument("--force", action="store_true", help="rebuild an already-built day")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    ap.add_argument("--dry", action="store_true", help="build without persisting")
    args = ap.parse_args()
    day = date.fromisoformat(args.date) if args.date else None
    ts = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    if args.cmd == "build":
        rep = build_portfolio(day=day, budget=args.budget, risk=args.risk,
                              persist=not args.dry, force=args.force, n_sims=args.sims)
        _print_sheet(rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "tickets"},
                         default=str, indent=2))
        print(f"[{ts()}] portfolio picks build COMPLETE")
    elif args.cmd == "settle":
        rep = settle_portfolio(today=day)
        print(json.dumps(rep, default=str, indent=2))
        print(f"[{ts()}] portfolio picks settle COMPLETE")
    else:
        df = bankroll_frame()
        if df.empty:
            print("bankroll picks: sin días registrados aún")
            return
        settled = df[df["end_bank"].notna()]
        pnl = float((settled["returned"] - settled["staked"]).sum()) if len(settled) else 0.0
        print(f"días: {len(df)} | banca actual: ${available_bank():,.0f} | "
              f"P&L acumulado: ${pnl:+,.0f} | último día: {df.iloc[-1]['date']}")


if __name__ == "__main__":
    main()
