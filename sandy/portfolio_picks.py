"""🎯 Portafolio B "Picks del Día" — PAPER-money FLAT-STAKE singles over the ✅ picks.

REDESIGNED 2026-09-28 after the portfolio audit. The original B ("bet our raw
probabilities, always bet, parlays up to 4 legs") was falsified by its own data:
  * the optimal weight on our model vs the market on our own picks was ZERO
    (log-loss grid over 577 settled value picks) — "raw probabilities" only
    meant bigger stakes on the same coin flips;
  * 63.6% of A's legs were also in B the same day (daily P&L correlation 0.80):
    two banks carrying one bet, no information gained;
  * the always-bet floor fired on 4 days ($10,000) and proved nothing;
  * 4-leg tickets since Aug 5 went 0/33 (believed 14%).

B is now the structurally DIFFERENT arm the A/B needs — the "more green days"
thesis, tested cleanly against A's Kelly arm:

  PORTFOLIO A (🎰 sandy/portfolio.py)          PORTFOLIO B (🎯 this module)
  ------------------------------------         ------------------------------------
  bets odds.value_log picks (edge 5–20pp)      bets the day's ✅ Picks del Día
  Kelly-shrunk sizing (E[log] optimizer)       FLAT stake: B_FLAT_FRACTION of bank
  singles + doubles                            SINGLES ONLY
  market must agree on the side                market must agree on the side
  odds.portfolio_log / odds.bankroll           odds.portfolio_picks_log / odds.bankroll_picks

Rules:
  * CANDIDATES  — every ✅ meta-approved pick of the day with a matched cuota,
    ONE per game (highest 🤖, EV tie-break — exactly the 🏁 Picks del Día list),
    that also passes the audit filters: market no-vig on OUR side ≥ 0.50
    (market-disagreeing picks hit 44% vs 58% believed), our calibrated
    probability ≥ market + 3pp (edge ≥ B_MIN_EDGE), ≥ B_MIN_BOOKS books quoting the
    exact line, and the league not blocked by portfolio.league_gate().
  * SIZING      — flat B_FLAT_FRACTION (2%) of the bank per pick, floored to
    $500, at most B_MAX_PICKS_PER_DAY picks (highest 🤖 first), never above the
    day's budget (BUDGET_FRACTION of the bank). No Kelly, no parlays, no
    forced deployment: a day without qualifying picks stakes $0.
  * PRICE       — cuota = MEDIAN price across books (what one real book pays),
    not the best of 24 (which inflated paper P&L by ~1.75pp of ROI).
  * SETTLING    — unchanged: legs re-grade straight from the prediction tables
    via sandy.betmeta._correct; A's settle_ticket rule (void leg drops out,
    all-void refunds).
  * STORAGE     — unchanged tables, fully separate from A.

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
# Shared portfolio plumbing — imported from A, never copy-pasted. B only adds its
# own candidate pool, the flat allocator and its own tables.
from sandy.portfolio import (
    BUDGET_FRACTION,
    DEFAULT_RISK,
    INITIAL_BANK,
    N_SIMS,
    RISKS,
    STEP,
    VOID_AFTER_DAYS,
    _dist_summary,
    _league_title,
    _market_label,
    _pick_label,
    default_budget,
    enumerate_tickets,
    floor500,
    league_gate,
    settle_ticket,
    started_games,
)

logger = logging.getLogger(__name__)

B_FLAT_FRACTION = 0.02          # flat stake per pick, as a fraction of B's bank (floored to $500)
B_MAX_PICKS_PER_DAY = 5         # highest-🤖 picks first; 5 × 2% = the 10% daily budget
B_REQUIRE_MARKET_SIDE = True    # market no-vig for our side must be ≥ 0.50
B_MIN_EDGE = 0.03               # our (calibrated) probability ≥ market + 3pp — the pool that replayed
                                # to 32 green / 23 red days as calibrated same-side singles
B_MIN_BOOKS = 2                 # a second book must quote the exact line (a real second opinion);
                                # A demands 3 because it SIZES on the edge — B stakes flat, and the
                                # flat replay at median price lost volume, not accuracy, at ≥3


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
    descending ev."""
    def _key(c):
        return (c.get("meta") if c.get("meta") is not None else -1.0, c["ev"])
    best: dict[tuple, dict] = {}
    for c in cands:
        g = c["game"]
        if g not in best or _key(c) > _key(best[g]):
            best[g] = c
    return sorted(best.values(), key=lambda c: -c["ev"])


def candidates_for_picks(day: date, engine, cfg: Config | None = None) -> list[dict]:
    """The day's ✅ accuracy picks with a matched cuota → one per game, filtered.

    Scan mirrors odds.log_value_picks (pending games, meta-approved picks,
    matched TheOddsAPI odds). Filters (audit 2026-09-28): market side
    agreement, edge ≥ B_MIN_EDGE, ≥ MIN_BOOKS books, league gate. Skips are
    logged with their reason."""
    cfg = cfg or load_config()
    _started = started_games(engine, day)
    gate = league_gate(engine, day)
    out: list[dict] = []
    game_best: dict[tuple, tuple] = {}   # (league,home,away) -> (best 🤖, market)
    skipped: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for league in SPECS:
        if gate.get(league, {}).get("blocked"):
            _skip(f"liga_bloqueada:{league}")
            continue
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
                cuota, novig, n_books = float(hit.price), hit.novig, int(hit.n)
                if novig is None:
                    _skip("sin_consenso"); continue
                novig = float(novig)
                if B_REQUIRE_MARKET_SIDE and novig < 0.5:
                    _skip("mercado_en_contra"); continue
                if prob - novig < B_MIN_EDGE:
                    _skip("bajo_el_mercado"); continue
                if n_books < B_MIN_BOOKS:
                    _skip("pocas_casas"); continue
                ev = prob * cuota - 1.0           # model EV per $1 at the median price
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
                    "cuota": round(cuota, 2),      # MEDIAN price across books
                    "prob": round(prob, 4),        # calibrated side prob
                    "p_bet": round(prob, 4),       # == prob (no shrink) — enumerate_tickets contract
                    "meta": round(float(mp), 4),   # 🤖 (selection key)
                    "mercado_pct": round(novig, 4),
                    "n_books": n_books,
                    "edge": round(prob - novig, 4),
                    "ev": round(ev, 4),
                })
    # ⚠️sust. — the selected candidate is NOT the game's overall best-🤖 pick
    # (the exact Picks del Día pick has no cuota); mark it in every label.
    for c in out:
        bm = game_best.get(c["game"])
        c["sustituto"] = bool(bm and bm[1] != c["market"])
        if c["sustituto"]:
            c["pick"] = f"{c['pick']} ⚠️sust."
    picks = best_per_game(out)
    if skipped:
        logger.info("picks candidates %s: %d kept, skipped %s", day, len(picks),
                    dict(sorted(skipped.items())))
    return picks


# ---------------------------------------------------------------- allocator --
def flat_stake(bank: float) -> float:
    """B's unit: B_FLAT_FRACTION of the bank, floored to $500 steps, min one step."""
    return max(floor500(B_FLAT_FRACTION * bank), STEP)


def allocate_flat(cands: list[dict], bank: float, budget: float,
                  max_picks: int = B_MAX_PICKS_PER_DAY) -> np.ndarray:
    """FLAT allocation: the same stake on each of the top-🤖 candidates (EV
    tie-break), at most `max_picks` picks (default B_MAX_PICKS_PER_DAY; a top-up
    passes the picks still allowed today), never above `budget`.
    Returns stakes aligned with `cands` (0 for unfunded picks). Pure + unit-tested."""
    stakes = np.zeros(len(cands))
    if not cands or budget < STEP or max_picks <= 0:
        return stakes
    unit = flat_stake(bank)
    order = sorted(range(len(cands)),
                   key=lambda i: (-(cands[i].get("meta") if cands[i].get("meta") is not None else -1.0),
                                  -cands[i]["ev"]))
    spent, n = 0.0, 0
    for i in order:
        if n >= max_picks or spent + unit > budget + 1e-9:
            break
        stakes[i] = unit
        spent += unit
        n += 1
    return stakes


def _simulate(cands: list[dict], stakes: np.ndarray, n_sims: int, seed: int) -> np.ndarray:
    """P&L distribution of the flat singles under the candidates' own probs."""
    rng = np.random.default_rng(seed)
    U = rng.random((n_sims, len(cands)))
    wins = U < np.array([c["prob"] for c in cands])
    pnl = np.zeros(n_sims)
    for i, s in enumerate(stakes):
        if s > 0:
            pnl += s * (cands[i]["cuota"] * wins[:, i] - 1.0)
    return pnl


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
def _materialize_singles(cands: list[dict], tickets: list[dict], stakes: np.ndarray,
                         first_id: int = 1) -> list[dict]:
    """Funded flat singles → persisted/display shape, highest 🤖 first."""
    out = []
    for t, s in zip(tickets, stakes):
        if s <= 0:
            continue
        legs = [{k: v for k, v in cands[i].items() if k not in ("game", "p_bet")}
                for i in t["legs_idx"]]
        out.append({
            "legs": legs, "n_legs": len(legs),
            "cuota": round(t["cuota"], 4),
            "prob": round(t["prob"], 4),      # calibrated model prob of the single
            "stake": float(s),
            "ev": round(float(s) * t["ev"], 2),
        })
    out.sort(key=lambda x: (-(x["legs"][0].get("meta") or 0.0), -x["ev"]))
    for i, t in enumerate(out, start=first_id):
        t["ticket_id"] = i
    return out


def build_portfolio(day: date | None = None, budget: float | None = None,
                    risk: str = DEFAULT_RISK, persist: bool = True,
                    force: bool = False, n_sims: int = N_SIMS,
                    cfg: Config | None = None, topup: bool = False) -> dict:
    """Build (and optionally persist) the day's 🎯 flat-stake picks portfolio.

    `risk` is accepted for interface compatibility with A / the dashboard but
    does not change anything: B stakes flat. persist=False → pure what-if
    recompute (deterministic, reproduces the persisted sheet with default controls).
    topup=True → if the day is already built, ADD flat singles for qualifying picks
    on games not yet bet today (within the remaining picks/budget) instead of skipping."""
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
            if topup:
                return _topup_picks(day, engine, cfg, n_sims)
            logger.info("portfolio picks %s: already built — skipping (idempotent)", day)
            return {"date": str(day), "skipped": "already_built"}

    bank = available_bank(engine, before=day)  # B's OWN bank, excludes the day
    budget = default_budget(bank) if budget is None else floor500(min(budget, bank))
    cands = candidates_for_picks(day, engine, cfg)
    result: dict = {"date": str(day), "bank": round(bank, 2), "budget": budget,
                    "risk": risk, "n_candidates": len(cands), "tickets": [],
                    "staked": 0.0, "forzado": False, "summary": None, "persisted": False,
                    "unidad": flat_stake(bank)}

    if cands and budget >= STEP:
        tickets = enumerate_tickets(cands, max_legs=1)     # singles only, aligned with cands
        seed = int(day.strftime("%Y%m%d"))
        stakes = allocate_flat(cands, bank, budget)
        out = _materialize_singles(cands, tickets, stakes)
        staked = float(sum(t["stake"] for t in out))
        result["tickets"] = out
        result["staked"] = staked
        if staked > 0:
            result["summary"] = _dist_summary(_simulate(cands, stakes, n_sims, seed))
    if not result["tickets"]:
        result["motivo"] = ("sin picks ✅ con cuota que pasen los filtros hoy" if not cands
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
            else:  # $0 day: logged AND settled on the spot
                conn.execute(text("""
                    INSERT INTO odds.bankroll_picks
                        (date, start_bank, staked, returned, end_bank, settled_at)
                    VALUES (:d, :b, 0, 0, :b, now())
                """), {"d": day, "b": bank})
        result["persisted"] = True
    return result


def _topup_picks(day: date, engine, cfg: Config, n_sims: int) -> dict:
    """ADD flat singles to an already-built day for qualifying picks that appeared
    since the build (games not yet bet), within the day's remaining budget and the
    remaining B_MAX_PICKS_PER_DAY slots. Existing tickets are never touched."""
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT ticket_id, legs, stake FROM odds.portfolio_picks_log WHERE date = :d"),
            {"d": day}).fetchall()
        bk = conn.execute(text(
            "SELECT start_bank, staked FROM odds.bankroll_picks WHERE date = :d"), {"d": day}).fetchone()
    existing_games: set = set()
    staked_so_far, max_tid = 0.0, 0
    for r in rows:
        legs = r.legs if isinstance(r.legs, list) else json.loads(r.legs)
        for l in legs:
            existing_games.add((l.get("liga"), (l.get("home") or "").strip(), (l.get("away") or "").strip()))
        staked_so_far += float(r.stake)
        max_tid = max(max_tid, int(r.ticket_id))
    bank = float(bk.start_bank) if bk is not None else available_bank(engine, before=day)
    budget_total = default_budget(bank)
    remaining = floor500(budget_total - staked_so_far)
    slots = B_MAX_PICKS_PER_DAY - len(rows)
    result: dict = {"date": str(day), "topup": True, "bank": round(bank, 2),
                    "budget": budget_total, "budget_restante": remaining, "tickets": [],
                    "staked": 0.0, "forzado": False, "summary": None, "persisted": False,
                    "unidad": flat_stake(bank), "tickets_previos": len(rows)}
    if remaining < STEP or slots <= 0:
        result["motivo"] = "cupo del día completo"
        return result
    cands = [c for c in candidates_for_picks(day, engine, cfg) if c["game"] not in existing_games]
    result["n_candidates"] = len(cands)
    if not cands:
        result["motivo"] = "sin picks nuevos que pasen los filtros"
        return result
    tickets = enumerate_tickets(cands, max_legs=1)
    stakes = allocate_flat(cands, bank, remaining, max_picks=slots)
    out = _materialize_singles(cands, tickets, stakes, first_id=max_tid + 1)
    if not out:
        result["motivo"] = "presupuesto restante menor a la unidad"
        return result
    staked = float(sum(t["stake"] for t in out))
    with engine.begin() as conn:
        for t in out:
            conn.execute(text("""
                INSERT INTO odds.portfolio_picks_log
                    (date, ticket_id, legs, ticket_cuota, ticket_prob, stake)
                VALUES (:d, :tid, :legs, :cuota, :prob, :stake)
            """), {"d": day, "tid": t["ticket_id"], "legs": json.dumps(t["legs"]),
                   "cuota": t["cuota"], "prob": t["prob"], "stake": t["stake"]})
        if bk is None:
            conn.execute(text("INSERT INTO odds.bankroll_picks (date, start_bank, staked) VALUES (:d, :b, :s)"),
                         {"d": day, "b": bank, "s": staked})
        else:  # grow the day's stake; a $0 day (settled on the spot) is re-opened
            conn.execute(text("""
                UPDATE odds.bankroll_picks SET staked = staked + :s, returned = NULL,
                       end_bank = NULL, settled_at = NULL WHERE date = :d
            """), {"d": day, "s": staked})
    seed = int(day.strftime("%Y%m%d")) + 1000 * (len(rows) + 1)
    result.update({"tickets": out, "staked": staked, "persisted": True,
                   "summary": _dist_summary(_simulate(cands, stakes, n_sims, seed))})
    logger.info("portfolio picks %s TOP-UP: +%d picks, +$%.0f (%d previos)",
                day, len(out), staked, len(rows))
    return result


def _clear_day(conn, day: date) -> None:
    """--force rebuild: wipe the day's B rows, refusing if anything settled OR in play
    (an in-flight leg is a frozen bet — same guard as portfolio A's _clear_day)."""
    settled = conn.execute(text(
        "SELECT count(*) FROM odds.portfolio_picks_log WHERE date = :d AND status != 'open'"
    ), {"d": day}).scalar()
    if settled:
        raise RuntimeError(f"{day}: picks tickets already settled — refusing to rebuild")
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
        # No outcome yet on a past date: ask the vertical's matches table whether the
        # game was postponed/moved — void immediately instead of waiting VOID_AFTER_DAYS.
        if date.fromisoformat(leg["date"]) < today:
            from sandy.odds import game_postponed
            if game_postponed(conn, spec, leg["date"], leg["home"], leg["away"]):
                return "void"
    if date.fromisoformat(leg["date"]) <= today - timedelta(days=VOID_AFTER_DAYS):
        return "void"  # never reconciled → postponed/cancelled (backstop timer)
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
            cuotas = [leg.get("cuota") for leg in legs]
            status, returned = settle_ticket(results, float(t.stake), float(t.ticket_cuota),
                                             leg_cuotas=cuotas if all(c is not None for c in cuotas) else None)
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
          f"presupuesto ${rep.get('budget', 0):,.0f} · unidad plana ${rep.get('unidad', 0):,.0f}")
    if rep.get("skipped"):
        print(f"  (omitido: {rep['skipped']})")
        return
    if not rep["tickets"]:
        print(f"  🙅 $0 apostado ({rep.get('motivo')})")
        return
    for t in rep["tickets"]:
        legs_txt = " + ".join(
            f"{l['liga_titulo']} {l['partido']}{' · ' + l['hora'] if l.get('hora') else ''}: {l['pick']} @{l['cuota']}"
            f" (🤖 {l.get('meta', 0):.2f}, mercado {l.get('mercado_pct', 0):.0%}, {l.get('n_books', '?')} libros)"
            for l in t["legs"])
        print(f"  Apuesta {t['ticket_id']}: Individual — ${t['stake']:,.0f} en {legs_txt}")
        print(f"      cuota {t['cuota']:.2f} · prob modelo {t['prob']:.1%} · EV modelo ${t['ev']:+,.0f}")
    s = rep["summary"] or {}
    print(f"  Σ apostado ${rep['staked']:,.0f}")
    print(f"  SEGÚN NUESTROS MODELOS (prob calibrada): esperado "
          f"${s.get('expected_profit', 0):+,.0f} · P(día verde) {s.get('p_green', 0):.1%} · "
          f"peor 5% ${s.get('p5', 0):+,.0f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Sandy 🎯 Portafolio B — flat-stake paper portfolio over the day's ✅ picks")
    ap.add_argument("cmd", choices=["build", "settle", "report"],
                    help="build = select+persist today; settle = grade open tickets "
                         "+ close bankroll days; report = ledger summary")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today America/Los_Angeles)")
    ap.add_argument("--budget", type=float, help=f"override the {BUDGET_FRACTION:.0%}-of-bank default")
    ap.add_argument("--risk", default=DEFAULT_RISK, choices=list(RISKS),
                    help="accepted for interface parity with A; B stakes flat")
    ap.add_argument("--force", action="store_true", help="rebuild an already-built day")
    ap.add_argument("--topup", action="store_true",
                    help="if the day is already built, add flat singles for picks logged since")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    ap.add_argument("--dry", action="store_true", help="build without persisting")
    args = ap.parse_args()
    day = date.fromisoformat(args.date) if args.date else None
    ts = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    if args.cmd == "build":
        rep = build_portfolio(day=day, budget=args.budget, risk=args.risk,
                              persist=not args.dry, force=args.force, n_sims=args.sims,
                              topup=args.topup)
        if rep.get("topup"):
            print(f"\n🎯 TOP-UP {rep['date']}: +{len(rep['tickets'])} picks, +${rep['staked']:,.0f} "
                  f"({rep.get('tickets_previos', 0)} previos)"
                  + (f" — {rep['motivo']}" if rep.get('motivo') else ""))
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
