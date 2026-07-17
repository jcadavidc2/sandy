"""🎰 Portafolio diario — PAPER-money daily betting-portfolio optimizer.

Simulates, with play money, what a disciplined bettor would do each day on a
BetPlay-style account. The user does NOT bet — this is analysis made "as real
as possible":

  * CANDIDATES  — today's odds.value_log picks (meta-approved ✅ with matched
    odds and edge ≥ 3pp, see sandy/odds.py) that are not yet settled. ONE
    candidate per GAME (the highest-EV market wins) so no ticket ever rides
    two correlated legs of the same match.
  * EDGE CONSERVATISM (edge shrinkage) — our calibrated probabilities can be
    optimistic exactly on the picks where we disagree most with the market
    (winner's curse). For every STAKING decision we use a shrunk probability
        p_bet = 0.7·prob_nuestra + 0.3·prob_mercado_sin_margen
    i.e. partial deference to market wisdom — standard practice when betting
    on model edges. The shrunk edge is then p_bet − mercado = 0.7·edge_raw.
    A pick only enters the portfolio if its SHRUNK edge (and shrunk EV) > 0.
    All Kelly/EV math uses p_bet; raw model probabilities are kept alongside
    so the page can show BOTH "estimación prudente" and "si nuestros modelos
    tienen razón".
  * TICKETS     — singles + parlays of 2..MAX_PARLAY_LEGS legs, ALL legs from
    DIFFERENT games. Parlay cuota = Π leg cuotas and parlay prob = Π leg
    probs — an INDEPENDENCE assumption: legs are distinct matches (usually
    distinct leagues), so cross-leg correlation is ~0; shared drivers like
    weather or slate effects are ignored (documented, not modeled). Tickets
    are ranked by (shrunk) EV; the optimizer sees the top ~TOP_TICKETS
    (singles always survive — they are the low-variance blocks Kelly needs).
  * OPTIMIZER   — fractional-Kelly expected-log-wealth via Monte Carlo:
    N_SIMS simulated days (each leg wins with its p_bet; tickets settle
    accordingly), then greedy coordinate ascent in $500 steps (BetPlay's
    minimum bet) until the day budget is spent or no step improves
    E[log wealth]. The risk dial (Conservador ⅛ / Balanceado ¼ / Agresivo ½)
    is the classic shrunk-virtual-bankroll trick: maximizing E[log(V + P&L)]
    with V = bank·f yields stakes ≈ f × full Kelly. Deterministic RNG seed
    (= YYYYMMDD) → the same day always rebuilds the same portfolio.
  * RISK CAP    — total exposure to any single GAME across all tickets ≤ 30%
    of the day budget.
  * BANKROLL    — compounding paper bankroll in odds.bankroll (initial
    $100,000). Daily stake budget = min(10% of the available bank floored to
    $500, available bank). Days without positive-edge picks stake $0 and are
    still logged (a no-bet day is a decision too).
  * SETTLING    — from OUR predictions' reconciled outcomes via
    odds.value_log.result (filled by sandy.odds.reconcile_value_log). All
    legs win → returned = stake × cuota; any leg loses → returned = 0. A leg
    whose game is postponed/moved (reconcile marks it void, or it never
    reconciles within VOID_AFTER_DAYS days) DROPS OUT and the parlay is
    re-priced over the remaining legs (void leg cuota → 1.0), exactly like
    real books; only a ticket whose EVERY leg is void refunds the stake.
  * PROJECTIONS — project_bankroll() Monte-Carlos the next N days assuming
    days like the recent logged ones (each simulated day replays a randomly
    chosen recent day's P&L-per-bankroll distribution, preserving within-day
    ticket correlation), with stakes compounding at 10% of the running bank.
  * AUDIT TRAIL — every persisted leg logs the exact assumptions used
    (prob raw, p_bet, cuota, mercado_pct, edge raw + prudente) so anyone can
    re-check the honesty of past portfolios.

Timing assumption: the persisted portfolio is built at 14:15 UTC right after
the odds fetch (scripts/odds_daily.sh); we assume the day's games have not
kicked off (the odds layer only stores pre-match prices). PROBABILITY
SEMANTICS: leg `prob` is the pick's OWN calibrated side probability from
odds.value_log — NOT the 🤖 meta P(correct).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.odds import DISPLAY_TZ

logger = logging.getLogger(__name__)

INITIAL_BANK = 100_000.0        # paper COP-like units
STEP = 500.0                    # BetPlay minimum bet / stake granularity
BUDGET_FRACTION = 0.30          # daily budget = 30% of available bankroll (user)
GAME_CAP_FRACTION = 0.30        # max exposure to ONE game across all tickets
MAX_PARLAY_LEGS = 4             # owner decision 2026-07-14: the cups expansion exploded the
                                # candidate pool and the optimizer started building x6/x7
                                # parlays (13 of 15 tickets on 07-14!) — sizes with ZERO
                                # settled evidence, compounding estimation error, and no
                                # recovery variability (one bad leg kills everything).
                                # Settled data: x1-x4 profitable in both portfolios.
MAX_PARLAY_POOL = 16            # enumerate parlays from the top-EV candidates only
MAX_TICKETS_PER_DAY = 15        # operability cap (owner, 2026-07-12): once 15 tickets are
                                # open, the greedy can only TOP UP existing ones — the
                                # objective (E[log wealth]) is untouched, only the day's
                                # weakest marginal tickets are declined. Shared by A and B.
TOP_TICKETS = 60
N_SIMS = 20_000
VOID_AFTER_DAYS = 3             # unreconciled leg older than this → void (postponed)
SHRINK_MODEL_WEIGHT = 0.70      # p_bet = 0.7·ours + 0.3·market (edge shrinkage)

# risk dial → Kelly fraction (virtual bankroll V = bank * fraction)
RISKS = {"Conservador": 1 / 8, "Balanceado": 1 / 4, "Agresivo": 1 / 2}
DEFAULT_RISK = "Agresivo"       # official daily portfolio risk (user choice, ½-Kelly)

_UNITS = {"mlb": "carreras", "nba": "puntos", "nfl": "puntos"}  # everything else: goles


# ------------------------------------------------------------------- schema --
def ensure_tables(engine) -> None:
    """Run the idempotent portfolio migration (CREATE ... IF NOT EXISTS)."""
    sql = (Path(__file__).parent / "migrations" / "add_portfolio_tables.sql").read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))


# -------------------------------------------------------------- probability --
def shrunk_prob(prob: float, novig: float) -> float:
    """Edge-shrunk staking probability: partial deference to market wisdom.

    p_bet = w·ours + (1−w)·market_novig with w = SHRINK_MODEL_WEIGHT. The
    shrunk edge is exactly w·(prob − novig) = w·edge_raw."""
    return SHRINK_MODEL_WEIGHT * prob + (1.0 - SHRINK_MODEL_WEIGHT) * novig


# ------------------------------------------------------------------- labels --
def _league_title(key: str) -> str:
    try:
        from sandy.dashboard.data import league_title
        return league_title(key)
    except Exception:
        return key


def _market_label(league: str, market: str) -> str:
    try:
        from sandy.betmeta import SPECS
        from sandy.dashboard.data import market_label
        kind = SPECS[league]["markets"][market][1]
        return market_label(market, kind)
    except Exception:
        return market


def _pick_label(league: str, market: str, side: str, line, home: str, away: str) -> str:
    unit = _UNITS.get(league, "goles")
    if side == "over":
        return f"Más de {line} {unit}"
    if side == "under":
        return f"Menos de {line} {unit}"
    if side == "home":
        return f"Gana {home}"
    if side == "home_or_draw":
        return f"{home} o empate (1X)"
    if side == "away" and market == "double_chance":
        return f"Gana {away} (2)"
    return f"Gana {away}"


# --------------------------------------------------------------- candidates --
def started_games(engine, day: date) -> set:
    """(league, home, away) of today's games whose kickoff already passed —
    the sims/builds must never (re)offer an in-progress game.
    Accepts an Engine or an already-open Connection (used inside _clear_day)."""
    from sqlalchemy import text as _t
    from sqlalchemy.engine import Connection
    sql, params = _t("""
        SELECT DISTINCT league, COALESCE(our_home, event_home), COALESCE(our_away, event_away)
        FROM odds.market_odds
        WHERE commence_utc <= now() AND commence_utc::date >= :d
    """), {"d": day}
    if isinstance(engine, Connection):
        rows = engine.execute(sql, params).fetchall()
    else:
        with engine.begin() as conn:
            rows = conn.execute(sql, params).fetchall()
    return {(r[0], (r[1] or '').strip(), (r[2] or '').strip()) for r in rows}


def hora_map(engine, day: date) -> dict[tuple, str]:
    """(league, home, away) -> kickoff/first-pitch hour in Bogota ('11:05 AM') for
    leagues whose prediction table exposes a time column (mlb: first_pitch_utc via
    the raw.games join). Lets tickets say WHICH game of a day they mean."""
    from sandy.betmeta import SPECS
    out: dict[tuple, str] = {}
    for lg, spec in SPECS.items():
        try:
            with engine.begin() as conn:
                rows = conn.execute(text(f"""
                    SELECT btrim(home_team) AS h, btrim(away_team) AS a, first_pitch_utc AS fp
                    FROM {spec['table']}
                    WHERE match_date = :d AND first_pitch_utc IS NOT NULL
                """), {"d": day}).fetchall()
        except Exception:
            continue  # this league's table has no time column — fine
        for r in rows:
            out[(lg, r.h, r.a)] = r.fp.astimezone(DISPLAY_TZ).strftime("%I:%M %p").lstrip("0")
    return out


def candidates_for(day: date, engine) -> list[dict]:
    """Value picks for the day → ONE candidate per game (highest shrunk EV).

    Source: odds.value_log (already meta-approved, matched odds, edge ≥ 3pp).
    Unsettled rows only — a pick whose outcome is already known can't be bet.
    Market no-vig prob is recovered exactly as prob − edge (how the value log
    defined edge at log time). Entry rule: SHRUNK edge > 0 and shrunk EV > 0.
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, date, league, home, away, market, side, line, prob, cuota, edge
            FROM odds.value_log
            WHERE date = :d AND result IS NULL AND edge > 0
            ORDER BY id
        """), {"d": day}).fetchall()
    _started = started_games(engine, day)
    _horas = hora_map(engine, day)
    best: dict[tuple, dict] = {}
    for r in rows:
        prob, cuota, edge = float(r.prob), float(r.cuota), float(r.edge)
        novig = prob - edge                       # consensus market prob (no vig)
        p_bet = shrunk_prob(prob, novig)          # staking probability
        edge_prudente = p_bet - novig             # = 0.7·edge
        ev_raw = prob * (cuota - 1.0) - (1.0 - prob)
        ev_prudente = p_bet * (cuota - 1.0) - (1.0 - p_bet)
        if edge_prudente <= 0 or ev_prudente <= 0:
            continue
        game = (r.league, (r.home or '').strip(), (r.away or '').strip())
        if game in _started:
            continue  # game already kicked off — not biddable anymore
        cand = {
            "vl_id": r.id, "date": str(r.date), "game": game,
            "liga": r.league, "liga_titulo": _league_title(r.league),
            "partido": f"{r.home} vs {r.away}", "home": r.home, "away": r.away,
            "hora": _horas.get(game),
            "market": r.market, "side": r.side,
            "line": None if r.line is None else float(r.line),
            "mercado": _market_label(r.league, r.market),
            "pick": _pick_label(r.league, r.market, r.side, r.line, r.home, r.away),
            "cuota": round(cuota, 2),
            "prob": round(prob, 4),                    # raw model side prob
            "p_bet": round(p_bet, 4),                  # shrunk staking prob
            "mercado_pct": round(novig, 4),            # market no-vig prob
            "edge": round(edge, 4),                    # raw edge
            "edge_prudente": round(edge_prudente, 4),  # shrunk edge (0.7·raw)
            "ev": round(ev_raw, 4),
            "ev_prudente": round(ev_prudente, 4),
        }
        # one candidate per game: keep the highest-shrunk-EV market (intra-game
        # legs of the same match are heavily correlated — never parlay them)
        if game not in best or cand["ev_prudente"] > best[game]["ev_prudente"]:
            best[game] = cand
    return sorted(best.values(), key=lambda c: -c["ev_prudente"])


# ------------------------------------------------------------------ tickets --
def enumerate_tickets(cands: list[dict], max_legs: int = MAX_PARLAY_LEGS) -> list[dict]:
    """Singles + cross-game parlays (2..max_legs legs), top-EV pruned.

    Every candidate is a different game by construction, so any combination
    satisfies the "all legs from different games" rule. Parlays are only
    enumerated over the MAX_PARLAY_POOL best candidates (combinatorics guard
    for the 1.9GB box); singles exist for every candidate and always survive
    the pruning. prob/prob_raw are Π of leg probs (independence assumption).
    `max_legs` lets a caller with a different probability regime cap the
    compounding (Portfolio B caps at 4 — see portfolio_picks.B_MAX_PARLAY_LEGS).
    """
    def _ticket(idxs) -> dict:
        cuota = p_bet = p_raw = 1.0
        for i in idxs:
            cuota *= cands[i]["cuota"]
            p_bet *= cands[i]["p_bet"]
            p_raw *= cands[i]["prob"]
        return {"legs_idx": tuple(idxs), "cuota": cuota, "prob": p_bet,
                "prob_raw": p_raw, "ev": p_bet * cuota - 1.0,
                "games": [cands[i]["game"] for i in idxs]}

    singles = [_ticket((i,)) for i in range(len(cands))]
    pool = range(min(len(cands), MAX_PARLAY_POOL))
    parlays = [_ticket(idxs)
               for k in range(2, min(len(pool), max_legs) + 1)
               for idxs in combinations(pool, k)]
    parlays.sort(key=lambda t: -t["ev"])
    keep = max(TOP_TICKETS - len(singles), 0)
    return singles + parlays[:keep]


# ---------------------------------------------------------------- optimizer --
def _optimize(cands: list[dict], tickets: list[dict], budget: float, bank: float,
              risk: str, n_sims: int, seed: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy $500 coordinate ascent on Monte-Carlo E[log(V + P&L)].

    Staking decisions use p_bet (shrunk); the SAME uniform draws also grade
    each leg under the raw model prob, so the caller gets both P&L
    distributions for the chosen stakes. Deterministic: fixed seed +
    first-best-index tie-breaking. Returns (stakes, pnl_prudente, pnl_modelo).
    """
    rng = np.random.default_rng(seed)
    U = rng.random((n_sims, len(cands)))
    wins_bet = U < np.array([c["p_bet"] for c in cands])
    wins_raw = U < np.array([c["prob"] for c in cands])
    V = max(bank * RISKS[risk], 1.0)              # fractional-Kelly virtual bank
    cap = GAME_CAP_FRACTION * budget
    # payoff per $1 staked on ticket i in each sim: cuota·win − 1
    pay = [t["cuota"] * wins_bet[:, list(t["legs_idx"])].all(axis=1) - 1.0
           for t in tickets]
    stakes = np.zeros(len(tickets))
    pnl = np.zeros(n_sims)
    expo: dict[tuple, float] = {}
    spent = 0.0
    n_open = 0
    cur_u = float(np.log(np.maximum(V + pnl, 1e-9)).mean())
    while spent + STEP <= budget + 1e-9:
        best_i, best_u = None, cur_u + 1e-12
        for i, t in enumerate(tickets):
            if stakes[i] == 0.0 and n_open >= MAX_TICKETS_PER_DAY:
                continue  # ticket cap reached: only top up already-open tickets
            if any(expo.get(g, 0.0) + STEP > cap + 1e-9 for g in t["games"]):
                continue  # 30%-per-game exposure cap
            u = float(np.log(np.maximum(V + pnl + STEP * pay[i], 1e-9)).mean())
            if u > best_u:
                best_i, best_u = i, u
        if best_i is None:
            break  # no $500 step improves expected log wealth
        if stakes[best_i] == 0.0:
            n_open += 1
        stakes[best_i] += STEP
        pnl += STEP * pay[best_i]
        for g in tickets[best_i]["games"]:
            expo[g] = expo.get(g, 0.0) + STEP
        spent += STEP
        cur_u = best_u
    pnl_raw = np.zeros(n_sims)
    for i, s in enumerate(stakes):
        if s > 0:
            t = tickets[i]
            pnl_raw += s * (t["cuota"] * wins_raw[:, list(t["legs_idx"])].all(axis=1) - 1.0)
    return stakes, pnl, pnl_raw


def floor500(x: float) -> float:
    return float(int(max(x, 0.0) // STEP) * STEP)


def available_bank(engine=None, before: date | None = None) -> float:
    """Cash available: chain end_bank; open days lock their stakes.

    `before=day` excludes that day's own row — the budget basis for building
    (or what-if rebuilding) day X, so a rebuild after persisting is identical.
    """
    engine = engine or create_engine(load_config())
    ensure_tables(engine)
    bank = INITIAL_BANK
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT start_bank, staked, returned, end_bank FROM odds.bankroll "
            "WHERE (CAST(:b AS DATE) IS NULL OR date < :b) ORDER BY date"
        ), {"b": before}).fetchall()
    for r in rows:
        bank = float(r.end_bank) if r.end_bank is not None else float(r.start_bank) - float(r.staked)
    return bank


def default_budget(bank: float) -> float:
    return min(floor500(BUDGET_FRACTION * bank), floor500(bank))


def _dist_summary(pnl: np.ndarray) -> dict:
    return {"expected_profit": round(float(pnl.mean()), 2),
            "p_green": round(float((pnl > 0).mean()), 4),
            "p5": round(float(np.percentile(pnl, 5)), 2)}


# -------------------------------------------------------------------- build --
def build_portfolio(day: date | None = None, budget: float | None = None,
                    risk: str = DEFAULT_RISK, persist: bool = True,
                    force: bool = False, n_sims: int = N_SIMS,
                    cfg: Config | None = None) -> dict:
    """Build (and optionally persist) the day's paper portfolio.

    persist=False → pure what-if recompute for the dashboard controls (same
    deterministic seed, so default budget+risk reproduces the persisted one).
    summary = P&L distribution under p_bet ("prudente") with a nested
    "modelo" block for the same stakes under our raw probabilities.
    """
    cfg = cfg or load_config()
    day = day or datetime.now(DISPLAY_TZ).date()
    engine = create_engine(cfg)
    ensure_tables(engine)
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {list(RISKS)}")

    if persist and not force:
        with engine.begin() as conn:
            built = conn.execute(text(
                "SELECT 1 FROM odds.bankroll WHERE date = :d UNION ALL "
                "SELECT 1 FROM odds.portfolio_log WHERE date = :d LIMIT 1"
            ), {"d": day}).fetchone()
        if built:
            logger.info("portfolio %s: already built — skipping (idempotent)", day)
            return {"date": str(day), "skipped": "already_built"}

    bank = available_bank(engine, before=day)  # excludes the day's own row
    budget = default_budget(bank) if budget is None else floor500(min(budget, bank))
    cands = candidates_for(day, engine)
    result: dict = {"date": str(day), "bank": round(bank, 2), "budget": budget,
                    "risk": risk, "n_candidates": len(cands), "tickets": [],
                    "staked": 0.0, "summary": None, "persisted": False}

    if cands and budget >= STEP:
        tickets = enumerate_tickets(cands)
        seed = int(day.strftime("%Y%m%d"))
        stakes, pnl, pnl_raw = _optimize(cands, tickets, budget, bank, risk, n_sims, seed)
        out = []
        for t, s in zip(tickets, stakes):
            if s <= 0:
                continue
            legs = [{k: v for k, v in cands[i].items() if k != "game"}
                    for i in t["legs_idx"]]
            out.append({
                "legs": legs, "n_legs": len(legs),
                "cuota": round(t["cuota"], 4),
                "prob": round(t["prob"], 4),          # prudente (p_bet product)
                "prob_modelo": round(t["prob_raw"], 4),
                "stake": float(s),
                "ev": round(float(s) * t["ev"], 2),   # $ EV, prudente
                "ev_modelo": round(float(s) * (t["prob_raw"] * t["cuota"] - 1.0), 2),
            })
        out.sort(key=lambda x: (-x["stake"], -x["ev"]))
        for i, t in enumerate(out, start=1):
            t["ticket_id"] = i
        staked = float(sum(t["stake"] for t in out))
        result["tickets"] = out
        result["staked"] = staked
        if staked > 0:
            result["summary"] = {**_dist_summary(pnl), "modelo": _dist_summary(pnl_raw)}
    if not result["tickets"]:
        result["motivo"] = ("sin picks con valor hoy" if not cands
                            else "presupuesto/utilidad no justifica apostar")
        logger.info("portfolio %s: $0 staked (%s)", day, result["motivo"])

    if persist:
        with engine.begin() as conn:
            if force:
                _clear_day(conn, day)
            for t in result["tickets"]:
                conn.execute(text("""
                    INSERT INTO odds.portfolio_log
                        (date, ticket_id, legs, ticket_cuota, ticket_prob, stake)
                    VALUES (:d, :tid, :legs, :cuota, :prob, :stake)
                """), {"d": day, "tid": t["ticket_id"], "legs": json.dumps(t["legs"]),
                       "cuota": t["cuota"], "prob": t["prob"], "stake": t["stake"]})
            if result["staked"] > 0:
                conn.execute(text("""
                    INSERT INTO odds.bankroll (date, start_bank, staked)
                    VALUES (:d, :b, :s)
                """), {"d": day, "b": bank, "s": result["staked"]})
            else:  # $0 day: logged AND settled on the spot (nothing at risk)
                conn.execute(text("""
                    INSERT INTO odds.bankroll
                        (date, start_bank, staked, returned, end_bank, settled_at)
                    VALUES (:d, :b, 0, 0, :b, now())
                """), {"d": day, "b": bank})
        result["persisted"] = True
    return result


def _clear_day(conn, day: date) -> None:
    """--force rebuild: wipe the day's rows, refusing if anything settled OR in play.
    A ticket whose game already kicked off is a REAL bet in flight — wiping it
    rewrites history (learned 2026-07-11: a mid-day rebuild silently dropped the
    morning's bets on six already-started games)."""
    settled = conn.execute(text(
        "SELECT count(*) FROM odds.portfolio_log WHERE date = :d AND status != 'open'"
    ), {"d": day}).scalar()
    if settled:
        raise RuntimeError(f"{day}: tickets already settled — refusing to rebuild")
    rows = conn.execute(text(
        "SELECT legs FROM odds.portfolio_log WHERE date = :d"), {"d": day}).fetchall()
    started = started_games(conn, day)
    for r in rows:
        legs = r.legs if isinstance(r.legs, list) else json.loads(r.legs)
        for l in legs:
            if (l.get("liga"), (l.get("home") or "").strip(), (l.get("away") or "").strip()) in started:
                raise RuntimeError(
                    f"{day}: ticket leg {l.get('partido')} already kicked off — refusing to "
                    "rebuild (in-flight bets are frozen; void the specific leg instead)")
    conn.execute(text("DELETE FROM odds.portfolio_log WHERE date = :d"), {"d": day})
    conn.execute(text(
        "DELETE FROM odds.bankroll WHERE date = :d AND (staked = 0 OR settled_at IS NULL)"
    ), {"d": day})


# ------------------------------------------------------------------- settle --
def settle_ticket(leg_results: list[str], stake: float, cuota: float,
                  leg_cuotas: list[float] | None = None) -> tuple[str, float | None]:
    """Pure grading rule (unit-tested). leg_results ∈ {win, lose, void, pending}.

    any lose → lost/0 (a dead leg kills the ticket, even with legs pending);
    any pending (and none lost) → still open;
    void legs (postponed/moved games) DROP OUT like real books re-price them:
    the parlay pays stake × Π(cuota of the remaining WINNING legs) — a void leg's
    cuota simply becomes 1.0. ALL legs void → ticket void → stake refunded.
    leg_cuotas gives the per-leg cuotas (same order as leg_results); without it
    (legacy rows missing per-leg prices) any void falls back to a full refund.
    all win → won → stake × cuota.
    """
    if any(r == "lose" for r in leg_results):
        return "lost", 0.0
    if any(r == "pending" for r in leg_results):
        return "open", None
    if any(r == "void" for r in leg_results):
        if all(r == "void" for r in leg_results) or leg_cuotas is None:
            return "void", round(float(stake), 2)
        repriced = 1.0
        for r, lc in zip(leg_results, leg_cuotas):
            if r == "win":
                repriced *= float(lc)
        return "won", round(float(stake) * repriced, 2)
    return "won", round(float(stake) * float(cuota), 2)


def _leg_result(conn, leg: dict, today: date) -> str:
    row = None
    if leg.get("vl_id") is not None:
        row = conn.execute(text("SELECT result FROM odds.value_log WHERE id = :i"),
                           {"i": leg["vl_id"]}).fetchone()
    if row is None:  # fallback: field match (survives value_log re-inserts)
        row = conn.execute(text("""
            SELECT result FROM odds.value_log
            WHERE date = :d AND league = :lg AND home = :h AND away = :a
              AND market = :m AND side = :s
              AND COALESCE(line, -1.0) = COALESCE(CAST(:ln AS REAL), -1.0)
            LIMIT 1
        """), {"d": leg["date"], "lg": leg["liga"], "h": leg["home"], "a": leg["away"],
               "m": leg["market"], "s": leg["side"], "ln": leg.get("line")}).fetchone()
    res = row.result if row else None
    if res == "win":
        return "win"
    if res == "lose":
        return "lose"
    if res == "void":
        return "void"  # reconcile marked the game postponed/moved — stake back
    if date.fromisoformat(leg["date"]) <= today - timedelta(days=VOID_AFTER_DAYS):
        return "void"  # never reconciled → postponed/cancelled
    return "pending"


def settle_portfolio(cfg: Config | None = None, today: date | None = None) -> dict:
    """Grade every open ticket from odds.value_log results, then close the
    bankroll rows of days with no tickets left open:
    end_bank = start_bank − staked + Σ returned."""
    cfg = cfg or load_config()
    today = today or datetime.now(DISPLAY_TZ).date()
    engine = create_engine(cfg)
    ensure_tables(engine)
    graded, still_open = {"won": 0, "lost": 0, "void": 0}, 0
    with engine.begin() as conn:
        open_rows = conn.execute(text(
            "SELECT * FROM odds.portfolio_log WHERE status = 'open' ORDER BY date, ticket_id"
        )).fetchall()
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
                UPDATE odds.portfolio_log
                SET status = :st, returned = :ret, settled_at = now()
                WHERE id = :id
            """), {"st": status, "ret": returned, "id": t.id})
            graded[status] += 1
            logger.info("ticket %s #%d: %s (stake %.0f → %.0f)",
                        t.date, t.ticket_id, status, t.stake, returned or 0.0)
        closed_days = []
        for (d,) in conn.execute(text(
                "SELECT date FROM odds.bankroll WHERE end_bank IS NULL ORDER BY date")):
            pending = conn.execute(text(
                "SELECT count(*) FROM odds.portfolio_log WHERE date = :d AND status = 'open'"
            ), {"d": d}).scalar()
            if pending:
                continue
            conn.execute(text("""
                UPDATE odds.bankroll b
                SET returned = r.tot,
                    end_bank = b.start_bank - b.staked + r.tot,
                    settled_at = now()
                FROM (SELECT COALESCE(SUM(returned), 0) AS tot
                      FROM odds.portfolio_log WHERE date = :d) r
                WHERE b.date = :d
            """), {"d": d})
            closed_days.append(str(d))
    rep = {"graded": graded, "still_open": still_open, "bankroll_days_closed": closed_days,
           "bank": available_bank(engine)}
    logger.info("portfolio settle: %s", rep)
    return rep


# --------------------------------------------------------------- projection --
def _day_pnl_frac(tickets: list[dict], start_bank: float, n_sims: int,
                  seed: int) -> np.ndarray:
    """One logged day → its P&L-as-fraction-of-bankroll distribution.

    Rebuilt from the persisted legs' p_bet (prudent probs), simulating LEGS
    first and settling tickets from them — so within-day correlation between
    tickets that share a leg is preserved exactly."""
    if not tickets:
        return np.zeros(1)
    leg_probs: dict[tuple, float] = {}
    for t in tickets:
        for leg in t["legs"]:
            key = (leg["liga"], leg["home"], leg["away"], leg["market"],
                   leg["side"], leg.get("line"))
            leg_probs[key] = float(leg.get("p_bet") or leg["prob"])
    keys = list(leg_probs)
    rng = np.random.default_rng(seed)
    wins = rng.random((n_sims, len(keys))) < np.array([leg_probs[k] for k in keys])
    kidx = {k: i for i, k in enumerate(keys)}
    pnl = np.zeros(n_sims)
    for t in tickets:
        cols = [kidx[(l["liga"], l["home"], l["away"], l["market"], l["side"], l.get("line"))]
                for l in t["legs"]]
        won = wins[:, cols].all(axis=1)
        pnl += float(t["stake"]) * (float(t["cuota"]) * won - 1.0)
    return pnl / max(start_bank, 1.0)


def project_bankroll(horizon: int = 90, n_paths: int = 2000, lookback: int = 30,
                     n_sims: int = 4000, cfg: Config | None = None) -> dict | None:
    """Monte-Carlo projection of the paper bankroll over the next `horizon`
    days, assuming days like the recent logged ones (bets/day and edges).

    Each simulated day picks a random recent day and samples from THAT day's
    P&L/bankroll distribution (no-value $0 days included — they model the
    real cadence), compounding: bank ← bank·(1 + frac). Continuous scaling of
    stakes with the bank approximates the $500 rounding. Returns per-day
    25/50/75 percentile paths + P(bank below start) at 30/`horizon` days.
    Deterministic seed → stable between reruns of the same day.
    """
    cfg = cfg or load_config()
    engine = create_engine(cfg)
    ensure_tables(engine)
    today = datetime.now(DISPLAY_TZ).date()
    with engine.begin() as conn:
        days = conn.execute(text(
            "SELECT date, start_bank FROM odds.bankroll ORDER BY date DESC LIMIT :n"
        ), {"n": lookback}).fetchall()
        if not days:
            return None
        templates = []
        for d in days:
            rows = conn.execute(text("""
                SELECT stake, ticket_cuota AS cuota, legs
                FROM odds.portfolio_log WHERE date = :d
            """), {"d": d.date}).fetchall()
            tickets = [{"stake": r.stake, "cuota": r.cuota,
                        "legs": r.legs if isinstance(r.legs, list) else json.loads(r.legs)}
                       for r in rows]
            templates.append(_day_pnl_frac(tickets, float(d.start_bank), n_sims,
                                           seed=int(d.date.strftime("%Y%m%d")) * 7 + 1))
    bank0 = available_bank(engine)
    rng = np.random.default_rng(int(today.strftime("%Y%m%d")) * 13 + 5)
    banks = np.full(n_paths, bank0)
    p25, p50, p75 = [], [], []
    below_30 = None
    for day_i in range(horizon):
        t_idx = rng.integers(0, len(templates), n_paths)
        frac = np.empty(n_paths)
        for ti in np.unique(t_idx):
            mask = t_idx == ti
            D = templates[ti]
            frac[mask] = D[rng.integers(0, len(D), int(mask.sum()))]
        banks = np.maximum(banks * (1.0 + frac), 0.0)
        p25.append(float(np.percentile(banks, 25)))
        p50.append(float(np.percentile(banks, 50)))
        p75.append(float(np.percentile(banks, 75)))
        if day_i + 1 == 30:
            below_30 = float((banks < bank0).mean())
    return {"bank0": bank0, "horizon": horizon, "n_templates": len(templates),
            "p25": p25, "p50": p50, "p75": p75,
            "p_below_start_30": below_30 if below_30 is not None else float((banks < bank0).mean()),
            "p_below_start_end": float((banks < bank0).mean())}


# --------------------------------------------------------------- dashboards --
def bankroll_frame(cfg: Config | None = None):
    """odds.bankroll ledger for the 🎰 page (date-ordered)."""
    import pandas as pd
    engine = create_engine(cfg or load_config())
    ensure_tables(engine)
    with engine.begin() as conn:
        return pd.read_sql(text("""
            SELECT date, start_bank, staked, returned, end_bank, settled_at
            FROM odds.bankroll ORDER BY date
        """), conn)


def tickets_frame(cfg: Config | None = None):
    """All persisted tickets, newest first, legs pre-formatted for display."""
    import pandas as pd
    engine = create_engine(cfg or load_config())
    ensure_tables(engine)
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT date, ticket_id, legs, ticket_cuota, ticket_prob, stake,
                   status, returned
            FROM odds.portfolio_log ORDER BY date DESC, ticket_id
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
    print(f"\n🎰 PORTAFOLIO {rep['date']} — banca ${rep.get('bank', 0):,.0f} · "
          f"presupuesto ${rep.get('budget', 0):,.0f} · riesgo {rep.get('risk')}")
    if rep.get("skipped"):
        print(f"  (omitido: {rep['skipped']})")
        return
    if not rep["tickets"]:
        print(f"  🙅 hoy no hay valor — $0 apostado ({rep.get('motivo')})")
        return
    for t in rep["tickets"]:
        kind = "Individual" if t["n_legs"] == 1 else f"Combinada x{t['n_legs']}"
        legs_txt = " + ".join(
            f"{l['liga_titulo']} {l['partido']}{' · ' + l['hora'] if l.get('hora') else ''}: {l['pick']} @{l['cuota']}"
            for l in t["legs"])
        print(f"  Apuesta {t['ticket_id']}: {kind} — ${t['stake']:,.0f} en {legs_txt}")
        print(f"      cuota total {t['cuota']:.2f} · prob prudente {t['prob']:.1%} "
              f"(modelo {t['prob_modelo']:.1%}) · EV prudente ${t['ev']:+,.0f} "
              f"(modelo ${t['ev_modelo']:+,.0f})")
    s = rep["summary"] or {}
    m = s.get("modelo", {})
    print(f"  Σ apostado ${rep['staked']:,.0f}")
    print(f"  PRUDENTE (p_bet = 0.7·nuestra + 0.3·mercado): esperado "
          f"${s.get('expected_profit', 0):+,.0f} · P(día verde) {s.get('p_green', 0):.1%} · "
          f"peor 5% ${s.get('p5', 0):+,.0f}")
    print(f"  SI EL MODELO TIENE RAZÓN: esperado ${m.get('expected_profit', 0):+,.0f} · "
          f"P(día verde) {m.get('p_green', 0):.1%} · peor 5% ${m.get('p5', 0):+,.0f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Sandy 🎰 paper-money portfolio optimizer")
    ap.add_argument("cmd", choices=["build", "settle", "report"],
                    help="build = optimize+persist today; settle = grade open tickets "
                         "+ close bankroll days; report = ledger summary")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today America/Los_Angeles)")
    ap.add_argument("--budget", type=float, help="override the 10%%-of-bank default")
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
        print(f"[{ts()}] portfolio build COMPLETE")
    elif args.cmd == "settle":
        rep = settle_portfolio(today=day)
        print(json.dumps(rep, default=str, indent=2))
        print(f"[{ts()}] portfolio settle COMPLETE")
    else:
        df = bankroll_frame()
        if df.empty:
            print("bankroll: sin días registrados aún")
            return
        settled = df[df["end_bank"].notna()]
        pnl = float((settled["returned"] - settled["staked"]).sum()) if len(settled) else 0.0
        print(f"días: {len(df)} | banca actual: ${available_bank():,.0f} | "
              f"P&L acumulado: ${pnl:+,.0f} | último día: {df.iloc[-1]['date']}")


if __name__ == "__main__":
    main()
