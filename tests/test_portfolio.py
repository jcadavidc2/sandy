"""Unit tests for the 🎰 paper-portfolio math (no DB needed).

Covers the settle grading rule, the edge-shrinkage formula and the Kelly
allocation edge cases: caps respected, negative edges → $0, budget below the
minimum bet → $0, and MC determinism (fixed seed).
"""
from __future__ import annotations

import numpy as np
import pytest

from sandy.portfolio import (
    BUDGET_FRACTION,
    DEFAULT_RISK,
    GAME_CAP_FRACTION,
    MAX_PARLAY_LEGS,
    RISKS,
    SHRINK_MODEL_WEIGHT,
    STEP,
    TICKET_CAP_FRACTION,
    _optimize,
    enumerate_tickets,
    settle_ticket,
    shrunk_prob,
)


# ------------------------------------------------------------------- settle --
def test_winning_double_pays_stake_times_cuota():
    status, returned = settle_ticket(["win", "win"], stake=1000.0, cuota=2.9388)
    assert status == "won"
    assert returned == pytest.approx(2938.8)


def test_losing_single_returns_zero():
    assert settle_ticket(["lose"], stake=500.0, cuota=1.86) == ("lost", 0.0)


def test_pending_leg_keeps_ticket_open():
    assert settle_ticket(["win", "pending"], stake=500.0, cuota=3.0) == ("open", None)


def test_void_leg_reprices_parlay_over_remaining_legs():
    # real-book rule: the postponed leg's cuota becomes 1.0; the rest still pay
    status, returned = settle_ticket(["win", "void"], stake=500.0, cuota=3.0,
                                     leg_cuotas=[1.24, 1.77])
    assert status == "won"
    assert returned == pytest.approx(500.0 * 1.24)


def test_void_leg_repricing_multi_leg():
    status, returned = settle_ticket(["win", "void", "win"], stake=1000.0, cuota=5.0,
                                     leg_cuotas=[1.5, 2.0, 1.8])
    assert status == "won"
    assert returned == pytest.approx(1000.0 * 1.5 * 1.8)


def test_all_legs_void_refunds_whole_stake():
    assert settle_ticket(["void", "void"], 500.0, 3.0, leg_cuotas=[1.5, 2.0]) == ("void", 500.0)


def test_void_without_leg_cuotas_falls_back_to_refund():
    # legacy rows without per-leg prices can't re-price — refund like before
    assert settle_ticket(["win", "void"], stake=500.0, cuota=3.0) == ("void", 500.0)


def test_void_leg_with_pending_sibling_stays_open():
    assert settle_ticket(["void", "pending"], 500.0, 3.0, leg_cuotas=[1.5, 2.0]) == ("open", None)


def test_lost_leg_settles_even_with_pending_or_void_siblings():
    assert settle_ticket(["lose", "pending", "void"], 500.0, 8.0) == ("lost", 0.0)


# -------------------------------------------------------------- edge shrink --
def test_shrunk_prob_is_partial_deference_to_market():
    p, m = 0.70, 0.60
    assert shrunk_prob(p, m) == pytest.approx(SHRINK_MODEL_WEIGHT * p
                                              + (1 - SHRINK_MODEL_WEIGHT) * m)
    # shrunk edge is exactly w·(raw edge)
    assert shrunk_prob(p, m) - m == pytest.approx(SHRINK_MODEL_WEIGHT * (p - m))


# ---------------------------------------------------------------- optimizer --
def _cand(i, cuota, prob, p_bet=None):
    return {"game": ("lg", f"H{i}", f"A{i}"), "cuota": cuota,
            "prob": prob, "p_bet": p_bet if p_bet is not None else prob,
            "ev_prudente": (p_bet or prob) * cuota - 1.0}


def _run(cands, budget, bank=100_000.0, risk="Balanceado", seed=42, n_sims=5000):
    tickets = enumerate_tickets(cands)
    stakes, pnl, pnl_raw = _optimize(cands, tickets, budget, bank, risk, n_sims, seed)
    return tickets, stakes, pnl, pnl_raw


def test_single_candidate_respects_budget_and_bank_caps():
    cands = [_cand(0, cuota=1.86, prob=0.65)]  # clear positive edge
    budget, bank = 10_000.0, 100_000.0
    _, stakes, _, _ = _run(cands, budget, bank=bank)
    assert stakes.sum() > 0                                    # it does bet
    assert stakes.sum() <= budget + 1e-9
    # caps are fractions of the BANK (audit 2026-09-28): one single → ticket cap binds
    assert stakes.max() <= TICKET_CAP_FRACTION * bank + 1e-9
    assert stakes.sum() <= GAME_CAP_FRACTION * bank + 1e-9     # one game → game cap
    assert all(s % STEP == 0 for s in stakes)                  # $500 granularity


def test_regime_constants_match_the_audit():
    """The 2026-09-28 regime, pinned so a casual edit cannot silently re-open the
    holes the audit closed (see the module docstring for the numbers)."""
    assert BUDGET_FRACTION == 0.10
    assert MAX_PARLAY_LEGS == 2
    assert SHRINK_MODEL_WEIGHT <= 0.30
    assert RISKS[DEFAULT_RISK] <= 0.25
    assert TICKET_CAP_FRACTION <= 0.02 and GAME_CAP_FRACTION <= 0.03


def test_negative_edge_bets_zero():
    # p·cuota < 1 for every ticket → E[log wealth] falls with any bet
    cands = [_cand(0, cuota=1.80, prob=0.50), _cand(1, cuota=1.50, prob=0.60)]
    _, stakes, pnl, _ = _run(cands, budget=10_000.0)
    assert stakes.sum() == 0.0
    assert np.all(pnl == 0.0)


def test_budget_below_minimum_bet_bets_zero():
    cands = [_cand(0, cuota=2.0, prob=0.70)]
    _, stakes, _, _ = _run(cands, budget=STEP - 100.0)
    assert stakes.sum() == 0.0


def test_multi_candidate_never_exceeds_budget_or_per_game_cap():
    cands = [_cand(0, 1.58, 0.726), _cand(1, 1.86, 0.686), _cand(2, 2.10, 0.55)]
    budget = 10_000.0
    tickets, stakes, _, _ = _run(cands, budget)
    assert stakes.sum() <= budget + 1e-9
    expo: dict[tuple, float] = {}
    for t, s in zip(tickets, stakes):
        for g in t["games"]:
            expo[g] = expo.get(g, 0.0) + s
    assert all(v <= GAME_CAP_FRACTION * 100_000.0 + 1e-9 for v in expo.values())
    assert all(s <= TICKET_CAP_FRACTION * 100_000.0 + 1e-9 for s in stakes)


def test_topup_respects_global_ticket_cap_and_prior_exposure():
    """A top-up pass counts the tickets and per-game exposure already placed today."""
    from sandy.portfolio import MAX_TICKETS_PER_DAY
    cands = [_cand(0, 1.90, 0.66), _cand(1, 1.86, 0.65)]
    tickets = enumerate_tickets(cands)
    # daily ticket cap already reached → nothing new can be opened
    stakes, _, _ = _optimize(cands, tickets, 10_000.0, 100_000.0, "Balanceado", 3000, 1,
                             n_open0=MAX_TICKETS_PER_DAY)
    assert stakes.sum() == 0.0
    # game 0 already carries the full per-game cap → only game 1 can be funded
    stakes, _, _ = _optimize(cands, tickets, 10_000.0, 100_000.0, "Balanceado", 3000, 1,
                             expo0={cands[0]["game"]: GAME_CAP_FRACTION * 100_000.0})
    funded_games = {g for t, s in zip(tickets, stakes) if s > 0 for g in t["games"]}
    assert cands[0]["game"] not in funded_games and cands[1]["game"] in funded_games


def test_fixed_seed_is_deterministic():
    cands = [_cand(0, 1.58, 0.726), _cand(1, 1.86, 0.686)]
    _, s1, p1, _ = _run(cands, budget=10_000.0, seed=20260705)
    _, s2, p2, _ = _run(cands, budget=10_000.0, seed=20260705)
    assert np.array_equal(s1, s2)
    assert np.array_equal(p1, p2)


def test_tickets_are_singles_plus_cross_game_parlays():
    cands = [_cand(i, 1.9, 0.6) for i in range(3)]
    tickets = enumerate_tickets(cands)           # default MAX_PARLAY_LEGS = 2
    sizes = sorted(len(t["legs_idx"]) for t in tickets)
    assert sizes == [1, 1, 1, 2, 2, 2]           # 3 singles, 3 doubles, NO triple
    assert max(sizes) <= MAX_PARLAY_LEGS
    tickets3 = enumerate_tickets(cands, max_legs=3)
    assert sorted(len(t["legs_idx"]) for t in tickets3) == [1, 1, 1, 2, 2, 2, 3]
    for t in tickets:                            # no repeated game inside a ticket
        assert len(set(t["games"])) == len(t["games"])
        # parlay cuota/prob are products of the legs (independence assumption)
        assert t["cuota"] == pytest.approx(1.9 ** len(t["legs_idx"]))
        assert t["prob"] == pytest.approx(0.6 ** len(t["legs_idx"]))
