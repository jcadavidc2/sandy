"""Unit tests for the 🎯 Portafolio B "Picks del Día" math (no DB needed).

B reuses A's math (enumerate_tickets / _optimize / settle_ticket — imported,
never copied); these tests cover what B ADDS: raw-probability candidates with
one-per-game pre-selection, and the ALWAYS-BET forced-deploy floor. The settle
tests go through B's imported grading to prove the wiring (winning combo,
losing single), matching A's test suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from sandy.portfolio_picks import (
    GAME_CAP_FRACTION,
    MIN_DEPLOY_FRACTION,
    STEP,
    allocate_day,
    best_per_game,
    enumerate_tickets,
    floor500,
    settle_ticket,
)


# ------------------------------------------------------------------- settle --
def test_winning_combo_pays_stake_times_cuota():
    # B grades with the same shared rule as A: all legs win → stake × cuota
    status, returned = settle_ticket(["win", "win", "win"], stake=1500.0, cuota=6.844)
    assert status == "won"
    assert returned == pytest.approx(10266.0)


def test_losing_single_returns_zero():
    assert settle_ticket(["lose"], stake=500.0, cuota=1.94) == ("lost", 0.0)


def test_void_and_pending_behave_like_a():
    assert settle_ticket(["win", "pending"], 500.0, 3.0) == ("open", None)
    assert settle_ticket(["win", "void"], 500.0, 3.0) == ("void", 500.0)


# ----------------------------------------------------- one pick per game (B) --
def _cand(i, cuota, prob, game=None):
    prob = float(prob)
    return {"game": game or ("mlb", f"H{i}", f"A{i}"), "cuota": cuota,
            "prob": prob, "p_bet": prob,          # RAW prob — no shrink (B's thesis)
            "ev": prob * cuota - 1.0}


def test_best_per_game_keeps_highest_raw_ev_pick():
    g = ("mlb", "NYY", "BOS")
    low = _cand(0, cuota=1.60, prob=0.60, game=g)    # ev = -0.04
    high = _cand(1, cuota=2.10, prob=0.55, game=g)   # ev = +0.155
    other = _cand(2, cuota=1.90, prob=0.55)          # ev = +0.045, distinct game
    out = best_per_game([low, high, other])
    assert len(out) == 2                             # one per game
    assert out[0] is high                            # highest raw EV wins the game
    assert [c["ev"] for c in out] == sorted((c["ev"] for c in out), reverse=True)


# ------------------------------------------------------- always-bet floor (B) --
def _alloc(cands, budget, bank=100_000.0, risk="Agresivo", seed=20260707, n_sims=5000):
    tickets = enumerate_tickets(cands)
    stakes, pnl, forced = allocate_day(cands, tickets, budget, bank, risk, n_sims, seed)
    return tickets, stakes, pnl, forced


def test_positive_ev_day_is_not_forced():
    cands = [_cand(0, cuota=1.94, prob=0.62)]        # clear positive raw EV
    _, stakes, _, forced = _alloc(cands, budget=10_000.0)
    assert stakes.sum() > 0
    assert not forced                                # normal Kelly day


def test_forced_deploy_when_every_candidate_is_negative_ev():
    # p·cuota < 1 for every ticket → plain Kelly stakes $0; B must still bet
    cands = [_cand(0, cuota=1.70, prob=0.56),        # ev = -0.048 (least bad)
             _cand(1, cuota=1.50, prob=0.55)]        # ev = -0.175
    tickets, stakes, pnl, forced = _alloc(cands, budget=10_000.0)
    assert forced
    expected = max(floor500(MIN_DEPLOY_FRACTION * 10_000.0), STEP)
    assert stakes.sum() == pytest.approx(expected)   # exactly 10% of the budget
    assert all(s % STEP == 0 for s in stakes)        # $500 granularity
    # the money goes to the least-bad best mix: the highest-EV single, never
    # the clearly worse candidate-1 single
    i_best = next(i for i, t in enumerate(tickets) if t["legs_idx"] == (0,))
    i_worst = next(i for i, t in enumerate(tickets) if t["legs_idx"] == (1,))
    assert stakes[i_best] > 0
    assert stakes[i_worst] == 0
    assert np.any(pnl != 0)                          # money is genuinely at risk


def test_forced_deploy_floors_to_min_one_step():
    # 10% of a $3,000 budget is $300 < $500 → still one $500 ticket
    cands = [_cand(0, cuota=1.70, prob=0.56)]
    _, stakes, _, forced = _alloc(cands, budget=3_000.0)
    assert forced
    assert stakes.sum() == pytest.approx(STEP)


def test_forced_deploy_respects_game_cap():
    cands = [_cand(0, cuota=1.70, prob=0.56), _cand(1, cuota=1.72, prob=0.56)]
    budget = 10_000.0
    tickets, stakes, _, forced = _alloc(cands, budget=budget)
    assert forced
    expo: dict[tuple, float] = {}
    for t, s in zip(tickets, stakes):
        for g in t["games"]:
            expo[g] = expo.get(g, 0.0) + s
    assert all(v <= GAME_CAP_FRACTION * budget + 1e-9 for v in expo.values())


def test_no_candidates_day_stakes_zero():
    tickets = enumerate_tickets([])
    stakes, pnl, forced = allocate_day([], tickets, 10_000.0, 100_000.0,
                                       "Agresivo", 1000, 1)
    assert stakes.sum() == 0.0
    assert not forced                                # a true $0 day, not a forced one
    assert np.all(pnl == 0.0)


def test_budget_below_minimum_bet_stakes_zero_even_with_candidates():
    cands = [_cand(0, cuota=1.70, prob=0.56)]
    tickets = enumerate_tickets(cands)
    stakes, _, forced = allocate_day(cands, tickets, STEP - 100.0, 100_000.0,
                                     "Agresivo", 1000, 1)
    assert stakes.sum() == 0.0 and not forced


def test_allocation_is_deterministic():
    cands = [_cand(0, 1.70, 0.56), _cand(1, 1.90, 0.54)]
    _, s1, p1, f1 = _alloc(cands, budget=10_000.0, seed=20260707)
    _, s2, p2, f2 = _alloc(cands, budget=10_000.0, seed=20260707)
    assert np.array_equal(s1, s2) and np.array_equal(p1, p2) and f1 == f2
