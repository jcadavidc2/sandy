"""Unit tests for 🎯 Portafolio B (sandy/portfolio_picks.py) — the FLAT-STAKE singles arm.

B reuses A's plumbing (enumerate_tickets / settle_ticket — imported, not copied);
these tests cover B's own pieces: the Picks-del-Día per-game selection, the
flat allocator, and the grading wiring. No database needed."""
from __future__ import annotations

import numpy as np
import pytest

from sandy.portfolio_picks import (
    B_FLAT_FRACTION,
    B_MAX_PICKS_PER_DAY,
    STEP,
    allocate_flat,
    best_per_game,
    enumerate_tickets,
    flat_stake,
    settle_ticket,
)


def _cand(i, cuota, prob, game=None, meta=0.8):
    return {"game": game or ("lg", f"H{i}", f"A{i}"), "cuota": cuota, "prob": prob,
            "p_bet": prob, "meta": meta, "ev": prob * cuota - 1.0}


# ------------------------------------------------------------------- settle --
def test_winning_combo_pays_stake_times_cuota():
    status, ret = settle_ticket(["win", "win"], 1500.0, 3.4)
    assert status == "won" and ret == pytest.approx(1500.0 * 3.4)


def test_losing_single_returns_zero():
    assert settle_ticket(["lose"], 500.0, 1.8) == ("lost", 0.0)


def test_void_and_pending_behave_like_a():
    assert settle_ticket(["void"], 500.0, 1.8) == ("void", 500.0)
    assert settle_ticket(["win", "pending"], 500.0, 3.0) == ("open", None)
    # a void leg drops out and the parlay re-prices over the remaining legs
    status, ret = settle_ticket(["win", "void"], 1000.0, 3.6, leg_cuotas=[1.8, 2.0])
    assert status == "won" and ret == pytest.approx(1800.0)


# --------------------------------------------------------- per-game selection --
def test_best_per_game_mirrors_picks_del_dia_highest_meta():
    g = ("mlb", "NYY", "BOS")
    hi_meta = _cand(0, cuota=1.60, prob=0.60, game=g, meta=0.91)   # ev = -0.04
    hi_ev = _cand(1, cuota=2.10, prob=0.55, game=g, meta=0.82)     # ev = +0.155
    other = _cand(2, cuota=1.90, prob=0.55, meta=0.85)             # distinct game
    out = best_per_game([hi_meta, hi_ev, other])
    assert len(out) == 2                              # one per game
    assert any(c is hi_meta for c in out)             # highest 🤖 wins the game
    assert all(c is not hi_ev for c in out)
    g2 = ("nba", "LAL", "BOS")
    a = _cand(3, cuota=2.0, prob=0.60, game=g2, meta=None)
    b = _cand(4, cuota=1.5, prob=0.60, game=g2, meta=None)
    out2 = best_per_game([a, b])
    assert len(out2) == 1 and out2[0] is a            # EV tie-break when no 🤖
    assert [c["ev"] for c in out] == sorted((c["ev"] for c in out), reverse=True)


# ------------------------------------------------------------ flat allocator --
def test_flat_stake_is_two_percent_of_bank_floored_to_steps():
    assert flat_stake(100_000.0) == pytest.approx(2000.0)
    assert flat_stake(133_333.0) == pytest.approx(2500.0)     # 2666 → floored to $2,500
    assert flat_stake(10_000.0) == STEP                        # never below one step
    assert B_FLAT_FRACTION == 0.02


def test_flat_allocation_same_stake_highest_meta_first_within_budget():
    bank, budget = 100_000.0, 10_000.0
    cands = [_cand(0, 1.9, 0.60, meta=0.70), _cand(1, 1.8, 0.62, meta=0.90),
             _cand(2, 2.0, 0.58, meta=0.80), _cand(3, 1.7, 0.65, meta=0.66),
             _cand(4, 1.9, 0.60, meta=0.75), _cand(5, 1.9, 0.60, meta=0.60),
             _cand(6, 1.9, 0.60, meta=0.95)]
    stakes = allocate_flat(cands, bank, budget)
    unit = flat_stake(bank)
    funded = [i for i, s in enumerate(stakes) if s > 0]
    assert len(funded) == min(B_MAX_PICKS_PER_DAY, int(budget // unit)) == 5
    assert all(stakes[i] == unit for i in funded)              # FLAT: identical stakes
    assert stakes.sum() <= budget + 1e-9
    # the five highest-🤖 candidates got the money (0.95, 0.90, 0.80, 0.75, 0.70)
    assert sorted(funded) == [0, 1, 2, 4, 6]
    assert stakes[5] == 0 and stakes[3] == 0


def test_flat_allocation_respects_a_small_budget_and_empty_days():
    cands = [_cand(0, 1.9, 0.60), _cand(1, 1.8, 0.62)]
    stakes = allocate_flat(cands, 100_000.0, budget=2_000.0)  # room for exactly one unit
    assert stakes.sum() == pytest.approx(2000.0) and (stakes > 0).sum() == 1
    assert allocate_flat(cands, 100_000.0, budget=STEP - 100.0).sum() == 0.0
    assert allocate_flat([], 100_000.0, budget=10_000.0).size == 0


def test_b_tickets_are_singles_only_aligned_with_candidates():
    cands = [_cand(i, 1.9, 0.6) for i in range(4)]
    tickets = enumerate_tickets(cands, max_legs=1)
    assert [t["legs_idx"] for t in tickets] == [(0,), (1,), (2,), (3,)]
    stakes = allocate_flat(cands, 100_000.0, 10_000.0)
    assert len(stakes) == len(tickets)


def test_flat_allocation_topup_slots():
    """A top-up pass only fills the picks still allowed today."""
    cands = [_cand(i, 1.9, 0.6, meta=0.9 - i / 100) for i in range(6)]
    assert (allocate_flat(cands, 100_000.0, 10_000.0, max_picks=2) > 0).sum() == 2
    assert allocate_flat(cands, 100_000.0, 10_000.0, max_picks=0).sum() == 0.0


def test_flat_allocation_is_deterministic():
    cands = [_cand(0, 1.70, 0.56, meta=0.7), _cand(1, 1.90, 0.54, meta=0.8)]
    s1 = allocate_flat(cands, 100_000.0, 10_000.0)
    s2 = allocate_flat(cands, 100_000.0, 10_000.0)
    assert np.array_equal(s1, s2)
