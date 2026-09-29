"""The 3-way h2h detection is per BOOK, never per sport (2026-09-29): a hockey book
that lists a Draw yields regulation no-vig probabilities and a derived 1X row; a
2-way book (moneyline incl. OT) yields no-vig summing to 1 and NO double chance."""
from __future__ import annotations

import pytest

from sandy.odds import _event_rows


def _event():
    return {
        "id": "ev1", "home_team": "Boston Bruins", "away_team": "New York Rangers",
        "commence_time": "2026-09-29T23:00:00Z",
        "bookmakers": [
            {"key": "unibet_se", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Bruins", "price": 2.32},
                {"name": "New York Rangers", "price": 2.39},
                {"name": "Draw", "price": 4.10}]}]},
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Bruins", "price": 1.95},
                {"name": "New York Rangers", "price": 1.95}]}]},
        ],
    }


def test_three_way_book_derives_double_chance_and_regulation_novig():
    rows = _event_rows("nhl", "icehockey_nhl", _event())
    uni = {(r["market"], r["side"]): r for r in rows if r["book"] == "unibet_se"}
    assert ("double_chance", "home_or_draw") in uni and ("double_chance", "away") in uni
    i_h, i_a, i_x = 1 / 2.32, 1 / 2.39, 1 / 4.10
    total = i_h + i_a + i_x
    assert uni[("h2h", "home")]["implied_novig"] == pytest.approx(i_h / total)
    assert uni[("double_chance", "home_or_draw")]["price"] == pytest.approx(1 / (i_h + i_x))
    assert uni[("double_chance", "home_or_draw")]["implied_novig"] == pytest.approx((i_h + i_x) / total)
    assert uni[("double_chance", "away")]["price"] == pytest.approx(2.39)


def test_two_way_book_derives_nothing_and_novig_sums_to_one():
    rows = _event_rows("nhl", "icehockey_nhl", _event())
    pin = {(r["market"], r["side"]): r for r in rows if r["book"] == "pinnacle"}
    assert not any(m == "double_chance" for m, _ in pin)
    assert pin[("h2h", "home")]["implied_novig"] + pin[("h2h", "away")]["implied_novig"] == pytest.approx(1.0)
