"""Telegram digest for the football vertical.

Reuses :func:`sandy.over_under.notifier.send_telegram` (same bot token / chat),
so football messages arrive alongside the baseball ones but are clearly a
separate digest. Pure formatting functions are unit-testable; the CLI fetches
the data and calls them.
"""
from __future__ import annotations

from datetime import date

from sandy.logging import get_logger
from sandy.over_under.notifier import send_telegram  # reuse existing sender

logger = get_logger("football.notifier")

_MEDALS = ["🥇", "🥈", "🥉"]


def _favorite(p: dict) -> tuple[str, float]:
    """Return (label, prob) of the most likely outcome."""
    options = [
        (p["home"], p["p_home_win"]),
        ("Draw", p["p_draw"]),
        (p["away"], p["p_away_win"]),
    ]
    return max(options, key=lambda x: x[1])


def format_daily_digest(
    predictions: list[dict],
    calibration: list[dict] | None = None,
    results: list[dict] | None = None,
) -> str:
    """Format the daily football Telegram message.

    ``predictions``: today's upcoming picks (dicts with home/away/p_*/most_likely
    /p_over_2_5/p_btts). ``calibration``: latest snapshots (trust signal).
    ``results``: last night's reconciled rows (optional).
    """
    today_str = date.today().strftime("%b %d")
    lines: list[str] = [f"⚽ World Cup Predictions ({today_str})"]

    # Trust signal from calibration.
    if calibration:
        by = {c["market"]: c for c in calibration}
        r = by.get("result")
        if r:
            lines.append(
                f"📊 Calibration ({r['sample_size']} backtested): "
                f"result {r['accuracy']*100:.0f}% · "
                f"O/U {by.get('over_2_5',{}).get('accuracy',0)*100:.0f}% · "
                f"BTTS {by.get('btts',{}).get('accuracy',0)*100:.0f}%. "
                f"Trust picks ≥{(r.get('recommended_threshold') or 0.7)*100:.0f}% confidence."
            )
    else:
        lines.append("📊 Calibration: not enough history yet.")

    # Last night's results.
    if results:
        correct = sum(1 for x in results if x.get("was_correct_result"))
        lines.append(f"\n🌙 Last night: {correct}/{len(results)} results correct")
        for x in results[:8]:
            mark = "✅" if x.get("was_correct_result") else "❌"
            lines.append(
                f"  {mark} {x['home']} {x['actual_home_goals']}-{x['actual_away_goals']} {x['away']}"
            )

    # Today's picks, sorted by confidence desc.
    if predictions:
        ranked = sorted(
            predictions, key=lambda p: max(p["p_home_win"], p["p_draw"], p["p_away_win"]),
            reverse=True,
        )
        lines.append(f"\n🔮 Today's picks ({len(ranked)}):")
        for i, p in enumerate(ranked):
            fav, prob = _favorite(p)
            medal = _MEDALS[i] if i < 3 else "  "
            conf = max(p["p_home_win"], p["p_draw"], p["p_away_win"])
            flag = "  ⚠️ coin-flip" if conf < 0.45 else ""
            lines.append(
                f"{medal} {p['home']} vs {p['away']} → {fav} {prob*100:.0f}% "
                f"| {p['most_likely_home']}-{p['most_likely_away']} "
                f"| O2.5 {p['p_over_2_5']*100:.0f}% | BTTS {p['p_btts']*100:.0f}%{flag}"
            )
    else:
        lines.append("\nNo World Cup matches scheduled today.")

    lines.append(
        "\nℹ️ O2.5 = chance of 3+ total goals · BTTS = both teams score · "
        "conf = how sure the top pick is (trust higher)."
    )
    return "\n".join(lines)


def notify_daily(
    predictions: list[dict],
    calibration: list[dict] | None = None,
    results: list[dict] | None = None,
) -> bool:
    """Format + send the daily digest. Returns True on send success."""
    msg = format_daily_digest(predictions, calibration, results)
    return send_telegram(msg)


__all__ = ["format_daily_digest", "notify_daily", "send_telegram"]
