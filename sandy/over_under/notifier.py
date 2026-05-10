"""Over/Under Notifier — Telegram message formatting and sending.

Formats morning digest, nightly report, and sends via Telegram bot API.

Requirements: 1.3, 1.5, 1.6, 4.1, 4.2, 4.3, 4.4, 4.5, 7.4, 7.5, 8.4, 10.5
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

import urllib.request
import urllib.parse

from sandy.logging import get_logger
from sandy.over_under.schemas import (
    CalibrationSnapshot,
    OverUnderPrediction,
    RetrainingResult,
)

logger = get_logger("over_under.notifier")


def format_morning_digest(
    predictions: list[OverUnderPrediction],
    calibration: CalibrationSnapshot | None,
) -> str:
    """Format the morning Telegram message.

    Includes:
    - Trust signal from latest calibration (or "insufficient data" note)
    - One line per game: HOMEvAWAY P(over 6.5)=XX%, sorted by game_time_utc ascending
    """
    today_str = date.today().strftime("%b %d")
    lines: list[str] = [f"⚾ Over/Under Predictions ({today_str})"]

    # Trust signal from calibration
    if calibration is not None:
        prob_thresholds = calibration.covariate_insights.get("probability_thresholds", {})

        # Build trust signal from the best findings
        trust_lines = []
        for t_str in ["5.5", "6.5"]:
            t_data = prob_thresholds.get(t_str, {})
            if t_data and not t_data.get("insufficient_data"):
                best_prob = t_data.get("best_prob_cutoff", 0.5)
                best_acc = t_data.get("accuracy_at_cutoff", 0.0)
                best_games = t_data.get("games_at_cutoff", 0)
                if best_games >= 3:
                    trust_lines.append(
                        f"O{t_str} at {best_prob:.0%}+ → {best_acc:.0%} accuracy ({best_games} games)"
                    )

        if trust_lines:
            lines.append("📊 Best picks this week: " + " | ".join(trust_lines))
        else:
            acc_6_5 = calibration.accuracy_by_threshold.get(6.5, 0.0)
            lines.append(f"📊 Overall 6.5 accuracy: {acc_6_5:.0%} (7-day)")
    else:
        lines.append("📊 Not enough history yet for calibration signal.")

    lines.append("")

    # Sort predictions by probability descending (highest confidence first) — O5.5
    sorted_preds = sorted(predictions, key=lambda p: p.p_over.get(5.5, 0.0), reverse=True)

    for pred in sorted_preds:
        p_over_6_5 = pred.p_over.get(6.5, 0.0)
        p_over_5_5 = pred.p_over.get(5.5, 0.0)
        # Convert UTC to PST (UTC-7)
        from datetime import timedelta
        pst_time = pred.game_time_utc - timedelta(hours=7)
        game_time_str = pst_time.strftime("%I:%M %p").lstrip("0") + " PST"
        fallback_marker = " (fallback)" if pred.pitcher_fallback else ""
        sigma_str = f"σ={pred.sigma_used:.2f}" if pred.sigma_used else ""
        lines.append(
            f"{pred.home_team_code} vs {pred.away_team_code}  "
            f"O5.5={p_over_5_5:.1%} | O6.5={p_over_6_5:.1%}  "
            f"{sigma_str}  ⏰ {game_time_str}{fallback_marker}"
        )

    lines.append(f"\n({len(predictions)} games total)")

    # Sigma summary
    sigmas = [p.sigma_used for p in predictions if p.sigma_used]
    if sigmas:
        lines.append(f"σ range: {min(sigmas):.2f}–{max(sigmas):.2f} (matchup-specific)")

    # Top 3 picks: highest probability + lowest σ (confidence score = prob / σ)
    scored = [
        (pred, pred.p_over.get(5.5, 0.0) / pred.sigma_used if pred.sigma_used else 0.0)
        for pred in predictions
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = scored[:3]
    if top3:
        lines.append("")
        lines.append("🏅 Top 3 picks (high prob + low σ):")
        for i, (pred, score) in enumerate(top3, 1):
            p55 = pred.p_over.get(5.5, 0.0)
            lines.append(
                f"  {i}. {pred.home_team_code} vs {pred.away_team_code}  "
                f"O5.5={p55:.1%}  σ={pred.sigma_used:.2f}"
            )

    return "\n".join(lines)


def format_nightly_report(
    outcomes: list[dict[str, Any]],
    calibration: CalibrationSnapshot | None,
    retraining: RetrainingResult | None,
) -> str:
    """Format the nightly Telegram message.

    Includes:
    - "Tonight's over/under (6.5): X/Y correct"
    - Per-game line with probability, actual total, ✅/❌, top-3 features
    - Calibration one-liner
    - Model retraining result
    """
    today_str = date.today().strftime("%b %d")
    lines: list[str] = [f"🌙 Over/Under Results ({today_str})"]

    # Count correct at 6.5 threshold
    total = len(outcomes)
    correct = sum(1 for o in outcomes if o.get("was_correct_6_5") is True)

    if total > 0:
        pct = correct / total * 100
        lines.append(
            f"Tonight's over/under (6.5): {correct}/{total} correct ({pct:.0f}%)"
        )
    else:
        lines.append("No outcomes to report.")

    lines.append("")

    # Per-game lines
    for o in outcomes:
        was_correct = o.get("was_correct_6_5")
        icon = "✅" if was_correct else "❌"
        home = o.get("home_team_code", "???")
        away = o.get("away_team_code", "???")
        p_over = o.get("p_over_6_5", 0.0)
        actual = o.get("actual_total_runs", "?")

        # Extract top-3 features from feature_vector
        fv = o.get("feature_vector", {})
        if isinstance(fv, str):
            import json
            try:
                fv = json.loads(fv)
            except (json.JSONDecodeError, TypeError):
                fv = {}

        feature_parts: list[str] = []
        for key in ["home_starter_era", "away_starter_era", "ballpark_id",
                    "home_trailing15_rpg", "away_trailing15_rpg"]:
            val = fv.get(key) or o.get(key)
            if val is not None:
                short_key = {
                    "home_starter_era": "ERA",
                    "away_starter_era": "ERA",
                    "ballpark_id": "BPK",
                    "home_trailing15_rpg": "RPG",
                    "away_trailing15_rpg": "RPG",
                }.get(key, key)
                feature_parts.append(f"{short_key}: {val}")
            if len(feature_parts) >= 3:
                break

        features_str = ", ".join(feature_parts) if feature_parts else ""
        features_display = f"  ({features_str})" if features_str else ""

        lines.append(
            f"{icon} {home} vs {away}  P={p_over:.0%}  "
            f"Actual: {actual}{features_display}"
        )

    lines.append("")

    # Calibration one-liner
    if calibration is not None:
        acc_6_5 = calibration.accuracy_by_threshold.get(6.5, 0.0)
        rec_t = calibration.recommended_threshold
        lines.append(
            f"Calibration: 6.5 accuracy = {acc_6_5:.0%} (7-day). "
            f"Optimal threshold: {rec_t}."
        )
    else:
        lines.append("Calibration: insufficient data.")

    # Retraining result
    if retraining is not None:
        if retraining.success:
            prev_str = f" (prev: {retraining.previous_mae:.2f})" if retraining.previous_mae else ""
            lines.append(
                f"🤖 Runs model retrained: {retraining.sample_size} games, "
                f"MAE = {retraining.new_mae:.2f}{prev_str}."
            )
        elif retraining.skipped_reason:
            lines.append(f"🤖 Retraining skipped: {retraining.skipped_reason}")

    return "\n".join(lines)


def format_no_games_message() -> str:
    """Returns message for no games scheduled."""
    return "No games scheduled today — no over/under predictions."


def format_no_finals_message() -> str:
    """Returns message for no final scores available."""
    return "No final scores yet for today's predictions — will retry tomorrow."


def send_telegram(message: str) -> bool:
    """Send message via Telegram bot API. Returns True on success.

    Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    Matches the pattern in daily_refresh.sh (HTTP POST to sendMessage endpoint).
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping notification",
            extra={"component": "over_under.notifier"},
        )
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning(
            f"Failed to send Telegram notification: {exc}",
            extra={"component": "over_under.notifier"},
        )
        return False


__all__ = [
    "format_morning_digest",
    "format_nightly_report",
    "format_no_finals_message",
    "format_no_games_message",
    "send_telegram",
]
