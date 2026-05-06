"""Confidence assessor — classifies predictions as HIGH or LOW confidence.

Phase 2, Tasks 2.2 + 6.1: Compares predictions to base rates and generates
natural-language explanations. Total function — never raises for valid inputs.

Requirements: 5.1–5.6, 6.3, 6.4
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sandy.live.schemas import ShutdownFeatures
from sandy.schemas import TopFeature


@dataclass(frozen=True)
class ConfidenceResult:
    """Result of confidence assessment for a prediction."""
    level: str                      # "HIGH" or "LOW"
    base_rate: float                # e.g., 0.72 for reached_base
    deviation: float                # prediction - base_rate (signed)
    explanation: str                # natural-language explanation
    shutdown_factors: list[str] = field(default_factory=list)


class ConfidenceAssessor:
    """Classifies predictions as HIGH or LOW confidence based on base rate deviation.

    HIGH confidence = the model is detecting something meaningfully different
    from the historical average. LOW confidence = close to base rate, the
    model isn't seeing anything unusual.

    Total function: assess() never raises for valid inputs.
    """

    BASE_RATES: dict[str, float] = {
        "reached_base": 0.72,
        "game_winner": 0.50,
        "runs": 4.5,
    }
    THRESHOLD: float = 0.05  # ±5 percentage points

    def assess(
        self,
        prediction: float,
        target: str,
        top_features: list[TopFeature] | None = None,
        shutdown_features: ShutdownFeatures | None = None,
    ) -> ConfidenceResult:
        """Classify prediction confidence. Total function — never raises.

        Parameters
        ----------
        prediction:        The predicted value (probability or expected runs)
        target:            "reached_base", "game_winner", or "runs"
        top_features:      Optional top contributing features for explanation
        shutdown_features: Optional shutdown indicators for below-base-rate explanations
        """
        # Handle edge cases (totality guarantee)
        if not isinstance(prediction, (int, float)):
            prediction = 0.5
        import math
        if math.isnan(prediction) or math.isinf(prediction):
            prediction = max(0.0, min(1.0, prediction)) if target != "runs" else 4.5

        base_rate = self.BASE_RATES.get(target, 0.50)

        # For runs target, normalize deviation to a percentage scale
        if target == "runs":
            deviation = (prediction - base_rate) / base_rate
        else:
            deviation = prediction - base_rate

        # Classify
        if abs(deviation) <= self.THRESHOLD:
            level = "LOW"
            explanation = "Close to base rate, nothing unusual detected."
        else:
            level = "HIGH"
            explanation = self._build_high_explanation(
                prediction, base_rate, deviation, target, top_features, shutdown_features
            )

        # Shutdown factors
        shutdown_factor_list: list[str] = []
        if shutdown_features and level == "HIGH" and deviation < 0:
            if shutdown_features.pitcher_zero_baserunner_innings >= 3:
                shutdown_factor_list.append(
                    f"Pitcher has {shutdown_features.pitcher_zero_baserunner_innings} "
                    f"consecutive shutout innings"
                )
            if shutdown_features.is_bottom_of_order:
                shutdown_factor_list.append("Bottom of the order due up (spots 7-8-9)")
            if shutdown_features.is_fresh_reliever:
                shutdown_factor_list.append("Fresh reliever (high velocity, unfamiliar)")
            if shutdown_features.pitcher_game_k_rate > 0.30:
                shutdown_factor_list.append(
                    f"Pitcher K-rate this game: {shutdown_features.pitcher_game_k_rate:.0%}"
                )

        return ConfidenceResult(
            level=level,
            base_rate=base_rate,
            deviation=round(deviation, 4),
            explanation=explanation,
            shutdown_factors=shutdown_factor_list,
        )

    def _build_high_explanation(
        self,
        prediction: float,
        base_rate: float,
        deviation: float,
        target: str,
        top_features: list[TopFeature] | None,
        shutdown_features: ShutdownFeatures | None,
    ) -> str:
        """Build a natural-language explanation for HIGH confidence predictions."""
        direction = "above" if deviation > 0 else "below"
        magnitude = abs(deviation)

        if target == "runs":
            explanation = (
                f"Predicted {prediction:.1f} runs vs average {base_rate:.1f} "
                f"({magnitude:.0%} {direction} average)."
            )
        else:
            explanation = (
                f"Predicted {prediction:.1%} vs base rate {base_rate:.1%} "
                f"({magnitude:.1%} {direction} average)."
            )

        # Add top feature context
        if top_features:
            top_2 = top_features[:2]
            feature_strs = []
            for f in top_2:
                direction_arrow = "↑" if f.contribution > 0 else "↓"
                feature_strs.append(f"{direction_arrow} {f.name}")
            explanation += f" Key factors: {', '.join(feature_strs)}."

        # Add shutdown context
        if shutdown_features and deviation < 0:
            if shutdown_features.pitcher_zero_baserunner_innings >= 3:
                explanation += (
                    f" Pitcher has been dominant "
                    f"({shutdown_features.pitcher_zero_baserunner_innings} clean innings)."
                )

        return explanation


__all__ = ["ConfidenceAssessor", "ConfidenceResult"]
