"""Dixon-Coles ratings for international football.

Fits team attack/defense strengths, a home-advantage term, and the Dixon-Coles
low-score correlation (rho) by weighted maximum likelihood over historical
matches. Matches are weighted by competition importance and exponential time
decay, so recent competitive form dominates stale friendlies.

The fitted model is the engine behind :mod:`sandy.football.predictor`: from
(attack, defense, home_adv, rho) it produces the expected goals for any
fixture, and the predictor turns those into a full scoreline distribution.

Pure-ish: :func:`fit_dixon_coles` takes plain arrays and returns a model with
no DB/network. :func:`fit_and_persist` is the DB-aware wrapper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.logging import get_logger

logger = get_logger("football.ratings")

# Exponential time-decay rate (per day). half-life = ln(2)/xi.
# xi=0.0019 -> ~365-day half-life: a match a year ago counts half as much.
DEFAULT_XI = 0.0019
# L2 regularization pulling attack/defense toward 0 (handles low-data teams).
DEFAULT_REG = 0.01
# Artifact filename for the fitted global model (loaded by the predictor).
ARTIFACT_NAME = "football_dixoncoles.json"


@dataclass(frozen=True)
class DixonColesModel:
    attack: dict[int, float]      # team_id -> attack strength
    defense: dict[int, float]     # team_id -> defense strength
    home_adv: float               # additive home advantage on log-rate
    rho: float                    # Dixon-Coles low-score correction
    as_of_date: date
    n_matches: int
    mean_goals: float             # baseline; folded into attack/defense intercept

    def expected_goals(
        self, home_team_id: int, away_team_id: int, *, neutral: bool = False
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) expected goals for a fixture.

        Unknown teams fall back to league-average (attack=defense=0).
        ``neutral`` drops the home-advantage term (WC matches at neutral venues).
        """
        ah = self.attack.get(home_team_id, 0.0)
        dh = self.defense.get(home_team_id, 0.0)
        aa = self.attack.get(away_team_id, 0.0)
        da = self.defense.get(away_team_id, 0.0)
        adv = 0.0 if neutral else self.home_adv
        lam = np.exp(self.mean_goals + ah - da + adv)
        mu = np.exp(self.mean_goals + aa - dh)
        return float(lam), float(mu)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_finished_matches(engine: Engine, as_of: date | None = None) -> pd.DataFrame:
    """Load finished matches with goals, weights, and dates.

    ``as_of`` (exclusive) supports walk-forward backtesting: only matches
    strictly before that date are returned (no leakage).
    """
    clause = "status IN ('FT','AET','PEN') AND home_goals IS NOT NULL AND away_goals IS NOT NULL"
    params: dict = {}
    if as_of is not None:
        clause += " AND match_date < :as_of"
        params["as_of"] = as_of
    sql = text(f"""
        SELECT fixture_id, match_date, home_team_id, away_team_id,
               home_goals, away_goals, COALESCE(competition_weight, 1.0) AS w
        FROM football.matches
        WHERE {clause}
        ORDER BY match_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return pd.DataFrame(rows, columns=[
        "fixture_id", "match_date", "home_team_id", "away_team_id",
        "home_goals", "away_goals", "w",
    ])


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score correction, vectorized. Clamped positive."""
    t = np.ones_like(lam)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    t = np.where(m00, 1.0 - lam * mu * rho, t)
    t = np.where(m01, 1.0 + lam * rho, t)
    t = np.where(m10, 1.0 + mu * rho, t)
    t = np.where(m11, 1.0 - rho, t)
    return np.clip(t, 1e-9, None)


def fit_dixon_coles(
    matches: pd.DataFrame,
    *,
    as_of_date: date,
    xi: float = DEFAULT_XI,
    reg: float = DEFAULT_REG,
    max_iter: int = 200,
) -> DixonColesModel:
    """Weighted-MLE fit of the Dixon-Coles model over ``matches``.

    ``matches`` needs columns: home_team_id, away_team_id, home_goals,
    away_goals, match_date, w (competition weight).
    """
    if matches.empty:
        raise ValueError("no matches to fit")

    teams = sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = matches["home_team_id"].map(idx).to_numpy()
    ai = matches["away_team_id"].map(idx).to_numpy()
    hg = matches["home_goals"].to_numpy(dtype=float)
    ag = matches["away_goals"].to_numpy(dtype=float)

    # Combined weight: competition importance x exponential time decay.
    md = pd.to_datetime(matches["match_date"])
    days_ago = (pd.Timestamp(as_of_date) - md).dt.days.to_numpy().astype(float)
    days_ago = np.clip(days_ago, 0, None)
    w = matches["w"].to_numpy(dtype=float) * np.exp(-xi * days_ago)

    mean_goals = float(np.log(max((hg.sum() + ag.sum()) / (2.0 * len(matches)), 1e-3)))

    # Param layout: [attack(n), defense(n), home_adv, rho]
    def unpack(p):
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

    def nll(p):
        attack, defense, adv, rho = unpack(p)
        loglam = mean_goals + attack[hi] - defense[ai] + adv
        logmu = mean_goals + attack[ai] - defense[hi]
        lam = np.exp(loglam)
        mu = np.exp(logmu)
        tau = _tau(hg, ag, lam, mu, rho)
        ll = np.log(tau) + hg * loglam - lam + ag * logmu - mu
        neg = -np.sum(w * ll)
        # Regularize team params; penalize attack drift (identifiability).
        neg += reg * (np.sum(attack ** 2) + np.sum(defense ** 2))
        neg += 100.0 * (attack.mean() ** 2)
        return neg

    x0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
    bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "maxfun": max_iter * 50})

    attack, defense, adv, rho = unpack(res.x)
    attack = attack - attack.mean()  # center for interpretability

    logger.info(
        "Dixon-Coles fit complete",
        extra={
            "component": "football.ratings", "n_teams": n, "n_matches": len(matches),
            "home_adv": round(float(adv), 4), "rho": round(float(rho), 4),
            "converged": bool(res.success), "nll": round(float(res.fun), 1),
        },
    )
    return DixonColesModel(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(adv), rho=float(rho), as_of_date=as_of_date,
        n_matches=len(matches), mean_goals=mean_goals,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: DixonColesModel, path: Path) -> None:
    payload = {
        "attack": {str(k): v for k, v in model.attack.items()},
        "defense": {str(k): v for k, v in model.defense.items()},
        "home_adv": model.home_adv, "rho": model.rho,
        "as_of_date": model.as_of_date.isoformat(),
        "n_matches": model.n_matches, "mean_goals": model.mean_goals,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def load_model(path: Path) -> DixonColesModel:
    d = json.loads(path.read_text())
    return DixonColesModel(
        attack={int(k): v for k, v in d["attack"].items()},
        defense={int(k): v for k, v in d["defense"].items()},
        home_adv=d["home_adv"], rho=d["rho"],
        as_of_date=date.fromisoformat(d["as_of_date"]),
        n_matches=d["n_matches"], mean_goals=d["mean_goals"],
    )


def _persist_team_ratings(engine: Engine, model: DixonColesModel) -> None:
    """Snapshot per-team attack/defense into football.team_ratings."""
    upsert = text("""
        INSERT INTO football.team_ratings
            (team_id, as_of_date, elo, attack_strength, defense_strength)
        VALUES (:team_id, :as_of_date, NULL, :attack, :defense)
        ON CONFLICT (team_id, as_of_date) DO UPDATE SET
            attack_strength = EXCLUDED.attack_strength,
            defense_strength = EXCLUDED.defense_strength,
            computed_at = now()
    """)
    with engine.begin() as conn:
        for team_id in model.attack:
            conn.execute(upsert, {
                "team_id": team_id, "as_of_date": model.as_of_date,
                "attack": model.attack[team_id], "defense": model.defense.get(team_id, 0.0),
            })


def fit_and_persist(
    config: Config | None = None,
    *,
    as_of: date | None = None,
    xi: float = DEFAULT_XI,
) -> DixonColesModel:
    """Fit Dixon-Coles on all finished matches and persist artifact + snapshot."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    as_of_date = as_of or date.today()
    matches = load_finished_matches(engine, as_of=as_of)
    model = fit_dixon_coles(matches, as_of_date=as_of_date, xi=xi)
    save_model(model, cfg.model.model_dir / ARTIFACT_NAME)
    _persist_team_ratings(engine, model)
    return model


__all__ = [
    "ARTIFACT_NAME",
    "DixonColesModel",
    "fit_and_persist",
    "fit_dixon_coles",
    "load_finished_matches",
    "load_model",
    "save_model",
]
