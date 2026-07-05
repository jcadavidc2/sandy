"""📋 Tablero — the sketch: meta matrices on top (always last-trained model),
games table below (date range, composite cells, multiple ✅ per game, results
for played games), and the final best-pick-per-game list."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from sandy.betmeta import SPECS, THRESHOLDS
from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Tablero", page_icon="📋", layout="wide")
st.title("📋 Tablero por liga")

GROUP_TITLES = {"result": "DOBLE OPORTUNIDAD", "goals": "META MODEL GOALS",
                "corners": "META MODEL CORNER KICKS", "winner": "GANADOR",
                "points": "META MODEL PUNTOS", "btts": "AMBOS ANOTAN"}

league = st.selectbox("Liga", list(D.LEAGUES) + ["mlb"],
                      format_func=lambda k: "⚾ MLB" if k == "mlb" else D.league_title(k))

# ---------------------------------------------------------------- MLB adapter
if league == "mlb":
    st.caption("⚾ MLB corre en su propio esquema histórico (derived.prediction_log). "
               "Probabilidades + resultado por target; su meta-modelo O5.5 sigue en el digest diario.")
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    @st.cache_data(ttl=600)
    def _mlb(start, end):
        e = create_engine(load_config())
        with e.begin() as c:
            return pd.read_sql(text("""
                SELECT p.predicted_at_utc::date AS fecha, p.target, p.team_code,
                       p.probability, p.was_correct
                FROM derived.prediction_log p
                WHERE p.predicted_at_utc::date BETWEEN :a AND :b
                ORDER BY p.predicted_at_utc"""), c, params={"a": start, "b": end})

    import datetime as _dt
    rng = st.date_input("Rango", ( _dt.date.today() - timedelta(days=7), _dt.date.today()))
    if len(rng) == 2:
        mlb = _mlb(*rng)
        if mlb.empty:
            st.info("Sin predicciones MLB en el rango.")
        else:
            mlb["✓"] = mlb["was_correct"].map({True: "✓", False: "✗", None: "—"})
            st.dataframe(mlb, use_container_width=True, hide_index=True,
                         column_config={"probability": st.column_config.ProgressColumn(
                             "prob", format="percent", min_value=0, max_value=1)})
    st.stop()

# ------------------------------------------------- Meta matrices (last model)
art = D.meta_artifact(league)
ebm = art.get("eval_by_market") or {}
thr_by = art.get("threshold_by_market") or {}
if not ebm:
    st.warning("Esta liga aún no tiene matrices por línea (meta sin re-entrenar).")
else:
    st.caption(f"Matrices del ÚLTIMO meta-modelo entrenado ({art.get('trained_at')}) — "
               "no cambian con el filtro de fechas. Celda verde = umbral recomendado de esa línea.")
    kinds: dict[str, list[str]] = {}
    for m, (_p, kind, _l) in SPECS[league]["markets"].items():
        kinds.setdefault(kind, []).append(m)
    for kind, mkts in kinds.items():
        st.subheader(GROUP_TITLES.get(kind, kind.upper()))
        rows = []
        for thr in THRESHOLDS:
            row = {"P(acierto)": f"≥{thr:.0%}"}
            for m in mkts:
                cell = next((t for t in ebm[m]["table"] if abs(t["thr"] - thr) < 1e-9), None)
                row[D.market_label(m)] = (f"{cell['acc']:.0%} ({cell['correct']}/{cell['n']})"
                                          if cell and cell["n"] else "—")
            rows.append(row)
        mdf = pd.DataFrame(rows).set_index("P(acierto)")

        def _hl(col):
            rec = thr_by.get(next(m for m in mkts if D.market_label(m) == col.name))
            mark = f"≥{rec:.0%}" if rec is not None else None
            return ["background-color: #1b5e20; color: white" if idx == mark else ""
                    for idx in col.index]

        st.dataframe(mdf.style.apply(_hl, axis=0), use_container_width=True)

# ------------------------------------------------------------- Games (range)
st.divider()
st.header("Games")
lo, hi = D.date_bounds(league)
default_hi = hi or __import__("datetime").date.today()
default_lo = max(lo, default_hi - timedelta(days=6)) if lo else default_hi
rng = st.date_input("Rango de fechas (pasado = con resultados ✓/✗; futuro = por jugar)",
                    (default_lo, default_hi), min_value=lo, max_value=hi)
if len(rng) != 2:
    st.stop()


@st.cache_data(ttl=300)
def _range(lg, a, b):
    return D.board_range(lg, a, b)


games, finals = _range(league, rng[0], rng[1])
if games.empty:
    st.info("Sin juegos en el rango.")
    st.stop()

st.caption("Cada celda: **prob del modelo** (🤖 P(acierto) / Th umbral de esa línea → "
           "acierto histórico a ese umbral) — **✅ = apuesta** (🤖 ≥ Th; puede haber varias por juego) — "
           "**✓/✗** = si ya se jugó, cómo salió ese pick.")
st.dataframe(games, use_container_width=True, hide_index=True, height=440)

with st.expander("🧠 Covariables de estos juegos"):
    days = pd.date_range(rng[0], rng[1]).date
    if len(days) > 10:
        st.caption("Rango largo — mostrando covariables de los últimos 10 días del rango.")
        days = days[-10:]
    cov = pd.concat([D.game_covariates(league, d) for d in days], ignore_index=True)
    if cov.empty:
        st.info("Sin covariables registradas en el rango.")
    else:
        st.dataframe(cov, use_container_width=True, hide_index=True)

# -------------------------------------------------------- Final picks per game
st.divider()
st.header("🏁 Picks finales (máx. 1 por juego — el mejor 🤖 entre los ✅)")
if finals.empty:
    st.info("Ningún juego del rango tiene picks ✅ — así debe ser cuando el meta no ve valor.")
else:
    st.dataframe(finals, use_container_width=True, hide_index=True,
                 column_config={
                     "prob": st.column_config.ProgressColumn("prob", format="percent", min_value=0, max_value=1),
                     "🤖": st.column_config.ProgressColumn("🤖 P(acierto)", format="percent", min_value=0, max_value=1),
                     "umbral": st.column_config.NumberColumn("umbral línea", format="percent"),
                     "acierto_hist": st.column_config.NumberColumn("acierto @Th", format="percent"),
                 })
    played = finals[finals["acertó"] != "—"]
    if len(played):
        st.metric("Tracking del rango (picks finales ya jugados)",
                  f"{(played['acertó'] == '✓').mean():.0%} de acierto ({(played['acertó'] == '✓').sum()}/{len(played)})")
