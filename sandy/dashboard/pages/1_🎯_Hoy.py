"""🎯 Every (game × market) candidate across all leagues for a chosen day."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Hoy", page_icon="🎯", layout="wide")
st.title("🎯 Picks del día — todas las ligas")

c1, c2, c3 = st.columns([1, 2, 1])
day = c1.date_input("Fecha", value=date.today(),
                    help="Fechas pasadas se muestran como replay del backtest (sin fuga de datos).")
leagues = c2.multiselect("Ligas", list(D.LEAGUES), default=list(D.LEAGUES),
                         format_func=D.league_title)
only_rec = c3.toggle("Solo ✅ recomendadas", value=False)


@st.cache_data(ttl=300)
def _board(lg: str, d: date) -> pd.DataFrame:
    return D.today_board(lg, d)


frames = [f for f in (_board(lg, day) for lg in leagues) if not f.empty]
if not frames:
    st.info("Sin partidos ese día en las ligas seleccionadas. Prueba otra fecha "
            "(ej. 2026-05-23 fútbol/MLS, 2026-04-14 NHL, 2026-04-12 NBA).")
    st.stop()

df = pd.concat(frames, ignore_index=True)
if df["replay"].any():
    st.caption("🧪 Replay histórico: predicciones reales del backtest walk-forward para esa fecha.")

k1, k2, k3 = st.columns(3)
k1.metric("Partidos", df["partido"].nunique())
k2.metric("Candidatos evaluados", len(df))
k3.metric("✅ Recomendadas", int(df["recomendada"].sum()))

if only_rec:
    df = df[df["recomendada"]]

df = df.sort_values(["recomendada", "meta"], ascending=[False, False])
st.dataframe(
    df[["liga", "partido", "mercado", "pick", "prob", "hist", "meta", "umbral", "recomendada"]],
    use_container_width=True, hide_index=True, height=600,
    column_config={
        "prob": st.column_config.ProgressColumn("prob modelo", format="percent", min_value=0, max_value=1),
        "hist": st.column_config.ProgressColumn("hist %", format="percent", min_value=0, max_value=1,
                                                help="Acierto real histórico de picks con esta confianza"),
        "meta": st.column_config.ProgressColumn("🤖 meta", format="percent", min_value=0, max_value=1,
                                                help="P(acierto) del meta-modelo"),
        "umbral": st.column_config.NumberColumn("umbral liga", format="percent",
                                                help="Umbral 🤖 que maximiza el acierto (holdout)"),
        "recomendada": st.column_config.CheckboxColumn("✅ apuesta"),
    },
)
st.caption("✅ apuesta = hist ≥ 60% (n≥30) **y** 🤖 ≥ umbral de la liga — el mismo filtro del digest de Telegram.")
