"""Sandy — landing page: today's recommended bets across every league at a glance.

Run:  scripts/dashboard.sh   (streamlit, port 8502)
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Predicciones", page_icon="🤖", layout="wide")

st.title("🤖 Sandy — Centro de predicciones")
st.caption("7 ligas · modelos base + meta-modelo P(acierto) por liga · usa el menú lateral: "
           "🎯 picks del día, 🤖 explorador del meta-modelo, 📊 resultados históricos, "
           "🧠 covariables, 📈 calibración, 🏆 Mundial 2026.")


@st.cache_data(ttl=300)
def _board(lg: str, d: date) -> pd.DataFrame:
    return D.today_board(lg, d)


today = date.today()
cols = st.columns(len(D.LEAGUES))
total_rec = 0
for col, lg in zip(cols, D.LEAGUES):
    df = _board(lg, today)
    n_games = df["partido"].nunique() if not df.empty else 0
    n_rec = int(df["recomendada"].sum()) if not df.empty else 0
    total_rec += n_rec
    col.metric(D.league_title(lg), f"{n_games} juegos", f"{n_rec} ✅ apuestas" if n_rec else "sin apuestas",
               delta_color="normal" if n_rec else "off")

if total_rec == 0:
    st.info("😴 Hoy no hay apuestas recomendadas (ligas en receso o sin picks que superen los filtros). "
            "Ve a **🎯 Hoy** y elige una fecha histórica para explorar cómo se ve un día con partidos.")
else:
    st.success(f"🎯 {total_rec} apuestas recomendadas hoy — detalle en la página **🎯 Hoy**.")

st.divider()
st.markdown(
    "**Cómo leer los números** — cada pick lleva tres señales:\n"
    "1. **prob**: probabilidad del modelo base (Dixon-Coles / normal NBA).\n"
    "2. **hist**: % de acierto real histórico de picks con esa confianza (calibración).\n"
    "3. **🤖 meta**: P(acierto) del meta-modelo (LightGBM + isotónica sobre covariables). "
    "Un pick es **✅ recomendada** cuando hist ≥ 60% (n≥30) **y** 🤖 ≥ umbral de la liga "
    "(el que maximiza el acierto en holdout)."
)
