"""🧠 Covariates: what each model sees per game + what the meta-model weighs."""
from __future__ import annotations

from datetime import date

import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Covariables", page_icon="🧠", layout="wide")
st.title("🧠 Covariables")

league = st.selectbox("Liga", list(D.LEAGUES), format_func=D.league_title)

st.subheader("Qué pesa el meta-modelo")
st.caption("Importancia (ganancia LightGBM, normalizada) de cada covariable al predecir P(acierto).")
imp = D.meta_importances(league)
if imp.empty:
    st.info("Sin meta-modelo entrenado para esta liga.")
else:
    c1, c2 = st.columns([1, 1])
    c1.dataframe(imp, use_container_width=True, hide_index=True,
                 column_config={"importancia %": st.column_config.ProgressColumn(
                     "importancia %", format="%.1f%%", min_value=0,
                     max_value=float(imp["importancia %"].max()))})
    c2.bar_chart(imp.set_index("covariable")["importancia %"], height=380, horizontal=True)

st.divider()
st.subheader("Covariables por partido")
day = st.date_input("Fecha", value=date.today(),
                    help="Fechas pasadas usan las covariables reales del backtest de ese día.")
cov = D.game_covariates(league, day)
if cov.empty:
    st.info("Sin partidos ese día. Prueba 2026-05-23 (fútbol/MLS), 2026-04-14 (NHL) o 2026-04-12 (NBA).")
else:
    st.caption("Exactamente lo que el modelo vio para cada partido: forma reciente (últ. 5/10), "
               "descanso, back-to-back, y las λ / totales esperados del modelo base.")
    st.dataframe(cov, use_container_width=True, hide_index=True, height=480)
