"""🏁 Picks del día — the best final pick per game ACROSS ALL LEAGUES for any
date or range. Default = today: one consolidated bet sheet."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Picks del día", page_icon="🏁", layout="wide")
st.title("🏁 Picks finales — todas las ligas")
st.caption("Máx. 1 pick por juego (el mejor 🤖 entre los ✅ de ese juego), consolidado "
           "entre todas las ligas y ordenado por 🤖. El mismo filtro del Tablero y de Telegram.")

c1, c2, c3, c4 = st.columns([1.4, 2, 1, 1])
rng = c1.date_input("Fecha / rango", (date.today(), date.today()),
                    help="HOY por defecto. Elige fechas pasadas para ver cómo salieron los picks.")
leagues = c2.multiselect("Ligas", list(D.LEAGUES), default=list(D.LEAGUES),
                         format_func=D.league_title)
min_meta = c3.slider("🤖 mínimo", 0.5, 0.99, 0.5, 0.01,
                     help="Sube esto para quedarte solo con la crema del día.")
solo_mejor = c4.toggle("Solo el mejor por juego", value=True,
                       help="Apagado = TODOS los picks ✅ (un juego puede tener varios).")
if len(rng) != 2:
    st.stop()


@st.cache_data(ttl=300)
def _finals(lg: str, a: date, b: date, best_only: bool) -> pd.DataFrame:
    _games, finals, todos = D.board_range(lg, a, b)
    sel = finals if best_only else todos
    if not sel.empty:
        sel.insert(0, "liga", D.league_title(lg))
    return sel


frames = [f for f in (_finals(lg, rng[0], rng[1], solo_mejor) for lg in leagues) if not f.empty]
if not frames:
    st.info("Ningún pick ✅ en ese rango — o no hay juegos, o el meta-modelo no ve valor "
            "(mejor no apostar). Prueba ampliar el rango o revisa una fecha con temporada activa.")
    st.stop()

df = pd.concat(frames, ignore_index=True)
df = df[df["🤖"] >= min_meta].sort_values("🤖", ascending=False).reset_index(drop=True)


pend = df[df["acertó"] == "—"]
played = df[df["acertó"] != "—"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Picks finales", len(df))
k2.metric("Por jugar", len(pend))
k3.metric("Ya jugados", len(played))
if len(played):
    k4.metric("Acierto (jugados)",
              f"{(played['acertó'] == '✓').mean():.0%} ({(played['acertó'] == '✓').sum()}/{len(played)})")

from sandy.dashboard import grid
grid.show(df[["liga", "fecha", "partido", "mercado", "pick", "prob", "🤖", "umbral",
              "acierto_hist", "resultado", "acertó"]], key="picks", height=520)
st.download_button("⬇️ Descargar picks (CSV)", df.to_csv(index=False).encode(),
                   "sandy_picks_finales.csv", "text/csv")