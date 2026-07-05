"""💰 Valor — our picks vs the betting market (TheOddsAPI, analytical only).

For every ✅ pick with matched odds: cuota (best decimal price across books),
mercado % (consensus no-vig implied prob, median across books),
edge = prob − mercado %, EV = prob·(cuota−1) − (1−prob).

`prob` is the pick's OWN base-model calibrated side probability — NOT 🤖
(which is P(pick correct), a second-stage reliability score).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Valor", page_icon="💰", layout="wide")
st.title("💰 Valor vs el mercado")
st.caption("Comparamos NUESTRA probabilidad (prob del modelo base, calibrada) contra la "
           "probabilidad implícita del mercado sin margen (mediana entre casas). "
           "edge = prob − mercado %; EV = prob·(cuota−1) − (1−prob). "
           "Mercados con feed: totales (goles/puntos/carreras, la línea debe coincidir "
           "exacto), ganador NBA y doble oportunidad en fútbol (cuota 1X derivada por casa "
           "del moneyline a 3 vías: 1/(1/local + 1/empate)). Corners y BTTS no tienen cuotas.")

c1, c2, c3 = st.columns([1.4, 2, 1.2])
day = c1.date_input("Fecha", date.today(), help="HOY por defecto — cuotas del día.")
leagues = c2.multiselect("Ligas", list(D.LEAGUES), default=list(D.LEAGUES),
                         format_func=D.league_title)
solo_valor = c3.toggle("Solo valor (edge ≥ 3pp)", value=False,
                       help="Enciende para quedarte solo con picks donde nuestro modelo ve "
                            "más probabilidad que el mercado (≥3 puntos).")


@st.cache_data(ttl=300)
def _todos(lg: str, d: date) -> pd.DataFrame:
    _games, _finals, todos = D.board_range(lg, d, d)
    if not todos.empty:
        todos.insert(0, "liga", D.league_title(lg))
    return todos


frames = [f for f in (_todos(lg, day) for lg in leagues) if not f.empty]
df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

if df.empty or "cuota" not in df.columns or df["cuota"].notna().sum() == 0:
    st.info("Sin picks con cuotas para esa fecha — o no hay juegos/picks ✅, o el feed de "
            "cuotas aún no corre hoy (corre 13:30 y 14:15 UTC), o ninguna línea nuestra "
            "coincide con la del mercado. Un día sin valor NO es un día perdido: es el "
            "sistema ahorrándote plata.")
    st.stop()

con_odds = df[df["cuota"].notna()].copy()
for c in ("cuota", "mercado %", "edge", "EV"):
    con_odds[c] = pd.to_numeric(con_odds[c], errors="coerce")
valor = con_odds[con_odds["edge"].fillna(-1) >= 0.03]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Picks ✅ del día", len(df))
k2.metric("Con cuotas", len(con_odds))
k3.metric("Con valor (edge ≥ 3pp)", len(valor))
best_ev = con_odds["EV"].max()
k4.metric("Mejor EV", f"{best_ev:+.2f} u" if pd.notna(best_ev) else "—")

show = (valor if solo_valor else con_odds).sort_values("EV", ascending=False)
st.dataframe(
    show[["liga", "partido", "mercado", "pick", "nivel", "prob", "🤖",
          "cuota", "mercado %", "edge", "EV"]],
    use_container_width=True, hide_index=True,
    column_config={
        "prob": st.column_config.NumberColumn("prob (nuestra)", format="percent"),
        "🤖": st.column_config.NumberColumn("🤖 P(acierto)", format="percent"),
        "cuota": st.column_config.NumberColumn("cuota", format="%.2f"),
        "mercado %": st.column_config.NumberColumn("mercado %", format="percent"),
        "edge": st.column_config.NumberColumn("edge", format="percent"),
        "EV": st.column_config.NumberColumn("EV (u por 1u)", format="%.2f"),
    },
)
if not solo_valor and valor.empty:
    st.caption("⚖️ Hoy el mercado y nosotros vemos lo mismo: ningún pick con edge ≥ 3pp. "
               "Los días sin valor son el sistema ahorrándote plata — no apostar también "
               "es una decisión rentable.")

# ---------------------------------------------------------------- ROI curve --
st.divider()
st.subheader("📒 Registro de valor — unidades acumuladas")
st.caption("Cada pick con edge ≥ 3pp se registra con stake plano de 1 unidad "
           "(gana: cuota−1; pierde: −1). Esto es análisis, no una invitación a apostar.")


@st.cache_data(ttl=300)
def _roi() -> pd.DataFrame:
    from sandy.odds import roi_frame
    return roi_frame()


try:
    log = _roi()
except Exception:
    log = pd.DataFrame()

if log.empty:
    st.info("Aún no hay picks de valor registrados — la curva aparece cuando el sistema "
            "encuentre (y liquide) los primeros.")
else:
    settled = log[log["units"].notna()].copy()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Picks de valor registrados", len(log))
    r2.metric("Liquidados", len(settled))
    if len(settled):
        units = settled["units"].sum()
        staked = settled["stake"].sum()
        r3.metric("Unidades netas", f"{units:+.2f} u")
        r4.metric("ROI", f"{units / staked * 100:+.1f}%",
                  help="unidades netas / unidades apostadas (stake plano 1u)")
        curve = settled.groupby("date")["units"].sum().cumsum().rename("unidades acumuladas")
        st.line_chart(curve, height=260)
        hit = (settled["result"] == "win").mean()
        st.caption(f"Acierto de los picks de valor liquidados: {hit:.0%} "
                   f"({(settled['result'] == 'win').sum()}/{len(settled)})")
    else:
        r3.metric("Unidades netas", "—")
        r4.metric("ROI", "—")
        st.caption("Los registrados de hoy se liquidan cuando sus juegos se reconcilien "
                   "(nightly de cada liga).")
    with st.expander("Ver registro completo"):
        st.dataframe(log.sort_values("date", ascending=False), use_container_width=True,
                     hide_index=True,
                     column_config={
                         "prob": st.column_config.NumberColumn(format="percent"),
                         "edge": st.column_config.NumberColumn(format="percent"),
                         "cuota": st.column_config.NumberColumn(format="%.2f"),
                     })
