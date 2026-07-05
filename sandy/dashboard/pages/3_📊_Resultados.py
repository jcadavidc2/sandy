"""📊 Historical results explorer across every league, with filters + CSV export."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Resultados", page_icon="📊", layout="wide")
st.title("📊 Resultados — todas las ligas, todos los mercados")


@st.cache_data(ttl=600)
def _scored(lg: str) -> pd.DataFrame:
    df = D.scored_results(lg)
    df["liga"] = D.league_title(lg)
    return df


leagues = st.multiselect("Ligas", list(D.LEAGUES), default=list(D.LEAGUES), format_func=D.league_title)
if not leagues:
    st.stop()
df = pd.concat([_scored(lg) for lg in leagues], ignore_index=True)
df["mercado"] = df["market"].map(D.market_label)
df["match_date"] = pd.to_datetime(df["match_date"])

c1, c2, c3, c4 = st.columns(4)
dmin, dmax = df["match_date"].min().date(), df["match_date"].max().date()
rango = c1.date_input("Rango de fechas", (dmin, dmax), min_value=dmin, max_value=dmax)
mercados = c2.multiselect("Mercados", sorted(df["mercado"].unique()), default=None,
                          placeholder="Todos")
meta_min = c3.slider("🤖 mínimo", 0.0, 1.0, 0.0, 0.05)
estado = c4.selectbox("Estado", ["Todos", "✅ Acierto", "❌ Fallo"])
solo_holdout = st.toggle("Solo holdout del meta (evaluación honesta)", value=False)

if len(rango) == 2:
    df = df[(df["match_date"].dt.date >= rango[0]) & (df["match_date"].dt.date <= rango[1])]
if mercados:
    df = df[df["mercado"].isin(mercados)]
df = df[df["meta"].fillna(0) >= meta_min]
if estado != "Todos":
    df = df[df["correct"] == estado.startswith("✅")]
if solo_holdout:
    df = df[df["holdout"]]

k1, k2, k3 = st.columns(3)
k1.metric("Picks", f"{len(df):,}")
k2.metric("Acierto", f"{df['correct'].mean():.1%}" if len(df) else "—")
k3.metric("Partidos", df.groupby(["liga", "home", "away", "match_date"]).ngroups if len(df) else 0)

show = df.sort_values("match_date", ascending=False).head(2000).copy()
show["✓"] = show["correct"].map({True: "✅", False: "❌"})
show["fecha"] = show["match_date"].dt.date
show["partido"] = show["home"] + " vs " + show["away"]
st.dataframe(
    show[["fecha", "liga", "partido", "mercado", "pick", "conf", "meta", "resultado", "✓"]],
    use_container_width=True, hide_index=True, height=520,
    column_config={
        "conf": st.column_config.ProgressColumn("prob modelo", format="percent", min_value=0, max_value=1),
        "meta": st.column_config.ProgressColumn("🤖 meta", format="percent", min_value=0, max_value=1),
    },
)
if len(df) > 2000:
    st.caption(f"Mostrando los 2,000 más recientes de {len(df):,} — usa los filtros o descarga el CSV completo.")
st.download_button("⬇️ Descargar CSV completo", df.to_csv(index=False).encode(),
                   "sandy_resultados.csv", "text/csv")

if len(df):
    st.subheader("Acierto por mes")
    mensual = df.set_index("match_date").groupby([pd.Grouper(freq="ME")])["correct"].mean()
    st.line_chart((mensual * 100).rename("acierto %"), height=240)
    st.subheader("Acierto por mercado")
    por_mercado = df.groupby("mercado")["correct"].agg(["mean", "size"])
    st.bar_chart((por_mercado["mean"] * 100).rename("acierto %"), height=260)
