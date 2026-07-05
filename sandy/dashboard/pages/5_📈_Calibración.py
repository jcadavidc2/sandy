"""📈 Calibration snapshots: per-market accuracy + reliability buckets (the 'hist %')."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Calibración", page_icon="📈", layout="wide")
st.title("📈 Calibración — el origen del «hist %»")

league = st.selectbox("Liga", list(D.LEAGUES), format_func=D.league_title)


@st.cache_data(ttl=600)
def _cal(lg: str) -> pd.DataFrame:
    return D.calibration_latest(lg)


cal = _cal(league)
if cal.empty:
    st.info("Sin snapshots de calibración.")
    st.stop()

cal = cal.copy()
cal["mercado"] = cal["market"].map(
    lambda m: "🤖 Meta (picks aprobados)" if m == "meta_pick" else D.market_label(m)
    if m in D.spec_markets(league) or m.startswith(("over_", "corners_", "double", "winner")) else m)
st.dataframe(
    cal[["mercado", "snapshot_date", "sample_size", "accuracy", "recommended_threshold"]]
    .sort_values("accuracy", ascending=False),
    use_container_width=True, hide_index=True,
    column_config={
        "accuracy": st.column_config.ProgressColumn("acierto global", format="percent",
                                                    min_value=0, max_value=1),
        "sample_size": st.column_config.NumberColumn("n evaluadas"),
        "recommended_threshold": st.column_config.NumberColumn("umbral confianza", format="%.2f"),
        "snapshot_date": st.column_config.DateColumn("snapshot"),
    },
)

st.subheader("Fiabilidad por cubeta de confianza")
st.caption("Para el mercado elegido: cuando el modelo dice X% de confianza, ¿cuánto acierta de verdad? "
           "De aquí sale el «hist %» de cada pick.")
mkt = st.selectbox("Mercado", list(cal["market"]),
                   format_func=lambda m: "🤖 Meta" if m == "meta_pick" else D.market_label(m))
rel = cal.loc[cal["market"] == mkt, "reliability"].iloc[0]
import json
rel = rel if isinstance(rel, list) else json.loads(rel or "[]")
if not rel:
    st.info("Sin cubetas para este mercado.")
else:
    rdf = pd.DataFrame(rel)
    if {"lo", "hi"}.issubset(rdf.columns):  # confidence buckets
        rdf["cubeta"] = rdf.apply(lambda r: f"{r['lo']:.0%}–{r['hi']:.0%}", axis=1)
        show_cols = [c for c in ["cubeta", "n", "acc"] if c in rdf.columns]
    else:  # meta_pick ladder {thr, n, acc}
        rdf["cubeta"] = rdf["thr"].map(lambda t: f"🤖 ≥{t:.0%}")
        show_cols = ["cubeta", "n", "acc"]
    st.dataframe(rdf[show_cols], use_container_width=True, hide_index=True,
                 column_config={"acc": st.column_config.ProgressColumn("acierto real", format="percent",
                                                                       min_value=0, max_value=1)})
    st.bar_chart(rdf.set_index("cubeta")["acc"].dropna() * 100, height=240)
