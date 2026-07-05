"""🤖 Meta-model explorer: play with thresholds, per-exact-line ladders."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Meta-modelo", page_icon="🤖", layout="wide")
st.title("🤖 Meta-modelo — explorador de umbrales")

league = st.selectbox("Liga", list(D.LEAGUES), format_func=D.league_title)
ms = D.meta_summary(league)
if not ms:
    st.warning("Esta liga aún no tiene meta-modelo entrenado.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Umbral recomendado", f"{ms['threshold']:.0%}" if ms["threshold"] else "—",
          help="El que maximiza el acierto en holdout (n≥50)")
m2.metric("AUC (holdout)", f"{ms['auc']:.3f}")
m3.metric("Filas holdout", f"{ms['holdout_rows']:,}")
m4.metric("Último entrenamiento", str(ms["trained_at"]))


@st.cache_data(ttl=600)
def _scored(lg: str) -> pd.DataFrame:
    return D.scored_results(lg)


scored = _scored(league)
if scored.empty or scored["meta"].isna().all():
    st.info("Sin datos puntuados.")
    st.stop()

st.divider()
c1, c2, c3 = st.columns([1, 2, 2])
scope = c1.radio("Datos", ["Solo holdout (honesto)", "Todo el backtest"], index=0,
                 help="El holdout es el 30% cronológico final que el meta-modelo NUNCA vio al entrenar — "
                      "es la evaluación honesta. El resto lo vio, así que su acierto está inflado.")
markets = c2.multiselect("Mercados (líneas exactas)", list(D.spec_markets(league)),
                         default=list(D.spec_markets(league)), format_func=D.market_label)
thr = c3.slider("Umbral 🤖 (juega con él)", 0.50, 0.95, float(ms["threshold"] or 0.8), 0.01)

f = scored[scored["market"].isin(markets)]
if scope.startswith("Solo"):
    f = f[f["holdout"]]

approved = f[f["meta"] >= thr]
base = f["correct"].mean() if len(f) else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Picks aprobados", f"{len(approved):,} de {len(f):,}")
k2.metric(f"Acierto con 🤖 ≥ {thr:.0%}", f"{approved['correct'].mean():.1%}" if len(approved) else "—")
k3.metric("Acierto sin filtro (base)", f"{base:.1%}")
if len(approved):
    k4.metric("Mejora vs base", f"{(approved['correct'].mean() - base) * 100:+.1f} pts")

st.subheader("Escalera por línea exacta")
st.caption("Cada celda: acierto % (n picks) de las apuestas con 🤖 ≥ umbral de la fila — "
           "exactamente lo que pediste: goles 1.5 / 2.5 / 3.5… cada una con sus propios números.")
grid = []
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    row = {"umbral 🤖": f"≥{t:.0%}" + (" ← recomendado" if ms["threshold"] and abs(t - ms["threshold"]) < 1e-9 else "")}
    for mk in markets:
        sub = f[(f["market"] == mk) & (f["meta"] >= t)]
        row[D.market_label(mk)] = f"{sub['correct'].mean():.0%} ({len(sub)})" if len(sub) else "—"
    grid.append(row)
st.dataframe(pd.DataFrame(grid), use_container_width=True, hide_index=True)

st.subheader(f"Por mercado con 🤖 ≥ {thr:.0%}")
per_mkt = (approved.groupby("market")
           .agg(picks=("correct", "size"), aciertos=("correct", "sum"), acierto=("correct", "mean"))
           .reset_index())
per_mkt["mercado"] = per_mkt["market"].map(D.market_label)
st.dataframe(per_mkt[["mercado", "picks", "aciertos", "acierto"]],
             use_container_width=True, hide_index=True,
             column_config={"acierto": st.column_config.ProgressColumn("acierto %", format="percent",
                                                                       min_value=0, max_value=1)})
chart = per_mkt.set_index("mercado")["acierto"]
st.bar_chart(chart, height=260)
