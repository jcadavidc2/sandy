"""🎯 Portafolio B "Picks del Día" — the A/B experiment page (paper money).

Second $100,000 paper bank that bets EVERY day on the day's ✅ accuracy picks
(one per game, matched cuota required) using OUR RAW calibrated probabilities
— no market shrink, plus an always-bet floor when no candidate has positive
EV. All math in sandy/portfolio_picks.py (shared optimizer with 🎰 portfolio
A, separate tables/bank). Layout mirrors page 4: official sheet → what-if
simulator → ledger (with A's curve overlaid for the head-to-head).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from sandy import portfolio as PA          # portfolio A (🎰 valor) — for the comparison
from sandy import portfolio_picks as P     # portfolio B (🎯 picks) — this page
from sandy.odds import DISPLAY_TZ

st.set_page_config(page_title="Sandy · Portafolio Picks", page_icon="🎯", layout="wide")
st.title("🎯 Portafolio B — Picks del Día (dinero de papel)")
st.caption("El experimento hermano del 🎰 Portafolio: una SEGUNDA banca de papel de $100.000 "
           "que apuesta TODOS los días EXACTAMENTE a los 🏁 Picks del Día que tengan cuota "
           "(el mejor 🤖 por partido), usando nuestra probabilidad cruda — sin recorte prudente y "
           "aunque el mercado no nos dé ventaja. Mismo optimizador (Kelly fraccional Monte "
           "Carlo, pasos de $500, tope 30% por partido), otra tesis. Las dos curvas, lado a "
           "lado, son la prueba A/B.")

with st.expander("📖 Cómo leer esta página (y en qué se diferencia del 🎰)"):
    st.markdown(f"""
- **Nadie apuesta plata real.** Es una banca de papel independiente de $100.000.
- **La tesis honesta de este portafolio:** apostamos los picks ✅ de nuestros modelos **aunque
  no haya ventaja sobre el mercado**. El 🎰 Portafolio A solo apuesta cuando el mercado paga de
  más (y con probabilidad recortada 70/30 hacia el mercado); este Portafolio B le cree **100% a
  nuestros modelos** y usa la probabilidad cruda para decidir cuánto poner.
- **Regla siempre-apostar:** si ningún candidato tiene valor esperado positivo ni siquiera bajo
  nuestras propias probabilidades, igual se despliega el {P.MIN_DEPLOY_FRACTION:.0%} del
  presupuesto del día (mínimo un tiquete de $500) en la mejor combinación disponible — "la
  menos mala". Solo un día SIN picks ✅ con cuota queda en $0.
- **⚠️sust. en un tiquete** = el pick exacto de Picks del Día para ese juego no tiene cuota
  en el mercado, así que B apuesta el MEJOR pick con cuota de ese mismo juego (el sustituto).
- **Esto es un experimento, no una recomendación.** Apostar sin ventaja de mercado pierde plata
  a largo plazo *si el mercado tiene razón*; gana *si nuestros modelos ven algo que el mercado
  no*. **Espera rachas perdedoras** — posiblemente largas. Para eso existe.
- **Lo único que importa:** comparar la curva de esta banca contra la del 🎰 Portafolio A
  (abajo van superpuestas). Si B supera a A sostenidamente, nuestras probabilidades crudas
  valen más que la prudencia; si A gana, el recorte prudente estaba protegiendo la plata.
- **Un pick por partido:** por cada juego entra solo el pick de mayor valor esperado con
  nuestra probabilidad (prob × cuota − 1). Picks sin cuota casada (corners, BTTS, 1X de NHL)
  no pueden apostarse.
""")

day = datetime.now(DISPLAY_TZ).date()


@st.cache_data(ttl=300)
def _bank() -> float:
    return P.available_bank()


@st.cache_data(ttl=300)
def _bank_before(d) -> float:
    # bank BEFORE the day's own stakes — default controls reproduce the
    # persisted portfolio exactly (deterministic seed)
    return P.available_bank(before=d)


@st.cache_data(ttl=300)
def _whatif(d, budget: float, risk: str) -> dict:
    return P.build_portfolio(day=d, budget=budget, risk=risk, persist=False)


@st.cache_data(ttl=300)
def _tickets_hist() -> pd.DataFrame:
    return P.tickets_frame()


@st.cache_data(ttl=300)
def _bankroll() -> pd.DataFrame:
    return P.bankroll_frame()


@st.cache_data(ttl=300)
def _bankroll_a() -> pd.DataFrame:
    return PA.bankroll_frame()


bank = _bank()
bank_basis = _bank_before(day)
hist = _tickets_hist()
persisted_today = hist[hist["date"] == day] if not hist.empty else pd.DataFrame()

# ----------------------------------------------------------- OFFICIAL first
st.subheader(f"📌 Portafolio B OFICIAL de hoy · {day.strftime('%d/%m/%Y')}")
o1, o2, o3, o4 = st.columns(4)
o1.metric("Banca disponible (B)", f"${bank:,.0f}",
          help="Plata de papel libre de ESTA banca (independiente del 🎰). Lo apostado en "
               "tiquetes abiertos está descontado.")
o2.metric("Presupuesto del día", f"${PA.default_budget(bank_basis):,.0f}",
          help=f"Misma regla del 🎰: {PA.BUDGET_FRACTION:.0%} de la banca con que amaneció el "
               f"día (${bank_basis:,.0f}), en pasos de $500.")
if not persisted_today.empty:
    o3.metric("Apostado hoy (oficial)", f"${persisted_today['stake'].sum():,.0f}")
    o4.metric("Tiquetes", f"{len(persisted_today)}")
    ICON_O = {"won": "✓ ganada", "lost": "✗ perdida", "open": "⏳ abierta", "void": "↩ anulada"}
    _tbl = pd.DataFrame({
        "Apuesta": persisted_today["ticket_id"].map(lambda i: f"Apuesta {i}"),
        "Tipo": persisted_today["tipo"],
        "Tiquete": persisted_today["tiquete"],
        "Cuota": persisted_today["ticket_cuota"],
        "Apostado": persisted_today["stake"],
        "Ganaría": persisted_today["stake"] * persisted_today["ticket_cuota"],
        "Estado": persisted_today["status"].map(ICON_O),
    })
    _tbl = pd.concat([_tbl, pd.DataFrame([{
        "Apuesta": "TOTAL", "Tipo": "", "Tiquete": f"{len(_tbl)} tiquetes",
        "Cuota": None, "Apostado": _tbl["Apostado"].sum(),
        "Ganaría": _tbl["Ganaría"].sum(), "Estado": "",
    }])], ignore_index=True)
    st.dataframe(_tbl, use_container_width=True, hide_index=True,
        column_config={
            "Tipo": st.column_config.TextColumn(help="Individual = un pick. Combinada xN = N "
                                                "partidos distintos; deben acertar TODOS."),
            "Cuota": st.column_config.NumberColumn(format="%.2f"),
            "Apostado": st.column_config.NumberColumn(format="$%.0f"),
            "Ganaría": st.column_config.NumberColumn(format="$%.0f",
                help="Lo que devuelve si acierta (apostado × cuota)."),
        })
    st.caption("Registro REAL del día del experimento B (congelado antes de los partidos, "
               "liquidado a la mañana siguiente).")
else:
    _bk0 = _bankroll()
    day_decided = (not _bk0.empty) and (day in set(_bk0["date"]))
    if day_decided:
        o3.metric("Apostado hoy (oficial)", "$0")
        o4.metric("Tiquetes", "0")
        st.success("✅ Decisión OFICIAL de hoy ya guardada: **$0 apostado** — hoy no hubo "
                   "ningún pick ✅ con cuota casada (la regla siempre-apostar necesita al "
                   "menos un candidato apostable).")
    else:
        o3.metric("Apostado hoy (oficial)", "—")
        o4.metric("Tiquetes", "—")
        st.info("💾 El portafolio B de hoy aún no se guarda — se arma solo a las 8:15 AM "
                "Bogotá, justo después del 🎰. Si no hay picks ✅ con cuota quedará "
                "'$0 apostado' (también es dato del experimento).")

st.divider()
st.subheader("🧪 Simulador «¿y si…?» — presupuesto y riesgo (NO cambia lo oficial)")
st.warning("⚠️ Esto es una CALCULADORA EN VIVO, no el registro: se recalcula al abrir la página, los partidos que ya empezaron desaparecen y las cuotas cambian. El registro REAL y congelado del día es ÚNICAMENTE la tabla 📌 de arriba — eso es lo que se liquida mañana.")
c1, c2, c3 = st.columns([2, 1.6, 1.2])
max_b = int(max(PA.floor500(bank_basis), PA.STEP))
budget = c1.slider("Presupuesto del día ($)", 0, max_b,
                   int(min(PA.default_budget(bank_basis), max_b)), step=int(PA.STEP),
                   help="Cuánto se permitiría apostar HOY como máximo en esta simulación. "
                        "El oficial usa el 30% de la banca B. Moverlo NO guarda nada.")
risk = c2.radio("Riesgo", list(PA.RISKS), index=2, horizontal=True,
                help="Fracción de Kelly: Conservador = ⅛, Balanceado = ¼, "
                     "Agresivo = ½ (el oficial).")
c3.metric("Presupuesto oficial", f"${PA.default_budget(bank_basis):,.0f}",
          help="El techo real del día: 30% de la banca B.")

res = _whatif(day, float(budget), risk)

if not res.get("tickets"):
    st.info("🙅 $0 apostado en esta simulación — no hay picks ✅ con cuota hoy (o el "
            "presupuesto no alcanza para la apuesta mínima de $500). La regla "
            "siempre-apostar solo aplica cuando existe al menos un candidato apostable.")
else:
    if res.get("forzado"):
        st.warning(f"⚠️ **Despliegue forzado (regla siempre-apostar):** hoy NINGÚN candidato "
                   f"tiene valor esperado positivo ni bajo nuestras propias probabilidades. "
                   f"El optimizador Kelly habría apostado $0; el experimento exige poner el "
                   f"{P.MIN_DEPLOY_FRACTION:.0%} del presupuesto en la mejor combinación "
                   f"disponible. Día con expectativa negativa asumida a conciencia.")
    s = res["summary"] or {}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Apostado hoy", f"${res['staked']:,.0f}",
              help="Suma de los tiquetes ($500 mínimo por paso, tope 30% por partido).")
    k2.metric("Ganancia esperada (modelo)", f"${s.get('expected_profit', 0):+,.0f}",
              help="Promedio de miles de días simulados usando nuestra probabilidad CRUDA "
                   "(sin recorte). Si el mercado tiene razón, el valor real es peor — ese "
                   "es exactamente el experimento.")
    k3.metric("P(día verde)", f"{s.get('p_green', 0):.0%}",
              help="Probabilidad de terminar el día ganando plata, según nuestros modelos.")
    k4.metric("Peor 5% del día", f"${s.get('p5', 0):+,.0f}",
              help="En el 5% de los días más malos se perdería esto o más — nunca más que "
                   "lo apostado.")

    rows = []
    for t in res["tickets"]:
        rows.append({
            "Apuesta": f"Apuesta {t['ticket_id']}",
            "Tipo": "Individual" if t["n_legs"] == 1 else f"Combinada x{t['n_legs']}",
            "Tiquete": " + ".join(f"{l['liga_titulo']} {l['partido']}: {l['pick']} @{l['cuota']}"
                                  for l in t["legs"]),
            "Cuota": t["cuota"], "Apostado": t["stake"],
            "Prob (modelo)": t["prob"],
            "Ganaría": t["stake"] * t["cuota"], "EV (modelo)": t["ev"],
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "Apuesta": st.column_config.TextColumn("Apuesta", help="Cada fila es un tiquete."),
            "Tipo": st.column_config.TextColumn("Tipo", help="Individual = un solo pick. "
                                                "Combinada = varios partidos: paga mucho más "
                                                "pero deben acertar TODOS."),
            "Tiquete": st.column_config.TextColumn("Tiquete", width="large",
                                                   help="Los picks del tiquete con su cuota."),
            "Cuota": st.column_config.NumberColumn("Cuota", format="%.2f",
                                                   help="Multiplicador del pago."),
            "Apostado": st.column_config.NumberColumn("Apostado", format="$%.0f",
                                                      help="Stake del tiquete (múltiplos de "
                                                           "$500)."),
            "Prob (modelo)": st.column_config.NumberColumn(
                "Prob (modelo)", format="percent",
                help="Probabilidad de que el tiquete gane según nuestra probabilidad CRUDA "
                     "— la única que usa este portafolio (sin mezcla con el mercado)."),
            "Ganaría": st.column_config.NumberColumn("Ganaría", format="$%.0f",
                                                     help="Stake × cuota si acierta."),
            "EV (modelo)": st.column_config.NumberColumn(
                "EV (modelo)", format="$%.0f",
                help="Ganancia promedio esperada del tiquete si nuestros modelos tienen "
                     "razón. Puede ser NEGATIVA en días de despliegue forzado."),
        })

# ------------------------------------------------------------------ history --
st.divider()
st.subheader("📒 Historial — B contra A, la curva que decide el experimento")

bk = _bankroll()
if bk.empty:
    st.info("Aún no hay días registrados en la banca B. El primer portafolio se guarda en "
            "el próximo run diario (8:15 AM Bogotá).")
else:
    bk = bk.copy()
    bk["banca"] = bk.apply(
        lambda r: r["end_bank"] if pd.notna(r["end_bank"]) else r["start_bank"] - r["staked"],
        axis=1)
    settled = bk[bk["end_bank"].notna()]
    pnl_total = float((settled["returned"] - settled["staked"]).sum()) if len(settled) else 0.0
    staked_total = float(settled["staked"].sum()) if len(settled) else 0.0
    verdes = int(((settled["returned"] - settled["staked"]) > 0).sum()) if len(settled) else 0
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Banca B actual", f"${bank:,.0f}",
              help="La banca de papel del experimento (arrancó en $100.000).")
    h2.metric("P&L acumulado", f"${pnl_total:+,.0f}",
              help="Ganancia/pérdida total de los días ya liquidados de B.")
    h3.metric("Días verdes", f"{verdes}/{len(settled)}",
              help="Días liquidados que terminaron en ganancia. En B espera menos que en A: "
                   "aquí se apuesta también sin ventaja.")
    h4.metric("ROI sobre lo apostado", f"{(pnl_total / staked_total * 100):+.1f}%"
              if staked_total else "—",
              help="P&L ÷ total apostado en días liquidados de B.")

    curva = bk.set_index("date")["banca"].rename("🎯 B (picks)")
    bka = _bankroll_a()
    if not bka.empty:
        bka = bka.copy()
        bka["banca"] = bka.apply(
            lambda r: r["end_bank"] if pd.notna(r["end_bank"])
            else r["start_bank"] - r["staked"], axis=1)
        curva = pd.concat([curva, bka.set_index("date")["banca"].rename("🎰 A (valor)")],
                          axis=1)
    st.line_chart(curva, height=300)
    st.caption("Las dos bancas de papel superpuestas — arrancan ambas en $100.000 (en fechas "
               "distintas). Si 🎯 B se despega hacia arriba, nuestras probabilidades crudas "
               "le ganan a la prudencia del 🎰 A; si se hunde, el recorte prudente y el "
               "filtro de valor estaban haciendo su trabajo.")

    st.markdown("**P&L por día (banca B)**")
    tabla = pd.DataFrame({
        "Fecha": bk["date"], "Banca inicio": bk["start_bank"], "Apostado": bk["staked"],
        "Devuelto": bk["returned"], "P&L": bk["returned"] - bk["staked"],
        "Banca fin": bk["end_bank"],
        "Estado": bk["end_bank"].map(lambda x: "✔ cerrado" if pd.notna(x) else "⏳ abierto"),
    })
    st.dataframe(tabla.sort_values("Fecha", ascending=False), use_container_width=True,
                 hide_index=True,
                 column_config={
                     "Banca inicio": st.column_config.NumberColumn(format="$%.0f"),
                     "Apostado": st.column_config.NumberColumn(format="$%.0f",
                         help="Total apostado ese día ($0 solo si no hubo picks con cuota)."),
                     "Devuelto": st.column_config.NumberColumn(format="$%.0f"),
                     "P&L": st.column_config.NumberColumn(format="$%.0f"),
                     "Banca fin": st.column_config.NumberColumn(format="$%.0f",
                         help="Vacío = aún hay tiquetes abiertos."),
                     "Estado": st.column_config.TextColumn(
                         help="⏳ abierto = esperando resultados; ✔ cerrado = día liquidado."),
                 })

    if not hist.empty:
        st.markdown("**Tiquetes (B)**")
        ICON = {"won": "✓ ganada", "lost": "✗ perdida", "open": "⏳ abierta", "void": "↩ anulada"}
        th = pd.DataFrame({
            "Fecha": hist["date"],
            "Apuesta": hist["ticket_id"].map(lambda i: f"Apuesta {i}"),
            "Tipo": hist["tipo"], "Tiquete": hist["tiquete"],
            "Cuota": hist["ticket_cuota"], "Apostado": hist["stake"],
            "Resultado": hist["status"].map(ICON),
            "Devuelto": hist["returned"],
        })
        st.dataframe(th, use_container_width=True, hide_index=True,
                     column_config={
                         "Cuota": st.column_config.NumberColumn(format="%.2f"),
                         "Apostado": st.column_config.NumberColumn(format="$%.0f"),
                         "Resultado": st.column_config.TextColumn(
                             help="✓ ganó · ✗ perdió · ⏳ esperando los partidos · "
                                  "↩ anulada (partido aplazado → se devuelve lo apostado)."),
                         "Devuelto": st.column_config.NumberColumn(format="$%.0f"),
                     })

st.caption("⚖️ Dinero 100% de papel — un experimento A/B de análisis, no una invitación a "
           "apostar. Este portafolio apuesta A PROPÓSITO sin exigir ventaja de mercado; su "
           "hermano prudente vive en 🎰 Portafolio. Liquidación diaria automática con los "
           "resultados reales; partido aplazado = tiquete anulado y stake devuelto.")
