"""🎰 Portafolio diario — paper-money BetPlay simulation (the user does NOT bet).

Shows TODAY's recommended ticket sheet (persisted daily at 14:15 UTC with the
default budget/risk), what-if controls that recompute the optimizer on the fly,
the Monte-Carlo day summary under BOTH the prudent (edge-shrunk) and raw-model
assumptions, a 30/90-day bankroll projection, and the full historical ledger.
All the math lives in sandy/portfolio.py (see its module doc for assumptions:
edge shrinkage 0.7/0.3, cross-game independence for parlays, fractional Kelly,
30% per-game cap, void = refund).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from sandy import portfolio as P
from sandy.odds import DISPLAY_TZ

st.set_page_config(page_title="Sandy · Portafolio", page_icon="🎰", layout="wide")
st.title("🎰 Portafolio diario (dinero de papel)")
st.caption("Simulación de una cuenta BetPlay con plata de papel — NADIE apuesta dinero real. "
           "Cada día el optimizador toma los picks con valor (ver 💰 Valor), arma tiquetes "
           "(individuales y combinadas de partidos DISTINTOS) y reparte el presupuesto en "
           "pasos de $500 (mínimo BetPlay) maximizando el crecimiento de la banca "
           "(Kelly fraccional por Monte Carlo). Tope de exposición por partido: 30% del "
           "presupuesto del día.")

with st.expander("📖 Cómo leer esta página"):
    st.markdown("""
- **Nadie está apostando plata real.** Es una banca de papel de $100.000 para medir, con reglas
  reales de casa de apuestas, si nuestros picks generarían dinero de verdad.
- **¿Qué es el *edge*?** Es cuánta más probabilidad le damos nosotros a un pick que el mercado.
  Si nosotros decimos 69% y las casas dicen 61%, hay 8 puntos de ventaja.
- **Somos prudentes a propósito:** para decidir cuánto apostar recortamos nuestra ventaja un 30%
  (mezclamos 70% nuestra probabilidad + 30% la del mercado), por si nuestros modelos pecan de
  optimistas justo donde más discrepan del mercado.
- **Por eso NO verás los picks "seguros":** un favorito obvio paga tan poquito que no deja
  ganancia. El valor está donde el mercado subestima — ahí es donde se gana plata a largo plazo.
- **Se pierde ~3 de cada 10 veces y está BIEN.** Apostamos con ventaja, no con certeza.
  Los días rojos son parte del plan; lo grave sería apostar sin ventaja.
- **Lo ÚNICO que importa es la curva de la banca** (abajo): si sube a lo largo de semanas,
  el sistema gana. Un día suelto no dice nada.
- **Qué mirar cada día:** los tiquetes de hoy, la ganancia esperada, el peor 5% (lo máximo que
  se perdería en un día muy malo) y la proyección a 30/90 días.
""")

day = datetime.now(DISPLAY_TZ).date()


@st.cache_data(ttl=300)
def _bank() -> float:
    return P.available_bank()


@st.cache_data(ttl=300)
def _bank_before(d) -> float:
    # budget basis for the day: the bank BEFORE the day's own stakes were
    # locked — with default controls the what-if reproduces the persisted
    # portfolio exactly (deterministic seed)
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


@st.cache_data(ttl=3600)
def _projection(d) -> dict | None:
    return P.project_bankroll()


bank = _bank()
bank_basis = _bank_before(day)
hist = _tickets_hist()
persisted_today = hist[hist["date"] == day] if not hist.empty else pd.DataFrame()

# ------------------------------------------------------------ today + what-if
st.subheader(f"📅 Portafolio de hoy · {day.strftime('%d/%m/%Y')}")
c1, c2, c3 = st.columns([2, 1.6, 1.2])
max_b = int(max(P.floor500(bank_basis), P.STEP))
budget = c1.slider("Presupuesto del día ($)", 0, max_b,
                   int(min(P.default_budget(bank_basis), max_b)), step=int(P.STEP),
                   help="Cuánto se permite apostar HOY como máximo. Por defecto: 10% de la "
                        "banca disponible. Moverlo recalcula el portafolio al vuelo (what-if); "
                        "el portafolio OFICIAL del día se guarda con el valor por defecto.")
risk = c2.radio("Riesgo", list(P.RISKS), index=1, horizontal=True,
                help="Qué tan agresivo es el tamaño de las apuestas (fracción de Kelly): "
                     "Conservador = ⅛, Balanceado = ¼ (el oficial), Agresivo = ½. "
                     "Más riesgo = más ganancia esperada pero bajones más feos.")
c3.metric("Banca disponible", f"${bank:,.0f}",
          help="Plata de papel libre ahora mismo (lo apostado en tiquetes abiertos está "
               "descontado hasta que se liquiden).")

if not persisted_today.empty:
    st.caption(f"💾 Portafolio OFICIAL de hoy ya guardado: {len(persisted_today)} tiquete(s), "
               f"${persisted_today['stake'].sum():,.0f} apostados (presupuesto y riesgo por "
               "defecto). Con los controles por defecto la tabla de abajo lo reproduce exacto.")
else:
    st.caption("💾 El portafolio oficial del día se guarda automáticamente a las 14:15 UTC "
               "(9:15 AM Bogotá) tras la corrida de cuotas. Lo de abajo es el cálculo en vivo.")

res = _whatif(day, float(budget), risk)

if not res.get("tickets"):
    st.info("🙅 Hoy no hay valor — $0 apostado. No encontramos picks donde nuestra ventaja "
            "sobreviva el recorte prudente, así que la mejor apuesta es NO apostar. "
            "Un día sin apostar también es el sistema cuidándote la plata.")
else:
    s = res["summary"] or {}
    m = s.get("modelo", {})
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Apostado hoy", f"${res['staked']:,.0f}",
              help="Suma de los tiquetes. Puede ser menos que el presupuesto: el optimizador "
                   "solo apuesta lo que mejora el crecimiento esperado de la banca (y el tope "
                   "del 30% por partido limita la concentración).")
    k2.metric("Ganancia esperada (prudente)", f"${s.get('expected_profit', 0):+,.0f}",
              delta=f"si el modelo acierta: ${m.get('expected_profit', 0):+,.0f}",
              help="Promedio de 20.000 días simulados usando la probabilidad PRUDENTE "
                   "(70% nuestra + 30% mercado). El deltica muestra el escenario si nuestros "
                   "modelos tienen toda la razón.")
    k3.metric("P(día verde)", f"{s.get('p_green', 0):.0%}",
              delta=f"modelo: {m.get('p_green', 0):.0%}",
              help="Probabilidad de terminar el día ganando plata. Puede ser <50% aunque el "
                   "día tenga valor: las combinadas ganan poco seguido pero pagan mucho.")
    k4.metric("Peor 5% del día", f"${s.get('p5', 0):+,.0f}",
              help="En el 5% de los días más malos se perdería esto o más — nunca más que lo "
                   "apostado. Sirve para dimensionar el riesgo real del día.")

    rows = []
    for t in res["tickets"]:
        rows.append({
            "Apuesta": f"Apuesta {t['ticket_id']}",
            "Tipo": "Individual" if t["n_legs"] == 1 else f"Combinada x{t['n_legs']}",
            "Tiquete": " + ".join(f"{l['liga_titulo']} {l['partido']}: {l['pick']} @{l['cuota']}"
                                  for l in t["legs"]),
            "Cuota": t["cuota"], "Apostado": t["stake"],
            "Prob (prudente)": t["prob"], "Prob (modelo)": t["prob_modelo"],
            "Ganaría": t["stake"] * t["cuota"], "EV (prudente)": t["ev"],
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "Apuesta": st.column_config.TextColumn("Apuesta", help="Como en un recibo de "
                                                   "BetPlay: cada fila es un tiquete."),
            "Tipo": st.column_config.TextColumn("Tipo", help="Individual = un solo pick. "
                                                "Combinada = varios partidos: paga mucho más "
                                                "pero deben acertar TODOS."),
            "Tiquete": st.column_config.TextColumn("Tiquete", width="large",
                                                   help="Los picks del tiquete con su cuota."),
            "Cuota": st.column_config.NumberColumn("Cuota", format="%.2f",
                                                   help="Multiplicador del pago: apostar $1.000 "
                                                        "a cuota 2.94 devuelve $2.940 si acierta."),
            "Apostado": st.column_config.NumberColumn("Apostado", format="$%.0f",
                                                      help="Stake del tiquete (múltiplos de "
                                                           "$500, mínimo BetPlay)."),
            "Prob (prudente)": st.column_config.NumberColumn(
                "Prob (prudente)", format="percent",
                help="Probabilidad de que el tiquete gane, con el recorte prudente "
                     "(70% nuestra + 30% mercado). Con esta se decide cuánto apostar."),
            "Prob (modelo)": st.column_config.NumberColumn(
                "Prob (modelo)", format="percent",
                help="La misma probabilidad pero creyéndole 100% a nuestros modelos."),
            "Ganaría": st.column_config.NumberColumn("Ganaría", format="$%.0f",
                                                     help="Lo que devuelve el tiquete si "
                                                          "acierta (stake × cuota)."),
            "EV (prudente)": st.column_config.NumberColumn(
                "EV (prudente)", format="$%.0f",
                help="Ganancia promedio esperada del tiquete a largo plazo, con la "
                     "probabilidad prudente."),
        })

# ------------------------------------------------------------------ project --
st.divider()
st.subheader("🔮 Proyección de la banca (30 y 90 días)")
st.caption("Monte Carlo: simulamos miles de futuros donde cada día se parece a los días "
           "recientes registrados (misma cantidad de apuestas y ventajas, incluyendo días sin "
           "valor), reinvirtiendo el 10% de la banca. Banda gris de la realidad: la mitad de "
           "los futuros cae entre la línea pesimista (25%) y la optimista (75%).")

proj = _projection(day)
if not proj:
    st.info("La proyección aparece cuando exista al menos un día registrado en la banca.")
else:
    pdf = pd.DataFrame({"Pesimista (25%)": proj["p25"], "Mediana": proj["p50"],
                        "Optimista (75%)": proj["p75"]},
                       index=pd.RangeIndex(1, proj["horizon"] + 1, name="día"))
    st.line_chart(pdf, height=300)
    j1, j2, j3 = st.columns(3)
    j1.metric("Banca hoy", f"${proj['bank0']:,.0f}")
    j2.metric("P(banca por debajo del inicio) a 30 días", f"{proj['p_below_start_30']:.0%}",
              help="Probabilidad de ir perdiendo tras 30 días. Con ventaja real este número "
                   "baja con el tiempo; si se mantiene alto, el sistema no está ganando.")
    j3.metric(f"P(por debajo del inicio) a {proj['horizon']} días",
              f"{proj['p_below_start_end']:.0%}",
              help="Lo mismo pero al final del horizonte simulado.")
    st.caption(f"Basada en los últimos {proj['n_templates']} día(s) registrados — con pocos "
               "días la proyección es apenas orientativa; gana precisión cada semana.")

# ------------------------------------------------------------------ history --
st.divider()
st.subheader("📒 Historial — la curva que importa")

bk = _bankroll()
if bk.empty:
    st.info("Aún no hay días registrados. El primer portafolio se guarda hoy a las 14:15 UTC.")
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
    h1.metric("Banca actual", f"${bank:,.0f}",
              help="La banca de papel hoy (arrancó en $100.000).")
    h2.metric("P&L acumulado", f"${pnl_total:+,.0f}",
              help="Ganancia/pérdida total de todos los días ya liquidados.")
    h3.metric("Días verdes", f"{verdes}/{len(settled)}",
              help="Días liquidados que terminaron en ganancia. No esperes 100%: basta con "
                   "que la curva suba.")
    h4.metric("ROI sobre lo apostado", f"{(pnl_total / staked_total * 100):+.1f}%"
              if staked_total else "—",
              help="P&L ÷ total apostado en días liquidados. Un ROI sostenido de +5% ya es "
                   "excelente en apuestas deportivas.")

    curva = bk.set_index("date")["banca"].rename("banca ($)")
    st.line_chart(curva, height=280)

    st.markdown("**P&L por día**")
    tabla = pd.DataFrame({
        "Fecha": bk["date"], "Banca inicio": bk["start_bank"], "Apostado": bk["staked"],
        "Devuelto": bk["returned"], "P&L": bk["returned"] - bk["staked"],
        "Banca fin": bk["end_bank"],
        "Estado": bk["end_bank"].map(lambda x: "✔ cerrado" if pd.notna(x) else "⏳ abierto"),
    })
    st.dataframe(tabla.sort_values("Fecha", ascending=False), use_container_width=True,
                 hide_index=True,
                 column_config={
                     "Banca inicio": st.column_config.NumberColumn(format="$%.0f",
                         help="Con cuánta plata de papel arrancó el día."),
                     "Apostado": st.column_config.NumberColumn(format="$%.0f",
                         help="Total apostado ese día ($0 en días sin valor)."),
                     "Devuelto": st.column_config.NumberColumn(format="$%.0f",
                         help="Lo que devolvieron los tiquetes al liquidarse."),
                     "P&L": st.column_config.NumberColumn(format="$%.0f",
                         help="Devuelto menos apostado: la ganancia o pérdida del día."),
                     "Banca fin": st.column_config.NumberColumn(format="$%.0f",
                         help="Banca al cerrar el día (vacío = aún hay tiquetes abiertos)."),
                     "Estado": st.column_config.TextColumn(
                         help="⏳ abierto = esperando resultados; ✔ cerrado = día liquidado."),
                 })

    if not hist.empty:
        st.markdown("**Tiquetes**")
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
                         "Cuota": st.column_config.NumberColumn(format="%.2f",
                             help="Multiplicador del pago del tiquete."),
                         "Apostado": st.column_config.NumberColumn(format="$%.0f"),
                         "Resultado": st.column_config.TextColumn(
                             help="✓ ganó · ✗ perdió · ⏳ esperando los partidos · "
                                  "↩ anulada (partido aplazado → se devuelve lo apostado)."),
                         "Devuelto": st.column_config.NumberColumn(format="$%.0f",
                             help="Ganada: apostado × cuota. Perdida: $0. Anulada: se "
                                  "devuelve lo apostado."),
                     })

st.caption("⚖️ Dinero 100% de papel — análisis, no una invitación a apostar. Liquidación "
           "diaria automática (13:30 UTC) con los resultados reales de los partidos; partido "
           "aplazado = tiquete anulado y stake devuelto.")
