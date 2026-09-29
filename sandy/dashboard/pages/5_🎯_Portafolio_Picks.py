"""🎯 Portafolio B "Picks del Día" — the A/B experiment page (paper money).

Second $100,000 paper bank. Since the 2026-09-28 audit B is the FLAT-STAKE
SINGLES arm: 2% of its bank on each of the day's top ✅ Picks del Día (one per
game, matched cuota, market no-vig on our side, ≥3 books, league gate), at
most 5 picks, no Kelly, no parlays, no forced bets. All math in
sandy/portfolio_picks.py (separate tables/bank). Layout mirrors page 4:
official sheet → what-if simulator → ledger (with A's curve overlaid).
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
st.caption(f"El experimento hermano del 🎰 Portafolio: una SEGUNDA banca de papel de $100.000 "
           f"que apuesta PLANO — {P.B_FLAT_FRACTION:.0%} de su banca en cada uno de los mejores "
           f"🏁 Picks del Día con cuota (máximo {P.B_MAX_PICKS_PER_DAY} individuales, el mejor 🤖 "
           f"por partido), solo cuando el mercado también está de nuestro lado. Sin Kelly, sin "
           f"combinadas, sin apuestas forzadas. Las dos curvas, lado a lado, son la prueba A/B: "
           f"Kelly (A) contra plano (B).")

with st.expander("📖 Cómo leer esta página (y en qué se diferencia del 🎰)"):
    st.markdown(f"""
- **Nadie apuesta plata real.** Es una banca de papel independiente de $100.000.
- **La tesis de este portafolio (rediseñada el 28/9):** apuesta **plana** — la misma cantidad
  ({P.B_FLAT_FRACTION:.0%} de la banca) en cada pick, solo individuales, máximo
  {P.B_MAX_PICKS_PER_DAY} al día, los de mayor 🤖 primero. Es la forma más directa de probar
  "más días verdes que rojos": sin combinadas que pierden casi siempre y sin tamaños Kelly que
  crecen con probabilidades infladas. El 🎰 A es el brazo Kelly (recortado 30/70 hacia el
  mercado, individuales y dobles).
- **Por qué cambió:** la B original ("creerle 100% al modelo, apostar siempre, combinadas de 4")
  quedó refutada por sus propios datos — el peso óptimo del modelo frente al mercado era ~0, el
  64% de sus patas eran las mismas de A el mismo día, la regla siempre-apostar solo actuó 4 días
  y las combinadas de 4 iban 0/33 desde agosto.
- **Filtros:** mercado % ≥ 50% en nuestro lado (en contra acertábamos 44% creyendo 58%), nuestra
  probabilidad ≥ la del mercado, ≥3 casas cotizando la línea, y ligas con ROI reciente < −5%
  por fuera. Un día sin picks que pasen los filtros queda en $0 — a conciencia.
- **⚠️sust. en un tiquete** = el pick exacto de Picks del Día para ese juego no tiene cuota
  en el mercado, así que B apuesta el MEJOR pick con cuota de ese mismo juego (el sustituto).
- **Esto es un experimento, no una recomendación.** Con cuotas ~1.5–2.0 y apuesta plana, un
  día es verde cuando aciertan más picks de los que fallan; la curva sube solo si nuestros
  picks ✅ aciertan de verdad más de lo que la cuota exige.
- **Lo único que importa:** comparar la curva de esta banca contra la del 🎰 Portafolio A
  (abajo van superpuestas). Si B sube más parejo, el tamaño plano protege mejor; si A gana,
  Kelly recortado está sacando más de la misma ventaja.
- **Un pick por partido:** por cada juego entra solo el pick de mayor 🤖 (el de Picks del Día).
  Picks sin cuota casada (corners, BTTS, 1X de NHL) no pueden apostarse.
""")

day = datetime.now(DISPLAY_TZ).date()


@st.cache_data(ttl=300)
def _bank_B(_page: str = "B") -> float:
    return P.available_bank()


@st.cache_data(ttl=300)
def _bank_before_B(d, _page: str = "B") -> float:
    # bank BEFORE the day's own stakes — default controls reproduce the
    # persisted portfolio exactly (deterministic seed)
    return P.available_bank(before=d)


@st.cache_data(ttl=300)
def _whatif_B(d, budget: float, risk: str, _page: str = "B") -> dict:
    return P.build_portfolio(day=d, budget=budget, risk=risk, persist=False)


@st.cache_data(ttl=300)
def _tickets_hist_B(_page: str = "B") -> pd.DataFrame:
    return P.tickets_frame()


@st.cache_data(ttl=300)
def _bankroll_B(_page: str = "B") -> pd.DataFrame:
    return P.bankroll_frame()


@st.cache_data(ttl=300)
def _bankroll_a() -> pd.DataFrame:
    return PA.bankroll_frame()


bank = _bank_B()
bank_basis = _bank_before_B(day)
hist = _tickets_hist_B()
persisted_today = hist[hist["date"] == day] if not hist.empty else pd.DataFrame()

# ----------------------------------------------------------- OFFICIAL first
st.subheader(f"📌 Portafolio B OFICIAL de hoy · {day.strftime('%d/%m/%Y')}")
o1, o2, o3, o4 = st.columns(4)
o1.metric("Banca disponible (B)", f"${bank:,.0f}",
          help="Plata de papel libre de ESTA banca (independiente del 🎰). Lo apostado en "
               "tiquetes abiertos está descontado.")
o2.metric("Presupuesto del día", f"${PA.default_budget(bank_basis):,.0f}",
          help=f"Misma regla del 🎰: {PA.BUDGET_FRACTION:.0%} de la banca con que amaneció el "
               f"día (${bank_basis:,.0f}), en pasos de $500 — {P.B_MAX_PICKS_PER_DAY} picks × "
               f"{P.B_FLAT_FRACTION:.0%} de la banca.")
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
    _bk0 = _bankroll_B()
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
                   help=f"Cuánto se permitiría apostar HOY como máximo en esta simulación. "
                        f"El oficial usa el {PA.BUDGET_FRACTION:.0%} de la banca B. Moverlo NO guarda nada.")
risk = PA.DEFAULT_RISK  # B stakes FLAT — the Kelly risk dial does not apply here
c2.metric("Apuesta por pick (plana)", f"${P.flat_stake(bank_basis):,.0f}",
          help=f"{P.B_FLAT_FRACTION:.0%} de la banca B redondeado a $500 (mínimo $500). "
               "La misma cantidad en cada pick — B no usa Kelly.")
c3.metric("Presupuesto oficial", f"${PA.default_budget(bank_basis):,.0f}",
          help=f"El techo real del día: {PA.BUDGET_FRACTION:.0%} de la banca B.")

res = _whatif_B(day, float(budget), risk)

if not res.get("tickets"):
    st.info("🙅 $0 apostado en esta simulación — hoy ningún pick ✅ con cuota pasa los filtros "
            "(mercado a favor, ≥3 casas, liga habilitada) o el presupuesto no alcanza para la "
            "apuesta mínima de $500. Un día sin apostar es una decisión, no una falla.")
else:
    s = res["summary"] or {}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Apostado hoy", f"${res['staked']:,.0f}",
              help=f"Individuales planas de ${res.get('unidad', 0):,.0f} cada una "
                   f"(máximo {P.B_MAX_PICKS_PER_DAY} picks, dentro del presupuesto).")
    k2.metric("Ganancia esperada (modelo)", f"${s.get('expected_profit', 0):+,.0f}",
              help="Promedio de miles de días simulados usando nuestra probabilidad calibrada. "
                   "Si el mercado tiene razón, el valor real es ~0 — por eso lo que decide es la "
                   "curva, no este número.")
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
                help="Probabilidad de que el pick gane según nuestra probabilidad calibrada "
                     "(el tamaño de la apuesta NO depende de ella: B apuesta plano)."),
            "Ganaría": st.column_config.NumberColumn("Ganaría", format="$%.0f",
                                                     help="Stake × cuota si acierta."),
            "EV (modelo)": st.column_config.NumberColumn(
                "EV (modelo)", format="$%.0f",
                help="Ganancia promedio esperada del pick si nuestros modelos tienen razón."),
        })

# ------------------------------------------------------------------ history --
st.divider()
st.subheader("📒 Historial — B contra A, la curva que decide el experimento")

bk = _bankroll_B()
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
              help="Días liquidados que terminaron en ganancia — la métrica que B existe para "
                   "mejorar (apuesta plana en individuales).")
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
               "distintas). Hasta el 28/9 las dos apostaban casi lo mismo (64% de patas en común); "
               "desde entonces 🎯 B es apuesta plana en individuales y 🎰 A es Kelly recortado con "
               "dobles — la comparación empieza ahí.")

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
           "apostar. Este portafolio apuesta PLANO en individuales con el mercado a favor; su "
           "hermano Kelly vive en 🎰 Portafolio. Liquidación diaria automática con los "
           "resultados reales; partido aplazado = tiquete anulado y stake devuelto.")
