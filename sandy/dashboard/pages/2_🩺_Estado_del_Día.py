"""🩺 Estado del día — did every daily job run? What's still pending/running?

Reads each pipeline's log for today's completion marker + every meta artifact's
trained_at, so at a glance you know whether the numbers on the other pages are
already TODAY's or still yesterday's.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from sandy.dashboard import data as D

st.set_page_config(page_title="Sandy · Estado", page_icon="🩺", layout="wide")
st.title("🩺 Estado del día")

BOG = timezone(timedelta(hours=-5))
LOGS = Path("/home/ec2-user/sandy/logs")
GRACE_MIN = 25

# job → (log file, scheduled UTC hh:mm, label)
JOBS = [
    ("⚾ MLB nightly (ingesta+modelo)", "nightly.log", (8, 0)),
    ("🌍 Fútbol ligas nightly", "soccer.log", (12, 15)),
    ("⚽ MLS nightly", "mls.log", (12, 30)),
    ("🏒 NHL nightly", "nhl.log", (12, 45)),
    ("🏀 NBA nightly", "nba.log", (12, 50)),
    ("🏆 Mundial nightly", "football.log", (13, 0)),
    ("🤖 Metas extra + refinador + refresh web", "metas_extra.log", (13, 30)),
    ("⚾ MLB predicciones del día", "over_under.log", (14, 0)),
    ("💰 Cuotas + valor (TheOddsAPI)", "odds.log", (14, 15)),
]
OK_PAT = re.compile(r"COMPLETE|complete|Done\b")
BAD_PAT = re.compile(r"FAILED|falló|Traceback|ERROR")
TS_PAT = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

now = datetime.now(timezone.utc)
today = now.date()

rows = []
all_ok = True
for label, logname, (hh, mm) in JOBS:
    sched = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    sched_str = sched.astimezone(BOG).strftime("%-I:%M %p")
    path = LOGS / logname
    ok_ts, bad_line = None, None
    if path.exists():
        tail = path.read_text(errors="ignore")[-20000:].splitlines()
        for ln in tail:
            m = TS_PAT.search(ln)
            ts = m.group(1) if m else None
            is_today = bool(ts and ts.startswith(str(today)))
            if is_today and OK_PAT.search(ln):
                ok_ts = ts
            if is_today and BAD_PAT.search(ln):
                bad_line = ln.strip()[:120]
    if ok_ts:
        done = datetime.fromisoformat(ok_ts).replace(tzinfo=timezone.utc).astimezone(BOG)
        estado, detalle = "✅ Completado", f"terminó {done.strftime('%-I:%M %p')}"
    elif bad_line:
        estado, detalle = "❌ Falló", bad_line
        all_ok = False
    elif now < sched:
        estado, detalle = "⏳ Programado", f"corre a las {sched_str}"
        all_ok = False
    elif now < sched + timedelta(minutes=GRACE_MIN):
        estado, detalle = "🔄 Corriendo…", f"empezó ~{sched_str}, dale unos minutos"
        all_ok = False
    else:
        estado, detalle = "❌ No corrió hoy", "revisa Telegram (alerta ❌) o los logs"
        all_ok = False
    rows.append({"pipeline": label, "hora (Bogotá)": sched_str, "estado": estado, "detalle": detalle})

if all_ok:
    st.success("✅ TODO lo de hoy corrió — lo que ves en las otras páginas ya es de HOY.")
else:
    pend = sum(1 for r in rows if not r["estado"].startswith("✅"))
    st.warning(f"⏳ {pend} trabajo(s) aún pendientes/corriendo — los números pueden ser de ayer "
               "hasta que terminen (el refresh web es el de las 8:30 AM).")

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Frescura de los meta-modelos")
st.caption("Cada meta debe decir HOY después de su nightly. La web carga estos artefactos en el refresh de las 8:30 AM.")
mrows = []
for lg in D.LEAGUES:
    art = D.meta_artifact(lg)
    ta = art.get("trained_at")
    fresh = "✅ hoy" if str(ta) == str(today) else f"🕐 {ta}"
    mrows.append({"liga": D.league_title(lg), "entrenado": fresh,
                  "filas": f"{(art.get('trained_rows') or 0) + (art.get('calib_rows') or 0) + (art.get('holdout_rows') or 0):,}",
                  "AUC test": art.get("auc")})
st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)

try:
    import pickle
    rp = Path("/home/ec2-user/sandy/models/refiner.pkl")
    if rp.exists():
        with open(rp, "rb") as f:
            ra = pickle.load(f)
        eng = {k: v for k, v in ra.items() if "engine" in str(k)}
        st.caption(f"🔬 Refinador: entrenado {ra.get('trained_at', '?')} · motores del día: {eng or 'aún por regla simple'}")
except Exception:
    pass

st.caption(f"Actualizado al abrir la página · hora servidor {now.astimezone(BOG).strftime('%-I:%M %p')} Bogotá. "
           "Cualquier ❌ también te llega como alerta en Telegram.")
