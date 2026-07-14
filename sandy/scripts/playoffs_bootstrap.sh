#!/usr/bin/env bash
# ONE-TIME playoffs-covariate bootstrap (2026-07-14). For col/mex/mls/nba/nfl the
# games history predates the stage/season_type columns — re-walk the (idempotent)
# backfills so every historical game carries its stage, then stamp the covariates
# onto ALL prediction rows set-based (no model re-runs: playoff flags are game
# properties, not model outputs), then retrain the six metas UNDER THE GATE
# (betmeta.retrain_gated: ship only if test AUC improves and acc@thr holds;
# regressing leagues revert to their backed-up artifacts automatically).
# NHL needs no re-walk (game_type always stored). Sequential + nice'd, 1.9GB box.
set -uo pipefail
SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
PY="/home/ec2-user/sandy/.venv/bin/python"
cd /home/ec2-user/sandy
set -a; source "$HOME/.sandy_env"; set +a
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" > /dev/null 2>&1 || true
}

echo "[$(date -Iseconds)] applying migrations (idempotent ALTERs)..."
nice -n 10 $PY - <<'PYEOF'
from pathlib import Path
from sandy.config import load_config
from sandy.db import create_engine
eng = create_engine(load_config())
base = Path('sandy/migrations')
for f in ('add_soccer_tables.sql', 'add_mls_tables.sql', 'add_nba_tables.sql',
          'add_nhl_tables.sql', 'add_nfl_tables.sql'):
    with eng.begin() as conn:
        conn.exec_driver_sql((base / f).read_text())
    print('applied', f)
PYEOF

echo "[$(date -Iseconds)] re-walk backfills (stage/season_type onto history)..."
nice -n 10 $SANDY mls backfill --start 2023-02-25 2>&1 | tail -1 || { tg "❌ Playoffs bootstrap: mls backfill falló"; exit 1; }
nice -n 10 $SANDY soccer backfill --league col --start 2023-07-14 2>&1 | tail -1 || { tg "❌ Playoffs bootstrap: col backfill falló"; exit 1; }
nice -n 10 $SANDY soccer backfill --league mex --start 2023-07-01 2>&1 | tail -1 || { tg "❌ Playoffs bootstrap: mex backfill falló"; exit 1; }
nice -n 10 $SANDY nba backfill --start 2022-10-01 2>&1 | tail -1 || { tg "❌ Playoffs bootstrap: nba backfill falló"; exit 1; }
nice -n 10 $SANDY nfl backfill --start 2022-09-01 2>&1 | tail -1 || { tg "❌ Playoffs bootstrap: nfl backfill falló"; exit 1; }

echo "[$(date -Iseconds)] stamping covariates onto all prediction rows..."
nice -n 10 $PY - <<'PYEOF'
from sandy.config import load_config
from sandy.db import create_engine
eng = create_engine(load_config())
from sandy.soccer.loop import stamp_stage_covariates
from sandy.mls.predictor import stamp_playoff_covariates as mls_stamp
from sandy.nhl.model import stamp_playoff_covariates as nhl_stamp
from sandy.nba.loop import stamp_playoff_covariates as nba_stamp
from sandy.nfl.loop import stamp_playoff_covariates as nfl_stamp
print('soccer:', stamp_stage_covariates(eng))
print('mls   :', mls_stamp(eng))
print('nhl   :', nhl_stamp(eng))
print('nba   :', nba_stamp(eng))
print('nfl   :', nfl_stamp(eng))
PYEOF

echo "[$(date -Iseconds)] GATED retrain of the six metas..."
nice -n 10 $PY - <<'PYEOF'
import json
from sandy.betmeta import retrain_gated
rep = retrain_gated(['soccer_col', 'soccer_mex', 'mls', 'nba', 'nhl', 'nfl'])
for lg, r in rep['leagues'].items():
    o, n = r['old'], r['new']
    print(f"{lg}: old auc={o.get('auc')} acc={o.get('acc_at_thr')} -> "
          f"new auc={n.get('auc')} acc={n.get('acc_at_thr')} err={n.get('error')}")
print('reverted:', rep.get('reverted'))
PYEOF

tg "🏁 Covariable de PLAYOFFS evaluada con compuerta en col/mex/MLS/NBA/NHL/NFL — resultados en models/tuning/meta_retrain_report.json (las ligas donde no mejora conservan su modelo actual)."
echo "[$(date -Iseconds)] PLAYOFFS BOOTSTRAP COMPLETE"
