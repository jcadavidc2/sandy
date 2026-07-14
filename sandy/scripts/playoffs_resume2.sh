#!/usr/bin/env bash
# Playoffs bootstrap — final stretch (2026-07-14): NFL re-walk, covariate stamps,
# gated retrains. Everything before this (migrations, mls/col/mex/nba re-walks)
# is already done and idempotent-safe.
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

echo "[$(date -Iseconds)] RESUME2: nfl re-walk..."
nice -n 10 $SANDY nfl backfill --start 2022-09-01 2>&1 | tail -1 || { tg "❌ Playoffs bootstrap: nfl falló"; exit 1; }

echo "[$(date -Iseconds)] stamping covariates..."
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

echo "[$(date -Iseconds)] GATED retrains..."
nice -n 10 $PY - <<'PYEOF'
from sandy.betmeta import retrain_gated
rep = retrain_gated(['soccer_col', 'soccer_mex', 'mls', 'nba', 'nhl', 'nfl'])
for lg, r in rep['leagues'].items():
    o, n = r['old'], r['new']
    print(f"{lg}: old auc={o.get('auc')} acc={o.get('acc_at_thr')} -> "
          f"new auc={n.get('auc')} acc={n.get('acc_at_thr')} err={n.get('error')}")
print('reverted:', rep.get('reverted'))
PYEOF

tg "🏁 Covariable de PLAYOFFS: evaluación con compuerta COMPLETA — reporte en models/tuning/meta_retrain_report.json"
echo "[$(date -Iseconds)] PLAYOFFS BOOTSTRAP COMPLETE"
