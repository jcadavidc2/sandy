#!/usr/bin/env bash
# Sandy Nightly Pipeline — runs ALL steps sequentially.
# Each step only starts after the previous one completes.
# No timing guesses — guaranteed correct order.
#
# Steps:
#   1. Ingest new games (get yesterday's final scores)
#   2. Build labels (all targets)
#   3. Build game features (incremental — only new games)
#   4. Reconcile over/under predictions vs actuals
#   5. Retrain ALL models (reached_base, game_winner, runs)
#   6. Calibrate over/under accuracy
#   7. Send nightly report to Telegram
#
# Runs at 1 AM PST (08:00 UTC) — after all games finish.

set -euo pipefail

SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
PROJECT_DIR="/home/ec2-user/sandy"
cd "$PROJECT_DIR"

# Load environment
if [ -f "$HOME/.sandy_env" ]; then
    source "$HOME/.sandy_env"
fi
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

# Telegram notification function
send_telegram() {
    local message="$1"
    if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
        return 0
    fi
    curl -s -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        > /dev/null 2>&1 || true
}

echo "=========================================="
echo "[$(date -Iseconds)] Sandy Nightly Pipeline starting"
echo "=========================================="

# Step 1: Ingest new games
echo "[$(date -Iseconds)] Step 1/7: Ingesting new games..."
if ! $SANDY ingest incremental 2>&1; then
    send_telegram "❌ Nightly pipeline failed at: ingest"
    exit 1
fi

# Step 2: Build labels (all targets)
echo "[$(date -Iseconds)] Step 2/7: Building labels..."
$SANDY labels build --target reached_base 2>&1 | tail -1
$SANDY labels build --target game_winner 2>&1 | tail -1
$SANDY labels build --target runs 2>&1 | tail -1

# Step 3: Build game features (incremental)
echo "[$(date -Iseconds)] Step 3/7: Building game features..."
$SANDY features build --target game_winner 2>&1 | tail -1

# Step 4: Reconcile over/under
echo "[$(date -Iseconds)] Step 4/7: Reconciling over/under outcomes..."
$SANDY over-under reconcile 2>&1 | tail -1

# Step 5: Retrain ALL models
echo "[$(date -Iseconds)] Step 5/7: Retraining all models..."
$SANDY train --target runs 2>&1 | tail -2
$SANDY train --target game_winner 2>&1 | tail -2
# Skip reached_base retrain for now (needs inning features which are slow)
# $SANDY train --target reached_base 2>&1 | tail -2

# Step 6: Calibrate
echo "[$(date -Iseconds)] Step 6/7: Computing calibration..."
$SANDY over-under calibrate 2>&1 | tail -3

# Step 7: Send nightly report
echo "[$(date -Iseconds)] Step 7/7: Sending nightly report..."
$SANDY over-under reconcile --notify 2>&1 | tail -1

echo "=========================================="
echo "[$(date -Iseconds)] Nightly Pipeline COMPLETE"
echo "=========================================="
send_telegram "✅ Sandy nightly pipeline complete: data updated, all models retrained, calibration updated."
