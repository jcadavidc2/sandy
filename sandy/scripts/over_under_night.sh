#!/usr/bin/env bash
# Sandy Over/Under Nightly Reconciliation + Retraining + Calibration (11 PM UTC)
#
# 1. Reconcile final scores
# 2. Retrain runs model
# 3. Compute calibration
# All with Telegram notifications.
#
# Required environment variables:
#   MLB_DB_HOST, MLB_DB_PORT, MLB_DB_NAME, MLB_DB_USER, MLB_DB_PASSWORD
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment
if [ -f "$HOME/.sandy_env" ]; then
    source "$HOME/.sandy_env"
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Activate virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Telegram notification function
send_telegram() {
    local message="$1"
    if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
        echo "[$(date -Iseconds)] WARNING: Telegram credentials not set"
        return 0
    fi
    curl -s -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML" \
        > /dev/null 2>&1 || echo "[$(date -Iseconds)] WARNING: Telegram send failed"
}

echo "=========================================="
echo "[$(date -Iseconds)] Starting over/under nightly pipeline"
echo "=========================================="

cd "$PROJECT_DIR"

# Step 1: Reconcile
echo "[$(date -Iseconds)] Step 1: Reconciling outcomes..."
if ! OUTPUT=$(sandy over-under reconcile --notify 2>&1); then
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] Reconciliation FAILED (exit code: $EXIT_CODE)"
    echo "$OUTPUT"
    ERROR_MSG=$(echo "$OUTPUT" | tail -5 | head -c 500)
    send_telegram "❌ over_under_night.sh failed at reconcile:
${ERROR_MSG}"
    exit $EXIT_CODE
fi
echo "$OUTPUT"

# Step 2: Retrain runs model
echo "[$(date -Iseconds)] Step 2: Retraining runs model..."
if ! OUTPUT=$(sandy train --target runs 2>&1); then
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] Retraining FAILED (exit code: $EXIT_CODE)"
    echo "$OUTPUT"
    ERROR_MSG=$(echo "$OUTPUT" | tail -5 | head -c 500)
    send_telegram "❌ over_under_night.sh failed at retrain:
${ERROR_MSG}"
    exit $EXIT_CODE
fi
echo "$OUTPUT"

# Step 3: Calibrate
echo "[$(date -Iseconds)] Step 3: Computing calibration..."
if ! OUTPUT=$(sandy over-under calibrate --notify 2>&1); then
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] Calibration FAILED (exit code: $EXIT_CODE)"
    echo "$OUTPUT"
    ERROR_MSG=$(echo "$OUTPUT" | tail -5 | head -c 500)
    send_telegram "❌ over_under_night.sh failed at calibrate:
${ERROR_MSG}"
    exit $EXIT_CODE
fi
echo "$OUTPUT"

echo "[$(date -Iseconds)] Nightly pipeline complete"
