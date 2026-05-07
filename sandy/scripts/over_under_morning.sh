#!/usr/bin/env bash
# Sandy Over/Under Morning Predictions (7 AM UTC)
#
# Runs over/under predictions for all scheduled games and sends
# a Telegram digest with trust signal from latest calibration.
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
echo "[$(date -Iseconds)] Starting over/under morning predictions"
echo "=========================================="

cd "$PROJECT_DIR"

if OUTPUT=$(sandy over-under predict --notify 2>&1); then
    echo "[$(date -Iseconds)] Morning predictions complete"
    echo "$OUTPUT"
else
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] Morning predictions FAILED (exit code: $EXIT_CODE)"
    echo "$OUTPUT"
    ERROR_MSG=$(echo "$OUTPUT" | tail -5 | head -c 500)
    send_telegram "❌ over_under_morning.sh failed:
${ERROR_MSG}"
    exit $EXIT_CODE
fi

echo "[$(date -Iseconds)] Done"
