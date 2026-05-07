#!/usr/bin/env bash
# Sandy Over/Under Weekly Deeper Analysis (Sunday 11:30 PM UTC)
#
# Runs deeper calibration with 4-week lookback and rolling trends.
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
echo "[$(date -Iseconds)] Starting over/under weekly analysis"
echo "=========================================="

cd "$PROJECT_DIR"

if OUTPUT=$(/home/ec2-user/sandy/.venv/bin/sandy over-under calibrate --notify --weekly 2>&1); then
    echo "[$(date -Iseconds)] Weekly analysis complete"
    echo "$OUTPUT"
else
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] Weekly analysis FAILED (exit code: $EXIT_CODE)"
    echo "$OUTPUT"
    ERROR_MSG=$(echo "$OUTPUT" | tail -5 | head -c 500)
    send_telegram "❌ over_under_weekly.sh failed:
${ERROR_MSG}"
    exit $EXIT_CODE
fi

echo "[$(date -Iseconds)] Done"
