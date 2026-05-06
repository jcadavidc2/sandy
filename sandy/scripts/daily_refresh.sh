#!/usr/bin/env bash
# Sandy daily refresh script with Telegram notifications.
#
# Runs `sandy refresh` and sends a Telegram notification on success or failure.
#
# Required environment variables:
#   TELEGRAM_BOT_TOKEN - Telegram bot API token
#   TELEGRAM_CHAT_ID   - Telegram chat ID to send notifications to
#
# These can be set in the environment, or in ~/.sandy_env:
#   export TELEGRAM_BOT_TOKEN="your-bot-token"
#   export TELEGRAM_CHAT_ID="your-chat-id"
#
# Usage:
#   ./sandy/scripts/daily_refresh.sh
#
# Cron example (see sandy/scripts/crontab.txt):
#   0 6 * * * /home/ec2-user/sandy/scripts/daily_refresh.sh >> /home/ec2-user/sandy/logs/cron.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment if config file exists
if [ -f "$HOME/.sandy_env" ]; then
    source "$HOME/.sandy_env"
fi

# Also try project-level .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Telegram notification function
send_telegram() {
    local message="$1"

    if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
        echo "[$(date -Iseconds)] WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping notification"
        return 0
    fi

    curl -s -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML" \
        > /dev/null 2>&1 || echo "[$(date -Iseconds)] WARNING: Failed to send Telegram notification"
}

# Main execution
echo "=========================================="
echo "[$(date -Iseconds)] Starting Sandy daily refresh"
echo "=========================================="

cd "$PROJECT_DIR"

# Run the refresh command and capture output
REFRESH_OUTPUT=""
if REFRESH_OUTPUT=$(sandy refresh 2>&1); then
    # Extract stats from output
    GAMES_ADDED=$(echo "$REFRESH_OUTPUT" | grep -oP '\d+ added' | head -1 | grep -oP '\d+' || echo "0")
    LABELS_COUNT=$(echo "$REFRESH_OUTPUT" | grep -oP '\d+ labels' | head -1 | grep -oP '\d+' || echo "0")
    FEATURES_COUNT=$(echo "$REFRESH_OUTPUT" | grep -oP '\d+ inning features' | head -1 | grep -oP '\d+' || echo "0")

    echo "[$(date -Iseconds)] Refresh completed successfully"
    echo "$REFRESH_OUTPUT"

    send_telegram "✅ Sandy daily refresh complete: ${GAMES_ADDED} games added, labels and features updated (${LABELS_COUNT} labels, ${FEATURES_COUNT} features)"
else
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] Refresh FAILED (exit code: $EXIT_CODE)"
    echo "$REFRESH_OUTPUT"

    # Truncate error message for Telegram (max 4096 chars)
    ERROR_MSG=$(echo "$REFRESH_OUTPUT" | tail -5 | head -c 500)
    send_telegram "❌ Sandy refresh failed: ${ERROR_MSG}"

    exit $EXIT_CODE
fi

echo "[$(date -Iseconds)] Done"
