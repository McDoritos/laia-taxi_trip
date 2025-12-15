#!/bin/sh
set -e

LOG_DIR="/app/logs"

mkdir -p "$LOG_DIR"

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}

chown -R $HOST_UID:$HOST_GID "$LOG_DIR"
chmod -R 777 "$LOG_DIR"

echo "Log directory ready:"
ls -ld "$LOG_DIR"

exec uv run python app.py
