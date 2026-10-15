#!/usr/bin/env bash
set -euo pipefail

# Example only. Run from the candidate repository on a reviewed GPU host.
ENV_FILE="${ENV_FILE:-.env}"
APP_PATH="${APP_PATH:-app.py}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Create an untracked $ENV_FILE from .env.example first." >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

exec python "$APP_PATH"
