#!/usr/bin/env bash
# Start live tracking on the Orange Pi AIpro. All arguments are passed to predict.py, e.g.
#   bash TrackNetV3/run_tracking.sh --csv output/d435i.csv
#   bash TrackNetV3/run_tracking.sh --headless --max-results 3
#   bash TrackNetV3/run_tracking.sh --list-devices
set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
PY="$ROOT/.venv/bin/python"

# Provides pyACL and the NPU runtime libraries.
[ -f "$ASCEND_ENV" ] && source "$ASCEND_ENV"
[ -x "$PY" ] || { echo "Missing .venv; run bash TrackNetV3/setup_aipro.sh" >&2; exit 1; }
exec "$PY" "$DIR/predict.py" "$@"
