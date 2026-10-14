#!/usr/bin/env bash
# Shared settings for experiment scripts.
set -euo pipefail
DEVICE=${DEVICE:-cpu}
TIME=${TIME:-600}
bench() { python -m rubiks.bench "$@"; }
