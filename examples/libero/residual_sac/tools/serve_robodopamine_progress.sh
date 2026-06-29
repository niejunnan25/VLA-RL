#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec "$ROOT/examples/libero/rlpd/tools/serve_robodopamine_progress.sh" "$@"
