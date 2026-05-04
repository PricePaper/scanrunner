#!/usr/bin/env bash
# scanrunner container entrypoint.
#
# Idempotently warms the uv and DocTR caches into a persistent volume,
# then drops privileges to the unprivileged `scanner` user via gosu and
# hands off to tini as PID 1.
#
# Cache layout (single host volume mounted at /var/cache/scanrunner):
#   $UV_CACHE_DIR    — uv-managed wheels (torch CPU, opencv, doctr, …)
#   $DOCTR_CACHE_DIR — DocTR model weights
#
# Marker files gate the warm steps:
#   $UV_CACHE_DIR/.warmed-<sha256-of-inline-deps-block>
#   $DOCTR_CACHE_DIR/.warmed
#
# The uv marker is keyed to a hash of docscanner.py's inline-script header
# so a dependency bump auto-invalidates without manual cache wipe.

set -euo pipefail

UV_CACHE_DIR="${UV_CACHE_DIR:-/var/cache/scanrunner/uv}"
DOCTR_CACHE_DIR="${DOCTR_CACHE_DIR:-/var/cache/scanrunner/doctr}"
DOCSCANNER="${DOCSCANNER:-/docscanner.py}"
SCANNER_USER="${SCANNER_USER:-scanner}"

mkdir -p "$UV_CACHE_DIR" "$DOCTR_CACHE_DIR"

# Production: started as root from the container CMD, so chown the volume
# (which may have come up empty + root-owned) and drop to scanner for every
# subsequent step. Tests: already running unprivileged, so skip both.
if [[ "$(id -u)" == "0" ]]; then
    chown -R "$SCANNER_USER:$SCANNER_USER" "$UV_CACHE_DIR" "$DOCTR_CACHE_DIR"
    as_user() { gosu "$SCANNER_USER" "$@"; }
else
    as_user() { "$@"; }
fi

deps_hash="$(awk '/^# \/\/\/ script/,/^# \/\/\/$/' "$DOCSCANNER" \
             | sha256sum | cut -c1-16)"
uv_marker="$UV_CACHE_DIR/.warmed-${deps_hash}"
doctr_marker="$DOCTR_CACHE_DIR/.warmed"

if [[ ! -f "$uv_marker" ]]; then
    echo "scanrunner: warming uv cache (deps ${deps_hash})" >&2
    rm -f "$UV_CACHE_DIR"/.warmed-*
    as_user "$DOCSCANNER" --help >/dev/null
    as_user touch "$uv_marker"
fi

if [[ ! -f "$doctr_marker" ]]; then
    echo "scanrunner: warming DocTR weights" >&2
    as_user "$DOCSCANNER" warm-models
    as_user touch "$doctr_marker"
fi

if [[ "$(id -u)" == "0" ]]; then
    exec gosu "$SCANNER_USER" /usr/bin/tini -- "$@"
else
    exec "$@"
fi
