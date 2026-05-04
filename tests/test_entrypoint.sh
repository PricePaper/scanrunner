#!/usr/bin/env bash
# Behavioral tests for /entrypoint.sh.
#
# Runs the entrypoint as a normal user with a fake docscanner.py, fake cache
# dirs, and asserts marker-file gating: cold start warms, hot start skips,
# dep-block change re-warms uv only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENTRYPOINT="$REPO_ROOT/entrypoint.sh"

if [[ ! -x "$ENTRYPOINT" ]]; then
    echo "FAIL: $ENTRYPOINT missing or not executable"
    exit 1
fi

setup() {
    TMPDIR=$(mktemp -d)
    export UV_CACHE_DIR="$TMPDIR/uv"
    export DOCTR_CACHE_DIR="$TMPDIR/doctr"
    export DOCSCANNER="$TMPDIR/docscanner.py"
    export WARM_LOG="$TMPDIR/warm.log"
    : > "$WARM_LOG"

    # Fake docscanner: appends a marker line to $WARM_LOG describing how it
    # was invoked. The warm steps and the final exec all flow through this,
    # so the log is a complete record of what the entrypoint decided to do.
    cat > "$DOCSCANNER" <<'PYEOF'
#!/usr/bin/env bash
case "$1" in
    --help)      echo "WARMED_UV"    >> "$WARM_LOG" ;;
    warm-models) echo "WARMED_DOCTR" >> "$WARM_LOG" ;;
    *)           echo "RAN: $*"      >> "$WARM_LOG" ;;
esac
PYEOF
    chmod +x "$DOCSCANNER"

    # Inline-deps block — the entrypoint hashes this to key the uv marker.
    cat >> "$DOCSCANNER" <<'PYEOF'
# /// script
# requires-python = ">=3.13"
# dependencies = ["pkg-a"]
# ///
PYEOF
}

teardown() {
    rm -rf "$TMPDIR"
}

assert_log() {
    local pattern="$1" expectation="$2" label="$3"
    if [[ "$expectation" == "present" ]]; then
        if ! grep -qx "$pattern" "$WARM_LOG"; then
            echo "FAIL ($label): expected '$pattern' in log:"
            cat "$WARM_LOG"
            exit 1
        fi
    else
        if grep -qx "$pattern" "$WARM_LOG"; then
            echo "FAIL ($label): unexpected '$pattern' in log:"
            cat "$WARM_LOG"
            exit 1
        fi
    fi
}

# --- test 1: cold start warms both caches and forwards args ----------------
setup
"$ENTRYPOINT" "$DOCSCANNER" daemon /scanner >/dev/null
assert_log "WARMED_UV"        present "test 1: cold uv warm"
assert_log "WARMED_DOCTR"     present "test 1: cold doctr warm"
assert_log "RAN: daemon /scanner" present "test 1: forwards args to CMD"
teardown
echo "PASS: cold start warms both caches and forwards CMD"

# --- test 2: second start with warm caches skips both warms ----------------
setup
"$ENTRYPOINT" "$DOCSCANNER" daemon /scanner >/dev/null  # cold start
: > "$WARM_LOG"                                         # reset log
"$ENTRYPOINT" "$DOCSCANNER" daemon /scanner >/dev/null  # warm start
assert_log "WARMED_UV"        absent  "test 2: skip uv warm on warm start"
assert_log "WARMED_DOCTR"     absent  "test 2: skip doctr warm on warm start"
assert_log "RAN: daemon /scanner" present "test 2: still forwards CMD"
teardown
echo "PASS: warm cache skips re-warm"

# --- test 3: dep-block change invalidates uv marker only -------------------
setup
"$ENTRYPOINT" "$DOCSCANNER" daemon /scanner >/dev/null  # cold start
: > "$WARM_LOG"
sed -i 's/pkg-a/pkg-b/' "$DOCSCANNER"                   # bump deps
"$ENTRYPOINT" "$DOCSCANNER" daemon /scanner >/dev/null
assert_log "WARMED_UV"    present "test 3: dep change re-warms uv"
assert_log "WARMED_DOCTR" absent  "test 3: dep change leaves doctr alone"
teardown
echo "PASS: dep block change invalidates uv marker only"

echo "ALL PASS"
