#!/bin/sh
# scanrunner v2 entrypoint. tini is PID 1 (set in the Dockerfile); we just
# exec the daemon so signals propagate cleanly. The script is +x and its
# shebang resolves through env to `uv run`.
exec /docscanner.py daemon /scanner -c "${DS_CONFIG}" -s "${DS_SERVER}"
