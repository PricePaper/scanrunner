#!/usr/bin/env bash
# scanrunner app image builder (thin layer over the base).
#
# Produces ${IMAGE_REPO}:3.5-<date> and ${IMAGE_REPO}:latest by adding
# docscanner.py + entrypoint.sh on top of ${IMAGE_REPO}:base-latest. Built
# every code change; the base (apt + uv + gosu + tini) does NOT rebuild.
#
# Caches (uv + DocTR) live in a single host-persistent volume mounted at
# /var/cache/scanrunner. The entrypoint warms them on first run, marker-
# gates subsequent runs, and re-warms uv automatically when docscanner.py's
# inline-deps block changes.
#
# Usage:
#   ./build.sh                 # uses base-latest
#   BASE_TAG=base-3.5-... ./build.sh
#
# Rebuild the base first via ./build-base.sh when apt deps / uv / gosu /
# tini / Python need to change.

set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-registry.digitalocean.com/pricepaper/scanrunner}"
BUILD_DATE="$(date +%Y%m%d%H%M)"
DATED_TAG="${IMAGE_REPO}:3.5-${BUILD_DATE}"
LATEST_TAG="${IMAGE_REPO}:latest"
BASE_TAG="${BASE_TAG:-${IMAGE_REPO}:base-latest}"

cd "$(dirname "$(readlink -f "$0")")"

ctr="$(buildah from "$BASE_TAG")"
trap 'buildah rm "$ctr" >/dev/null 2>&1 || true' EXIT

# App source + entrypoint. Both readable by anyone, executable by all so
# gosu-dropped scanner can run them.
buildah copy --chmod 0755 "$ctr" docscanner.py  /docscanner.py
buildah copy --chmod 0755 "$ctr" entrypoint.sh  /entrypoint.sh

# Image metadata. Note: USER is intentionally left as root — the entrypoint
# chowns the cache volume and then drops to scanner via gosu before exec'ing
# tini as PID 1. The cache volume is the single source of truth for both
# the uv cache and the DocTR weights.
buildah config \
  --env PYTHONUNBUFFERED=1 \
  --env UV_CACHE_DIR=/var/cache/scanrunner/uv \
  --env DOCTR_CACHE_DIR=/var/cache/scanrunner/doctr \
  --env DS_CONFIG=/etc/docscanner/config.yaml \
  --env DS_SERVER=production \
  --env OMP_NUM_THREADS=2 \
  --env MKL_NUM_THREADS=2 \
  --env OPENBLAS_NUM_THREADS=2 \
  --volume /var/cache/scanrunner \
  --workingdir /scanner \
  --entrypoint '["/entrypoint.sh"]' \
  --cmd '["/docscanner.py","daemon","/scanner"]' \
  --label "maintainer=Ean J Price <ean@pricepaper.com>" \
  --label "org.opencontainers.image.title=scanrunner" \
  --label "org.opencontainers.image.description=Invoice scan ingest daemon (DocTR + layered v3)." \
  --label "org.opencontainers.image.version=3.5-${BUILD_DATE}" \
  --label "org.opencontainers.image.source=https://github.com/PricePaper/scanrunner" \
  --label "org.opencontainers.image.base.name=${BASE_TAG}" \
  "$ctr"

buildah commit --rm "$ctr" "$DATED_TAG"
trap - EXIT
buildah tag "$DATED_TAG" "$LATEST_TAG"

size_bytes="$(podman image inspect --format '{{.Size}}' "$DATED_TAG")"
size_mb="$(awk -v b="$size_bytes" 'BEGIN { printf "%.0f", b/1024/1024 }')"
n_layers="$(podman image inspect --format '{{len .RootFS.Layers}}' "$DATED_TAG")"

cat <<EOF

Built ${DATED_TAG}
       ${LATEST_TAG}
Base:    ${BASE_TAG}
Image:   ${size_mb} MB, ${n_layers} layer(s)

Push:
  podman push ${DATED_TAG}
  podman push ${LATEST_TAG}
EOF
