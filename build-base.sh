#!/usr/bin/env bash
# scanrunner base image builder.
#
# Produces ${IMAGE_REPO}:base-3.5-<date> and ${IMAGE_REPO}:base-latest. The
# base contains everything that doesn't change with the application source:
# debian apt deps, gosu, tini, system Python 3.13, and the uv binary. No
# docscanner.py and no warmed caches — those move to the thin app layer
# (build.sh) and to the runtime entrypoint, respectively.
#
# Rebuild this image only when bumping Debian, Python, gosu, tini, or uv.
# Code changes go through build.sh and produce a small layer on top.
#
# Usage:
#   ./build-base.sh
#
# Override registry / image base via env if needed:
#   IMAGE_REPO=registry.example/team/scanrunner ./build-base.sh

set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-registry.digitalocean.com/pricepaper/scanrunner}"
BUILD_DATE="$(date +%Y%m%d%H%M)"
DATED_TAG="${IMAGE_REPO}:base-3.5-${BUILD_DATE}"
LATEST_TAG="${IMAGE_REPO}:base-latest"
BASE="debian:trixie-slim"

cd "$(dirname "$(readlink -f "$0")")"

ctr="$(buildah from --pull=newer "$BASE")"
trap 'buildah rm "$ctr" >/dev/null 2>&1 || true' EXIT

# Runtime apt deps + scanner user. gosu drops privileges by exec()ing the
# target (no fork/wait, unlike runuser+PAM), letting tini land at PID 1.
buildah run "$ctr" -- bash -c '
  set -eux
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
      tini gosu python3 libmagic1 libgl1 libglib2.0-0 \
      ca-certificates curl
  groupadd --gid 1001 scanner
  useradd --uid 1001 --gid 1001 --shell /usr/sbin/nologin -d /scanner -m scanner
'

# Install uv (needs curl), then purge curl (build-only).
buildah run "$ctr" -- bash -c '
  set -eux
  export DEBIAN_FRONTEND=noninteractive
  curl -fsSL https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh
  /usr/local/bin/uv --version
  apt-get purge -y curl
  apt-get autoremove -y
  apt-get clean
  rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*
'

buildah config \
  --env DEBIAN_FRONTEND=noninteractive \
  --env PYTHONUNBUFFERED=1 \
  --env UV_PYTHON_PREFERENCE=only-system \
  --env UV_PYTHON=/usr/bin/python3.13 \
  --label "maintainer=Ean J Price <ean@pricepaper.com>" \
  --label "org.opencontainers.image.title=scanrunner-base" \
  --label "org.opencontainers.image.description=Base image for scanrunner: Debian trixie + Python 3.13 + uv + gosu + tini." \
  --label "org.opencontainers.image.version=base-3.5-${BUILD_DATE}" \
  --label "org.opencontainers.image.source=https://github.com/PricePaper/scanrunner" \
  "$ctr"

buildah commit --rm --squash "$ctr" "$DATED_TAG"
trap - EXIT
buildah tag "$DATED_TAG" "$LATEST_TAG"

size_bytes="$(podman image inspect --format '{{.Size}}' "$DATED_TAG")"
size_mb="$(awk -v b="$size_bytes" 'BEGIN { printf "%.0f", b/1024/1024 }')"
n_layers="$(podman image inspect --format '{{len .RootFS.Layers}}' "$DATED_TAG")"

cat <<EOF

Built ${DATED_TAG}
       ${LATEST_TAG}

Image:   ${size_mb} MB, ${n_layers} layer(s)

Push:
  podman push ${DATED_TAG}
  podman push ${LATEST_TAG}
EOF
