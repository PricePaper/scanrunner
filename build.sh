#!/usr/bin/env bash
# scanrunner v2 image builder.
#
# Builds a single-layer (squashed) container image, tags it with both a
# date-stamped tag and `latest`, and prints the push commands. Designed to
# be re-run idempotently — produces a fresh date tag each call.
#
# Layout choices for efficiency:
#   * One squashed layer at commit time → smallest possible image.
#   * `curl` is installed only to fetch uv, then purged before commit.
#   * apt cache + /root/.cache wiped in the same RUN that uses them.
#   * `buildah copy --chmod` sets perms inline (no extra layer for chmod).
#   * Pre-warms the uv cache against /usr/bin/python3.13 so first daemon
#     start does no network I/O.
#
# Usage:
#   ./build.sh
#
# Override registry / image base via env if needed:
#   IMAGE_REPO=registry.example/team/scanrunner ./build.sh

set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-registry.digitalocean.com/pricepaper/scanrunner}"
BUILD_DATE="$(date +%Y%m%d%H%M)"
DATED_TAG="${IMAGE_REPO}:3.5-${BUILD_DATE}"
LATEST_TAG="${IMAGE_REPO}:latest"
BASE="debian:trixie-slim"

cd "$(dirname "$(readlink -f "$0")")"

ctr="$(buildah from --pull=newer "$BASE")"
trap 'buildah rm "$ctr" >/dev/null 2>&1 || true' EXIT

# 1. Runtime apt deps + scanner user.
buildah run "$ctr" -- bash -c '
  set -eux
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
      tini python3 libmagic1 libgl1 libglib2.0-0 \
      ca-certificates curl
  groupadd --gid 1001 scanner
  useradd --uid 1001 --gid 1001 --shell /usr/sbin/nologin -d /scanner -m scanner
'

# 2. Install uv (needs curl), then purge curl (build-only).
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

# 3. App + uv cache warm-up + perms.
# DocTR weights (~124 MB) are NOT preloaded — they download to
# /opt/doctr-cache on first inference. Mount a persistent volume at
# /opt/doctr-cache (or run `docscanner.py warm-models` once after start)
# to avoid re-downloading across container restarts.
buildah copy --chmod 0755 "$ctr" docscanner.py /docscanner.py
buildah run "$ctr" -- bash -c '
  set -eux
  mkdir -p /opt/uv-cache /opt/doctr-cache
  # Pull every uv-managed wheel (torch CPU build, opencv, doctr, …)
  # into /opt/uv-cache. --help exits before any model load.
  UV_CACHE_DIR=/opt/uv-cache UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=/usr/bin/python3.13 /docscanner.py --help >/dev/null
  chown -R scanner:scanner /opt/uv-cache /opt/doctr-cache
'

# 4. Image metadata.
buildah config \
  --env DEBIAN_FRONTEND=noninteractive \
  --env PYTHONUNBUFFERED=1 \
  --env UV_CACHE_DIR=/opt/uv-cache \
  --env UV_PYTHON_PREFERENCE=only-system \
  --env UV_PYTHON=/usr/bin/python3.13 \
  --env DOCTR_CACHE_DIR=/opt/doctr-cache \
  --env DS_CONFIG=/etc/docscanner/config.yaml \
  --env DS_SERVER=production \
  --env OMP_NUM_THREADS=2 \
  --env MKL_NUM_THREADS=2 \
  --env OPENBLAS_NUM_THREADS=2 \
  --volume /opt/doctr-cache \
  --user scanner \
  --workingdir /scanner \
  --cmd '["/usr/bin/tini","--","/docscanner.py","daemon","/scanner"]' \
  --label "maintainer=Ean J Price <ean@pricepaper.com>" \
  --label "org.opencontainers.image.title=scanrunner" \
  --label "org.opencontainers.image.description=Invoice scan ingest daemon (DocTR + layered v3)." \
  --label "org.opencontainers.image.version=3.5-${BUILD_DATE}" \
  --label "org.opencontainers.image.source=https://github.com/PricePaper/scanrunner" \
  "$ctr"

# 5. Squash to a single layer, commit under the dated tag, then alias `latest`.
buildah commit --rm --squash "$ctr" "$DATED_TAG"
trap - EXIT
buildah tag "$DATED_TAG" "$LATEST_TAG"

# Report.
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
