FROM debian:trixie-slim

LABEL maintainer="Ean J Price <ean@pricepaper.com>"
LABEL org.opencontainers.image.title="scanrunner"
LABEL org.opencontainers.image.description="Invoice scan ingest daemon — clean-room v2.0 rewrite (single-file PEP 723 uv script, watchdog + ProcessPoolExecutor, JSON-RPC over httpx)."

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=/usr/bin/python3.13

# System runtime: tini for PID 1 signals, python3 (Debian trixie ships
# 3.13.5 — no need for uv to download its own), tesseract for OCR,
# libmagic for mime sniffing, libgl1 for opencv-python-headless's image
# codecs, ca-certs for HTTPS-bound JSON-RPC.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        tini \
        python3 \
        tesseract-ocr \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1001 scanner && \
    useradd --uid 1001 --gid 1001 --shell /usr/sbin/nologin -d /scanner -m scanner

# Install uv. The script writes to /usr/local/bin/uv when UV_INSTALL_DIR is set.
RUN curl -fsSL https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR=/usr/local/bin sh && \
    /usr/local/bin/uv --version

# Pre-resolve the script's PEP 723 deps into the uv cache so the first
# daemon start doesn't pay a network round-trip. uv uses the apt-installed
# /usr/bin/python3.13 (UV_PYTHON above) instead of downloading its own.
COPY docscanner.py /docscanner.py
RUN chmod 0755 /docscanner.py && \
    mkdir -p /opt/uv-cache && \
    /docscanner.py --help >/dev/null && \
    chown -R scanner:scanner /opt/uv-cache

COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh

# Default config location; override at runtime with `-e DS_CONFIG=...` or
# bind-mount over /etc/docscanner/config.yaml.
ENV DS_CONFIG=/etc/docscanner/config.yaml \
    DS_SERVER=production

USER scanner
WORKDIR /scanner

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/entrypoint.sh"]
