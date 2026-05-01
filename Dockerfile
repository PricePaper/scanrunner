FROM debian:trixie-slim AS base

LABEL maintainer="Ean J Price <ean@pricepaper.com>"

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    groupadd --gid 1001 scanner && \
    useradd --uid 1001 --gid 1001 --shell /bin/bash -d /scanner -m scanner && \
    apt-get install -y --no-install-recommends \
       tini \
       tesseract-ocr \
       python3-magic \
       python3-yaml \
       python3-numpy \
       python3-opencv \
       python3-psutil \
       python3-pil \
       python3-packaging \
       python3-pyparsing \
       python3-pip && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache -q install --break-system-packages --root-user-action ignore pytesseract

FROM base

COPY entrypoint.sh docscanner.py  /

# Note: The path for tini on Debian is /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["/entrypoint.sh"]
