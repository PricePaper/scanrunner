FROM alpine:3.20 AS base

LABEL maintainer="Ean J Price <ean@pricepaper.com>"

RUN apk upgrade && \
   /usr/sbin/addgroup -g 1001 scanner &&\
   /usr/sbin/adduser -G scanner -s /bin/sh -D -u 1001 scanner &&\
   mkdir /scanner &&\
   apk add --no-cache \
      tini \
      tesseract-ocr \
      py3-pip \
      py3-magic \
      py3-yaml \
      py3-numpy \
      py3-opencv \
      py3-psutil \
      py3-pillow \
      py3-packaging \
      py3-parsing &&\
      pip3 --no-cache -q install --break-system-packages pytesseract

FROM base

COPY entrypoint.sh docscanner.py  /

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["/entrypoint.sh"]
