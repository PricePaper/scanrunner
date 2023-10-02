FROM alpine:3.18 AS base

LABEL maintainer="Ean J Price <ean@pricepaper.com>"

RUN apk upgrade && \
   /usr/sbin/addgroup -g 1001 scanner &&\
   /usr/sbin/adduser -G scanner -s /bin/sh -D -u 1001 scanner &&\
   mkdir /scanner &&\
   apk add --no-cache \
      tini \
      tesseract-ocr \
      poppler-utils \
      py3-pip \
      py3-magic \
      py3-yaml \
      py3-numpy \
      py3-opencv \
      py3-psutil \
      py3-pillow \
      py3-packaging \
      py3-parsing \
      && pip3 --no-cache -q install pytesseract pdf2image

FROM base

COPY entrypoint.sh docscanner.py  /

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["/entrypoint.sh"]
