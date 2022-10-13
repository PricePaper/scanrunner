FROM alpine:3.16

LABEL maintainer="Ean J Price <ean@pricepaper.com>"

RUN apk upgrade && \
   addgroup -g 1001 scanner &&\
   adduser -G scanner -s /bin/sh -D -u 1001 scanner &&\
   mkdir /scanner &&\
   apk add --no-cache \
      tini \
      su-exec \
      tesseract-ocr \
      py3-pip \
      py3-yaml \
      py3-numpy \
      py3-opencv \
      py3-psutil \
      py3-pillow \
      py3-packaging \
      py3-parsing &&\
      pip3 --no-cache -q install pytesseract

COPY entrypoint.sh docscanner.py config.yaml /

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["/entrypoint.sh"]
