# scanrunner v2.0

Watch-and-process daemon for scanned invoices (and, eventually, other
document types). Reads scans from a watched inbox, OCRs the document
number, attaches a cleaned reproduction-quality image to the matching
Odoo record, and archives the result under a structured tree.

A single-file Python script: `docscanner.py`. Inline PEP 723 dependency
metadata; `uv` brings the interpreter and resolves deps. All production
code is OOP; the only top-level callable is `if __name__ == "__main__"`.

Designed to run as a tini-supervised container daemon. Designed for
extension — new document types subclass `DocumentType` and register with
`DocumentTypeRegistry`.

## What it does

1. **Watches** an inbox directory via `watchdog` (`IN_CLOSE_WRITE`).
2. **Submits** each new file to a `ProcessPoolExecutor` (default
   `nproc - 1` workers — desktop tools like the calibration sweep cap
   at `nproc // 3` instead). The pool drains cleanly on SIGTERM.
3. **Classifies** by filename glob + magic-detected MIME type.
4. **Preprocesses** per type. `Invoice` strips the yellow paper cast.
5. **OCRs** with Tesseract on the per-type search regions in priority
   order. First regex hit wins; an `R?INV/...` regex naturally rejects
   `SO/...` source-document numbers.
6. **Looks up** the Odoo record by name via JSON-RPC over `httpx`
   (HTTP/2 when available; never `xmlrpc.client`).
7. **Encodes** a storage-grade image with calibrated, fixed encoder
   settings — pure-white background, faithful grayscale foreground,
   200 dpi, JPEG q=85 ≤ 300 KB or 8-bit grayscale PNG fallback.
8. **Attaches** to the Odoo record and links to a `documents.document`
   in the configured folder with the configured tag.
9. **Archives** the cleaned bytes to
   `{inbox}/done/{TYPE}/{YYYY}/{N//100*100:05d}/{TYPE}-{YYYY}-{N:05d}_id-{odoo_id}_aid-{aid}_{orig_stem}.{ext}`.
10. **Records** a SHA-256 in a SQLite ledger so a re-drop is a no-op.
11. **Routes failures** to `{inbox}/done/unreadable/` and (if
    configured) emails the office.

## Quick start

```bash
# Smoke run on the host (requires uv, tesseract, libmagic).
DS_DEV_PASSWORD='...'
uv run --script docscanner.py daemon /path/to/inbox \
    -c ./config.yaml -s development
```

Container (recommended for production):

```bash
podman build -t scanrunner:dev .

# Bind-mount the Samba inbox at /scanner and the config at
# /etc/docscanner/. --userns=keep-id maps the host user to the container's
# `scanner` (uid 1001) so bind-mount permissions match. --network host is
# the simplest way to reach the Odoo server on the same host; in production
# use proper container networking.
podman run -d --name scanrunner \
    --userns=keep-id:uid=1001,gid=1001 \
    --network host \
    -e DS_SERVER=production \
    -e DS_PROD_PASSWORD='...' \
    -e DS_PROD_SMTP_PASSWORD='...' \
    -v /srv/scanner-inbox:/scanner:Z \
    -v /etc/docscanner:/etc/docscanner:Z,ro \
    scanrunner:dev
```

Config secrets use shell-style `${VAR}` interpolation, so `config.yaml`
stays committable while passwords live in env / a secrets manager.

## CLI

```
docscanner daemon <inbox>          # watch + process forever
docscanner process <file> <inbox>  # one-shot — process a single file
```

Both subcommands accept `-c <config>` (default `/etc/docscanner/config.yaml`
or `$DS_CONFIG`), `-s <server>` (default `production` or `$DS_SERVER`),
and `-v` for verbose logging.

## Storage encoder calibration

The storage image's encoder settings are **fixed**, not adjustable at
runtime. They were calibrated against the 168-sample population in
`inv/good/` and baked into `Encoder` / `StoragePreparer`:

- Downsample to 200 dpi (≈2/3 of the 300 dpi source).
- JPEG quality 85, grayscale, progressive — used when ≤ 300 KB.
- Otherwise PNG, grayscale, `optimize=True compress_level=9`.

Re-run `calibrate_storage.py inv/good/` if the substrate or scanner DPI
shifts; pick the new winner and bake it back into `Encoder` constants.

## Tests

Test invocation pins Python 3.13 explicitly so production-target
features are validated:

```bash
SCANRUNNER_TEST_ODOO_LOGIN='ean@pricepaper.com' \
SCANRUNNER_TEST_ODOO_PASSWORD='...' \
uv run --python 3.13 --with-requirements docscanner.py --with pytest \
    pytest tests/ -k "not test_all_samples"
```

`test_all_samples_produce_valid_output` runs the storage pipeline on
all 168 calibration samples sequentially (~7 min). Filter it out for
fast iterations.

All tests run against real artifacts (real images, the live
`pp-odoo-test` harness on `127.0.0.1:58069`, real SQLite) — no mocks.

## Adding a new document type

1. Subclass `DocumentType` in `docscanner.py`.
2. Provide `name`, `config`, `regex`, and override `preprocess` if the
   substrate needs cleanup (yellow paper, thermal fade, carbon-copy blue).
3. Register in `DocumentTypeRegistry.from_config`'s `match` block.
4. Add a `documents.<TypeName>` entry to `config.yaml` with its
   `file-name-match`, `mime-types`, `ocr_regex`, `search_regions`, and
   Odoo identifiers.
