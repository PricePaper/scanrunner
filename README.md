# ScanRunner OCR Pipeline (OpenCV + Tesseract)

This project reads invoice numbers from scanned images and attaches them to records in Odoo. It is tuned for the new
yellow‑paper invoice format and provides configurable OCR and output pipelines.

## Outputs at a Glance

- OCR image: in‑memory only (not saved). Aggressive preprocessing for accuracy.
- Odoo attachment and error email: high‑quality black & white PNG (1‑bit), denoised, printable, same dimensions as
  original.
- Local archive: the original input file (e.g., JPEG) is moved into the done folder structure; it is no longer
  reformatted.

Default invoice regex: `R?INV(/20[0-9]{2}/[0-9]{4,5})` — examples: `INV/2025/13102` or `RINV/2025/13102`.

---

## Configuration Reference (config.yaml)

This section documents all supported keys, their types, defaults, and effects.

### Root‑level settings

- `retry` (int, default 5): Number of retries for Odoo operations.
- `retry_sleep` (int seconds, default 5): Delay between retries.
- `statistics-file` (string, default `statistics.yaml`): Path to write region hit statistics when `--stats` is used.
- `tesseract-bin` (string): Path to the Tesseract binary.
- `done-path` (string, default `/done`): Subdirectory where processed documents are stored.
- `error-email` (string): Address to receive unreadable-document notifications.
- `error-mail-message` (multiline string): Message body for error emails.

### Servers

Under `servers`, provide one or more environments (e.g., `development`, `production`).

- `url` (string): Odoo server URL.
- `username` (string): Odoo username.
- `password` (string): Odoo password.
- `database` (string): Odoo database name.
- `smtp-server` (string): SMTP host for error emails.
- `smtp-port` (int): SMTP port.
- `smtp-use-tls` (bool): Use STARTTLS.
- `smtp-user` (string): SMTP username.
- `smtp-password` (string): SMTP password.

Security note: Never commit real credentials to VCS; use environment‑specific secrets management in production.

### Per‑document settings

Keys live under `documents.<DocType>` (e.g., `documents.Invoice`). Each document type can override the following:

#### Identification / acceptance

- `file-name-match` (glob): Only process files whose names match this pattern.
- `mime-types` (list[string]): Accepted MIME types.

#### OCR and Tesseract

- `ocr_regex` (regex): Pattern used to extract/validate the document number.
- `ocr_top_half_only` (bool, default False): If true, OCR only scans the top 50% of the page.
- `threshold_region_ignore` (int percent, default 80): Starting threshold for ignoring overly bright regions during
  region scans (used internally; higher is stricter).
- `threshold_region_ignore_min` (int percent, default 40): Minimum threshold floor.
- `threshold_region_ignore_decrement` (int percent, default 20): Step size to relax the ignore threshold if a match
  isn’t found.
- `search_regions` (list of `[x1%, y1%, x2%, y2%]`): Regions to OCR, evaluated in order. Coordinates are percentages of
  page width/height. Example: `[70, 2, 99, 18]` means right‑top box.
    - Backward compatibility: the legacy key `regions` is still accepted but deprecated; prefer `search_regions`.
- `tesseract_config` (string): Passed directly to Tesseract (e.g., `--oem 1 --psm 6 -l eng`).
- `ocr_whitelist` (string, optional): Tesseract character whitelist (e.g., `RINV/0123456789-`).
- `keep_intermediate_ocr_image` (bool, default False): If true, writes the intermediate OCR image to disk for debugging;
  otherwise, OCR image stays in memory only.

#### Odoo/Email black & white output (printable PNG)

- `odoo_storage_bw_method` (string, default `adaptive_gaussian`): Method for binarizing the page for Odoo/email.
  Options:
    - `adaptive_gaussian`: Robust general default; local Gaussian adaptive threshold.
    - `adaptive_mean`: Alternative local mean adaptive threshold.
    - `background_subtract`: Denoise → dilate → median blur → difference from background → hard threshold (threshold set
      via `odoo_bw_threshold`). Good for uneven backgrounds but may require tuning.
    - `background_otsu`: Illumination normalization (divide by smoothed background) then Otsu threshold; robust on
      yellow paper.
- `odoo_bw_use_clahe` (bool, default True): Apply CLAHE on the L channel to normalize illumination before thresholding.
- `odoo_bw_denoise_h` (int, default 10): Strength for `fastNlMeansDenoising` (higher removes more noise but can thin
  strokes).
- `odoo_bw_dilate_kernel` (int pixels, default 7): Kernel size for pre‑threshold dilation (elliptical). Larger
  approximates a smoother background.
- `odoo_bw_median_ksize` (odd int, default 21): Kernel size for median blur when estimating background; must be odd (
  auto‑corrected to next odd).
- `odoo_bw_threshold` (0–255, default 210): Hard threshold used only by `background_subtract`.
- `odoo_bw_open_kernel` (int pixels, default 2): Size for morphological open to remove speckle (0 disables).
- `odoo_bw_open_iterations` (int, default 1): Number of open iterations.
- `odoo_bw_post_dilate_kernel` (int pixels, default 0): Optional dilation after threshold to thicken strokes (0
  disables).
- Saving: Output is always saved as 1‑bit PNG (`mode="1"`) with `storage_png_compress_level` compression and original
  dimensions.
- Guardrails (automatic): The pipeline auto‑corrects polarity to white background and falls back to a safe adaptive
  method if the result is too sparse/dense.

- `storage_png_compress_level` (int 0–9, default 6): PNG compression level for the B&W PNG (higher is smaller but
  slower; 6–9 are typical).

#### Local archival

- The original input file (e.g., JPEG) is moved as-is to the done folder structure. There are no configuration options
  for reformatting or compressing the local archive.

#### Odoo integration

- `odoo_sequence` (string): Prefix sequence used in Odoo (e.g., `INV`).
- `odoo_object` (string): Odoo model for the document (e.g., `account.move`).
- `odoo_attachment_tag_id` (int): Tag ID to apply to attachments.
- `odoo_folder_id` (int): Folder ID to store documents.

---

## Typical Invoice Configuration Example

```yaml
documents:
  Invoice:
    file-name-match: "*Customer_Invoice*"
    mime-types: ["image/jpeg", "image/png"]
    ocr_regex: "R?INV(/20[0-9]{2}/[0-9]{4,5})"
    ocr_top_half_only: Yes
    search_regions:
      - [70, 2, 99, 18]
      - [55, 2, 85, 20]
      - [30, 2, 60, 18]
    tesseract_config: "--oem 1 --psm 6 -l eng"
    ocr_whitelist: "RINV/0123456789-"
    # Local archival (color)
    # Odoo/email B&W PNG
    storage_png_compress_level: 6
    odoo_storage_bw_method: "adaptive_gaussian"
    odoo_bw_use_clahe: Yes
    odoo_bw_denoise_h: 10
    odoo_bw_dilate_kernel: 7
    odoo_bw_median_ksize: 21
    odoo_bw_open_kernel: 2
    odoo_bw_open_iterations: 1
    odoo_bw_post_dilate_kernel: 0
    # Odoo integration
    odoo_sequence: "INV"
    odoo_object: "account.move"
    odoo_attachment_tag_id: 1
    odoo_folder_id: 7
```

## Notes on OCR Regions (Percent Coordinates)

- Coordinates are percentages of the full page: `[x1%, y1%, x2%, y2%]`.
- Values are clamped to page bounds internally. Regions are evaluated in list order; OCR stops at the first regex match.
- Start with a tight top‑right region for invoice number and add 1–2 broader fallbacks.

## Running

- Install dependencies (OpenCV, Tesseract, Pillow, etc.).
- Example: `python docscanner.py -s development -c config.yaml --stats new_invoices/*.jpg`
- Preserve originals: add `--keep` to preserve the original input files in place. By default, after successful
  processing and archiving, the original input file is deleted.
- With `--stats`, a `statistics.yaml` file is updated with region hit counts to help you refine `search_regions`.

## Tests

Tests run against `new_invoices/` and verify:

- OCR succeeds for first pages with ≥99% hit rate.
- Odoo/email output is a 1‑bit PNG with original dimensions.
- Successful runs move the original input file into the appropriate done folder structure; failure runs move the B&W PNG
  into done/unreadable.

## Troubleshooting & Tuning

- If Odoo PNG looks too thin, increase `odoo_bw_post_dilate_kernel` to 1 or 2, or switch method to `background_otsu`.
- If background speckle appears, raise `odoo_bw_open_kernel` to 3 and/or increase `odoo_bw_denoise_h` slightly.
- If OCR misses, widen the first `search_regions` rectangle and ensure `ocr_whitelist` includes all expected characters.
- For large color local files, try `storage_format: webp` with lower `storage_webp_quality` if available; otherwise, set
  `storage_format: jpeg` and lower `storage_jpeg_quality`.
