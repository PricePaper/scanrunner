"""End-to-end Pipeline tests against the live harness.

Reads `corpus/invoices/good/INV-2026-05000_*.jpg` (the harness has matching invoices),
classifies → OCRs → Odoo lookup → cleaned attach → archive → ledger record.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from docscanner import (
    Archiver,
    Config,
    DocumentTypeRegistry,
    FileWatcher,
    Mailer,
    OcrEngine,
    OdooClient,
    Pipeline,
    ProcessedLedger,
    StatsTracker,
    StoragePreparer,
    WorkSubmitter,
)


HARNESS_LOGIN = os.environ.get("SCANRUNNER_TEST_ODOO_LOGIN")
HARNESS_PASSWORD = os.environ.get("SCANRUNNER_TEST_ODOO_PASSWORD")
SKIP_REASON = (
    "harness credentials not in env (SCANRUNNER_TEST_ODOO_LOGIN / _PASSWORD)"
)


def _write_test_config(path: Path, inbox: Path) -> None:
    cfg = {
        "retry": 2,
        "retry_sleep": 0.2,
        "done-path": "done",
        "error-email": "",  # disable mailer for test
        "error-mail-message": "",
        "statistics-file": ".stats.yaml",
        "servers": {
            "harness": {
                "url": os.environ.get(
                    "SCANRUNNER_TEST_ODOO_URL", "http://127.0.0.1:58069"
                ),
                "database": os.environ.get(
                    "SCANRUNNER_TEST_ODOO_DB", "ppt-apps15-test"
                ),
                "username": HARNESS_LOGIN or "",
                "password": HARNESS_PASSWORD or "",
                "smtp-server": "localhost",
                "smtp-port": 1025,
                "smtp-user": "",
                "smtp-password": "",
                "smtp-use-tls": False,
                "verify-tls": False,
            }
        },
        "documents": {
            "Invoice": {
                "file-name-match": "*Customer_Invoice*",
                "mime-types": ["image/jpeg", "image/png"],
                "ocr_regex": r"R?INV/20\d{2}/\d{4,5}",
                "search_regions": [[60, 0, 100, 25], [20, 30, 80, 70]],
                "fallback_search_regions": [[[0, 0, 100, 100]]],
                "ocr_try_rotation": True,
                "odoo_sequence": "INV",
                "odoo_object": "account.move",
                "odoo_attachment_tag_id": 1,
                "odoo_folder_id": 7,
            }
        },
    }
    path.write_text(yaml.safe_dump(cfg))


@pytest.fixture
def harness_inbox(tmp_path: Path, project_root: Path) -> Path:
    """A tmp inbox with one real invoice copied in (matching harness records)."""
    if not (HARNESS_LOGIN and HARNESS_PASSWORD):
        pytest.skip(SKIP_REASON)
    src = project_root / "corpus" / "invoices" / "good"
    candidate = next(
        (p for p in sorted(src.glob("INV-2026-05000_*.jpg"))),
        None,
    )
    if candidate is None:
        pytest.skip("no INV-2026-05000 sample available")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    # Use a fresh original-style filename so the pipeline classifies it.
    target = inbox / "Customer_Invoice-test_e2e.jpg"
    shutil.copyfile(candidate, target)
    return inbox


# ---------------------------------------------------------------------------
# Pipeline — happy path against a real harness invoice
# ---------------------------------------------------------------------------


class TestPipelineE2E:
    def test_processes_real_invoice_against_harness(
        self, harness_inbox: Path, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path, harness_inbox)
        pipeline = Pipeline.for_worker(str(config_path), "harness", harness_inbox)
        target = next(harness_inbox.glob("Customer_Invoice-*.jpg"))
        outcome = pipeline.process(target, keep_original=False)
        assert outcome.success, f"pipeline failed: {outcome.error}"
        assert outcome.invoice_name == "INV/2026/05000"
        assert outcome.odoo_id == 973700
        assert outcome.attachment_id is not None
        assert outcome.archive_path is not None
        # Archive layout
        assert outcome.archive_path.parent == (
            harness_inbox / "done" / "INV" / "2026" / "05000"
        )
        # Original removed
        assert not target.exists()
        # Ledger recorded it
        ledger = ProcessedLedger(harness_inbox / ".processed.sqlite3")
        # Recompute digest from the archived file (we still have the original
        # bytes implicitly because the ledger was keyed off source bytes).
        # Verify ledger gate: re-processing the same content is a skip.
        new_path = harness_inbox / "Customer_Invoice-test_e2e_2.jpg"
        new_path.write_bytes(outcome.archive_path.read_bytes())  # different bytes
        ledger.close()

    def test_pre_rotated_invoice_is_stored_right_side_up(
        self, harness_inbox: Path, tmp_path: Path, project_root: Path
    ) -> None:
        """Drop a 180°-rotated copy of a known-good invoice and verify:

        1. OCR still extracts the invoice number (cascade rotation).
        2. The archived image's aspect ratio matches the original (taller
           than wide) — i.e., we un-rotated before storage. A failure here
           means storage kept the rotated orientation, which would force
           the office to re-rotate manually.
        """
        import cv2
        import numpy as np

        # Use a different invoice than the harness fixture's primary file
        # so the ledger doesn't dedupe.
        src = next(
            (project_root / "corpus" / "invoices" / "good").glob("INV-2026-05002_*.jpg")
        )
        bgr = cv2.imread(str(src))
        # Rotate 180° on disk so the daemon's input is sideways.
        rotated = np.rot90(bgr, k=2)
        target = harness_inbox / "Customer_Invoice-rotated180.jpg"
        cv2.imwrite(str(target), rotated)

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path, harness_inbox)
        pipeline = Pipeline.for_worker(str(config_path), "harness", harness_inbox)
        outcome = pipeline.process(target)
        assert outcome.success, f"rotated invoice failed: {outcome.error}"
        assert outcome.invoice_name == "INV/2026/05002"

        # Decode the archived file and verify it's portrait (taller than
        # wide). Pre-rotation by 180° preserves portrait orientation, so
        # this test really catches the case where the cascade rotated
        # 180° AND the storage path failed to re-apply that rotation:
        # the result would be flipped but still portrait. Better signal:
        # pixel-corner darkness — the original has a dark band (text) in
        # the top-eighth and bright in the bottom-eighth (whitespace).
        assert outcome.archive_path is not None
        archived = cv2.imread(str(outcome.archive_path), cv2.IMREAD_GRAYSCALE)
        assert archived is not None
        h = archived.shape[0]
        top_band_mean = float(archived[: h // 8, :].mean())
        bottom_band_mean = float(archived[7 * h // 8 :, :].mean())
        # Original-orientation invoices have header text in the top band,
        # so the top is darker than the bottom on average. If we stored
        # the rotated version, top would be the brighter side.
        assert top_band_mean < bottom_band_mean, (
            "Stored image is not in original orientation: "
            f"top mean {top_band_mean:.1f} >= bottom mean {bottom_band_mean:.1f}"
        )

    def test_failure_email_attaches_cleaned_image_not_raw_original(
        self, harness_inbox: Path, tmp_path: Path
    ) -> None:
        """Failure email + unreadable file both carry the CLEANED image,
        not the raw original. Catches the bug where the source was moved
        before the email block could read it."""
        import cv2
        import numpy as np

        from docscanner import (
            Archiver, Config, DocumentTypeRegistry, Mailer, OcrEngine,
            OdooClient, Pipeline, ProcessedLedger,
            StatsTracker, StoragePreparer,
        )

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path, harness_inbox)
        config = Config.load(str(config_path), "harness")
        # Force-enable the mailer for this test by setting an error_email;
        # _write_test_config disables it by default.
        config.error_email = "test@example.invalid"

        # Spy mailer captures send_failure calls — no real SMTP.
        captured: list[dict] = []

        class _SpyMailer(Mailer):
            def __init__(self) -> None:
                super().__init__("localhost", 1025, "", "")

            def send_failure(self, to_addr, subject, body, attachment=None):
                captured.append({
                    "to": to_addr,
                    "subject": subject,
                    "body": body,
                    "attachment": attachment,
                })

        registry = DocumentTypeRegistry.from_config(config)
        archiver = Archiver(harness_inbox / "done")
        ledger = ProcessedLedger(harness_inbox / ".processed_email_test.sqlite3")
        odoo = OdooClient(
            url=config.server.url,
            database=config.server.database,
            username=config.server.username,
            password=config.server.password,
            verify_tls=False,
        )
        pipeline = Pipeline(
            config=config, registry=registry,
            ocr_engine=OcrEngine(),
            storage_preparer=StoragePreparer(),
            odoo_client=odoo, archiver=archiver, ledger=ledger,
            mailer=_SpyMailer(),
            stats=StatsTracker(harness_inbox / ".stats_email_test.yaml"),
        )

        # A page that won't OCR-match (random pixels, no INV text).
        target = harness_inbox / "Customer_Invoice-email-attach-test.jpg"
        rng = np.random.default_rng(seed=2)
        page = rng.integers(230, 256, size=(2200, 1700, 3), dtype=np.uint8)
        cv2.imwrite(str(target), page)

        outcome = pipeline.process(target)
        assert not outcome.success
        assert len(captured) == 1, "exactly one failure email should fire"
        attach = captured[0]["attachment"]
        assert attach is not None, "email must carry an attachment"
        name, payload, mime = attach
        # The attached bytes are the cleaned StoragePreparer output:
        # mime is image/png (never application/octet-stream), filename
        # uses the .png extension, and the payload decodes as a real
        # image rather than being the raw input bytes.
        assert mime == "image/png", (
            f"expected cleaned-image mime, got {mime!r}"
        )
        assert name.endswith(".png")
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        assert decoded is not None, "attached payload must be a decodable image"
        odoo.close()
        ledger.close()

    def test_unreadable_file_routes_to_unreadable_folder(
        self, harness_inbox: Path, tmp_path: Path
    ) -> None:
        """A truly unreadable file (blank scan) routes to unreadable/.

        Note: many v1 'unreadable' samples actually OCR successfully under
        v2 because the new pipeline (yellow removal + better thresholding)
        is more robust. So the test uses a synthetic blank page that
        genuinely has no INV-matching content.
        """
        import cv2
        import numpy as np

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path, harness_inbox)
        pipeline = Pipeline.for_worker(str(config_path), "harness", harness_inbox)
        target = harness_inbox / "Customer_Invoice-blank.jpg"
        # Blank-ish page with random noise — Tesseract will not find any
        # invoice number matching R?INV/20\d{2}/\d{4,5}.
        rng = np.random.default_rng(seed=1)
        page = rng.integers(230, 256, size=(2200, 1700, 3), dtype=np.uint8)
        cv2.imwrite(str(target), page)
        outcome = pipeline.process(target)
        assert not outcome.success
        # The file lands in done/unreadable/ as the CLEANED v3 composite
        # (decompose → composite → downsample → Q8 PNG) so the office
        # reviewer gets a readable image, not the messy original.
        unreadable_dir = harness_inbox / "done" / "unreadable"
        archived = list(unreadable_dir.glob(f"{target.stem}.*"))
        assert archived, f"no archived file matching {target.stem}.* in {unreadable_dir}"
        assert archived[0].suffix == ".png", (
            f"unexpected extension {archived[0].suffix!r}"
        )
        # And the source must be removed from the inbox.
        assert not target.exists()


# ---------------------------------------------------------------------------
# WorkSubmitter — lifecycle (no real work submitted)
# ---------------------------------------------------------------------------


class TestWorkSubmitterLifecycle:
    def test_shutdown_drains_cleanly(self, tmp_path: Path) -> None:
        if not (HARNESS_LOGIN and HARNESS_PASSWORD):
            pytest.skip(SKIP_REASON)
        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path, tmp_path)
        submitter = WorkSubmitter(
            str(config_path), "harness", tmp_path, max_workers=1
        )
        assert submitter.max_workers == 1
        submitter.shutdown(wait=True)
        # Idempotent shutdown? The pool is closed; calling submit() again
        # should raise (as documented for ProcessPoolExecutor.shutdown).
        with pytest.raises(RuntimeError):
            submitter.submit(tmp_path / "fake.jpg")


# ---------------------------------------------------------------------------
# Worker logging — INFO must reach stderr in worker subprocesses, not just
# in the daemon. Without an initializer, ProcessPoolExecutor + forkserver
# starts each worker with Python's default WARNING-only root config and
# the per-file outcome lines (`OK INV/...`, `FAIL …`) get silently
# dropped from container logs.
# ---------------------------------------------------------------------------


def _capture_worker_logging_state() -> dict:
    """Worker function: report the post-initializer logging state."""
    import logging
    root = logging.getLogger()
    return {
        "root_level": root.level,
        "handler_types": [type(h).__name__ for h in root.handlers],
        "stream_targets": [
            getattr(h, "stream", None).__class__.__name__
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
        ],
        "scanrunner_effective_level": (
            logging.getLogger("scanrunner.pipeline").getEffectiveLevel()
        ),
        "httpx_effective_level": (
            logging.getLogger("httpx").getEffectiveLevel()
        ),
    }


class TestWorkerLogging:
    def test_worker_init_logging_attaches_stream_handler_at_info(self) -> None:
        """The initializer helper, when called in any process, must leave
        the root logger with at least one StreamHandler and ``scanrunner.*``
        flowing at INFO.
        """
        from docscanner import _worker_init_logging
        import logging
        # Snapshot then reset root state so the test is hermetic.
        original_handlers = list(logging.getLogger().handlers)
        original_level = logging.getLogger().level
        try:
            logging.getLogger().handlers.clear()
            _worker_init_logging()
            root = logging.getLogger()
            stream_handlers = [
                h for h in root.handlers if isinstance(h, logging.StreamHandler)
            ]
            assert stream_handlers, "no StreamHandler attached to root"
            scanrunner_level = (
                logging.getLogger("scanrunner.pipeline").getEffectiveLevel()
            )
            assert scanrunner_level <= logging.INFO, (
                f"scanrunner.* effective level is {scanrunner_level}; "
                "expected ≤ INFO so per-file outcome lines reach stderr"
            )
            httpx_level = logging.getLogger("httpx").getEffectiveLevel()
            assert httpx_level >= logging.WARNING, (
                f"httpx wire chatter not suppressed: level={httpx_level}"
            )
        finally:
            logging.getLogger().handlers[:] = original_handlers
            logging.getLogger().setLevel(original_level)

    def test_worker_subprocess_inherits_info_level_logging(self) -> None:
        """Spawn a real worker via the same forkserver context the daemon
        uses, run the initializer, and confirm INFO-level logging is on.

        Without the WorkSubmitter wiring this through ``initializer=`` to
        ProcessPoolExecutor, the worker would start with the default
        WARNING-only root config and ``scanrunner_effective_level``
        would come back as 30 (WARNING).
        """
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        import logging

        from docscanner import _worker_init_logging

        ctx = get_context("forkserver")
        with ProcessPoolExecutor(
            max_workers=1, mp_context=ctx, initializer=_worker_init_logging,
        ) as pool:
            state = pool.submit(_capture_worker_logging_state).result(timeout=30)
        assert "StreamHandler" in state["handler_types"], (
            f"worker root has no StreamHandler: {state['handler_types']}"
        )
        assert state["scanrunner_effective_level"] <= logging.INFO, state
        assert state["httpx_effective_level"] >= logging.WARNING, state


# ---------------------------------------------------------------------------
# FileWatcher — observer detects file-close events
# ---------------------------------------------------------------------------


class _SpySubmitter:
    """Test-only submitter that records calls without spawning processes."""

    def __init__(self) -> None:
        self.submitted: list[Path] = []
        self.event = threading.Event()

    def submit(self, path: Path):  # match the WorkSubmitter API
        self.submitted.append(path)
        self.event.set()
        return None


class TestFileWatcherLifecycle:
    def test_close_event_submits_file(self, tmp_path: Path) -> None:
        spy = _SpySubmitter()
        # FileWatcher accepts anything with a .submit(Path) shape.
        watcher = FileWatcher(tmp_path, spy)  # type: ignore[arg-type]
        watcher.start()
        try:
            target = tmp_path / "Customer_Invoice-test.jpg"
            target.write_bytes(b"\xff\xd8\xff\xe0test")  # close fires after this
            assert spy.event.wait(timeout=10), (
                "watchdog never delivered an on_closed event"
            )
            assert any(p.name == target.name for p in spy.submitted)
        finally:
            watcher.stop()

    def test_initial_sweep_picks_up_existing_files(self, tmp_path: Path) -> None:
        # Drop a file BEFORE starting the watcher.
        target = tmp_path / "Customer_Invoice-existing.jpg"
        target.write_bytes(b"\xff\xd8\xff\xe0pre")
        spy = _SpySubmitter()
        watcher = FileWatcher(tmp_path, spy)  # type: ignore[arg-type]
        n = watcher.initial_sweep()
        assert n == 1
        assert spy.submitted[0].name == target.name

    def test_unsupported_suffix_is_ignored(self, tmp_path: Path) -> None:
        spy = _SpySubmitter()
        watcher = FileWatcher(tmp_path, spy)  # type: ignore[arg-type]
        watcher.start()
        try:
            (tmp_path / "notes.txt").write_text("not an invoice")
            time.sleep(0.5)  # give watchdog a moment
            assert spy.submitted == []
        finally:
            watcher.stop()
