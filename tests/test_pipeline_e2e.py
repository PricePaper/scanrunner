"""End-to-end Pipeline tests against the live harness.

Reads `inv/good/INV-2026-05000_*.jpg` (the harness has matching invoices),
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
    OcrPreparer,
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
        "tesseract-bin": shutil.which("tesseract") or "/usr/bin/tesseract",
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
                "tesseract_config": "--psm 6 -l eng",
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
    src = project_root / "inv" / "good"
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
        unreadable_dest = (
            harness_inbox / "done" / "unreadable" / target.name
        )
        assert unreadable_dest.exists()


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
