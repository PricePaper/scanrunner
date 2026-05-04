"""Contract tests for Archiver, ParsedName, ProcessedLedger, StatsTracker."""

from pathlib import Path

import pytest
import yaml

from docscanner import (
    Archiver,
    ParsedName,
    ProcessedLedger,
    Region,
    StatsTracker,
)


# ---------------------------------------------------------------------------
# ParsedName
# ---------------------------------------------------------------------------


class TestParsedName:
    def test_parses_invoice_name(self) -> None:
        p = ParsedName.parse("INV/2026/05000")
        assert p == ParsedName("INV", 2026, 5000)
        assert p.number_range == 5000

    def test_parses_return_invoice_name(self) -> None:
        p = ParsedName.parse("RINV/2026/12345")
        assert p == ParsedName("RINV", 2026, 12345)
        assert p.number_range == 12300

    def test_rejects_so_name(self) -> None:
        with pytest.raises(ValueError):
            ParsedName.parse("SO/2026/05000")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            ParsedName.parse("not-an-invoice-name")


# ---------------------------------------------------------------------------
# Archiver
# ---------------------------------------------------------------------------


class TestArchiver:
    def test_archive_path_follows_type_year_grouping(self, tmp_path: Path) -> None:
        archiver = Archiver(tmp_path)
        parsed = ParsedName("INV", 2026, 5000)
        dest = archiver.archive(
            parsed,
            odoo_id=973700,
            attachment_id=430436,
            original_filename="Customer_Invoice-20260429_153602_0020.jpg",
            payload=b"\xff\xd8\xff\xe0fake",
            ext="jpg",
        )
        expected = tmp_path / "INV" / "2026" / "05000" / (
            "INV-2026-05000_id-973700_aid-430436_"
            "Customer_Invoice-20260429_153602_0020.jpg"
        )
        assert dest == expected
        assert dest.read_bytes() == b"\xff\xd8\xff\xe0fake"

    def test_grouping_by_hundred(self, tmp_path: Path) -> None:
        archiver = Archiver(tmp_path)
        parsed = ParsedName("INV", 2026, 5132)
        dest = archiver.archive(
            parsed, 1, 2, "src.jpg", b"x", "jpg"
        )
        # 5132 → 5100 grouping
        assert dest.parent.name == "05100"

    def test_archive_idempotent_overwrite(self, tmp_path: Path) -> None:
        archiver = Archiver(tmp_path)
        parsed = ParsedName("INV", 2026, 5000)
        d1 = archiver.archive(parsed, 1, 2, "src.jpg", b"first", "jpg")
        d2 = archiver.archive(parsed, 1, 2, "src.jpg", b"second", "jpg")
        assert d1 == d2
        assert d2.read_bytes() == b"second"

    def test_archive_unreadable_moves_source_file(self, tmp_path: Path) -> None:
        src = tmp_path / "broken.jpg"
        src.write_bytes(b"junk")
        archiver = Archiver(tmp_path / "done")
        dest = archiver.archive_unreadable(src)
        assert dest == tmp_path / "done" / "unreadable" / "broken.jpg"
        assert dest.read_bytes() == b"junk"
        assert not src.exists()

    def test_png_extension_used_when_mime_is_png(self, tmp_path: Path) -> None:
        archiver = Archiver(tmp_path)
        parsed = ParsedName("INV", 2026, 5000)
        dest = archiver.archive(parsed, 1, 2, "src.jpg", b"png-bytes", "png")
        assert dest.suffix == ".png"


# ---------------------------------------------------------------------------
# ProcessedLedger
# ---------------------------------------------------------------------------


class TestProcessedLedger:
    def test_records_and_recalls_success(self, tmp_path: Path) -> None:
        ledger = ProcessedLedger(tmp_path / "ledger.sqlite")
        digest = "deadbeef" * 8
        assert not ledger.has(digest)
        ledger.record_success(
            digest, "src.jpg", odoo_id=42, attachment_id=99,
            res_model="account.move", ocr_name="INV/2026/00001",
            archive_path=tmp_path / "done.png",
        )
        assert ledger.has(digest)
        ledger.close()

    def test_persists_across_open_close(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.sqlite"
        l1 = ProcessedLedger(path)
        digest = "a" * 64
        l1.record_success(
            digest, "src.jpg", 1, 2,
            res_model="account.move", ocr_name="INV/2026/00001",
            archive_path=path.parent / "done.png",
        )
        l1.close()
        l2 = ProcessedLedger(path)
        assert l2.has(digest)
        l2.close()

    def test_failure_does_not_gate_dedup_so_retry_is_possible(
        self, tmp_path: Path
    ) -> None:
        """`has()` is the dedup gate in Pipeline.process. Failures must
        not lock a file out — an infra glitch (Odoo down, model cache
        unwritable, …) should be retryable just by re-dropping the file
        into the inbox. Operator surgery on the SQLite ledger to retry
        a previously-failed file would be a wretched UX.
        """
        ledger = ProcessedLedger(tmp_path / "ledger.sqlite")
        digest = "b" * 64
        ledger.record_failure(digest, "broken.jpg")
        assert not ledger.has(digest), (
            "failure must not gate dedup: re-dropping the same file should retry"
        )
        ledger.close()

    def test_record_success_persists_res_model_ocr_name_archive_path(
        self, tmp_path: Path
    ) -> None:
        """Ledger persists everything Phase-3 verification needs to
        confirm a duplicate is really attached to the right Odoo
        record without having to re-OCR."""
        ledger = ProcessedLedger(tmp_path / "ledger.sqlite")
        digest = "d" * 64
        ledger.record_success(
            digest,
            source_name="Customer_Invoice-foo.jpg",
            odoo_id=42,
            attachment_id=99,
            res_model="account.move",
            ocr_name="INV/2026/05001",
            archive_path=tmp_path / "done" / "INV" / "2026" / "05000" / "x.png",
        )
        row = ledger.get_success_row(digest)
        assert row is not None
        assert row.odoo_id == 42
        assert row.attachment_id == 99
        assert row.res_model == "account.move"
        assert row.ocr_name == "INV/2026/05001"
        assert str(row.archive_path).endswith("x.png")
        ledger.close()

    def test_get_success_row_returns_none_for_failure_or_missing(
        self, tmp_path: Path
    ) -> None:
        ledger = ProcessedLedger(tmp_path / "ledger.sqlite")
        # Failure row exists but is not a success → get_success_row None.
        ledger.record_failure("e" * 64, "broken.jpg")
        assert ledger.get_success_row("e" * 64) is None
        # Never-seen digest → None.
        assert ledger.get_success_row("f" * 64) is None
        ledger.close()

    def test_delete_removes_row_and_re_enables_processing(
        self, tmp_path: Path
    ) -> None:
        ledger = ProcessedLedger(tmp_path / "ledger.sqlite")
        digest = "g" * 64
        ledger.record_success(
            digest, "x.jpg", odoo_id=1, attachment_id=1,
            res_model="account.move", ocr_name="INV/2026/00001",
            archive_path=tmp_path / "x.png",
        )
        assert ledger.has(digest)
        ledger.delete(digest)
        assert not ledger.has(digest)
        assert ledger.get_success_row(digest) is None
        ledger.close()

    def test_existing_row_without_phase2_columns_is_readable(
        self, tmp_path: Path
    ) -> None:
        """Schema migration safety: a ledger written by a pre-Phase-2
        binary (no res_model / ocr_name / archive_path columns) must
        still load without error. Older rows surface as success rows
        with empty strings for the new fields — Phase 3's verify
        treats those as "can't confirm" → reprocess (Phase 2's
        idempotency catches the redundant upload)."""
        # Simulate an old ledger by creating just the original schema.
        import sqlite3
        path = tmp_path / "legacy.sqlite"
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE processed (
                sha256 TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                outcome TEXT NOT NULL,
                odoo_id INTEGER,
                attachment_id INTEGER,
                recorded_at REAL NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO processed VALUES (?, ?, ?, ?, ?, ?)",
            ("h" * 64, "old.jpg", "success", 1, 2, 0.0),
        )
        conn.commit()
        conn.close()
        # Now open with the new code; migration should run silently.
        ledger = ProcessedLedger(path)
        assert ledger.has("h" * 64)
        row = ledger.get_success_row("h" * 64)
        assert row is not None
        assert row.odoo_id == 1
        assert row.attachment_id == 2
        assert row.res_model == ""
        assert row.ocr_name == ""
        ledger.close()

    def test_success_after_failure_correctly_records_and_gates(
        self, tmp_path: Path
    ) -> None:
        """Realistic recovery sequence: file fails the first attempt
        (e.g. transient OCR cache PermissionError), operator re-drops
        after fix, second attempt succeeds. From then on dedup gates
        further re-drops."""
        ledger = ProcessedLedger(tmp_path / "ledger.sqlite")
        digest = "c" * 64
        ledger.record_failure(digest, "scan.jpg")
        assert not ledger.has(digest)  # retryable
        ledger.record_success(
            digest, "scan.jpg", odoo_id=42, attachment_id=99,
            res_model="account.move", ocr_name="INV/2026/00042",
            archive_path=tmp_path / "x.png",
        )
        assert ledger.has(digest)       # now dedup-gated
        ledger.close()

    def test_file_digest_is_stable(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        d1 = ProcessedLedger.file_digest(f)
        d2 = ProcessedLedger.file_digest(f)
        assert d1 == d2 and len(d1) == 64


# ---------------------------------------------------------------------------
# StatsTracker
# ---------------------------------------------------------------------------


class TestStatsTracker:
    def test_records_and_flushes(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.yaml"
        tracker = StatsTracker(path)
        r1 = Region(60, 0, 100, 25)
        r2 = Region(20, 30, 80, 70)
        tracker.record("Invoice", r1)
        tracker.record("Invoice", r1)
        tracker.record("Invoice", r2)
        tracker.flush()
        loaded = yaml.safe_load(path.read_text())
        assert loaded == {
            "Invoice": {
                "60,0,100,25": 2,
                "20,30,80,70": 1,
            }
        }

    def test_loads_existing_stats(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.yaml"
        path.write_text(yaml.safe_dump({"Invoice": {"60,0,100,25": 5}}))
        tracker = StatsTracker(path)
        snap = tracker.snapshot()
        assert snap["Invoice"]["60,0,100,25"] == 5
        tracker.record("Invoice", Region(60, 0, 100, 25))
        assert tracker.snapshot()["Invoice"]["60,0,100,25"] == 6
