"""Contract tests for Config, Region, DocumentType, Invoice, and DocumentTypeRegistry."""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from docscanner import (
    Config,
    ConfigError,
    DocumentType,
    DocumentTypeRegistry,
    Invoice,
    Region,
)


SAMPLE_CONFIG: dict = {
    "retry": 3,
    "retry_sleep": 1.0,
    "tesseract-bin": "/usr/bin/tesseract",
    "done-path": "done",
    "error-email": "ops@example.com",
    "error-mail-message": "Failed to read invoice",
    "statistics-file": "statistics.yaml",
    "servers": {
        "harness": {
            "url": "http://127.0.0.1:58069",
            "database": "ppt-apps15-test",
            "username": "admin",
            "password": "admin",
            "smtp-server": "localhost",
            "smtp-port": 1025,
            "smtp-user": "u",
            "smtp-password": "p",
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


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(SAMPLE_CONFIG))
    return p


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------


class TestRegion:
    def test_crop_returns_correct_subregion(self) -> None:
        img = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        r = Region(0, 0, 50, 50)
        crop = r.crop(img)
        assert crop.shape == (50, 50)
        assert crop[0, 0] == img[0, 0]
        assert crop[49, 49] == img[49, 49]

    def test_invalid_percent_raises(self) -> None:
        with pytest.raises(ValueError):
            Region(-1, 0, 100, 100)
        with pytest.raises(ValueError):
            Region(0, 0, 101, 100)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            Region(50, 0, 50, 100)
        with pytest.raises(ValueError):
            Region(0, 50, 100, 50)

    def test_from_list(self) -> None:
        r = Region.from_list([10, 20, 30, 40])
        assert r == Region(10, 20, 30, 40)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_loads_from_yaml(self, config_path: Path) -> None:
        cfg = Config.load(config_path, "harness")
        assert cfg.server_name == "harness"
        assert cfg.server.url == "http://127.0.0.1:58069"
        assert cfg.server.database == "ppt-apps15-test"
        assert cfg.server.smtp_use_tls is False
        assert cfg.server.verify_tls is False
        assert cfg.retry == 3
        assert cfg.tesseract_bin == "/usr/bin/tesseract"

    def test_loads_invoice_document_type(self, config_path: Path) -> None:
        cfg = Config.load(config_path, "harness")
        inv = cfg.documents["Invoice"]
        assert inv.name == "Invoice"
        assert inv.file_name_match == "*Customer_Invoice*"
        assert inv.mime_types == ("image/jpeg", "image/png")
        assert inv.odoo_object == "account.move"
        assert len(inv.search_regions) == 2
        assert inv.search_regions[0] == Region(60, 0, 100, 25)

    def test_unknown_server_raises(self, config_path: Path) -> None:
        with pytest.raises(ConfigError, match="server 'production' not in config"):
            Config.load(config_path, "production")

    def test_missing_server_key_raises(self, tmp_path: Path) -> None:
        broken = dict(SAMPLE_CONFIG)
        broken_servers = dict(broken["servers"])
        broken_servers["broken"] = {"url": "http://x"}  # missing required keys
        broken["servers"] = broken_servers
        p = tmp_path / "broken.yaml"
        p.write_text(yaml.safe_dump(broken))
        with pytest.raises(ConfigError, match="missing key"):
            Config.load(p, "broken")

    def test_missing_document_key_raises(self, tmp_path: Path) -> None:
        broken = dict(SAMPLE_CONFIG)
        broken_docs = dict(broken["documents"])
        broken_docs["BadType"] = {"file-name-match": "*"}
        broken["documents"] = broken_docs
        p = tmp_path / "broken.yaml"
        p.write_text(yaml.safe_dump(broken))
        with pytest.raises(ConfigError, match="documents.BadType missing key"):
            Config.load(p, "harness")


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------


class TestInvoice:
    def test_regex_matches_invoice_number(self, config_path: Path) -> None:
        cfg = Config.load(config_path, "harness")
        inv = Invoice(cfg.documents["Invoice"])
        assert inv.regex.search("Some text INV/2026/05000 more text")
        assert inv.regex.search("RINV/2026/12345")

    def test_regex_does_not_match_source_document_so_number(
        self, config_path: Path
    ) -> None:
        cfg = Config.load(config_path, "harness")
        inv = Invoice(cfg.documents["Invoice"])
        # SO is a sale order — must NOT be picked up as the invoice name.
        assert not inv.regex.search("SO/2026/05000")
        # Prefix that isn't (R)INV either.
        assert not inv.regex.search("CUST/2026/05000")

    def test_matches_filename_and_mime(self, config_path: Path) -> None:
        cfg = Config.load(config_path, "harness")
        inv = Invoice(cfg.documents["Invoice"])
        assert inv.matches(
            Path("Customer_Invoice-20260101_120000_0001.jpg"), "image/jpeg"
        )
        assert not inv.matches(
            Path("Customer_Invoice-20260101_120000_0001.jpg"), "application/pdf"
        )
        assert not inv.matches(Path("Receipt-20260101.jpg"), "image/jpeg")

    def test_preprocess_removes_yellow(self, config_path: Path) -> None:
        cfg = Config.load(config_path, "harness")
        inv = Invoice(cfg.documents["Invoice"])
        yellow = np.full((100, 100, 3), (140, 230, 240), dtype=np.uint8)
        out = inv.preprocess(yellow)
        # YellowRemover snaps yellow paper to white.
        assert out[10:50, 10:50].mean() > 220


# ---------------------------------------------------------------------------
# DocumentTypeRegistry
# ---------------------------------------------------------------------------


class TestDocumentTypeRegistry:
    def test_classifies_real_invoice_sample(
        self, config_path: Path, first_good_invoice: Path
    ) -> None:
        cfg = Config.load(config_path, "harness")
        registry = DocumentTypeRegistry.from_config(cfg)
        dt = registry.classify(first_good_invoice)
        assert dt is not None
        assert dt.name == "Invoice"

    def test_returns_none_for_unmatched_file(
        self, config_path: Path, tmp_path: Path
    ) -> None:
        cfg = Config.load(config_path, "harness")
        registry = DocumentTypeRegistry.from_config(cfg)
        random = tmp_path / "junk.txt"
        random.write_text("not an invoice")
        assert registry.classify(random) is None

    def test_unknown_type_in_config_raises(self, tmp_path: Path) -> None:
        cfg_data = dict(SAMPLE_CONFIG)
        cfg_data["documents"] = {
            **cfg_data["documents"],
            "Receipt": {
                "file-name-match": "*Receipt*",
                "mime-types": ["application/pdf"],
                "ocr_regex": r"REC/\d+",
                "search_regions": [[0, 0, 100, 50]],
                "tesseract_config": "--psm 6 -l eng",
                "odoo_sequence": "REC",
                "odoo_object": "account.move",
                "odoo_attachment_tag_id": 1,
                "odoo_folder_id": 7,
            },
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(cfg_data))
        cfg = Config.load(p, "harness")
        with pytest.raises(ConfigError, match="unknown document type 'Receipt'"):
            DocumentTypeRegistry.from_config(cfg)
