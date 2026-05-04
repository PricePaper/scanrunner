"""Integration tests for OdooClient against the live test harness.

Credentials come from environment variables (NEVER hardcode):

    SCANRUNNER_TEST_ODOO_URL       (default: http://127.0.0.1:58069)
    SCANRUNNER_TEST_ODOO_DB        (default: ppt-apps15-test)
    SCANRUNNER_TEST_ODOO_LOGIN     (required)
    SCANRUNNER_TEST_ODOO_PASSWORD  (required)

Tests skip if the required env vars are unset.

Includes the xmlrpc-guard test: docscanner.py must NOT import xmlrpc.
"""

import os
import re
from pathlib import Path

import pytest

from docscanner import OdooClient, OdooError


def _harness_settings() -> dict[str, str]:
    login = os.environ.get("SCANRUNNER_TEST_ODOO_LOGIN")
    password = os.environ.get("SCANRUNNER_TEST_ODOO_PASSWORD")
    if not (login and password):
        pytest.skip(
            "harness credentials not in env (SCANRUNNER_TEST_ODOO_LOGIN / _PASSWORD)"
        )
    return {
        "url": os.environ.get("SCANRUNNER_TEST_ODOO_URL", "http://127.0.0.1:58069"),
        "database": os.environ.get("SCANRUNNER_TEST_ODOO_DB", "ppt-apps15-test"),
        "username": login,
        "password": password,
    }


@pytest.fixture
def harness_client() -> OdooClient:
    s = _harness_settings()
    client = OdooClient(
        url=s["url"],
        database=s["database"],
        username=s["username"],
        password=s["password"],
        verify_tls=False,  # localhost test pod uses HTTP
        retry=2,
        retry_sleep=0.1,
    )
    yield client
    client.close()


# ---------------------------------------------------------------------------
# xmlrpc guard — must remain green forever
# ---------------------------------------------------------------------------


class TestXmlrpcGuard:
    def test_docscanner_does_not_import_xmlrpc(self, project_root: Path) -> None:
        source = (project_root / "docscanner.py").read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+xmlrpc", source, re.MULTILINE), (
            "docscanner.py must use httpx + JSON-RPC, never xmlrpc.client"
        )


# ---------------------------------------------------------------------------
# OdooClient — happy path against the harness
# ---------------------------------------------------------------------------


class TestOdooClientAuth:
    def test_authenticate_returns_uid(self, harness_client: OdooClient) -> None:
        uid = harness_client.authenticate()
        assert uid > 0

    def test_authenticate_caches_uid(self, harness_client: OdooClient) -> None:
        first = harness_client.authenticate()
        second = harness_client.authenticate()
        assert first == second

    def test_invalid_credentials_raise(self) -> None:
        s = _harness_settings()
        client = OdooClient(
            url=s["url"],
            database=s["database"],
            username=s["username"],
            password="definitely_not_the_password",
            verify_tls=False,
        )
        try:
            with pytest.raises(OdooError, match="authentication failed"):
                client.authenticate()
        finally:
            client.close()


class TestOdooClientLookup:
    def test_search_read_returns_records(self, harness_client: OdooClient) -> None:
        rows = harness_client.search_read(
            "account.move",
            [["name", "like", "INV/2026"]],
            ["id", "name"],
            limit=3,
        )
        assert 1 <= len(rows) <= 3
        for row in rows:
            assert "id" in row and "name" in row
            assert row["name"].startswith(("INV/2026", "RINV/2026"))

    def test_find_record_returns_id_for_known_invoice(
        self, harness_client: OdooClient
    ) -> None:
        # Pick an existing invoice off the harness, then look it up by name.
        rows = harness_client.search_read(
            "account.move", [["name", "like", "INV/2026"]], ["id", "name"], limit=1
        )
        assert rows
        rid = harness_client.find_record("account.move", rows[0]["name"])
        assert rid == int(rows[0]["id"])

    def test_find_record_returns_none_for_missing(
        self, harness_client: OdooClient
    ) -> None:
        assert harness_client.find_record(
            "account.move", "INV/9999/00000-does-not-exist"
        ) is None


class TestOdooClientAttachAndLink:
    def test_attach_and_link_round_trip(self, harness_client: OdooClient) -> None:
        rows = harness_client.search_read(
            "account.move", [["name", "like", "INV/2026"]], ["id", "name"], limit=1
        )
        assert rows
        invoice_id = int(rows[0]["id"])
        invoice_name = rows[0]["name"]
        payload = b"\xff\xd8\xff\xe0fake-jpeg-payload"
        attachment_id = harness_client.attach(
            "account.move",
            invoice_id,
            f"{invoice_name.replace('/', '-')}_test_attach.jpg",
            "image/jpeg",
            payload,
        )
        assert attachment_id > 0

        # Verify by reading it back.
        rows = harness_client.search_read(
            "ir.attachment",
            [["id", "=", attachment_id]],
            ["res_model", "res_id", "mimetype", "name"],
            limit=1,
        )
        assert rows
        attached = rows[0]
        assert attached["res_model"] == "account.move"
        assert attached["res_id"] == invoice_id
        assert attached["mimetype"] == "image/jpeg"

        # Link to documents app.
        document_id = harness_client.link_to_documents_app(
            attachment_id=attachment_id,
            folder_id=7,  # Customer Invoices on the harness
            tag_id=1,  # Inbox
        )
        assert document_id > 0


class TestOdooClientIdempotencyAndVerify:
    def test_find_attachment_by_checksum_matches_recent_upload(
        self, harness_client: OdooClient
    ) -> None:
        """find_attachment_by_checksum returns the aid of an existing
        ir.attachment on the SAME (res_model, res_id) whose sha1
        checksum matches the bytes the caller is about to upload.
        Lets Pipeline.process skip a redundant create_attachment when
        a previous attempt already uploaded but crashed before
        recording success in the local ledger.
        """
        import hashlib
        rows = harness_client.search_read(
            "account.move", [["name", "like", "INV/2026"]], ["id", "name"], limit=1
        )
        invoice_id = int(rows[0]["id"])
        payload = b"\xff\xd8\xff\xe0idempotency-fixture"
        aid = harness_client.attach(
            "account.move", invoice_id,
            "INV-2026-idem-test.jpg", "image/jpeg", payload,
        )
        sha1 = hashlib.sha1(payload).hexdigest()
        # First lookup must find it.
        found = harness_client.find_attachment_by_checksum(
            "account.move", invoice_id, sha1,
        )
        assert found == aid

        # Different res_id → no match (scope is per-record).
        other_invoice_id = invoice_id + 1
        not_found = harness_client.find_attachment_by_checksum(
            "account.move", other_invoice_id, sha1,
        )
        assert not_found is None

        # Different checksum on same record → no match.
        not_found = harness_client.find_attachment_by_checksum(
            "account.move", invoice_id, "deadbeef" * 5,
        )
        assert not_found is None

    def test_verify_attachment_confirms_aid_is_on_expected_record(
        self, harness_client: OdooClient
    ) -> None:
        """verify_attachment is the Phase-3 duplicate-handling primitive.
        It must return True only when the aid is still attached to the
        expected (res_model, res_id) AND the attachment name still
        contains the expected slug. Mismatches return False; transport
        errors are NOT swallowed (caller's catchall handles)."""
        rows = harness_client.search_read(
            "account.move", [["name", "like", "INV/2026"]], ["id", "name"], limit=1
        )
        invoice_id = int(rows[0]["id"])
        invoice_name = rows[0]["name"]
        slug = invoice_name.replace("/", "-")
        payload = b"\xff\xd8\xff\xe0verify-fixture"
        aid = harness_client.attach(
            "account.move", invoice_id,
            f"{slug}_verify-test.jpg", "image/jpeg", payload,
        )

        # Happy path — all three conditions match.
        assert harness_client.verify_attachment(
            attachment_id=aid,
            expected_res_model="account.move",
            expected_res_id=invoice_id,
            expected_name_contains=slug,
        )
        # Wrong model → False.
        assert not harness_client.verify_attachment(
            aid, "res.partner", invoice_id, slug,
        )
        # Wrong res_id → False.
        assert not harness_client.verify_attachment(
            aid, "account.move", invoice_id + 1, slug,
        )
        # Slug not in name → False (defensive against operator-edited names).
        assert not harness_client.verify_attachment(
            aid, "account.move", invoice_id, "INV-9999-99999",
        )
        # Non-existent aid → False.
        assert not harness_client.verify_attachment(
            999_999_999, "account.move", invoice_id, slug,
        )


# ---------------------------------------------------------------------------
# Lifecycle — closed/reopened client transparently recovers (per global TDD rule)
# ---------------------------------------------------------------------------


class TestOdooClientLifecycle:
    def test_call_after_close_reopens_transparently(
        self, harness_client: OdooClient
    ) -> None:
        # First call opens and authenticates.
        first = harness_client.authenticate()
        # Simulate a degraded state: close the underlying httpx client.
        harness_client.close()
        # Next call must rebuild and re-authenticate without error.
        second = harness_client.authenticate()
        assert first == second

    def test_close_is_idempotent(self) -> None:
        s = _harness_settings()
        client = OdooClient(
            url=s["url"],
            database=s["database"],
            username=s["username"],
            password=s["password"],
            verify_tls=False,
        )
        client.close()
        client.close()  # must not raise

    def test_context_manager_closes(self) -> None:
        s = _harness_settings()
        with OdooClient(
            url=s["url"],
            database=s["database"],
            username=s["username"],
            password=s["password"],
            verify_tls=False,
        ) as client:
            client.authenticate()
        # Re-use after exit transparently reopens.
        client.authenticate()
        client.close()
