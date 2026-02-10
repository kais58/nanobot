"""Tests for GDPR consent store."""

import pytest
from pathlib import Path

from nanobot.marketing.consent import ConsentStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_consent.db"
    s = ConsentStore(db_path=db_path)
    yield s
    s.close()


class TestConsent:
    def test_no_consent_by_default(self, store):
        assert (
            store.check_consent("test@example.com", "marketing")
            is False
        )

    def test_record_and_check(self, store):
        store.record_consent(
            "test@example.com", "marketing", "web_form"
        )
        assert (
            store.check_consent("test@example.com", "marketing")
            is True
        )

    def test_withdraw(self, store):
        store.record_consent("test@example.com", "marketing")
        store.withdraw_consent("test@example.com", "marketing")
        assert (
            store.check_consent("test@example.com", "marketing")
            is False
        )

    def test_case_insensitive(self, store):
        store.record_consent(
            "Test@Example.COM", "marketing"
        )
        assert (
            store.check_consent("test@example.com", "marketing")
            is True
        )


class TestB2BFirstContact:
    def test_first_contact_allowed(self, store):
        """B2B first contact allowed under legitimate interest."""
        assert (
            store.can_send_marketing("new@company.com") is True
        )

    def test_second_contact_blocked_without_consent(
        self, store
    ):
        store.record_contact("contacted@company.com")
        assert (
            store.can_send_marketing("contacted@company.com")
            is False
        )

    def test_second_contact_allowed_with_consent(self, store):
        store.record_contact("contacted@company.com")
        store.record_consent(
            "contacted@company.com", "marketing"
        )
        assert (
            store.can_send_marketing("contacted@company.com")
            is True
        )


class TestTransactional:
    def test_always_allowed(self, store):
        assert (
            store.can_send_transactional("anyone@example.com")
            is True
        )


class TestGDPRDelete:
    def test_cascading_delete(self, store):
        store.record_consent(
            "delete@example.com", "marketing"
        )
        store.record_contact("delete@example.com")
        stats = store.gdpr_delete("delete@example.com")
        assert stats["consent"] == 1
        assert stats["contacts"] == 1
        assert (
            store.check_consent(
                "delete@example.com", "marketing"
            )
            is False
        )

    def test_audit_log_preserved(self, store):
        store.record_consent(
            "delete@example.com", "marketing"
        )
        store.gdpr_delete("delete@example.com")
        log = store.get_audit_log("delete@example.com")
        assert any(
            e["action"] == "data_deleted" for e in log
        )
