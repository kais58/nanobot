"""Tests for web dashboard authentication."""

from unittest.mock import MagicMock

from nanobot.web.auth import AuthManager


class TestAuthManager:
    def test_no_hash_accepts_any_password(self):
        auth = AuthManager(username="admin", password_hash="")
        assert auth.verify_password("anything") is True

    def test_no_hash_rejects_empty_password(self):
        auth = AuthManager(username="admin", password_hash="")
        assert auth.verify_password("") is False

    def test_plain_text_fallback(self):
        # Mock bcrypt away so the plain text fallback is exercised
        import nanobot.web.auth as auth_mod

        orig = auth_mod.bcrypt
        auth_mod.bcrypt = None
        try:
            auth = AuthManager(
                username="admin",
                password_hash="secret123",
            )
            assert auth.verify_password("secret123") is True
            assert auth.verify_password("wrong") is False
        finally:
            auth_mod.bcrypt = orig

    def test_create_session(self):
        auth = AuthManager(secret_key="test-secret")
        token = auth.create_session("admin")
        assert token is not None
        assert len(token) > 0

    def test_get_current_user_no_cookie(self):
        auth = AuthManager()
        request = MagicMock()
        request.cookies = {}
        assert auth.get_current_user(request) is None

    def test_get_current_user_valid_session(self):
        auth = AuthManager(secret_key="test-secret")
        token = auth.create_session("admin")
        request = MagicMock()
        request.cookies = {AuthManager.COOKIE_NAME: token}
        assert auth.get_current_user(request) == "admin"

    def test_get_current_user_invalid_token(self):
        auth = AuthManager()
        request = MagicMock()
        request.cookies = {AuthManager.COOKIE_NAME: "garbage"}
        assert auth.get_current_user(request) is None

    def test_require_auth_returns_user(self):
        auth = AuthManager(secret_key="test-secret")
        token = auth.create_session("admin")
        request = MagicMock()
        request.cookies = {AuthManager.COOKIE_NAME: token}
        assert auth.require_auth(request) == "admin"

    def test_require_auth_no_session(self):
        auth = AuthManager()
        request = MagicMock()
        request.cookies = {}
        assert auth.require_auth(request) is None
