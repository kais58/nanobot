"""Tests for the email channel."""

from unittest.mock import MagicMock

from nanobot.config.schema import EmailConfig


class TestEmailConfig:
    def test_defaults(self):
        config = EmailConfig()
        assert config.enabled is False
        assert config.imap_host == "imap.gmail.com"
        assert config.imap_port == 993
        assert config.smtp_host == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.poll_interval == 60
        assert config.folder == "INBOX"

    def test_from_alias(self):
        config = EmailConfig(
            **{
                "imapHost": "imap.example.com",
                "smtpHost": "smtp.example.com",
                "pollInterval": 30,
            }
        )
        assert config.imap_host == "imap.example.com"
        assert config.smtp_host == "smtp.example.com"
        assert config.poll_interval == 30


class TestEmailChannel:
    def test_extract_body_plain(self):
        import email

        msg = email.message_from_string(
            "From: test@example.com\r\n"
            "Subject: Test\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "Hello World"
        )
        from nanobot.channels.email import EmailChannel

        assert EmailChannel._extract_body(msg) == "Hello World"

    def test_is_allowed_empty_list(self):
        from nanobot.channels.email import EmailChannel

        config = EmailConfig(enabled=True)
        bus = MagicMock()
        channel = EmailChannel(config, bus)
        assert channel.is_allowed("anyone@example.com") is True

    def test_is_allowed_with_list(self):
        from nanobot.channels.email import EmailChannel

        config = EmailConfig(
            **{
                "enabled": True,
                "allowFrom": ["allowed@example.com"],
            }
        )
        bus = MagicMock()
        channel = EmailChannel(config, bus)
        assert channel.is_allowed("allowed@example.com") is True
        assert channel.is_allowed("denied@example.com") is False
