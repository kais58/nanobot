"""Email channel for IMAP/SMTP communication."""

import asyncio
import email as email_lib
import email.mime.multipart
import email.mime.text
from email.header import decode_header
from email.utils import parseaddr

import aiosmtplib
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import EmailConfig


class EmailChannel(BaseChannel):
    """Email channel using IMAP for inbound and SMTP for outbound."""

    name = "email"

    def __init__(self, config: EmailConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: EmailConfig = config
        self._poll_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start polling IMAP for new emails."""
        self._running = True
        logger.info(
            f"Email channel starting "
            f"(IMAP: {self.config.imap_host}, "
            f"poll: {self.config.poll_interval}s)"
        )
        self._poll_task = asyncio.create_task(self._poll_loop())
        await self._poll_task

    async def stop(self) -> None:
        """Stop the email channel."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("Email channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send an email via SMTP."""
        try:
            mime_msg = email.mime.multipart.MIMEMultipart("alternative")
            mime_msg["From"] = self.config.username
            mime_msg["To"] = msg.chat_id
            mime_msg["Subject"] = msg.metadata.get("subject", "Message from Nanobot")

            # Reply threading
            if msg.reply_to:
                mime_msg["In-Reply-To"] = msg.reply_to
                mime_msg["References"] = msg.reply_to

            # Plain text part
            text_part = email.mime.text.MIMEText(msg.content, "plain", "utf-8")
            mime_msg.attach(text_part)

            # HTML part (simple conversion)
            html_content = msg.content.replace("\n", "<br>\n")
            html_part = email.mime.text.MIMEText(
                f"<html><body>{html_content}</body></html>",
                "html",
                "utf-8",
            )
            mime_msg.attach(html_part)

            await aiosmtplib.send(
                mime_msg,
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                username=self.config.username,
                password=self.config.password,
                start_tls=True,
            )
            logger.info(f"Email sent to {msg.chat_id}")
        except Exception as e:
            logger.error(f"Failed to send email to {msg.chat_id}: {e}")
            raise

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages."""
        last_seen_uid: int = 0

        while self._running:
            try:
                loop = asyncio.get_event_loop()
                messages = await loop.run_in_executor(None, self._fetch_new_emails, last_seen_uid)

                for uid, sender, subject, body in messages:
                    if uid > last_seen_uid:
                        last_seen_uid = uid

                    _, sender_email = parseaddr(sender)
                    if not sender_email:
                        continue

                    content = f"[Subject: {subject}]\n\n{body}" if subject else body

                    await self._handle_message(
                        sender_id=sender_email,
                        chat_id=sender_email,
                        content=content,
                        metadata={
                            "subject": subject,
                            "message_id": str(uid),
                            "raw_sender": sender,
                        },
                    )

            except Exception as e:
                logger.error(f"IMAP poll error: {e}")

            await asyncio.sleep(self.config.poll_interval)

    def _fetch_new_emails(self, last_seen_uid: int) -> list[tuple[int, str, str, str]]:
        """Fetch new emails from IMAP (runs in executor thread).

        Returns list of (uid, sender, subject, body) tuples.
        """
        import imaplib

        results: list[tuple[int, str, str, str]] = []

        try:
            mail = imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port)
            mail.login(self.config.username, self.config.password)
            mail.select(self.config.folder, readonly=True)

            if last_seen_uid > 0:
                status, data = mail.uid("search", None, f"UID {last_seen_uid + 1}:*")
            else:
                status, data = mail.uid("search", None, "UNSEEN")

            if status != "OK" or not data[0]:
                mail.logout()
                return results

            uid_list = data[0].split()

            for uid_bytes in uid_list:
                uid = int(uid_bytes)
                if uid <= last_seen_uid:
                    continue

                status, msg_data = mail.uid("fetch", uid_bytes, "(RFC822)")
                if status != "OK" or not msg_data[0]:
                    continue

                raw_email = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw_email)

                sender = msg.get("From", "")
                subject = msg.get("Subject", "")

                # Decode subject if encoded
                if subject:
                    decoded_parts = decode_header(subject)
                    subject = ""
                    for part, charset in decoded_parts:
                        if isinstance(part, bytes):
                            subject += part.decode(charset or "utf-8", errors="replace")
                        else:
                            subject += part

                body = self._extract_body(msg)
                results.append((uid, sender, subject, body))

            mail.logout()
        except Exception as e:
            logger.error(f"IMAP fetch error: {e}")

        return results

    @staticmethod
    def _extract_body(msg: email_lib.message.Message) -> str:
        """Extract plaintext body from email message."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")
            # Fallback to HTML if no plain text
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        return ""
