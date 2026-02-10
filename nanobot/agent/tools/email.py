"""Send email tool for the agent."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class SendEmailTool(Tool):
    """Tool to send emails with subject, body, and optional template rendering."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        consent_store: Any = None,
        report_generator: Any = None,
    ):
        self._send_callback = send_callback
        self._consent_store = consent_store
        self._report_generator = report_generator

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending emails."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return (
            "Send an email. Supports plain text, HTML, templates, "
            "and reply threading. Checks GDPR consent before "
            "sending marketing emails."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Email body (plain text or markdown)",
                },
                "cc": {
                    "type": "string",
                    "description": "CC recipient (optional)",
                },
                "reply_to_message_id": {
                    "type": "string",
                    "description": ("Message ID to reply to (for threading)"),
                },
                "template": {
                    "type": "string",
                    "description": ("Template name to render (e.g., 'outreach_change.md')"),
                },
                "template_vars": {
                    "type": "object",
                    "description": "Variables for template rendering",
                },
                "is_marketing": {
                    "type": "boolean",
                    "description": (
                        "Whether this is a marketing email (requires GDPR consent check)"
                    ),
                },
            },
            "required": ["to", "subject"],
        }

    async def execute(
        self,
        to: str,
        subject: str,
        body: str = "",
        cc: str | None = None,
        reply_to_message_id: str | None = None,
        template: str | None = None,
        template_vars: dict[str, Any] | None = None,
        is_marketing: bool = True,
        **kwargs: Any,
    ) -> str:
        try:
            # GDPR consent check for marketing emails
            if is_marketing and self._consent_store:
                if not self._consent_store.can_send_marketing(to):
                    return (
                        f"Error: Cannot send marketing email to {to} - "
                        "no consent and not a first contact. "
                        "Use is_marketing=false for transactional emails."
                    )

            # Render template if provided
            if template and self._report_generator:
                try:
                    body = self._report_generator.render_outreach(template, template_vars or {})
                except Exception as e:
                    return f"Error rendering template '{template}': {e}"
            elif template and not self._report_generator:
                return "Error: Template rendering not available (report generator not configured)"

            if not body:
                return "Error: Email body is required (or provide a template)"

            if not self._send_callback:
                return "Error: Email sending not configured"

            # Build metadata for the email channel
            metadata: dict[str, Any] = {"subject": subject}
            if cc:
                metadata["cc"] = cc

            msg = OutboundMessage(
                channel="email",
                chat_id=to,
                content=body,
                reply_to=reply_to_message_id,
                metadata=metadata,
            )

            await self._send_callback(msg)

            # Record contact for GDPR tracking
            if self._consent_store:
                self._consent_store.record_contact(to)
                self._consent_store.log_audit(
                    "email_sent",
                    to,
                    f"subject={subject}, marketing={is_marketing}",
                )

            return f"Email sent to {to} (subject: {subject})"

        except Exception as e:
            return f"Error sending email: {e}"
