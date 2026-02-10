"""GDPR consent tracking and audit logging."""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class ConsentStore:
    """GDPR-compliant consent tracking with audit logging."""

    def __init__(
        self,
        db_path: str | Path = "~/.nanobot/data/consent.db",
    ):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS consent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_hash TEXT NOT NULL,
                consent_type TEXT NOT NULL,
                consent_given INTEGER NOT NULL DEFAULT 0,
                consent_date TEXT NOT NULL,
                source TEXT DEFAULT '',
                withdrawn_date TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                entity_email_hash TEXT NOT NULL,
                details TEXT DEFAULT '',
                performed_by TEXT DEFAULT 'system'
            );

            CREATE TABLE IF NOT EXISTS contacts (
                email_hash TEXT PRIMARY KEY,
                first_contact_date TEXT DEFAULT '',
                last_contact_date TEXT DEFAULT '',
                contact_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_consent_hash
                ON consent(email_hash);
            CREATE INDEX IF NOT EXISTS idx_consent_type
                ON consent(email_hash, consent_type);
            CREATE INDEX IF NOT EXISTS idx_audit_hash
                ON audit_log(entity_email_hash);
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _hash_email(email: str) -> str:
        return hashlib.sha256(email.lower().strip().encode("utf-8")).hexdigest()

    def check_consent(
        self,
        email: str,
        consent_type: str = "marketing",
    ) -> bool:
        """Check if consent exists and is active."""
        email_hash = self._hash_email(email)
        row = self._conn.execute(
            """SELECT consent_given FROM consent
            WHERE email_hash = ? AND consent_type = ?
            AND consent_given = 1
            AND withdrawn_date = ''
            ORDER BY consent_date DESC LIMIT 1""",
            (email_hash, consent_type),
        ).fetchone()
        return row is not None

    def record_consent(
        self,
        email: str,
        consent_type: str,
        source: str = "manual",
    ) -> None:
        """Record a consent grant."""
        email_hash = self._hash_email(email)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO consent
            (email_hash, consent_type, consent_given,
             consent_date, source)
            VALUES (?, ?, 1, ?, ?)""",
            (email_hash, consent_type, now, source),
        )
        self._conn.commit()
        self.log_audit(
            "consent_given",
            email,
            f"type={consent_type}, source={source}",
        )

    def withdraw_consent(self, email: str, consent_type: str) -> None:
        """Withdraw consent (GDPR opt-out)."""
        email_hash = self._hash_email(email)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """UPDATE consent SET withdrawn_date = ?
            WHERE email_hash = ? AND consent_type = ?
            AND withdrawn_date = ''""",
            (now, email_hash, consent_type),
        )
        self._conn.commit()
        self.log_audit(
            "consent_withdrawn",
            email,
            f"type={consent_type}",
        )

    def can_send_marketing(self, email: str) -> bool:
        """Check if marketing email can be sent.

        Returns True if:
        - Explicit marketing consent exists, OR
        - B2B first-contact exception (Art. 6(1)(f) DSGVO):
          never contacted before AND legitimate business interest
        """
        if self.check_consent(email, "marketing"):
            return True
        # B2B first-contact exception
        email_hash = self._hash_email(email)
        contact = self._conn.execute(
            "SELECT contact_count FROM contacts WHERE email_hash = ?",
            (email_hash,),
        ).fetchone()
        if contact is None or contact["contact_count"] == 0:
            return True  # First contact allowed
        return False

    def can_send_transactional(self, email: str) -> bool:
        """Transactional emails are always allowed."""
        return True

    def record_contact(self, email: str) -> None:
        """Record that a contact was made."""
        email_hash = self._hash_email(email)
        now = datetime.utcnow().isoformat()
        existing = self._conn.execute(
            "SELECT email_hash FROM contacts WHERE email_hash = ?",
            (email_hash,),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE contacts SET last_contact_date = ?, "
                "contact_count = contact_count + 1 "
                "WHERE email_hash = ?",
                (now, email_hash),
            )
        else:
            self._conn.execute(
                """INSERT INTO contacts
                (email_hash, first_contact_date,
                 last_contact_date, contact_count)
                VALUES (?, ?, ?, 1)""",
                (email_hash, now, now),
            )
        self._conn.commit()

    def log_audit(
        self,
        action: str,
        email: str,
        details: str = "",
        performed_by: str = "system",
    ) -> None:
        """Log an audit entry."""
        email_hash = self._hash_email(email)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO audit_log
            (timestamp, action, entity_email_hash,
             details, performed_by)
            VALUES (?, ?, ?, ?, ?)""",
            (now, action, email_hash, details, performed_by),
        )
        self._conn.commit()

    def get_audit_log(
        self,
        email: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get audit log entries."""
        if email:
            email_hash = self._hash_email(email)
            rows = self._conn.execute(
                """SELECT * FROM audit_log
                WHERE entity_email_hash = ?
                ORDER BY timestamp DESC LIMIT ?""",
                (email_hash, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def gdpr_delete(self, email: str) -> dict[str, int]:
        """Execute GDPR right-to-erasure.

        Cascading delete across all tables.
        Returns dict of deletion counts per table.
        """
        email_hash = self._hash_email(email)
        stats: dict[str, int] = {}

        cursor = self._conn.execute(
            "DELETE FROM consent WHERE email_hash = ?",
            (email_hash,),
        )
        stats["consent"] = cursor.rowcount

        cursor = self._conn.execute(
            "DELETE FROM contacts WHERE email_hash = ?",
            (email_hash,),
        )
        stats["contacts"] = cursor.rowcount

        self._conn.commit()

        # Audit logs are kept for compliance
        self.log_audit("data_deleted", email, f"deleted: {stats}")

        return stats
