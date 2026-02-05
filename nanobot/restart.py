"""Restart signal utilities for nanobot."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

RESTART_SIGNAL_FILE = ".restart_signal"


def write_restart_signal(
    workspace: Path,
    reason: str,
    verify_job: dict[str, Any] | None = None,
) -> None:
    """
    Write a restart signal to trigger nanobot restart.

    The parent process (e.g., Docker entrypoint) should monitor for this file
    and restart nanobot when it appears.

    Args:
        workspace: Path to the workspace directory.
        reason: Human-readable reason for the restart.
        verify_job: Optional verification job to schedule after restart.
            Should include: name, at_time (ISO), message, deliver, channel, to
    """
    signal_file = workspace / RESTART_SIGNAL_FILE

    data: dict[str, Any] = {
        "reason": reason,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    if verify_job:
        data["verify_job"] = verify_job

    signal_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Wrote restart signal: {reason}")


def check_and_clear_restart_signal(workspace: Path) -> dict[str, Any] | None:
    """
    Check for restart signal and clear it if found.

    Should be called on startup to handle any pending restart actions.

    Args:
        workspace: Path to the workspace directory.

    Returns:
        Signal data dictionary if found, None otherwise.
    """
    signal_file = workspace / RESTART_SIGNAL_FILE

    if not signal_file.exists():
        return None

    try:
        data = json.loads(signal_file.read_text(encoding="utf-8"))
        signal_file.unlink()
        logger.debug(f"Cleared restart signal: {data.get('reason', 'unknown')}")
        return data
    except Exception as e:
        logger.warning(f"Failed to read restart signal: {e}")
        # Try to remove the file anyway
        try:
            signal_file.unlink()
        except Exception:
            pass
        return None


def has_restart_signal(workspace: Path) -> bool:
    """
    Check if a restart signal file exists.

    Args:
        workspace: Path to the workspace directory.

    Returns:
        True if restart signal exists, False otherwise.
    """
    return (workspace / RESTART_SIGNAL_FILE).exists()
