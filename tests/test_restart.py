"""Tests for restart signal utilities."""

import json
from pathlib import Path

import pytest

from nanobot.restart import (
    RESTART_SIGNAL_FILE,
    check_and_clear_restart_signal,
    has_restart_signal,
    write_restart_signal,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def test_write_restart_signal(workspace: Path) -> None:
    """Test writing restart signal."""
    write_restart_signal(workspace, reason="Test restart")

    signal_file = workspace / RESTART_SIGNAL_FILE
    assert signal_file.exists()

    data = json.loads(signal_file.read_text(encoding="utf-8"))
    assert data["reason"] == "Test restart"
    assert "timestamp" in data


def test_write_restart_signal_with_verify_job(workspace: Path) -> None:
    """Test writing restart signal with verification job."""
    verify_job = {
        "name": "verify_test",
        "at_time": "2025-02-05T12:00:00Z",
        "message": "Verify test installation",
        "deliver": True,
        "channel": "telegram",
        "to": "123456",
    }

    write_restart_signal(workspace, reason="MCP installed", verify_job=verify_job)

    signal_file = workspace / RESTART_SIGNAL_FILE
    data = json.loads(signal_file.read_text(encoding="utf-8"))

    assert data["reason"] == "MCP installed"
    assert data["verify_job"]["name"] == "verify_test"
    assert data["verify_job"]["at_time"] == "2025-02-05T12:00:00Z"
    assert data["verify_job"]["deliver"] is True


def test_check_and_clear_restart_signal(workspace: Path) -> None:
    """Test checking and clearing restart signal."""
    write_restart_signal(workspace, reason="Test clear")

    signal_file = workspace / RESTART_SIGNAL_FILE
    assert signal_file.exists()

    data = check_and_clear_restart_signal(workspace)
    assert data is not None
    assert data["reason"] == "Test clear"
    assert not signal_file.exists()


def test_check_and_clear_no_signal(workspace: Path) -> None:
    """Test checking when no restart signal exists."""
    data = check_and_clear_restart_signal(workspace)
    assert data is None


def test_has_restart_signal(workspace: Path) -> None:
    """Test has_restart_signal check."""
    assert has_restart_signal(workspace) is False

    write_restart_signal(workspace, reason="Test")
    assert has_restart_signal(workspace) is True

    check_and_clear_restart_signal(workspace)
    assert has_restart_signal(workspace) is False


def test_check_and_clear_invalid_json(workspace: Path) -> None:
    """Test handling of invalid JSON in signal file."""
    signal_file = workspace / RESTART_SIGNAL_FILE
    signal_file.write_text("not valid json", encoding="utf-8")

    data = check_and_clear_restart_signal(workspace)
    assert data is None
    # File should be removed even if invalid
    assert not signal_file.exists()
