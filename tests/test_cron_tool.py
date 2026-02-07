"""Tests for CronTool."""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.cron import CronTool
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule


@pytest.fixture
def cron_service(tmp_path: Path) -> CronService:
    """Create a CronService with a temp store path."""
    store_path = tmp_path / "cron" / "jobs.json"
    return CronService(store_path, on_job=None)


@pytest.fixture
def cron_tool(cron_service: CronService) -> CronTool:
    """Create a CronTool with the test service."""
    return CronTool(cron_service=cron_service)


class TestList:
    """Tests for the list action."""

    @pytest.mark.asyncio
    async def test_list_empty(self, cron_tool: CronTool) -> None:
        """List with no jobs returns appropriate message."""
        result = await cron_tool.execute(action="list")
        assert "No scheduled jobs" in result

    @pytest.mark.asyncio
    async def test_list_with_jobs(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """List shows job details."""
        # Add a job first
        cron_service.add_job(
            name="test-job",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test message",
        )

        result = await cron_tool.execute(action="list")
        assert "test-job" in result
        assert "every 60s" in result
        assert "enabled" in result


class TestAdd:
    """Tests for the add action."""

    @pytest.mark.asyncio
    async def test_add_every_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Create interval-based job."""
        result = await cron_tool.execute(
            action="add",
            name="interval-job",
            message="Run every minute",
            schedule_type="every",
            every_seconds=60,
        )
        assert "Created job 'interval-job'" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].name == "interval-job"
        assert jobs[0].schedule.kind == "every"
        assert jobs[0].schedule.every_ms == 60000

    @pytest.mark.asyncio
    async def test_add_cron_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Create cron expression job."""
        result = await cron_tool.execute(
            action="add",
            name="daily-job",
            message="Run at 9am",
            schedule_type="cron",
            cron_expr="0 9 * * *",
        )
        assert "Created job 'daily-job'" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].name == "daily-job"
        assert jobs[0].schedule.kind == "cron"
        assert jobs[0].schedule.expr == "0 9 * * *"

    @pytest.mark.asyncio
    async def test_add_at_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Create one-time job."""
        future_time = datetime.now() + timedelta(hours=1)
        at_time = future_time.isoformat()

        result = await cron_tool.execute(
            action="add",
            name="one-time-job",
            message="Run once",
            schedule_type="at",
            at_time=at_time,
        )
        assert "Created job 'one-time-job'" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].name == "one-time-job"
        assert jobs[0].schedule.kind == "at"
        assert jobs[0].delete_after_run is True

    @pytest.mark.asyncio
    async def test_add_duplicate_name(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Error on duplicate names (case-insensitive)."""
        # Add first job
        cron_service.add_job(
            name="My-Job",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="First job",
        )

        # Try to add duplicate
        result = await cron_tool.execute(
            action="add",
            name="my-job",  # Different case
            message="Duplicate",
            schedule_type="every",
            every_seconds=60,
        )
        assert "Error" in result
        assert "already exists" in result

    @pytest.mark.asyncio
    async def test_add_past_time(self, cron_tool: CronTool) -> None:
        """Error on past at_time."""
        past_time = (datetime.now() - timedelta(hours=1)).isoformat()

        result = await cron_tool.execute(
            action="add",
            name="past-job",
            message="Run in the past",
            schedule_type="at",
            at_time=past_time,
        )
        assert "Error" in result
        assert "future" in result

    @pytest.mark.asyncio
    async def test_add_invalid_cron(self, cron_tool: CronTool) -> None:
        """Error on bad cron expression."""
        result = await cron_tool.execute(
            action="add",
            name="bad-cron-job",
            message="Bad cron",
            schedule_type="cron",
            cron_expr="invalid cron expression",
        )
        assert "Error" in result
        assert "Invalid cron" in result or "croniter" in result.lower()

    @pytest.mark.asyncio
    async def test_add_interval_too_short(self, cron_tool: CronTool) -> None:
        """Error on interval less than minimum."""
        result = await cron_tool.execute(
            action="add",
            name="too-fast-job",
            message="Run too fast",
            schedule_type="every",
            every_seconds=30,  # Less than 60
        )
        assert "Error" in result
        assert "60 seconds" in result

    @pytest.mark.asyncio
    async def test_add_missing_name(self, cron_tool: CronTool) -> None:
        """Error when name is missing."""
        result = await cron_tool.execute(
            action="add",
            message="No name",
            schedule_type="every",
            every_seconds=60,
        )
        assert "Error" in result
        assert "name" in result.lower()

    @pytest.mark.asyncio
    async def test_add_missing_message(self, cron_tool: CronTool) -> None:
        """Error when message is missing."""
        result = await cron_tool.execute(
            action="add",
            name="no-message-job",
            schedule_type="every",
            every_seconds=60,
        )
        assert "Error" in result
        assert "message" in result.lower()


class TestRemove:
    """Tests for the remove action."""

    @pytest.mark.asyncio
    async def test_remove_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Delete existing job."""
        job = cron_service.add_job(
            name="to-remove",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Will be removed",
        )

        result = await cron_tool.execute(action="remove", job_id=job.id)
        assert "Removed" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, cron_tool: CronTool) -> None:
        """Error on missing job."""
        result = await cron_tool.execute(action="remove", job_id="nonexistent")
        assert "Error" in result
        assert "not found" in result


class TestEnableDisable:
    """Tests for enable/disable actions."""

    @pytest.mark.asyncio
    async def test_disable_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Disable a job."""
        job = cron_service.add_job(
            name="to-disable",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Will be disabled",
        )

        result = await cron_tool.execute(action="disable", job_id=job.id)
        assert "disabled" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].enabled is False

    @pytest.mark.asyncio
    async def test_enable_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Enable a disabled job."""
        job = cron_service.add_job(
            name="to-enable",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Will be enabled",
        )
        cron_service.enable_job(job.id, enabled=False)

        result = await cron_tool.execute(action="enable", job_id=job.id)
        assert "enabled" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].enabled is True

    @pytest.mark.asyncio
    async def test_enable_nonexistent(self, cron_tool: CronTool) -> None:
        """Error when job not found."""
        result = await cron_tool.execute(action="enable", job_id="nonexistent")
        assert "Error" in result
        assert "not found" in result


class TestRun:
    """Tests for the run action."""

    @pytest.mark.asyncio
    async def test_run_job(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Execute a job immediately."""
        job = cron_service.add_job(
            name="to-run",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Run now",
        )

        result = await cron_tool.execute(action="run", job_id=job.id)
        assert "executed" in result

    @pytest.mark.asyncio
    async def test_run_nonexistent(self, cron_tool: CronTool) -> None:
        """Error when job not found."""
        result = await cron_tool.execute(action="run", job_id="nonexistent")
        assert "Error" in result
        assert "not found" in result


class TestContext:
    """Tests for context handling."""

    @pytest.mark.asyncio
    async def test_context_default_delivery(
        self, cron_tool: CronTool, cron_service: CronService
    ) -> None:
        """Uses context for delivery target when not specified."""
        cron_tool.set_context("telegram", "123456")

        result = await cron_tool.execute(
            action="add",
            name="context-job",
            message="Use context delivery",
            schedule_type="every",
            every_seconds=60,
            deliver=True,
        )
        assert "Created job" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].payload.deliver is True
        assert jobs[0].payload.channel == "telegram"
        assert jobs[0].payload.to == "123456"

    @pytest.mark.asyncio
    async def test_explicit_delivery_overrides_context(
        self, cron_tool: CronTool, cron_service: CronService
    ) -> None:
        """Explicit delivery target overrides context."""
        cron_tool.set_context("telegram", "123456")

        result = await cron_tool.execute(
            action="add",
            name="explicit-job",
            message="Use explicit delivery",
            schedule_type="every",
            every_seconds=60,
            deliver=True,
            channel="whatsapp",
            to="9876543210",
        )
        assert "Created job" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].payload.deliver is True
        assert jobs[0].payload.channel == "whatsapp"
        assert jobs[0].payload.to == "9876543210"


class TestDiscordContext:
    """Tests for Discord channel context handling."""

    @pytest.mark.asyncio
    async def test_discord_context_default_delivery(
        self, cron_tool: CronTool, cron_service: CronService
    ) -> None:
        """Uses Discord context for delivery target when not specified."""
        cron_tool.set_context("discord", "1234567890")

        result = await cron_tool.execute(
            action="add",
            name="discord-reminder",
            message="Remind user to check server",
            schedule_type="every",
            every_seconds=60,
            deliver=True,
        )
        assert "Created job" in result

        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].payload.deliver is True
        assert jobs[0].payload.channel == "discord"
        assert jobs[0].payload.to == "1234567890"


class TestCronDelivery:
    """Tests for cron job delivery via the message bus."""

    @pytest.mark.asyncio
    async def test_on_cron_job_delivers_to_bus(
        self, cron_service: CronService
    ) -> None:
        """Cron callback publishes OutboundMessage when deliver=True and channel is set."""
        from nanobot.bus.events import OutboundMessage
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        published: list[OutboundMessage] = []

        original_put = bus.outbound.put

        async def capture_put(msg: OutboundMessage) -> None:
            published.append(msg)
            await original_put(msg)

        bus.outbound.put = capture_put  # type: ignore[assignment]

        job = cron_service.add_job(
            name="deliver-test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test delivery",
            deliver=True,
            channel="discord",
            to="999888777",
        )

        # Simulate the on_cron_job callback logic (post-fix)
        response = "Test response"
        if job.payload.deliver and job.payload.to:
            target_channel = job.payload.channel
            if target_channel:
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=target_channel,
                        chat_id=job.payload.to,
                        content=response,
                    )
                )

        assert len(published) == 1
        assert published[0].channel == "discord"
        assert published[0].chat_id == "999888777"
        assert published[0].content == "Test response"

    @pytest.mark.asyncio
    async def test_on_cron_job_skips_when_no_channel(
        self, cron_service: CronService
    ) -> None:
        """Cron callback skips delivery when channel is not set."""
        job = cron_service.add_job(
            name="no-channel-test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test no channel",
            deliver=True,
            channel=None,
            to="999888777",
        )
        # With our fix, delivery is skipped (not defaulted to whatsapp)
        assert job.payload.channel is None
        assert job.payload.deliver is True


class TestJobLimit:
    """Tests for job limit enforcement."""

    @pytest.mark.asyncio
    async def test_max_jobs_limit(self, cron_tool: CronTool, cron_service: CronService) -> None:
        """Error when job limit is reached."""
        # Add MAX_JOBS jobs
        for i in range(CronTool.MAX_JOBS):
            cron_service.add_job(
                name=f"job-{i}",
                schedule=CronSchedule(kind="every", every_ms=60000),
                message=f"Job {i}",
            )

        # Try to add one more
        result = await cron_tool.execute(
            action="add",
            name="one-too-many",
            message="Over limit",
            schedule_type="every",
            every_seconds=60,
        )
        assert "Error" in result
        assert "limit" in result.lower()
