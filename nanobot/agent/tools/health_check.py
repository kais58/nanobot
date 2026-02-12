"""Health check tool for nanobot error metrics and system status."""

from pathlib import Path
from typing import Any

from nanobot.agent.errors import ErrorLogger, get_error_logger


class HealthCheckTool:
    """
    Health check tool for monitoring nanobot status.

    Provides error metrics, recovery rates, and system health information.
    """

    name = "health_check"
    description = "Check nanobot health status including error metrics and recovery rates"

    def __init__(self, workspace: Path):
        self._workspace = workspace

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "What to check: 'all', 'errors', 'recent', or 'summary'",
                    "enum": ["all", "errors", "recent", "summary"],
                    "default": "summary",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
                "minutes": {
                    "type": "integer",
                    "description": "Time window in minutes for recent errors",
                    "default": 30,
                },
            },
        }

    def validate_params(self, params: dict) -> list[str]:
        errors = []
        scope = params.get("scope", "summary")
        limit = params.get("limit", 10)
        minutes = params.get("minutes", 30)

        if scope not in ["all", "errors", "recent", "summary"]:
            errors.append(f"Invalid scope: {scope}")

        if limit < 1 or limit > 100:
            errors.append(f"Limit must be between 1 and 100, got {limit}")

        if minutes < 1 or minutes > 1440:  # 24 hours
            errors.append(f"Minutes must be between 1 and 1440, got {minutes}")

        return errors

    async def execute(self, **kwargs) -> str:
        """Execute health check and return formatted report."""
        scope = kwargs.get("scope", "summary")
        limit = kwargs.get("limit", 10)
        minutes = kwargs.get("minutes", 30)

        logger = get_error_logger()
        if not logger:
            return "Error logger not initialized."

        if scope == "summary":
            return self._format_summary(logger.get_metrics())
        elif scope == "errors":
            return self._format_top_errors(logger.get_top_errors(limit), limit)
        elif scope == "recent":
            return self._format_recent_errors(logger.get_recent_errors(minutes), minutes, limit)
        elif scope == "all":
            return self._format_full_report(logger, limit, minutes)

        return "Unknown scope."

    def _format_summary(self, metrics: dict[str, Any]) -> str:
        """Format health summary."""
        total_errors = metrics.get("total_errors", 0)
        errors_last_hour = metrics.get("errors_last_hour", 0)

        # Health status
        if errors_last_hour == 0:
            status = "✅ Healthy - No errors in last hour"
        elif errors_last_hour < 5:
            status = "✅ Good - Less than 5 errors in last hour"
        elif errors_last_hour < 20:
            status = "⚠️ Warning - 5-20 errors in last hour"
        else:
            status = "❌ Critical - 20+ errors in last hour"

        recovery_rates = metrics.get("recovery_rates", {})

        # Find worst recovery rate
        worst_recovery = None
        worst_value = 1.0
        for category, rate in recovery_rates.items():
            if rate < worst_value:
                worst_value = rate
                worst_recovery = category

        # Build report
        report = [
            "# 🏥 nanobot Health Summary",
            "",
            f"**Status:** {status}",
            "",
            f"**Total Errors:** {total_errors}",
            f"**Errors Last Hour:** {errors_last_hour}",
            f"**Error Rate:** {errors_last_hour} per hour",
            "",
        ]

        if worst_recovery and worst_value < 0.5:
            report.extend(
                [
                    "⚠️ **Low Recovery Rate**",
                    f"Category '{worst_recovery}' only recovers {worst_value * 100:.0f}% of the time.",
                    "This may indicate a persistent issue that needs attention.",
                    "",
                ]
            )

        if errors_last_hour > 0:
            report.append("## Error Categories (Last Hour)")
            for category, count in metrics.get("errors_by_category", {}).items():
                if count > 0:
                    report.append(f"- {category}: {count} errors")

        return "\n".join(report)

    def _format_top_errors(self, top_errors: list[dict[str, Any]], limit: int) -> str:
        """Format top error categories."""
        report = [
            "# 📊 Top Error Categories",
            "",
        ]

        if not top_errors:
            report.append("No errors recorded.")
            return "\n".join(report)

        report.append(f"Showing top {len(top_errors)} error categories:")
        report.append("")

        for i, err in enumerate(top_errors, 1):
            category = err.get("category", "unknown")
            count = err.get("count", 0)
            recovery = err.get("recovery_rate", 0.0) * 100

            recovery_status = "✅" if recovery > 80 else "⚠️" if recovery > 50 else "❌"

            report.append(
                f"{i}. **{category}** ({count} errors, {recovery:.0f}% recovery) {recovery_status}"
            )

        report.append("")
        report.append(
            "Recovery rate indicates how often the system automatically recovers from this error type."
        )
        report.append("- ✅ >80%: Good automatic recovery")
        report.append("- ⚠️ 50-80%: Partial recovery, some manual intervention needed")
        report.append("- ❌ <50%: Poor recovery, needs investigation")

        return "\n".join(report)

    def _format_recent_errors(
        self, recent_errors: list[dict[str, Any]], minutes: int, limit: int
    ) -> str:
        """Format recent errors."""
        report = [
            f"# 🕐 Recent Errors (Last {minutes} minutes)",
            "",
        ]

        if not recent_errors:
            report.append("No errors recorded in this time window.")
            return "\n".join(report)

        report.append(f"Showing last {min(len(recent_errors), limit)} errors:")
        report.append("")

        from datetime import datetime

        for i, err in enumerate(recent_errors[:limit], 1):
            timestamp = err.get("timestamp", 0)
            if timestamp:
                ts = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            else:
                ts = "unknown"

            category = err.get("category", "unknown")
            message = err.get("error_message", "")[:100]
            tool = err.get("tool_name")
            severity = err.get("severity", "info")

            severity_emoji = (
                "🔴" if severity == "critical" else "🟠" if severity == "error" else "🟡"
            )

            tool_info = f" ({tool})" if tool else ""

            report.append(f"{i}. {severity_emoji} [{ts}] {category}{tool_info}")
            report.append(f"   {message}")

        return "\n".join(report)

    def _format_full_report(self, logger: ErrorLogger, limit: int, minutes: int) -> str:
        """Format full health report."""
        sections = []

        # Summary
        sections.append(self._format_summary(logger.get_metrics()))

        # Top errors
        sections.append("")
        sections.append(self._format_top_errors(logger.get_top_errors(limit), limit))

        # Recent errors
        sections.append("")
        sections.append(
            self._format_recent_errors(logger.get_recent_errors(minutes), minutes, limit)
        )

        return "\n".join(sections)


def format_health_summary(metrics: dict[str, Any]) -> str:
    """Convenience function to format health metrics."""
    total_errors = metrics.get("total_errors", 0)
    errors_last_hour = metrics.get("errors_last_hour", 0)

    if errors_last_hour == 0:
        status = "✅ Healthy"
    elif errors_last_hour < 5:
        status = "✅ Good"
    elif errors_last_hour < 20:
        status = "⚠️ Warning"
    else:
        status = "❌ Critical"

    return f"{status} | {errors_last_hour} errors/hour | {total_errors} total"
