"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import (
        CompactionConfig,
        ContextConfig,
        ExecToolConfig,
        MCPConfig,
        MemoryConfig,
    )
    from nanobot.cron.service import CronService
    from nanobot.mcp import MCPManager
    from nanobot.memory.vectors import VectorStore
    from nanobot.session.manager import Session


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        channel_manager: ChannelManager | None = None,
        cron_service: "CronService | None" = None,
        mcp_config: "MCPConfig | None" = None,
        context_config: "ContextConfig | None" = None,
        compaction_config: "CompactionConfig | None" = None,
        memory_config: "MemoryConfig | None" = None,
        api_key: str | None = None,
    ):
        from nanobot.config.schema import (
            CompactionConfig,
            ContextConfig,
            ExecToolConfig,
            MCPConfig,
            MemoryConfig,
        )

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.channel_manager = channel_manager
        self.cron_service = cron_service
        self.mcp_config = mcp_config or MCPConfig()
        self.context_config = context_config or ContextConfig()
        self.compaction_config = compaction_config or CompactionConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.api_key = api_key

        # Vector store for semantic memory (initialized in run())
        self.vector_store: "VectorStore | None" = None

        self.context = ContextBuilder(workspace, memory_enabled=self.memory_config.enabled)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
        )

        # MCP manager (initialized in run())
        self.mcp_manager: "MCPManager | None" = None

        self._running = False
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())

        # Shell tool
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.exec_config.restrict_to_workspace,
            )
        )

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for self-scheduling)
        if self.cron_service:
            from nanobot.agent.tools.cron import CronTool

            self.tools.register(CronTool(cron_service=self.cron_service))

        # Discord config tool (if Discord channel is enabled)
        if self.channel_manager and "discord" in self.channel_manager.channels:
            from nanobot.agent.tools.discord_config import DiscordSetNotificationChannelTool
            from nanobot.channels.discord import DiscordChannel

            discord_channel = self.channel_manager.channels["discord"]
            if isinstance(discord_channel, DiscordChannel):
                self.tools.register(DiscordSetNotificationChannelTool(discord_channel))

        # MCP install tool (always available for self-installation)
        from nanobot.agent.tools.mcp_install import InstallMCPServerTool

        self.tools.register(InstallMCPServerTool(workspace=self.workspace))

    def _truncate_tool_result(self, result: str) -> str:
        """Truncate tool result to fit within budget."""
        from nanobot.utils.tokens import count_tokens, truncate_to_token_limit

        max_tokens = self.context_config.tool_result_budget
        current_tokens = count_tokens(result)

        if current_tokens <= max_tokens:
            return result

        logger.debug(f"Truncating tool result from {current_tokens} to {max_tokens} tokens")
        return truncate_to_token_limit(result, max_tokens)

    def _check_context_budget(self, messages: list[dict]) -> None:
        """Log warnings if approaching context limits."""
        from nanobot.utils.tokens import count_messages_tokens

        total_tokens = count_messages_tokens(messages)
        max_tokens = self.context_config.max_context_tokens
        threshold = max_tokens * 0.9  # 90% warning threshold

        if total_tokens > threshold:
            logger.warning(
                f"Context approaching limit: {total_tokens}/{max_tokens} tokens "
                f"({total_tokens / max_tokens * 100:.1f}%)"
            )
        elif total_tokens > max_tokens * 0.8:
            logger.debug(f"Context at {total_tokens / max_tokens * 100:.1f}% capacity")

    async def _maybe_compact(
        self,
        messages: list[dict],
        session: "Session",
    ) -> list[dict]:
        """Run compaction if context is near capacity and compaction is enabled."""
        from nanobot.agent.compaction import MessageCompactor, should_compact

        if not self.compaction_config.enabled:
            return messages

        max_tokens = self.context_config.max_context_tokens
        threshold = self.compaction_config.threshold

        if not await should_compact(messages, max_tokens, threshold):
            return messages

        # Create compactor
        compactor = MessageCompactor(
            provider=self.provider,
            model=self.compaction_config.model or self.model,
            keep_recent=self.compaction_config.keep_recent,
        )

        # Get previous rolling summary
        previous_summary = session.get_rolling_summary()

        # Run compaction
        target_tokens = int(max_tokens * 0.6)  # Compact to 60% capacity
        compacted, new_summary = await compactor.compact_messages(
            messages, target_tokens, previous_summary
        )

        # Store new rolling summary
        if new_summary != previous_summary:
            session.set_rolling_summary(new_summary)

        return compacted

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True

        # Initialize MCP if enabled
        await self._init_mcp()

        # Initialize memory if enabled
        self._init_memory()

        # Check for restart signal (e.g., after MCP server installation)
        await self._check_restart_signal()

        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response (include metadata for reaction/typing cleanup)
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"Sorry, I encountered an error: {str(e)}",
                            metadata=msg.metadata,
                        )
                    )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False

        # Stop MCP servers
        if self.mcp_manager:
            await self.mcp_manager.stop()

        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.

        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")

        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)

        # Update cron tool context
        from nanobot.agent.tools.cron import CronTool

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        # Update MCP install tool context
        from nanobot.agent.tools.mcp_install import InstallMCPServerTool

        mcp_install_tool = self.tools.get("install_mcp_server")
        if isinstance(mcp_install_tool, InstallMCPServerTool):
            mcp_install_tool.set_context(msg.channel, msg.chat_id)

        # Extract channel context from metadata (e.g., Discord channel history)
        channel_context = msg.metadata.get("channel_context", "") if msg.metadata else ""

        # Auto-recall relevant memories
        memory_context = await self._auto_recall(msg.content)
        if memory_context:
            # Prepend memory context to channel context
            if channel_context:
                channel_context = f"{memory_context}\n\n{channel_context}"
            else:
                channel_context = memory_context

        # Build initial messages (use token-aware history)
        messages = self.context.build_messages(
            history=session.get_history(max_tokens=self.context_config.history_budget),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel_context=channel_context,
        )

        # Run compaction if needed
        messages = await self._maybe_compact(messages, session)

        # Agent loop
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            # Check context budget before LLM call
            self._check_context_budget(messages)

            # Call LLM
            response = await self.provider.chat(
                messages=messages, tools=self.tools.get_definitions(), model=self.model
            )

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),  # Must be JSON string
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    # Truncate large tool results to fit within budget
                    result = self._truncate_tool_result(result)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        # Index conversation to memory (async, don't block response)
        await self._index_conversation(msg.session_key, msg.content, final_content)

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content, metadata=msg.metadata
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        # Update cron tool context
        from nanobot.agent.tools.cron import CronTool

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)

        # Update MCP install tool context
        from nanobot.agent.tools.mcp_install import InstallMCPServerTool

        mcp_install_tool = self.tools.get("install_mcp_server")
        if isinstance(mcp_install_tool, InstallMCPServerTool):
            mcp_install_tool.set_context(origin_channel, origin_chat_id)

        # Build messages with the announce content (token-aware history)
        messages = self.context.build_messages(
            history=session.get_history(max_tokens=self.context_config.history_budget),
            current_message=msg.content,
        )

        # Run compaction if needed
        messages = await self._maybe_compact(messages, session)

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            # Check context budget before LLM call
            self._check_context_budget(messages)

            response = await self.provider.chat(
                messages=messages, tools=self.tools.get_definitions(), model=self.model
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    # Truncate large tool results to fit within budget
                    result = self._truncate_tool_result(result)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "Background task completed."

        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel, chat_id=origin_chat_id, content=final_content
        )

    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """
        Process a message directly (for CLI usage).

        Args:
            content: The message content.
            session_key: Session identifier.

        Returns:
            The agent's response.
        """
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=content)

        response = await self._process_message(msg)
        return response.content if response else ""

    async def _auto_recall(self, user_message: str) -> str | None:
        """
        Automatically search memory for context relevant to the user's message.

        Args:
            user_message: The user's incoming message.

        Returns:
            Relevant memory context, or None if disabled or no results.
        """
        if not self.memory_config.enabled or not self.memory_config.auto_recall:
            return None

        if not self.vector_store:
            return None

        try:
            results = await self.vector_store.search(
                query=user_message,
                top_k=self.memory_config.search_top_k,
                min_similarity=self.memory_config.min_similarity,
            )

            if not results:
                return None

            # Format results as context
            memory_context = ["[Relevant memories from past conversations]"]
            for r in results:
                text = r.get("text", "")[:300]  # Limit length
                memory_context.append(f"- {text}")

            return "\n".join(memory_context)

        except Exception as e:
            logger.warning(f"Auto-recall failed: {e}")
            return None

    async def _index_conversation(
        self,
        session_key: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Index a conversation turn to the vector store.

        Args:
            session_key: Session identifier.
            user_message: The user's message.
            assistant_message: The assistant's response.
        """
        if not self.memory_config.enabled or not self.memory_config.index_conversations:
            return

        if not self.vector_store:
            return

        try:
            # Create conversation turn text
            turn_text = f"User: {user_message}\nAssistant: {assistant_message}"

            metadata = {
                "session_key": session_key,
                "type": "conversation",
            }

            await self.vector_store.add(turn_text, metadata)

            # Extract and index facts if enabled
            if self.memory_config.extract_facts:
                await self._extract_and_index_facts(session_key, user_message, assistant_message)

        except Exception as e:
            logger.warning(f"Failed to index conversation: {e}")

    async def _extract_and_index_facts(
        self,
        session_key: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Extract and index key facts from a conversation turn."""
        try:
            from nanobot.memory.extractor import FactExtractor

            # Use extraction model or fall back to compaction model or main model
            extraction_model = (
                self.memory_config.extraction_model or self.compaction_config.model or self.model
            )

            extractor = FactExtractor(
                provider=self.provider,
                model=extraction_model,
            )

            facts = await extractor.extract_from_turn(user_message, assistant_message)

            for fact in facts:
                metadata = {
                    "session_key": session_key,
                    "type": "fact",
                }
                await self.vector_store.add(fact, metadata)

            if facts:
                logger.debug(f"Indexed {len(facts)} facts from conversation")

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

    def _init_memory(self) -> None:
        """Initialize semantic memory system."""
        if not self.memory_config.enabled:
            return

        try:
            from nanobot.agent.tools.memory_search import MemorySearchTool
            from nanobot.llm.embeddings import EmbeddingService
            from nanobot.memory.vectors import VectorStore

            # Create embedding service
            embedding_service = EmbeddingService(
                model=self.memory_config.embedding_model,
                api_key=self.api_key,
            )

            # Create vector store
            self.vector_store = VectorStore(
                db_path=self.memory_config.db_path,
                embedding_service=embedding_service,
            )

            # Register memory search tool
            self.tools.register(MemorySearchTool(self.vector_store))

            logger.info(
                f"Memory initialized: {self.vector_store.count()} entries, "
                f"model={self.memory_config.embedding_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self.vector_store = None

    async def _init_mcp(self) -> None:
        """Initialize MCP manager and register MCP tools."""
        if not self.mcp_config.enabled:
            return

        from nanobot.mcp import MCPManager, create_mcp_tool_proxies

        self.mcp_manager = MCPManager(self.mcp_config)
        await self.mcp_manager.start()

        # Register MCP tool proxies
        mcp_tools = create_mcp_tool_proxies(self.mcp_manager)
        for tool in mcp_tools:
            self.tools.register(tool)
            logger.debug(f"Registered MCP tool: {tool.name}")

        # Log any server errors for diagnosis
        for server_name, error in self.mcp_manager.server_errors.items():
            logger.warning(f"MCP server {server_name} failed: {error}")

        logger.info(f"MCP initialized: {len(mcp_tools)} tools available")

    async def _check_restart_signal(self) -> None:
        """Check for restart signal and schedule verification job if needed."""
        from nanobot.restart import check_and_clear_restart_signal

        signal = check_and_clear_restart_signal(self.workspace)
        if not signal:
            return

        logger.info(f"Restart signal detected: {signal.get('reason', 'unknown')}")

        # Schedule verification job if present
        verify_job = signal.get("verify_job")
        if verify_job and self.cron_service:
            from datetime import datetime

            from nanobot.cron.types import CronSchedule

            at_time = verify_job.get("at_time")
            if at_time:
                try:
                    dt = datetime.fromisoformat(at_time.replace("Z", "+00:00"))
                    at_ms = int(dt.timestamp() * 1000)

                    self.cron_service.add_job(
                        name=verify_job.get("name", "verify_mcp"),
                        schedule=CronSchedule(kind="at", at_ms=at_ms),
                        message=verify_job.get("message", "Verify MCP installation"),
                        deliver=verify_job.get("deliver", True),
                        channel=verify_job.get("channel"),
                        to=verify_job.get("to"),
                        delete_after_run=True,
                    )
                    logger.info(f"Scheduled verification job: {verify_job.get('name')}")
                except Exception as e:
                    logger.error(f"Failed to schedule verification job: {e}")
