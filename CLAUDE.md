# nanobot

You are working on **nanobot**, an ultra-lightweight personal AI assistant framework (~5,145 LOC Python).

## Critical: Question Ambiguities

Before implementing any feature or task, you MUST aggressively clarify ambiguities. Do not assume. Do not guess. Ask the user directly:

- **Scope**: "What exactly should this feature do? What should it NOT do?"
- **Edge cases**: "How should this handle [specific edge case]?"
- **Integration**: "How does this interact with existing [component]?"
- **Error states**: "What should happen when [failure scenario]?"
- **User expectations**: "What does success look like for this feature?"
- **Priorities**: "If we can't have everything, what's essential vs nice-to-have?"

Create detailed feature specifications BEFORE writing code. Push back on vague requirements. A well-defined feature prevents wasted implementation cycles.

## Architecture

```
User -> Channel -> MessageBus -> AgentLoop -> LLM
                        |              |
                        v              v
                   SessionStore    ToolRegistry
```

- **Channels**: Telegram, WhatsApp, Feishu/Lark - extend `BaseChannel`
- **Message Bus**: Decouples channels from agent via async queues
- **Agent Loop**: Processes messages, manages tool calls, streams responses
- **Tools**: Extend `Tool` base class with JSON Schema validation

## Key Directories

```
nanobot/
├── agent/          # Core agent loop, context assembly, tool execution
├── channels/       # Telegram, WhatsApp, Feishu implementations
├── bus/            # Message bus (async queue-based)
├── config/         # Pydantic config schema and loading
├── llm/            # LiteLLM provider wrapper
├── session/        # JSONL-based conversation storage
└── cli/            # Typer CLI commands
bridge/             # Node.js WhatsApp Web.js bridge
tests/              # pytest test suite
```

## Tech Stack

- Python 3.11+ with Hatchling build system
- Node.js 20+ for WhatsApp bridge only
- LiteLLM for LLM integration (100+ models)
- Pydantic for configuration validation
- Typer for CLI
- Config location: `~/.nanobot/config.json`

## Commands You Should Run

```bash
# Testing
pytest                          # Run all tests
pytest tests/test_agent.py      # Run specific test file

# Linting & Formatting
ruff check nanobot/             # Check linting
ruff format nanobot/            # Format code

# Full verification - run after any code changes
pytest && ruff check nanobot/ && ruff format --check nanobot/

# WhatsApp Bridge
cd bridge && npm run build && npm start
```

## Key Files

| File | Purpose |
|------|---------|
| `nanobot/agent/loop.py` | Core agent processing loop |
| `nanobot/agent/context.py` | System prompt assembly |
| `nanobot/agent/tools/base.py` | Tool interface (`Tool` ABC) |
| `nanobot/channels/base.py` | Channel interface (`BaseChannel` ABC) |
| `nanobot/config/schema.py` | Pydantic config models |
| `nanobot/bus/queue.py` | Message bus implementation |
| `nanobot/llm/provider.py` | LiteLLM wrapper |

## Code Patterns You Must Follow

### Async-First
Everything is async. Use `async def` and `await` consistently:
```python
async def process_message(self, message: Message) -> AsyncIterator[str]:
    async for chunk in self.llm.stream(messages):
        yield chunk
```

### Abstract Base Classes
Extend `BaseChannel` for new channels, `Tool` for new tools:
```python
class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {...}}

    async def execute(self, **kwargs) -> str:
        return "result"
```

### Error Handling in Tools
Return error strings. Do not raise exceptions:
```python
async def execute(self, **kwargs) -> str:
    try:
        result = await do_work()
        return result
    except SomeError as e:
        return f"Error: {e}"
```

### Type Hints
Use Python 3.11+ union syntax. Do not use `Optional`:
```python
def process(self, data: str | None = None) -> dict[str, Any]:
    ...
```

## Rules You Must Follow

- **Line length**: 100 characters max
- **Logging**: Use loguru. No emojis in debug messages.
- **File I/O**: Always use `encoding="utf-8"`
- **Single implementation**: When refactoring, remove legacy code completely. Do not leave alternatives.
- **Type hints**: Required on all function signatures
- **Imports**: Use absolute imports from `nanobot.` package
- **Verification**: Run `pytest && ruff check nanobot/` after any code change

## Workflow

1. **Use git worktrees**: Always work in a git worktree to enable parallel work with other agents. Create a worktree before starting any implementation:
   ```bash
   git worktree add ../nanobot-feature-name feature-branch-name
   cd ../nanobot-feature-name
   ```
   This prevents conflicts when multiple agents work on the codebase simultaneously.
2. **Clarify first**: Question the user on all ambiguities before implementing
3. **Plan complex tasks**: Use Plan mode to explore the codebase before proposing changes
4. **Verify changes**: Run tests and linting after every modification
5. **Prove correctness**: Demonstrate that changes work with tests or examples
6. **Use subagents**: For multi-file exploration or complex refactoring

## GitHub

This is a fork of `HKUDS/nanobot`. PRs must target **MTAAP/nanobot**, not the upstream.

After cloning, run once: `gh repo set-default MTAAP/nanobot`

## Mistakes to Avoid

- Forgetting `encoding="utf-8"` on file operations
- Using `Optional[X]` instead of `X | None`
- Raising exceptions in tool `execute()` methods instead of returning error strings
- Leaving emoji in debug/log messages
- Implementing features without clarifying requirements first
- Assuming user intent instead of asking
- Creating PRs against the upstream `HKUDS/nanobot` instead of `MTAAP/nanobot`
