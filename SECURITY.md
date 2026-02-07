# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it through one of these channels:

1. **GitHub Issue**: Open an issue at [MTAAP/nanobot](https://github.com/MTAAP/nanobot/issues) with the `security` label
2. **Discord**: Send a message in our Discord server
3. Include: description, reproduction steps, and potential impact

## API Key Management

- **Never** commit API keys to version control
- Store keys in `~/.nanobot/config.json` (mode `0600`)
- Use environment variables (`NANOBOT_PROVIDERS__OPENROUTER__API_KEY`) as an alternative
- Rotate keys immediately if exposed
- Each provider (OpenRouter, Anthropic, OpenAI, etc.) should use its own key

## Channel Access Control

### Telegram
- Set `allowFrom` to restrict which user IDs can interact
- Use a unique bot token per deployment
- Enable proxy if operating in restricted networks

### WhatsApp
- Set `allowFrom` to restrict which phone numbers can interact
- The bridge runs locally; do not expose port 3001 to the internet

### Discord
- Set `allowFrom` to restrict which user IDs can interact
- Use `guildId` to limit to a single server
- Bot token should have minimal permissions (Send Messages, Read Message History)

### Feishu/Lark
- Set `allowFrom` to restrict which user open_ids can interact
- Store `appSecret` and `encryptKey` securely

## Shell Execution Safety

The `exec` tool runs shell commands. Mitigate risk with:

- **`restrictToWorkspace`**: When `true`, blocks commands accessing paths outside the workspace. Set this at the `tools` level in config to apply to all file-access tools (read, write, edit, exec).
- **Deny patterns**: Dangerous commands (`rm -rf`, `dd`, `shutdown`, fork bombs) are blocked by default.
- **Timeout**: Commands are killed after the configured timeout (default: 60s).
- **Working directory**: Commands run in the configured workspace directory.

### Recommended production config

```json
{
  "tools": {
    "restrictToWorkspace": true,
    "exec": {
      "timeout": 30
    }
  }
}
```

## File System Access

File tools (read_file, write_file, edit_file, list_dir) operate on the local filesystem. When `tools.restrictToWorkspace` is enabled, these tools are restricted to the workspace directory.

- Avoid granting access to sensitive directories (`~/.ssh`, `~/.gnupg`, `/etc`)
- Use workspace isolation to contain agent file operations

## Memory Storage Security

- **Vector database**: Stored at `~/.nanobot/memory/vectors.db` (SQLite). Contains conversation history and extracted facts. Protect with file permissions.
- **Core memory**: Stored in the workspace as plaintext. Contains user preferences and key context.
- **Entity database**: Stored at `~/.nanobot/memory/entities.db`. Contains extracted entities and relations.
- **Recommendation**: Encrypt the `~/.nanobot` directory at rest on shared systems.

## MCP Server Trust Model

MCP (Model Context Protocol) servers extend nanobot's capabilities:

- Only install MCP servers from trusted sources
- Review server packages before installation (`npx`, `uvx`, `pip`)
- MCP servers run as child processes with the same permissions as nanobot
- Use `tools.mcp.tools` config to disable specific MCP tools
- Set `enabled: false` on unused MCP servers

## Daemon Mode Safety

The heartbeat/daemon system runs autonomous agent iterations:

- **Strategy file** (`HEARTBEAT.md`): Review and audit regularly. The agent executes tasks listed here.
- **Max iterations**: Limits agent loop depth per heartbeat (default: 25).
- **Cooldown**: Prevents rapid-fire execution after actions (default: 600s).
- **Triage model**: Use a cheaper model for triage to limit cost exposure.
- Disable daemon mode (`daemon.enabled: false`) if unattended execution is not desired.

## Provider Key Isolation

The `ProviderResolver` routes different subsystems to different providers:

- Subsystem providers (compaction, extraction, embedding) can use separate API keys
- This limits blast radius if a single key is compromised
- Configure per-subsystem providers in `agents.defaults.compaction.provider`, `agents.defaults.memory.embeddingProvider`, etc.

## Network Security

- nanobot gateway binds to `0.0.0.0:18790` by default. Restrict to `127.0.0.1` if not serving external clients.
- The WhatsApp bridge uses a local WebSocket (`ws://localhost:3001`). Do not expose externally.
- Web tools (`web_search`, `web_fetch`) make outbound HTTP requests. Use network policies to restrict if needed.

## Deployment Hardening

1. Run nanobot as a non-root user
2. Use `restrictToWorkspace: true` at the tools level
3. Set `allowFrom` on all enabled channels
4. Set file permissions: `chmod 600 ~/.nanobot/config.json`
5. Review `HEARTBEAT.md` before enabling daemon mode
6. Monitor logs (`loguru` output) for unexpected tool calls
7. Keep dependencies updated: `pip install --upgrade nanobot`

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| < latest | Best effort |
