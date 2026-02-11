# Self-Evolution Roadmap for nanobot

**Goal**: Continuously improve nanobot through spec-driven development using OpenSpec workflows.

## Current State Analysis

### Architecture Overview
- **Core**: Agent loop (~7600 LOC) with tool execution and LLM orchestration
- **Heartbeat**: Three-tier daemon (gather context → triage → execute)
- **Memory**: Vector store (SQLite + embeddings), core memory, entities, proactive reminders
- **MCP**: Extensible tool system for external servers
- **Channels**: Telegram, Discord, WhatsApp, Feishu
- **Self-Evolution**: `SelfEvolveManager` class for PR-based code changes
- **Cron**: Scheduled tasks with SQLite persistence

### Existing Workflows
- `.github/workflows/deploy.yml` - Deployment automation
- `.github/workflows/pr-check.yml` - PR validation
- **No OpenSpec workflow** (need to add)

### Recent PRs (from git log)
- Discord message fetch fix
- Discord mention context
- Cron timezone awareness
- Heartbeat duplicate spawn prevention
- Action intent tool enforcement

---

## Self-Evolution Goals

### Phase 1: OpenSpec Integration
- [ ] Add OpenSpec workflow to `.github/workflows/openspec-interview-pr.yml`
- [ ] Add OpenSpec workflow to `.github/workflows/openspec-proposal-from-pr.yml`
- [ ] Create `temp/` directory structure
- [ ] Test interview workflow on a feature issue

### Phase 2: Enhanced Error Logging
- [ ] Create structured error logger in `nanobot/agent/errors.py`
- [ ] Add error categorization (tool failures, LLM errors, MCP errors, memory errors)
- [ ] Add self-evaluation metrics (error rates, recovery rates)
- [ ] Create dashboard/health check endpoint for error stats

### Phase 3: Skill Discovery & Learning
- [ ] Automated web search for new AI/agent technologies weekly
- [ ] Create feature request issues based on research
- [ ] Use `/learn` skill for new technologies before implementation
- [ ] Auto-generate skills from research findings

### Phase 4: Performance & Reliability
- [ ] Add APM integration (OpenTelemetry or similar)
- [ ] Implement circuit breakers for external API calls
- [ ] Add retry logic with exponential backoff
- [ ] Optimize memory extraction/consolidation pipeline

### Phase 5: New Capabilities
- [ ] Multi-modal support (images, voice) - from roadmap
- [ ] Better reasoning with multi-step planning (ReWOO?)
- [ ] Long-term memory improvements
- [ ] Slack and email integrations

---

## Agentic Technologies to Evaluate

Based on prior research, 5 technologies to evaluate:

1. **ReWOO** (Reasoning Without Observation)
   - Benefit: Decouple planning from execution
   - Implementation: Add planning phase before tool execution loop
   - Use case: HEARTBEAT.md task batching

2. **AutoGen** (Microsoft)
   - Benefit: Multi-agent parallel processing
   - Implementation: Add multi-agent coordinator
   - Use case: WedPilot interview parallelization

3. **Cognee** (Memory for AI Agents)
   - Benefit: Knowledge graph extraction
   - Implementation: Replace/add to vector store
   - Use case: Better context retrieval for WedPilot architecture

4. **LangGraph / LATS**
   - Benefit: Tree-based reasoning with backtracking
   - Implementation: Add tree search to decision making
   - Use case: Self-evolution task planning

5. **Langbase Memory** (Serverless RAG)
   - Benefit: Cheaper vector storage
   - Implementation: Alternative embedding backend
   - Use case: Cost optimization for large-scale deployments

---

## Current Error Patterns (from TOOLS.md)

Review TOOLS.md for recurring patterns and create fixes.

---

## HEARTBEAT.md Tasks

Background monitoring and self-improvement tasks go here (below).
