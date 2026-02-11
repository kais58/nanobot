---
name: self-evolution
description: Self-evolution capabilities for nanobot - automated learning, error analysis, and spec-driven improvement
version: 1.0.0
---

# Self-Evolution Skill

**Purpose**: Enable nanobot to continuously improve itself through automated learning, error analysis, and spec-driven development using OpenSpec workflows.

---

## When to Use

This skill activates when:
- Self-evaluating capabilities or planning improvements
- Analyzing error patterns and creating fixes
- Learning new technologies for feature implementation
- Creating spec-driven features via OpenSpec workflows
- Evaluating agentic AI technologies for integration

---

## Self-Evolution Principles

### 1. Spec-Driven Development
All code changes go through OpenSpec workflows:
1. **Create issue**: Describe the feature or fix needed
2. **Interview**: Use `/interview` workflow to gather requirements
3. **Proposal**: Generate spec files (proposal.md, spec.md, tasks.md)
4. **PR**: Create pull request via `SelfEvolveManager`
5. **Review**: Peer review before merge to main
6. **Never push directly** to protected branches

### 2. Continuous Learning
- Use `/learn` skill for new technologies
- Web search weekly for new AI/agent frameworks
- Evaluate against SELF_EVOLUTION.md goals
- Create feature requests based on research

### 3. Error-Driven Improvement
- Track tool failure patterns in TOOLS.md
- Analyze recurring errors weekly
- Create fix PRs for common issues
- Improve error handling and recovery

---

## Self-Evolution Workflow

### Adding New Features

```bash
# 1. Create issue via gh CLI
gh issue create --repo MTAAP/nanobot \
  --title "[FEATURE] Add multi-modal support" \
  --body "Description of the feature"

# 2. Add "start-interview" label
gh issue edit <issue_number> --add-label "start-interview"

# 3. Answer interview questions with /interview
# 4. Workflow auto-creates PR to main

# 5. Review and merge PR
```

### Fixing Bugs

```bash
# 1. Create issue with bug report
gh issue create --repo MTAAP/nanobot \
  --title "[BUG] Tool timeout not handled correctly" \
  --body "Steps to reproduce..."

# 2. Use SelfEvolveManager to create PR
# - Run tests: pytest tests/ -v
# - Run lint: ruff check nanobot/ && ruff format --check nanobot/
# - Commit changes
# - Create PR: not to main (feature branch)
```

### Evaluating New Technologies

Before integrating any new AI/agent technology:

1. **Research**: Use web search to find official docs
2. **Learn**: Use `/learn <technology>` to create skill
3. **Evaluate**: Consider:
   - **Complexity**: Implementation effort vs benefit
   - **Compatibility**: Fits nanobot architecture?
   - **Maintenance**: Long-term support and updates
   - **Value**: Solves real problem or nice-to-have?
4. **Propose**: Create issue with evaluation findings
5. **Interview**: Work through implementation details

---

## Current Evolution Goals

### Phase 1: OpenSpec Integration
- Add OpenSpec workflow for interview-driven development
- Enable spec file generation (proposal.md, spec.md, tasks.md)
- Auto-merge PRs with validation

### Phase 2: Enhanced Error Logging
- Structured error logger (`nanobot/agent/errors.py`)
- Error categorization system
- Self-evaluation metrics dashboard
- Health check endpoint

### Phase 3: Skill Discovery
- Weekly automated web search
- Auto-generate feature requests
- `/learn` integration for new skills

### Phase 4: Performance & Reliability
- APM integration (OpenTelemetry)
- Circuit breakers for external APIs
- Retry with exponential backoff
- Memory pipeline optimization

### Phase 5: New Capabilities
- Multi-modal (images, voice)
- Better reasoning (ReWOO, tree search)
- Long-term memory improvements
- Slack/email integrations

---

## Technologies Under Evaluation

| Tech | Priority | Benefit | Status |
|-------|-----------|---------|--------|
| ReWOO | High | Planning decoupling | Research |
| AutoGen | Medium | Multi-agent parallelization | Research |
| Cognee | Medium | Knowledge graph memory | Research |
| LangGraph | Medium | Tree-based reasoning | Research |
| Langbase | Low | Cheaper vector storage | Research |

---

## File Structure

```
nanobot/
├── SELF_EVOLUTION.md        # Evolution roadmap and goals
├── .github/
│   └── workflows/
│       ├── openspec-interview-pr.yml    # Interview workflow
│       └── openspec-proposal-from-pr.yml  # Proposal workflow
├── nanobot/
│   ├── registry/
│   │   └── evolve.py               # Self-evolution manager
│   └── agent/
│       ├── errors.py                # Structured error logger (TODO)
│       └── evaluation.py            # Self-evaluation (TODO)
└── HEARTBEAT.md                # Daemon task list
```

---

## Commands

### Self-Evolution Commands

| Command | Description |
|---------|-------------|
| `/evolve status` | Check current evolution progress |
| `/evolve plan` | Show upcoming features and fixes |
| `/evolve errors` | Analyze recent error patterns |
| `/evolve learn <topic>` | Research and learn new technology |

---

## Best Practices

### 1. Before Any Change
```python
# Always pull latest main first
cd /root/.nanobot/workspace/nanobot
git pull origin main
```

### 2. Test Before PR
```bash
# Run full test suite
pytest tests/ -v

# Run linting
ruff check nanobot/
ruff format --check nanobot/
```

### 3. Branch Naming
- Features: `feat/feature-name`
- Fixes: `fix/bug-description`
- Refactor: `refactor/component-name`

### 4. Commit Messages
```
feat: add multi-modal support
fix: handle tool timeout gracefully
refactor: optimize memory extraction
```

### 5. PR Guidelines
- Include tests for new features
- Update documentation (README.md, SELF_EVOLUTION.md)
- Link related issues
- Keep PRs focused (single change per PR)

---

## Monitoring

### Error Metrics
- Tool failure rate per tool type
- LLM error rate by provider
- MCP server failure rate
- Memory system health

### Self-Evaluation
- Weekly error pattern analysis
- Monthly capability gap analysis
- Quarterly roadmap review

### Success Metrics
- Feature delivery time
- Bug fix turnaround time
- Test coverage percentage
- Mean time to recovery (MTTR)

---

## References

- SELF_EVOLUTION.md - Full roadmap and goals
- TOOLS.md - Tool usage lessons and patterns
- skills/self-learning/SKILL.md - Learning new technologies
- skills/github-advanced/SKILL.md - GitHub workflow patterns
