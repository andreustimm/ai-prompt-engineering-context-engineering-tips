# AI-Assisted Development Study Roadmap

A comprehensive guide for developers who want to master AI-assisted development, connecting the 30 techniques in this project with modern tools like Claude Code.

> **Language / Idioma:** [Português Brasileiro](ROADMAP.pt-BR.md) | English

---

## Table of Contents

1. [Overview: The Future of Development](#1-overview-the-future-of-development)
2. [Technique Connection Map](#2-technique-connection-map)
3. [Learning Tracks](#3-learning-tracks)
4. [AI-Assisted Development - Project Organization](#4-ai-assisted-development---project-organization)
5. [Modern Tools - Claude Code](#5-modern-tools---claude-code)
6. [Templates and Examples](#6-templates-and-examples)
7. [Additional Resources](#7-additional-resources)

---

## 1. Overview: The Future of Development

```
┌─────────────────────────────────────────────────────────────┐
│                    AI-ASSISTED DEVELOPMENT                  │
├─────────────────────────────────────────────────────────────┤
│  PROMPT ENGINEERING    │    CONTEXT ENGINEERING             │
│  (How to talk to AI)   │    (How to give context to AI)     │
├────────────────────────┴────────────────────────────────────┤
│                     MODERN TOOLING                          │
│  Claude Code │ Agents │ Skills │ MCP │ Hooks                │
└─────────────────────────────────────────────────────────────┘
```

### What is AI-Assisted Development?

AI-assisted development is a paradigm where developers work alongside AI tools to increase productivity, code quality, and learning speed. It combines:

- **Prompt Engineering**: The art of communicating effectively with AI models
- **Context Engineering**: The science of providing the right information to AI
- **Modern Tooling**: Integration of AI into the development workflow

### Why Learn This?

| Benefit | Description |
|---------|-------------|
| **Productivity** | Automate repetitive tasks, generate boilerplate, accelerate debugging |
| **Code Quality** | AI-assisted code review, best practices suggestions, security checks |
| **Learning** | Instant explanations, documentation generation, concept exploration |
| **Problem Solving** | Multiple approaches to problems, trade-off analysis, architecture decisions |

---

## 2. Technique Connection Map

Understanding how the 30 techniques relate to each other helps you choose the right tool for each situation.

```
FOUNDATION (01-06)
    │
    ├── 01 Zero-Shot ──────────────────────────────┐
    │       │                                      │
    ├── 02 Chain of Thought ───┐                   │
    │       │                  │                   │
    ├── 03 Few-Shot ──────────┐│                   │
    │       │                 ││                   │
    ├── 04 Tree of Thoughts ──┼┼─── 07 Self-Consistency
    │       │                 ││          │
    ├── 05 Skeleton of Thought┼┼─── 08 Least-to-Most
    │       │                 ││          │
    └── 06 ReAct ─────────────┼┴─── 09 Self-Refine
            │                 │           │
            │                 └──── 10 Prompt Chaining
            │                             │
            ▼                             ▼
    ┌──────────────────────────────────────────────┐
    │          CONTEXT ENGINEERING (11-30)         │
    ├──────────────────────────────────────────────┤
    │                                              │
    │   11 RAG Basic ◄─────────────────────────┐   │
    │       │                                  │   │
    │       ├── 12 RAG + Reranking             │   │
    │       │                                  │   │
    │       └── 13 RAG Conversational          │   │
    │               │                          │   │
    │               ▼                          │   │
    │   ┌─────────────────────────────────┐    │   │
    │   │  21 Advanced Chunking ◄─────────┼────┘   │
    │   │      │                          │        │
    │   │      ├── 22 Hybrid Search       │        │
    │   │      │                          │        │
    │   │      ├── 23 Query Transform     │        │
    │   │      │                          │        │
    │   │      ├── 24 Contextual Compress │        │
    │   │      │                          │        │
    │   │      └── 25 Self-Query          │        │
    │   │              │                  │        │
    │   └──────────────┼──────────────────┘        │
    │                  ▼                           │
    │   ┌─────────────────────────────────┐        │
    │   │  26 Parent-Document             │        │
    │   │  27 Multi-Vector                │        │
    │   │  28 Ensemble Retrieval          │        │
    │   │  29 Long Context                │        │
    │   │  30 Time-Weighted               │        │
    │   └─────────────────────────────────┘        │
    └──────────────────────────────────────────────┘
```

### Technique Categories

| Category | Techniques | Use When |
|----------|-----------|----------|
| **Basic Prompting** | 01-06 | Starting out, direct tasks |
| **Advanced Reasoning** | 07-10 | Complex problems, multiple steps |
| **RAG Fundamentals** | 11-13 | Document-based Q&A |
| **Local Models** | 14-15 | Privacy, cost reduction, offline |
| **Structured Output** | 16-17 | API integrations, tool use |
| **Advanced Features** | 18-20 | Vision, memory, prompt optimization |
| **Chunking & Retrieval** | 21-25 | RAG optimization |
| **Context Management** | 26-30 | Advanced RAG patterns |

---

## 3. Learning Tracks

Choose your track based on your current level and goals.

### Beginner Track (4 weeks)

Perfect for developers new to AI-assisted development.

```
Week 1: Fundamentals
├── 01 Zero-Shot
│   └── Learn to write clear, direct prompts
├── 02 Chain of Thought
│   └── Guide AI through step-by-step reasoning
└── 03 Few-Shot
    └── Use examples to shape responses

Week 2: Advanced Reasoning
├── 04 Tree of Thoughts
│   └── Explore multiple solution paths
├── 05 Skeleton of Thought
│   └── Structure first, details later
└── 06 ReAct
    └── Combine reasoning with actions

Week 3: RAG Basics
├── 11 RAG Basic
│   └── Document retrieval fundamentals
├── 12 RAG + Reranking
│   └── Improve retrieval relevance
└── 13 RAG Conversational
    └── Multi-turn document Q&A

Week 4: Practice Project
├── 16 Structured Output
│   └── JSON mode + Pydantic models
├── 17 Tool Calling
│   └── Custom function tools
└── Project: Build a Chatbot with RAG
    └── Combine techniques 11-13, 16-17
```

### Intermediate Track (4 weeks)

For developers familiar with basic prompting who want to master RAG.

```
Week 1: Advanced Prompt Techniques
├── 07 Self-Consistency
│   └── Multiple samples + voting
├── 08 Least-to-Most
│   └── Progressive decomposition
├── 09 Self-Refine
│   └── Iterative improvement
└── 10 Prompt Chaining
    └── Connected prompt pipelines

Week 2: Chunking & Retrieval
├── 21 Advanced Chunking
│   └── Semantic, recursive, token-based strategies
├── 22 Hybrid Search
│   └── BM25 + Vector with RRF
└── 23 Query Transformation
    └── HyDE, Multi-Query, Step-Back

Week 3: Context Optimization
├── 24 Contextual Compression
│   └── Extract relevant portions only
├── 25 Self-Query
│   └── Auto-generate metadata filters
└── 26 Parent-Document
    └── Small chunks search, large context return

Week 4: Production Project
└── Build a Production RAG System
    ├── Combine chunking strategies
    ├── Implement hybrid search
    ├── Add query transformation
    └── Deploy with monitoring
```

### Advanced Track (4 weeks)

For developers building production AI systems.

```
Week 1: Multi-Vector & Ensemble
├── 27 Multi-Vector
│   └── Multiple document representations
├── 28 Ensemble Retrieval
│   └── Combine retrievers with weighted RRF
└── 29 Long Context
    └── Map-Reduce, Refine, Map-Rerank

Week 2: Advanced Features
├── 18 Vision/Multimodal
│   └── Image analysis with GPT-4o
├── 19 Memory/Conversation
│   └── Persistent conversation context
├── 20 Meta-Prompting
│   └── LLM generating/optimizing prompts
└── 30 Time-Weighted
    └── Recency bias in retrieval

Week 3: Local Models
├── 14 Ollama Basic
│   └── Local LLMs without API costs
├── 15 Ollama + RAG
│   └── 100% offline RAG
└── Cost Analysis
    └── Compare local vs. API costs

Week 4: Production Setup
├── Claude Code Configuration
│   └── CLAUDE.md, settings, hooks
├── Agents & Skills
│   └── Custom subagents and workflows
└── Complete System
    └── Full AI-assisted development environment
```

---

## 4. AI-Assisted Development - Project Organization

### Recommended Project Structure

```
my-project/
├── src/                          # Application code
├── tests/                        # Tests (unit, integration)
├── docs/                         # Documentation
│   └── adr/                      # Architecture Decision Records
│       ├── 0000-template.md
│       ├── 0001-llm-choice.md
│       └── 0002-rag-strategy.md
├── CLAUDE.md                     # Project context for AI
├── .claude/                      # Claude Code configuration
│   ├── agents/                   # Custom subagents
│   │   ├── code-reviewer.md
│   │   └── debugger.md
│   ├── skills/                   # Project skills
│   │   ├── commit/SKILL.md
│   │   └── deploy/SKILL.md
│   └── rules/                    # Path-based rules
├── .mcp.json                     # MCP servers
└── scripts/                      # Automation scripts
```

### Feature Decomposition Framework

When working on features with AI assistance, follow this structured approach:

```
FEATURE REQUEST
      │
      ▼
┌─────────────────────────────────────────┐
│  1. UNDERSTANDING (Use 02 CoT)          │
│  - What does the user want?             │
│  - What are the requirements?           │
│  - What are the constraints?            │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  2. DECOMPOSITION (Use 08 Least-to-Most)│
│  - Break into sub-tasks                 │
│  - Identify dependencies                │
│  - Prioritize by value                  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  3. EXPLORATION (Use 04 ToT)            │
│  - Consider alternatives                │
│  - Evaluate trade-offs                  │
│  - Document in ADR                      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  4. IMPLEMENTATION                      │
│  - Use Claude Code with context         │
│  - Iterate with Self-Refine (09)        │
│  - Validate with tests                  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  5. REVIEW                              │
│  - Code review with subagent            │
│  - Update documentation                 │
│  - ADR if architectural decision made   │
└─────────────────────────────────────────┘
```

---

## 5. Modern Tools - Claude Code

Claude Code is a CLI tool that brings AI assistance directly into your development workflow.

### Key Features

| Feature | Description |
|---------|-------------|
| **Agents** | Specialized subprocesses for specific tasks |
| **Skills** | Reusable commands for common workflows |
| **MCP** | Model Context Protocol for external integrations |
| **Hooks** | Automated actions triggered by events |

### Subagents

Create custom agents for specialized tasks:

```yaml
# .claude/agents/code-reviewer.md
---
name: code-reviewer
description: Code quality and security reviewer
tools: Read, Grep, Glob
model: sonnet
---

You are an expert code reviewer. Analyze:
1. Code quality and readability
2. Security vulnerabilities
3. Performance issues
4. Adherence to project conventions

Provide actionable feedback with specific line references.
```

### Skills

Define reusable workflows:

```yaml
# .claude/skills/commit/SKILL.md
---
name: smart-commit
description: Creates semantic commits with context
---

To create a commit:
1. Analyze changes with `git diff --staged`
2. Identify type: feat|fix|docs|refactor|test
3. Write message following Conventional Commits
4. Include Co-Authored-By
```

### MCP Servers

Connect to external services:

```json
// .mcp.json
{
  "servers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@bytebase/dbhub", "--dsn", "${DATABASE_URL}"]
    }
  }
}
```

### Hooks

Automate actions on tool use:

```json
// In settings.json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "npx prettier --write \"$file_path\""
      }]
    }]
  }
}
```

### Agentic Workflows - Loops and Orchestration

AI agents that run in loops are a fundamental pattern for building autonomous systems. Understanding these patterns is essential for developing sophisticated AI-assisted workflows.

#### What are Agentic Loops?

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENTIC LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│    │  PERCEIVE │───▶│  REASON  │───▶│   ACT    │            │
│    │  (Input)  │    │  (Think) │    │ (Output) │            │
│    └──────────┘    └──────────┘    └────┬─────┘             │
│         ▲                               │                   │
│         │         ┌──────────┐          │                   │
│         └─────────│ OBSERVE  │◀─────────┘                   │
│                   │(Feedback)│                              │
│                   └──────────┘                              │
│                                                             │
│    Loop continues until: goal reached OR max iterations     │
└─────────────────────────────────────────────────────────────┘
```

#### Agentic Workflow Patterns

**Pattern 1: ReAct Loop (Technique 06)**

```python
while not done:
    thought = llm.think(observation)      # Reasoning
    action = llm.decide_action(thought)   # Decision
    observation = execute(action)         # Execution
    done = check_completion(observation)  # Verification
```

**Pattern 2: Plan-Execute Loop**

```python
plan = llm.create_plan(goal)              # Initial planning
for step in plan:
    result = execute(step)                # Execute step
    if needs_replan(result):
        plan = llm.replan(goal, result)   # Replan if needed
```

**Pattern 3: Self-Refine Loop (Technique 09)**

```python
response = llm.generate(task)
while not satisfactory:
    critique = llm.critique(response)     # Self-critique
    response = llm.improve(response, critique)  # Improvement
    satisfactory = evaluate(response)
```

**Pattern 4: Multi-Agent Orchestration**

```
┌──────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                         │
│                    (Coordinator)                         │
├──────────────────────────────────────────────────────────┤
│         │              │             │                   │
│    ┌────▼─────┐   ┌────▼────┐   ┌────▼────┐              │
│    │ Agent 1  │   │ Agent 2 │   │ Agent 3 │              │
│    │(Research)│   │ (Code)  │   │(Review) │              │
│    └────┬─────┘   └────┬────┘   └────┬────┘              │
│         │              │             │                   │
│         └──────────────┴─────────────┘                   │
│                        │                                 │
│                   ┌────▼────┐                            │
│                   │ COMBINE │                            │
│                   │ RESULTS │                            │
│                   └─────────┘                            │
└──────────────────────────────────────────────────────────┘
```

#### Practical Implementation with LangChain/LangGraph

```python
# Example: ReAct Loop with LangChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_tool, calculator_tool]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,      # Loop iteration limit
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

# Loop happens automatically inside the executor
result = agent_executor.invoke({"input": "Research and calculate..."})
```

#### Recommended Development Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              AI DEVELOPMENT WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DEFINITION                                              │
│     └── Define clear objective and success criteria         │
│                                                             │
│  2. DECOMPOSITION                                           │
│     └── Break into sub-tasks (use Least-to-Most)            │
│                                                             │
│  3. PATTERN SELECTION                                       │
│     ├── Simple task → Single Agent (ReAct)                  │
│     ├── Complex task → Plan-Execute                         │
│     ├── Critical quality → Self-Refine Loop                 │
│     └── Multi-domain → Multi-Agent Orchestration            │
│                                                             │
│  4. IMPLEMENTATION                                          │
│     ├── Configure max_iterations (avoid infinite loops)     │
│     ├── Define clear stopping criteria                      │
│     └── Add logging/observability                           │
│                                                             │
│  5. VALIDATION                                              │
│     ├── Test with edge cases                                │
│     ├── Monitor costs (tokens per loop)                     │
│     └── Adjust parameters as needed                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Decision Table: Which Pattern to Use?

| Scenario | Recommended Pattern | Base Technique |
|----------|-------------------|--------------|
| Search + Reasoning | ReAct Loop | 06 ReAct |
| Multi-step tasks | Plan-Execute | 08 Least-to-Most |
| Content generation | Self-Refine | 09 Self-Refine |
| Data pipeline | Prompt Chaining | 10 Prompt Chaining |
| Multiple perspectives | Multi-Agent | 07 Self-Consistency |
| Complex decisions | Tree of Thoughts | 04 ToT |

#### Practical Tips

- **Always define `max_iterations`** - Avoids infinite loops and exploding costs
- **Logging is essential** - Record each iteration for debugging
- **Clear stopping criteria** - The agent needs to know when to finish
- **Fallbacks** - Have a plan B if the loop doesn't converge
- **Costs** - Each iteration = more tokens = more cost

---

## 6. Templates and Examples

### CLAUDE.md Template

```markdown
# [Project Name]

## Overview
[2-3 sentence description]

## Tech Stack
- **Backend**: Python 3.11, FastAPI
- **LLM**: OpenAI GPT-4o-mini via LangChain
- **Vector DB**: ChromaDB
- **Tests**: pytest

## Architecture
[Diagram or architecture description]

## Conventions
- Code in English, bilingual documentation
- Google Style docstrings
- Type hints required
- Tests for every new feature

## Common Commands
```bash
# Run tests
pytest tests/ -v

# Development server
uvicorn src.main:app --reload

# Production build
docker build -t app .
```

## Sensitive Areas
- `.env` - Never commit
- `src/auth/` - Review security changes
- `migrations/` - Test in staging first
```

### ADR Template (Architecture Decision Record)

```markdown
# ADR [Number]: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context
[Technical situation requiring a decision]

## Decision
[The decision made and its rationale]

## Alternatives Considered
1. **[Alternative 1]**: [Pros/Cons]
2. **[Alternative 2]**: [Pros/Cons]

## Consequences
### Positive
- [Benefit 1]
- [Benefit 2]

### Negative
- [Trade-off 1]
- [Trade-off 2]

## References
- [Relevant links]
```

### Technique Selection Guide

| Situation | Recommended Technique | Why |
|-----------|----------------------|-----|
| Simple classification | 01 Zero-Shot | Direct, no examples needed |
| Math/logic problems | 02 Chain of Thought | Step-by-step reasoning |
| Specific output format | 03 Few-Shot | Examples guide format |
| Complex decisions | 04 Tree of Thoughts | Explore multiple paths |
| Long-form content | 05 Skeleton of Thought | Structure first |
| Need external data | 06 ReAct | Reasoning + actions |
| High accuracy needed | 07 Self-Consistency | Multiple samples + voting |
| Complex problem | 08 Least-to-Most | Progressive decomposition |
| Quality improvement | 09 Self-Refine | Iterative critique |
| Multi-step workflow | 10 Prompt Chaining | Connected pipeline |
| Document Q&A | 11-13 RAG | Retrieval-augmented |
| Privacy/offline | 14-15 Ollama | Local models |
| API integration | 16 Structured Output | Guaranteed schema |
| External tools | 17 Tool Calling | Function execution |
| Image analysis | 18 Vision | Multimodal |
| Chat context | 19 Memory | Conversation history |
| Prompt optimization | 20 Meta-Prompting | LLM creates prompts |
| Better chunks | 21 Advanced Chunking | Strategy selection |
| Keyword + semantic | 22 Hybrid Search | Combined retrieval |
| Poor retrieval | 23 Query Transform | Improve queries |
| Token reduction | 24 Compression | Extract relevant |
| Metadata filtering | 25 Self-Query | Auto-generate filters |
| Need more context | 26 Parent-Document | Large parents |
| Multiple views | 27 Multi-Vector | Summary + questions |
| Multiple methods | 28 Ensemble | Combine retrievers |
| Large documents | 29 Long Context | Map-Reduce/Refine |
| Time-sensitive | 30 Time-Weighted | Recency bias |

---

## 7. Additional Resources

### This Project

- [README.md](README.md) - Complete technique documentation
- [techniques/en/](techniques/en/) - English implementations
- [techniques/pt-br/](techniques/pt-br/) - Portuguese implementations
- [sample_data/](sample_data/) - Sample data for testing

### External Resources

#### Documentation
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [Ollama Documentation](https://ollama.ai/docs)

#### Learning
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Academy](https://academy.langchain.com/)
- [Anthropic's Prompt Engineering Course](https://docs.anthropic.com/claude/docs/prompt-engineering)

#### Communities
- [LangChain Discord](https://discord.gg/langchain)
- [Ollama Discord](https://discord.gg/ollama)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

### Next Steps

1. **Choose Your Track**: Start with the track that matches your level
2. **Run Examples**: Execute the scripts in this project
3. **Build Projects**: Apply techniques to real projects
4. **Share Knowledge**: Contribute back to the community

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNIQUE QUICK REFERENCE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROMPT ENGINEERING                                         │
│  ├── Zero-Shot (01)      → Direct prompts                   │
│  ├── Chain of Thought (02) → Step-by-step                   │
│  ├── Few-Shot (03)       → Examples guide                   │
│  ├── Tree of Thoughts (04) → Multiple paths                 │
│  ├── Skeleton (05)       → Structure first                  │
│  ├── ReAct (06)          → Reasoning + Actions              │
│  ├── Self-Consistency (07) → Vote on answers                │
│  ├── Least-to-Most (08)  → Decompose problems               │
│  ├── Self-Refine (09)    → Iterate & improve                │
│  └── Prompt Chaining (10) → Pipeline prompts                │
│                                                             │
│  CONTEXT ENGINEERING                                        │
│  ├── RAG Basic (11)      → Document retrieval               │
│  ├── RAG Reranking (12)  → Better relevance                 │
│  ├── RAG Conversational (13) → Chat + docs                  │
│  ├── Ollama Basic (14)   → Local LLMs                       │
│  ├── Ollama RAG (15)     → Offline RAG                      │
│  ├── Structured (16)     → JSON/Pydantic                    │
│  ├── Tool Calling (17)   → Functions                        │
│  ├── Vision (18)         → Images                           │
│  ├── Memory (19)         → Conversation                     │
│  ├── Meta-Prompting (20) → Optimize prompts                 │
│  ├── Chunking (21)       → Split strategies                 │
│  ├── Hybrid Search (22)  → BM25 + Vector                    │
│  ├── Query Transform (23) → Better queries                  │
│  ├── Compression (24)    → Reduce tokens                    │
│  ├── Self-Query (25)     → Auto filters                     │
│  ├── Parent-Doc (26)     → More context                     │
│  ├── Multi-Vector (27)   → Multiple representations         │
│  ├── Ensemble (28)       → Combine retrievers               │
│  ├── Long Context (29)   → Large docs                       │
│  └── Time-Weighted (30)  → Recency bias                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*This roadmap is a living document. As new techniques and tools emerge, it will be updated to reflect best practices in AI-assisted development.*
