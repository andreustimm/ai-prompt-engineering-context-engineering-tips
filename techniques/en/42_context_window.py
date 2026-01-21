"""
Technique 42: Context Window Management

Demonstrates intelligent management of the context window to prevent
information loss and maintain quality over long conversations.

Key concepts from FullCycle AI Tech Week (Aula 2):
- Context window has a limit - when full, oldest info is lost
- Summarization degrades quality ("summary of summary" problem)
- Each summarization compounds the loss
- Subagents help by having separate context windows

Use cases:
- Long-running coding agents
- Complex multi-step tasks
- Maintaining context in extended conversations
"""

import tiktoken
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Context window limits (approximate)
CONTEXT_LIMITS = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}


@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: str
    tokens: int = 0
    is_summary: bool = False
    original_messages: int = 0  # How many messages this summarizes


@dataclass
class ContextWindow:
    """
    Manages the context window with smart summarization strategies.

    Key insight from FullCycle: Summarization is lossy. Each time you
    summarize, you lose information. "Summary of summary" compounds this.
    """
    max_tokens: int
    messages: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    summarization_count: int = 0  # Track how many times we've summarized
    encoding: any = field(default=None)

    def __post_init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def add_message(self, role: str, content: str) -> bool:
        """
        Add a message to the context window.
        Returns True if summarization was needed.
        """
        tokens = self.count_tokens(content)
        message = Message(role=role, content=content, tokens=tokens)

        # Check if we need to summarize
        if self.total_tokens + tokens > self.max_tokens * 0.8:  # 80% threshold
            self._smart_summarize()
            self.summarization_count += 1

        self.messages.append(message)
        self.total_tokens += tokens
        return self.summarization_count > 0

    def _smart_summarize(self):
        """
        Intelligent summarization that preserves critical information.

        Strategy:
        1. Keep system message intact
        2. Keep recent messages intact
        3. Summarize old messages with critical info extraction
        """
        if len(self.messages) < 5:
            return

        # Find messages to summarize (keep last 3)
        to_summarize = self.messages[:-3]
        to_keep = self.messages[-3:]

        if not to_summarize:
            return

        # Extract content for summarization
        summary_content = "\n".join([
            f"{m.role}: {m.content}" for m in to_summarize
            if not m.is_summary  # Don't re-summarize summaries (loses more info)
        ])

        # Create summary with critical info preservation
        summary_prompt = f"""Summarize this conversation, preserving:
1. Key decisions made
2. Important code/technical details
3. Unresolved questions
4. Current context/state

Conversation to summarize:
{summary_content}

Create a concise but complete summary that maintains all critical context."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500
        )

        summary_text = response.choices[0].message.content

        # Create summary message
        summary_msg = Message(
            role="system",
            content=f"[Previous conversation summary - summarization #{self.summarization_count + 1}]\n{summary_text}",
            tokens=self.count_tokens(summary_text),
            is_summary=True,
            original_messages=len(to_summarize)
        )

        # Replace old messages with summary
        self.messages = [summary_msg] + to_keep
        self.total_tokens = sum(m.tokens for m in self.messages)

    def get_messages_for_api(self) -> list[dict]:
        """Get messages formatted for OpenAI API."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_stats(self) -> dict:
        """Get context window statistics."""
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": (self.total_tokens / self.max_tokens) * 100,
            "message_count": len(self.messages),
            "summarization_count": self.summarization_count,
            "has_summaries": any(m.is_summary for m in self.messages),
        }


class ContextAwareAgent:
    """
    An agent that monitors and manages its context window.

    Key insight: Instead of letting context overflow silently,
    actively manage it to maintain quality.
    """

    def __init__(self, max_tokens: int = 4000):
        self.context = ContextWindow(max_tokens=max_tokens)
        self.quality_warnings: list[str] = []

    def chat(self, user_message: str) -> str:
        """
        Process a chat message with context management.

        Monitors context window and warns about potential quality degradation.
        """
        # Add user message
        needed_summarization = self.context.add_message("user", user_message)

        if needed_summarization:
            self.quality_warnings.append(
                f"Context summarized (#{self.context.summarization_count}). "
                "Some information may be lost."
            )

        # Generate response
        response = client.chat.completions.create(
            model=MODEL,
            messages=self.context.get_messages_for_api()
        )

        assistant_message = response.choices[0].message.content

        # Add assistant response
        self.context.add_message("assistant", assistant_message)

        return assistant_message

    def get_context_health(self) -> dict:
        """
        Assess the health of the current context.

        Higher summarization count = lower quality context.
        """
        stats = self.context.get_stats()

        # Calculate health score (lower is worse)
        health_score = 100
        health_score -= stats["summarization_count"] * 15  # Each summarization hurts
        health_score -= max(0, stats["usage_percent"] - 70)  # High usage hurts

        return {
            **stats,
            "health_score": max(0, health_score),
            "quality_warnings": self.quality_warnings.copy(),
            "recommendation": self._get_recommendation(health_score)
        }

    def _get_recommendation(self, health_score: int) -> str:
        if health_score >= 80:
            return "Context is healthy. Continue normally."
        elif health_score >= 50:
            return "Context degrading. Consider starting a new session or using subagents."
        else:
            return "Context severely degraded. Start new session or checkpoint important info."


def demonstrate_context_degradation():
    """
    Demonstrate how context degrades over time with summarization.

    This shows the "summary of summary" problem mentioned in FullCycle.
    """
    print("=" * 60)
    print("Context Window Management - Degradation Demo")
    print("=" * 60)

    # Use small context window to demonstrate
    agent = ContextAwareAgent(max_tokens=2000)

    # Simulate a long conversation
    messages = [
        "Let's build a user management system. Start with the User model.",
        "Now add authentication with JWT tokens.",
        "Implement password hashing with bcrypt.",
        "Add email verification flow.",
        "Create the login endpoint.",
        "Add refresh token functionality.",
        "Implement logout that invalidates tokens.",
        "Add rate limiting to prevent brute force.",
        "Create password reset flow.",
        "Add two-factor authentication.",
    ]

    print("\nSimulating long conversation with small context window...")
    print(f"Max tokens: {agent.context.max_tokens}")
    print()

    for i, msg in enumerate(messages, 1):
        print(f"Message {i}: {msg[:50]}...")
        response = agent.chat(msg)
        stats = agent.context.get_stats()
        print(f"  Tokens: {stats['total_tokens']}/{stats['max_tokens']} ({stats['usage_percent']:.1f}%)")
        print(f"  Summarizations: {stats['summarization_count']}")
        if stats['summarization_count'] > 0:
            print(f"  ⚠️ Context has been summarized - information loss possible")
        print()

    # Show final context health
    health = agent.get_context_health()
    print("\n" + "=" * 60)
    print("Final Context Health Report")
    print("=" * 60)
    print(f"Health Score: {health['health_score']}/100")
    print(f"Total Summarizations: {health['summarization_count']}")
    print(f"Recommendation: {health['recommendation']}")
    if health['quality_warnings']:
        print("\nWarnings:")
        for warning in health['quality_warnings']:
            print(f"  - {warning}")


def demonstrate_subagent_strategy():
    """
    Demonstrate using subagents to avoid context degradation.

    Key insight: Each subagent has its own context window.
    """
    print("\n" + "=" * 60)
    print("Subagent Strategy - Avoiding Context Degradation")
    print("=" * 60)

    print("""
The subagent strategy avoids context degradation by:

1. Main agent has its own context window
2. Each subagent task runs with a FRESH context window
3. Only RESULTS are passed back (not full context)
4. This prevents "summary of summary" degradation

Example flow:
┌─────────────────────────────────────────────────────────┐
│ Main Agent (Context Window A)                           │
│                                                         │
│  Task: "Build user system"                             │
│    │                                                    │
│    ├─> Subagent 1 (Fresh Context B)                    │
│    │     Task: "Create User model"                     │
│    │     Returns: {model_code, decisions}              │
│    │                                                    │
│    ├─> Subagent 2 (Fresh Context C)                    │
│    │     Task: "Add auth" + results from Subagent 1    │
│    │     Returns: {auth_code, decisions}               │
│    │                                                    │
│    └─> Main agent aggregates results                   │
│                                                         │
│  Each subagent: Fresh context = No degradation         │
│  Main agent: Only stores results = Minimal context     │
└─────────────────────────────────────────────────────────┘
""")

    # Simulate subagent pattern
    class SubagentOrchestrator:
        def __init__(self):
            self.results = []

        def execute_subtask(self, task: str, context: str = "") -> dict:
            """Execute a subtask with fresh context (simulated)."""
            prompt = f"""Execute this task and return a structured result.

Task: {task}
{"Context from previous steps:" + context if context else ""}

Return a JSON with: {{"result": "...", "key_decisions": ["..."]}}"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            return response.choices[0].message.content

    orchestrator = SubagentOrchestrator()

    subtasks = [
        "Design the User data model with fields for name, email, password",
        "Implement password hashing function",
        "Create JWT token generation function",
    ]

    print("\nExecuting subtasks with fresh contexts...")
    accumulated_context = ""

    for i, task in enumerate(subtasks, 1):
        print(f"\nSubtask {i}: {task}")
        result = orchestrator.execute_subtask(task, accumulated_context)
        print(f"Result: {result[:200]}...")

        # Only pass essential info forward (not full context)
        accumulated_context += f"\n- Completed: {task}"

    print("\n✅ Each subtask ran with fresh context - no degradation!")


def main():
    print("=" * 60)
    print("Context Window Management")
    print("Key Concepts from FullCycle AI Tech Week")
    print("=" * 60)

    print("""
The context window is like a bucket of memory:
- When full, oldest information overflows (is lost)
- Summarization compresses but LOSES information
- "Summary of summary" compounds the loss exponentially

Wesley's illustration from the video:
"I love you" -> [context fills] -> "love you" (lost "I")
The meaning completely changes!

Strategies to manage:
1. Monitor context usage proactively
2. Use subagents for isolated tasks
3. Checkpoint critical information
4. Start fresh sessions when degraded
""")

    # Demo 1: Context degradation
    demonstrate_context_degradation()

    # Demo 2: Subagent strategy
    demonstrate_subagent_strategy()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. Context window management is CRITICAL for long-running agents
2. Summarization is lossy - avoid when possible
3. Subagents provide fresh context windows
4. Monitor and warn about context degradation
5. Checkpoint important information before it's lost
""")


if __name__ == "__main__":
    main()
