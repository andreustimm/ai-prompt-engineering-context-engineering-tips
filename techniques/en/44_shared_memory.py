"""
Technique 44: Shared Memory Between Agents

Demonstrates a Memory DB pattern where multiple agents can share knowledge,
including errors/problems found, for cross-agent learning and feedback.

Key concepts from FullCycle AI Tech Week (Aula 2):
- Memory DB allows agents to share knowledge
- Short-term, medium-term, and long-term memory categories
- Feedback loop: agent A finds problem → stored → agent B avoids it
- Semantic search over shared memories

Use cases:
- Multi-agent development teams
- Error pattern recognition
- Cross-project learning
- Organizational knowledge base
"""

import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class Memory:
    """A single memory entry in the shared memory system."""
    id: str
    content: str
    memory_type: str  # "short", "medium", "long"
    category: str  # "error", "solution", "pattern", "decision", "context"
    source_agent: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    access_count: int = 0
    relevance_score: float = 1.0

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "category": self.category,
            "source_agent": self.source_agent,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "relevance_score": self.relevance_score,
        }


class MemoryDB:
    """
    Shared memory database for multi-agent systems.

    Key insight from FullCycle: Memory is categorized by duration:
    - Short-term: Current task context (expires quickly)
    - Medium-term: Session/project context (days)
    - Long-term: Organizational knowledge (permanent)
    """

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.embeddings_cache: dict[str, list[float]] = {}

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory."""
        return hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    def _get_expiration(self, memory_type: str) -> Optional[datetime]:
        """Get expiration time based on memory type."""
        now = datetime.now()
        if memory_type == "short":
            return now + timedelta(hours=1)
        elif memory_type == "medium":
            return now + timedelta(days=7)
        else:  # long-term
            return None  # Never expires

    def store(
        self,
        content: str,
        memory_type: str,
        category: str,
        source_agent: str,
        metadata: dict = None
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The memory content
            memory_type: "short", "medium", or "long"
            category: "error", "solution", "pattern", "decision", "context"
            source_agent: Name of the agent storing the memory
            metadata: Additional metadata
        """
        memory = Memory(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            category=category,
            source_agent=source_agent,
            created_at=datetime.now(),
            expires_at=self._get_expiration(memory_type),
            metadata=metadata or {},
        )
        self.memories[memory.id] = memory
        return memory

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 5
    ) -> list[Memory]:
        """
        Recall memories based on query and filters.

        Uses semantic matching via the LLM to find relevant memories.
        """
        # Filter by category and type
        candidates = [
            m for m in self.memories.values()
            if not m.is_expired()
            and (category is None or m.category == category)
            and (memory_type is None or m.memory_type == memory_type)
        ]

        if not candidates:
            return []

        # Use LLM to rank relevance
        memories_text = "\n".join([
            f"[{m.id}] ({m.category}): {m.content[:200]}"
            for m in candidates
        ])

        ranking_prompt = f"""Given the query and memories, rank the memories by relevance.
Return a JSON object with memory IDs and relevance scores (0-1).

Query: {query}

Memories:
{memories_text}

Return: {{"rankings": [{{"id": "...", "score": 0.9}}, ...]}}
Only include memories with score > 0.3"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": ranking_prompt}],
            response_format={"type": "json_object"}
        )

        rankings = json.loads(response.choices[0].message.content)

        # Update access counts and return ranked memories
        results = []
        for ranking in rankings.get("rankings", [])[:limit]:
            memory_id = ranking.get("id")
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.access_count += 1
                memory.relevance_score = ranking.get("score", 0.5)
                results.append(memory)

        return results

    def get_recent(self, limit: int = 10) -> list[Memory]:
        """Get most recent memories."""
        valid = [m for m in self.memories.values() if not m.is_expired()]
        return sorted(valid, key=lambda m: m.created_at, reverse=True)[:limit]

    def get_by_category(self, category: str) -> list[Memory]:
        """Get all memories in a category."""
        return [
            m for m in self.memories.values()
            if m.category == category and not m.is_expired()
        ]

    def cleanup_expired(self) -> int:
        """Remove expired memories."""
        expired_ids = [
            m.id for m in self.memories.values() if m.is_expired()
        ]
        for mid in expired_ids:
            del self.memories[mid]
        return len(expired_ids)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        valid = [m for m in self.memories.values() if not m.is_expired()]
        return {
            "total_memories": len(valid),
            "by_type": {
                "short": len([m for m in valid if m.memory_type == "short"]),
                "medium": len([m for m in valid if m.memory_type == "medium"]),
                "long": len([m for m in valid if m.memory_type == "long"]),
            },
            "by_category": {
                cat: len([m for m in valid if m.category == cat])
                for cat in ["error", "solution", "pattern", "decision", "context"]
            },
            "total_accesses": sum(m.access_count for m in valid),
        }


class MemoryAwareAgent:
    """
    An agent that uses shared memory for learning and feedback.

    Key insight: Agents can learn from each other's experiences
    without sharing full context - just storing key learnings.
    """

    def __init__(self, name: str, memory_db: MemoryDB, specialty: str = "general"):
        self.name = name
        self.memory_db = memory_db
        self.specialty = specialty

    def execute_with_memory(self, task: str) -> dict:
        """
        Execute a task while utilizing and contributing to shared memory.

        Flow:
        1. Recall relevant memories before starting
        2. Execute task with memory context
        3. Store learnings back to memory
        """
        # Step 1: Recall relevant memories
        relevant_memories = self.memory_db.recall(
            query=task,
            limit=5
        )

        memory_context = ""
        if relevant_memories:
            memory_context = "\n\n## Relevant Memories from Other Agents:\n"
            for mem in relevant_memories:
                memory_context += f"- [{mem.category}] {mem.content}\n"
                memory_context += f"  (from {mem.source_agent}, accessed {mem.access_count}x)\n"

        # Step 2: Execute task
        execution_prompt = f"""You are {self.name}, a {self.specialty} specialist.

Execute this task and learn from previous agent experiences.

## Task:
{task}
{memory_context}

## Instructions:
1. Consider any relevant memories when making decisions
2. Execute the task
3. Note any errors encountered or solutions found
4. Document key decisions and patterns

Return JSON:
{{
    "result": "your execution result",
    "errors_found": ["error 1", ...],
    "solutions_applied": ["solution 1", ...],
    "patterns_identified": ["pattern 1", ...],
    "key_decisions": ["decision 1", ...]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": execution_prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Step 3: Store learnings back to memory
        self._store_learnings(task, result)

        return result

    def _store_learnings(self, task: str, result: dict):
        """Store learnings from task execution to shared memory."""

        # Store errors as short-term (might be task-specific)
        for error in result.get("errors_found", []):
            self.memory_db.store(
                content=f"Error in task '{task[:50]}': {error}",
                memory_type="short",
                category="error",
                source_agent=self.name,
                metadata={"task": task}
            )

        # Store solutions as medium-term (useful for similar tasks)
        for solution in result.get("solutions_applied", []):
            self.memory_db.store(
                content=solution,
                memory_type="medium",
                category="solution",
                source_agent=self.name,
                metadata={"task": task}
            )

        # Store patterns as long-term (organizational knowledge)
        for pattern in result.get("patterns_identified", []):
            self.memory_db.store(
                content=pattern,
                memory_type="long",
                category="pattern",
                source_agent=self.name,
                metadata={"task": task}
            )

        # Store key decisions as medium-term
        for decision in result.get("key_decisions", []):
            self.memory_db.store(
                content=decision,
                memory_type="medium",
                category="decision",
                source_agent=self.name,
                metadata={"task": task}
            )


def demonstrate_shared_memory():
    """Demonstrate the shared memory pattern with multiple agents."""
    print("=" * 60)
    print("Shared Memory Between Agents")
    print("=" * 60)

    print("""
Flow:
┌─────────────────────────────────────────────────────────────┐
│                      Memory DB                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ Short   │  │ Medium  │  │  Long   │                     │
│  │ (1hr)   │  │ (7days) │  │ (perm)  │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │                           │
│       └────────────┴────────────┘                           │
│                    │                                         │
│      ┌─────────────┼─────────────┐                          │
│      │             │             │                          │
│      ▼             ▼             ▼                          │
│  ┌───────┐    ┌───────┐    ┌───────┐                       │
│  │Agent A│    │Agent B│    │Agent C│                       │
│  │Backend│    │Frontend│   │Testing│                       │
│  └───────┘    └───────┘    └───────┘                       │
│                                                              │
│  Agent A finds error → stores → Agent B avoids it          │
└─────────────────────────────────────────────────────────────┘
""")

    # Create shared memory
    memory_db = MemoryDB()

    # Create agents that share memory
    backend_agent = MemoryAwareAgent("BackendAgent", memory_db, "backend development")
    frontend_agent = MemoryAwareAgent("FrontendAgent", memory_db, "frontend development")
    testing_agent = MemoryAwareAgent("TestingAgent", memory_db, "testing and QA")

    # Task 1: Backend agent works on API
    print("\n" + "=" * 60)
    print("Task 1: Backend Agent - Create User API")
    print("=" * 60)

    result1 = backend_agent.execute_with_memory(
        "Create a REST API endpoint for user registration with email validation"
    )
    print(f"\nResult: {result1.get('result', '')[:200]}...")
    print(f"Errors found: {result1.get('errors_found', [])}")
    print(f"Solutions: {result1.get('solutions_applied', [])}")

    # Task 2: Frontend agent - can learn from backend's experience
    print("\n" + "=" * 60)
    print("Task 2: Frontend Agent - Create Registration Form")
    print("(Should benefit from Backend's learnings)")
    print("=" * 60)

    result2 = frontend_agent.execute_with_memory(
        "Create a user registration form that calls the registration API"
    )
    print(f"\nResult: {result2.get('result', '')[:200]}...")
    print(f"Patterns identified: {result2.get('patterns_identified', [])}")

    # Task 3: Testing agent - learns from both
    print("\n" + "=" * 60)
    print("Task 3: Testing Agent - Write Registration Tests")
    print("(Should benefit from both Backend and Frontend learnings)")
    print("=" * 60)

    result3 = testing_agent.execute_with_memory(
        "Write integration tests for the user registration flow"
    )
    print(f"\nResult: {result3.get('result', '')[:200]}...")
    print(f"Key decisions: {result3.get('key_decisions', [])}")

    # Show memory statistics
    print("\n" + "=" * 60)
    print("Memory DB Statistics")
    print("=" * 60)

    stats = memory_db.get_stats()
    print(f"\nTotal memories: {stats['total_memories']}")
    print(f"\nBy type:")
    for mtype, count in stats['by_type'].items():
        print(f"  {mtype}: {count}")
    print(f"\nBy category:")
    for cat, count in stats['by_category'].items():
        if count > 0:
            print(f"  {cat}: {count}")

    # Query memories
    print("\n" + "=" * 60)
    print("Querying Shared Memory")
    print("=" * 60)

    patterns = memory_db.get_by_category("pattern")
    print(f"\nLong-term patterns learned ({len(patterns)}):")
    for p in patterns[:5]:
        print(f"  - {p.content[:100]}...")
        print(f"    (from {p.source_agent})")


def demonstrate_feedback_loop():
    """Demonstrate the feedback loop pattern."""
    print("\n" + "=" * 60)
    print("Feedback Loop Pattern")
    print("=" * 60)

    print("""
Feedback Loop:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Agent A executes task                                   │
│      │                                                   │
│      ▼                                                   │
│  Finds error/problem                                     │
│      │                                                   │
│      ▼                                                   │
│  Stores in Memory DB                                     │
│      │                                                   │
│      ▼                                                   │
│  Agent B starts similar task                             │
│      │                                                   │
│      ▼                                                   │
│  Recalls relevant memories                               │
│      │                                                   │
│      ▼                                                   │
│  AVOIDS the same error!                                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
""")

    memory_db = MemoryDB()

    # Seed with some known errors
    memory_db.store(
        content="TypeError when comparing None with string in validation - always check for None first",
        memory_type="long",
        category="error",
        source_agent="PreviousAgent"
    )

    memory_db.store(
        content="Rate limiting should be implemented at API gateway level, not application code",
        memory_type="long",
        category="pattern",
        source_agent="ArchitectAgent"
    )

    memory_db.store(
        content="Use Pydantic for request validation - catches errors early with clear messages",
        memory_type="long",
        category="solution",
        source_agent="BackendAgent"
    )

    # New agent benefits from historical knowledge
    new_agent = MemoryAwareAgent("NewDeveloper", memory_db, "full-stack development")

    print("\nNew agent executing task with historical memories...")
    result = new_agent.execute_with_memory(
        "Implement input validation for the user profile update endpoint"
    )

    print(f"\nResult benefited from {len(memory_db.recall('validation', limit=10))} previous memories")
    print(f"New patterns discovered: {result.get('patterns_identified', [])}")


def main():
    print("=" * 60)
    print("Shared Memory Between Agents")
    print("Key Concepts from FullCycle AI Tech Week")
    print("=" * 60)

    print("""
Why Shared Memory?

1. Learning Accumulation
   - Each agent contributes to collective knowledge
   - Errors found once are avoided by all

2. Memory Categories
   - Short-term: Current task context (1 hour)
   - Medium-term: Project context (7 days)
   - Long-term: Organizational knowledge (permanent)

3. Feedback Loop
   - Agent A finds problem → Memory DB
   - Agent B recalls → Avoids problem
   - Continuous improvement

4. Cross-Agent Communication
   - Without sharing full context
   - Just key learnings and patterns
""")

    # Demo 1: Shared memory with multiple agents
    demonstrate_shared_memory()

    # Demo 2: Feedback loop pattern
    demonstrate_feedback_loop()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. Memory DB enables cross-agent learning
2. Categorize by duration: short/medium/long term
3. Categorize by type: error/solution/pattern/decision
4. Semantic search enables relevant recall
5. Feedback loops improve collective performance

"The more agents use the system, the smarter it becomes"
- FullCycle AI Tech Week
""")


if __name__ == "__main__":
    main()
