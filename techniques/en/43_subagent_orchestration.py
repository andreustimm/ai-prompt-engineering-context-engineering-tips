"""
Technique 43: Subagent Orchestration with Context Isolation

Demonstrates orchestrating multiple subagents, each with isolated context windows,
to avoid context degradation and enable parallel processing.

Key concepts from FullCycle AI Tech Week (Aula 2):
- Each subagent has its OWN context window (isolated)
- Main agent only receives RESULTS (not full context)
- This amplifies the number of context windows available
- Enables parallel development with multiple agents

Use cases:
- Long-running development tasks
- Parallel feature development
- Complex multi-step workflows
- Avoiding context window limits
"""

import json
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    agent_name: str
    task: str
    result: str
    key_decisions: list[str]
    files_created: list[str]
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class Subagent:
    """
    A specialized subagent with its own isolated context.

    Key insight: Each subagent starts with a FRESH context window.
    It doesn't inherit the main agent's full context - only what's
    explicitly passed to it. This prevents context degradation.
    """
    name: str
    specialty: str
    system_prompt: str
    messages: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Start with fresh context - only system prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def execute(self, task: str, context: str = "") -> SubagentResult:
        """
        Execute a task with isolated context.

        The subagent only knows:
        1. Its specialty/system prompt
        2. The specific task
        3. Minimal context passed explicitly

        It does NOT know the full conversation history!
        """
        import time
        start_time = time.time()

        # Build task message with only necessary context
        task_content = f"""## Task
{task}

## Context (from previous steps)
{context if context else "This is the first task, no previous context."}

## Instructions
1. Complete the task according to your specialty
2. Be specific and implementation-ready
3. Document key decisions made
4. List any files you would create

Respond with JSON:
{{
    "result": "detailed implementation/solution",
    "key_decisions": ["decision 1", "decision 2"],
    "files_created": ["file1.py", "file2.py"]
}}"""

        self.messages.append({"role": "user", "content": task_content})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                response_format={"type": "json_object"}
            )

            result_data = json.loads(response.choices[0].message.content)
            execution_time = time.time() - start_time

            return SubagentResult(
                agent_name=self.name,
                task=task,
                result=result_data.get("result", ""),
                key_decisions=result_data.get("key_decisions", []),
                files_created=result_data.get("files_created", []),
                execution_time=execution_time,
                success=True
            )
        except Exception as e:
            return SubagentResult(
                agent_name=self.name,
                task=task,
                result="",
                key_decisions=[],
                files_created=[],
                execution_time=time.time() - start_time,
                success=False,
                error=str(e)
            )


class SubagentOrchestrator:
    """
    Orchestrates multiple subagents for complex tasks.

    Key pattern from FullCycle:
    - Main orchestrator has minimal context
    - Each subagent gets fresh context
    - Only results are aggregated
    - Can run multiple agents in parallel
    """

    def __init__(self):
        self.subagents: dict[str, Subagent] = {}
        self.results: list[SubagentResult] = []
        self.accumulated_context: str = ""

    def register_subagent(self, name: str, specialty: str, system_prompt: str):
        """Register a specialized subagent."""
        self.subagents[name] = Subagent(
            name=name,
            specialty=specialty,
            system_prompt=system_prompt
        )

    def execute_sequential(self, tasks: list[tuple[str, str]]) -> list[SubagentResult]:
        """
        Execute tasks sequentially, passing results between steps.

        Args:
            tasks: List of (agent_name, task_description) tuples
        """
        results = []

        for agent_name, task in tasks:
            agent = self.subagents.get(agent_name)
            if not agent:
                print(f"Warning: Agent '{agent_name}' not found")
                continue

            print(f"\n🤖 Executing with {agent_name}: {task[:50]}...")

            # Pass accumulated context (just results, not full history)
            result = agent.execute(task, self.accumulated_context)
            results.append(result)

            if result.success:
                # Add only key info to context (not everything!)
                self.accumulated_context += f"\n\n## Completed: {task}\n"
                self.accumulated_context += f"Key decisions: {', '.join(result.key_decisions)}\n"
                self.accumulated_context += f"Files: {', '.join(result.files_created)}\n"
                print(f"   ✅ Completed in {result.execution_time:.2f}s")
            else:
                print(f"   ❌ Failed: {result.error}")

        return results

    def execute_parallel(self, tasks: list[tuple[str, str, str]]) -> list[SubagentResult]:
        """
        Execute independent tasks in parallel.

        Each task gets its own subagent instance = fresh context window.

        Args:
            tasks: List of (agent_type, task_description, shared_context) tuples
        """
        print("\n🚀 Executing tasks in parallel (separate context windows)...")

        results = []
        for agent_name, task, context in tasks:
            # Create fresh agent instance for each task
            template = self.subagents.get(agent_name)
            if not template:
                continue

            fresh_agent = Subagent(
                name=f"{template.name}-{len(results)}",
                specialty=template.specialty,
                system_prompt=template.system_prompt
            )

            result = fresh_agent.execute(task, context)
            results.append(result)

            status = "✅" if result.success else "❌"
            print(f"   {status} {fresh_agent.name}: {task[:40]}...")

        return results


def create_development_agents() -> SubagentOrchestrator:
    """Create a set of specialized development subagents."""
    orchestrator = SubagentOrchestrator()

    # Backend Agent
    orchestrator.register_subagent(
        name="backend",
        specialty="Backend Development",
        system_prompt="""You are a backend development specialist.
Focus on:
- Clean architecture patterns
- API design best practices
- Database modeling
- Error handling
- Security considerations

Be specific and implementation-ready in your responses."""
    )

    # Frontend Agent
    orchestrator.register_subagent(
        name="frontend",
        specialty="Frontend Development",
        system_prompt="""You are a frontend development specialist.
Focus on:
- Component architecture
- State management
- User experience
- Accessibility
- Performance optimization

Be specific and implementation-ready in your responses."""
    )

    # Testing Agent
    orchestrator.register_subagent(
        name="testing",
        specialty="Testing & Quality",
        system_prompt="""You are a testing and quality assurance specialist.
Focus on:
- Unit test design
- Integration testing
- Test coverage
- Edge cases
- Test data management

Be specific and implementation-ready in your responses."""
    )

    # DevOps Agent
    orchestrator.register_subagent(
        name="devops",
        specialty="DevOps & Infrastructure",
        system_prompt="""You are a DevOps and infrastructure specialist.
Focus on:
- CI/CD pipelines
- Container configuration
- Monitoring & logging
- Security hardening
- Deployment strategies

Be specific and implementation-ready in your responses."""
    )

    return orchestrator


def demonstrate_sequential():
    """Demonstrate sequential execution with context passing."""
    print("=" * 60)
    print("Sequential Execution with Context Passing")
    print("=" * 60)

    print("""
Flow:
┌──────────────────────────────────────────────────────────┐
│ Task 1 (Backend) ─────► Result 1                        │
│                              │                           │
│                              ▼                           │
│ Task 2 (Frontend) ◄──── Context from Result 1           │
│         │                                                │
│         ▼                                                │
│ Task 3 (Testing) ◄──── Context from Results 1,2         │
└──────────────────────────────────────────────────────────┘

Each agent has FRESH context, only receiving essential info.
""")

    orchestrator = create_development_agents()

    tasks = [
        ("backend", "Create a User model with fields: id, email, name, created_at"),
        ("backend", "Create an endpoint POST /users to create new users"),
        ("testing", "Write unit tests for the User model and endpoint"),
    ]

    results = orchestrator.execute_sequential(tasks)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for result in results:
        print(f"\n📦 {result.agent_name}: {result.task[:40]}...")
        print(f"   Success: {result.success}")
        print(f"   Time: {result.execution_time:.2f}s")
        print(f"   Decisions: {result.key_decisions[:2] if result.key_decisions else 'None'}")
        print(f"   Files: {result.files_created[:3] if result.files_created else 'None'}")


def demonstrate_parallel():
    """Demonstrate parallel execution for independent tasks."""
    print("\n" + "=" * 60)
    print("Parallel Execution with Isolated Contexts")
    print("=" * 60)

    print("""
Flow:
┌──────────────────────────────────────────────────────────┐
│             Foundation Complete                          │
│                    │                                     │
│        ┌──────────┼──────────┬──────────┐               │
│        ▼          ▼          ▼          ▼               │
│    Feature A  Feature B  Feature C  Feature D           │
│   (Backend)  (Backend)  (Frontend) (DevOps)             │
│        │          │          │          │               │
│        └──────────┴──────────┴──────────┘               │
│                    │                                     │
│              Aggregation                                 │
└──────────────────────────────────────────────────────────┘

All features execute with SEPARATE context windows!
10 agents = 10 context windows = No degradation
""")

    orchestrator = create_development_agents()

    # Shared context from foundation
    foundation_context = """
## Foundation Complete
- User model created with id, email, name, created_at
- Database schema initialized
- Base API structure ready
"""

    # Independent tasks that can run in parallel
    parallel_tasks = [
        ("backend", "Implement user authentication with JWT", foundation_context),
        ("backend", "Implement user profile update endpoint", foundation_context),
        ("frontend", "Create login form component", foundation_context),
        ("devops", "Set up Docker configuration for the API", foundation_context),
    ]

    results = orchestrator.execute_parallel(parallel_tasks)

    print("\n" + "=" * 60)
    print("Parallel Results Summary")
    print("=" * 60)

    total_time = sum(r.execution_time for r in results)
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"If sequential: ~{total_time:.2f}s")
    print(f"Parallel advantage: Each agent had fresh context!")

    for result in results:
        print(f"\n📦 {result.agent_name}")
        print(f"   Task: {result.task[:40]}...")
        print(f"   Success: {result.success}")


def demonstrate_feature_breakdown():
    """Demonstrate the feature breakdown pattern from FullCycle."""
    print("\n" + "=" * 60)
    print("Feature Breakdown Pattern")
    print("(From FullCycle: Planning enables parallelization)")
    print("=" * 60)

    print("""
Wesley's insight: Plan your features to identify dependencies.

Example Feature Breakdown:
┌─────────────────────────────────────────────────────────┐
│ Feature 1: Foundation ──► MUST be first (dependency)   │
│     │                                                   │
│     ▼                                                   │
│ Feature 2: Auth ──► Depends on Feature 1               │
│     │                                                   │
│     ▼                                                   │
│ Feature 3: Base UI ──► Depends on Feature 2            │
│     │                                                   │
│ ════════════ PARALLEL ZONE ════════════                │
│     │                                                   │
│     ├──► Feature 4: User Profile (independent)         │
│     ├──► Feature 5: Settings (independent)             │
│     ├──► Feature 6: Dashboard (independent)            │
│     └──► Feature 7: Reports (independent)              │
│                                                         │
│ After Feature 3, all others can run in PARALLEL!       │
└─────────────────────────────────────────────────────────┘

"If I have 10 agents, I can run 10 features simultaneously"
- Wesley Williams, FullCycle AI Tech Week
""")

    orchestrator = create_development_agents()

    # Phase 1: Sequential dependencies
    print("\n📍 Phase 1: Sequential (Dependencies)")
    sequential_tasks = [
        ("backend", "Create database schema and User model"),
        ("backend", "Implement authentication system"),
    ]
    orchestrator.execute_sequential(sequential_tasks)

    # Phase 2: Parallel independent features
    print("\n📍 Phase 2: Parallel (Independent Features)")
    foundation = orchestrator.accumulated_context

    parallel_tasks = [
        ("backend", "Implement user profile management", foundation),
        ("backend", "Implement settings/preferences", foundation),
        ("frontend", "Create dashboard layout", foundation),
        ("testing", "Write integration tests for auth", foundation),
    ]
    orchestrator.execute_parallel(parallel_tasks)


def main():
    print("=" * 60)
    print("Subagent Orchestration with Context Isolation")
    print("Key Concepts from FullCycle AI Tech Week")
    print("=" * 60)

    print("""
Why Subagents?

1. Each subagent has its OWN context window
   - Main agent: Context Window A
   - Subagent 1: Context Window B (fresh!)
   - Subagent 2: Context Window C (fresh!)

2. This MULTIPLIES your context capacity
   - Instead of 1 window getting full and summarized
   - You have N fresh windows

3. Enables parallelization
   - 10 branches in parallel = 10 agents working
   - Each with fresh context = no degradation

4. Results aggregation
   - Main agent only receives RESULTS
   - Not the full conversation history
   - Keeps main context clean
""")

    # Demo 1: Sequential execution
    demonstrate_sequential()

    # Demo 2: Parallel execution
    demonstrate_parallel()

    # Demo 3: Feature breakdown pattern
    demonstrate_feature_breakdown()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. Subagents have ISOLATED context windows
2. Pass only RESULTS, not full context
3. Plan features to identify parallelization opportunities
4. Sequential for dependencies, parallel for independents
5. This is how you scale without context degradation

"You can have 10 Cloud Code sessions working on 10 branches"
- Wesley Williams, FullCycle AI Tech Week
""")


if __name__ == "__main__":
    main()
