"""
Multi-Agent Applications with LLMs

Multi-agent systems allow multiple AI agents to collaborate
to solve complex tasks. Each agent can have different specializations,
tools, and responsibilities.

Multi-Agent Patterns:
- Orchestrator: A central agent coordinates other agents
- Pipeline: Agents process in sequence
- Debate: Agents discuss to reach consensus
- Hierarchical: Agents organized in authority levels

Use cases:
- Software development (planner, coder, reviewer, tester)
- Research (collector, analyzer, synthesizer)
- Customer service (triage, specialists, supervisor)
- Data analysis (ETL, analysis, visualization)

Requirements:
- pip install openai
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.openai_client import get_openai_client


class AgentRole(Enum):
    """Possible roles for agents."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class AgentMessage:
    """Message exchanged between agents."""
    sender: str
    receiver: str
    content: str
    message_type: str = "task"  # task, response, feedback, question
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class Agent:
    """Represents an individual agent in the system."""
    name: str
    role: AgentRole
    system_prompt: str
    tools: list = field(default_factory=list)
    memory: list = field(default_factory=list)

    def add_to_memory(self, message: AgentMessage):
        """Adds a message to the agent's memory."""
        self.memory.append(message)

    def get_context(self, max_messages: int = 10) -> str:
        """Returns recent context from memory."""
        recent = self.memory[-max_messages:]
        return "\n".join([
            f"[{m.sender} -> {m.receiver}]: {m.content}"
            for m in recent
        ])


class MultiAgentSystem:
    """
    Collaborative multi-agent system.

    This system demonstrates how to coordinate multiple AI agents
    to solve complex tasks collaboratively.
    """

    def __init__(self, name: str = "multi-agent-system"):
        self.name = name
        self.agents: dict[str, Agent] = {}
        self.message_history: list[AgentMessage] = []
        self.client = get_openai_client()

    def add_agent(self, agent: Agent):
        """Adds an agent to the system."""
        self.agents[agent.name] = agent
        print(f"   Agent added: {agent.name} ({agent.role.value})")

    def send_message(self, message: AgentMessage):
        """Sends a message between agents."""
        self.message_history.append(message)

        # Add to sender's and receiver's memory
        if message.sender in self.agents:
            self.agents[message.sender].add_to_memory(message)
        if message.receiver in self.agents:
            self.agents[message.receiver].add_to_memory(message)

    def agent_respond(self, agent_name: str, task: str, context: str = "") -> str:
        """Makes an agent respond to a task."""
        if agent_name not in self.agents:
            return f"Agent '{agent_name}' not found"

        agent = self.agents[agent_name]

        # Build prompt with context
        full_prompt = f"""
Conversation context:
{context if context else agent.get_context()}

Current task:
{task}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content


def create_software_development_team() -> MultiAgentSystem:
    """Creates a software development team with multiple agents."""

    system = MultiAgentSystem(name="dev-team")

    # Planner Agent
    planner = Agent(
        name="Planner",
        role=AgentRole.PLANNER,
        system_prompt="""You are an experienced software architect.
Your role is to:
- Analyze software requirements
- Create detailed implementation plans
- Break tasks into smaller subtasks
- Identify dependencies and risks

Always respond in a structured format with:
1. Problem analysis
2. Implementation plan
3. Specific tasks for the team"""
    )

    # Coder Agent
    coder = Agent(
        name="Coder",
        role=AgentRole.EXECUTOR,
        system_prompt="""You are an experienced Python developer.
Your role is to:
- Implement code based on specifications
- Follow programming best practices
- Write clean and well-documented code
- Create reusable functions and classes

Always include:
- Explanatory docstrings
- Type hints
- Basic error handling"""
    )

    # Reviewer Agent
    reviewer = Agent(
        name="Reviewer",
        role=AgentRole.REVIEWER,
        system_prompt="""You are a software quality engineer.
Your role is to:
- Review code for bugs
- Verify best practices
- Suggest performance improvements
- Evaluate code security

Provide constructive feedback with:
1. Positive points
2. Issues found
3. Improvement suggestions"""
    )

    # Tester Agent
    tester = Agent(
        name="Tester",
        role=AgentRole.SPECIALIST,
        system_prompt="""You are a software testing specialist.
Your role is to:
- Create comprehensive test cases
- Identify edge cases
- Write unit tests
- Validate test coverage

Always include:
- Tests for normal cases
- Tests for edge cases
- Error tests"""
    )

    system.add_agent(planner)
    system.add_agent(coder)
    system.add_agent(reviewer)
    system.add_agent(tester)

    return system


def demonstrate_pipeline_pattern():
    """Demonstrates the Pipeline multi-agent pattern."""

    print("\n" + "=" * 60)
    print("PIPELINE PATTERN")
    print("=" * 60)

    print("""
    In the Pipeline pattern, tasks flow sequentially:

    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Planner  │──▶│  Coder   │──▶│ Reviewer │──▶│  Tester  │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘

    Each agent processes and passes to the next.
    """)

    system = create_software_development_team()
    task = "Create a function that validates email addresses"

    print(f"\nTask: {task}")
    print("-" * 40)

    # 1. Planner analyzes
    print("\n1. PLANNER analyzes the task:")
    plan = system.agent_respond("Planner", task)
    print(f"   {plan[:500]}...")

    # Record the message
    system.send_message(AgentMessage(
        sender="Planner",
        receiver="Coder",
        content=plan,
        message_type="task"
    ))

    # 2. Coder implements
    print("\n2. CODER implements:")
    code = system.agent_respond(
        "Coder",
        f"Implement based on this plan:\n{plan}"
    )
    print(f"   {code[:500]}...")

    system.send_message(AgentMessage(
        sender="Coder",
        receiver="Reviewer",
        content=code,
        message_type="response"
    ))

    # 3. Reviewer evaluates
    print("\n3. REVIEWER evaluates the code:")
    review = system.agent_respond(
        "Reviewer",
        f"Review this code:\n{code}"
    )
    print(f"   {review[:500]}...")

    system.send_message(AgentMessage(
        sender="Reviewer",
        receiver="Tester",
        content=f"Code:\n{code}\n\nReview:\n{review}",
        message_type="feedback"
    ))

    # 4. Tester creates tests
    print("\n4. TESTER creates tests:")
    tests = system.agent_respond(
        "Tester",
        f"Create tests for:\n{code}"
    )
    print(f"   {tests[:500]}...")

    return system


def demonstrate_debate_pattern():
    """Demonstrates the Debate multi-agent pattern."""

    print("\n" + "=" * 60)
    print("DEBATE PATTERN")
    print("=" * 60)

    print("""
    In the Debate pattern, agents discuss to reach consensus:

         ┌──────────┐
         │Moderator │
         └────┬─────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Expert1│◀▶│Expert2│◀▶│Expert3│
└───────┘ └───────┘ └───────┘

    Agents debate until converging on a solution.
    """)

    system = MultiAgentSystem(name="debate-system")

    # Create specialists with different perspectives
    expert_performance = Agent(
        name="Expert_Performance",
        role=AgentRole.SPECIALIST,
        system_prompt="""You are a software performance specialist.
Always prioritize and argue in favor of:
- Execution speed
- Efficient memory usage
- Algorithm optimization
Defend your positions with technical arguments."""
    )

    expert_maintainability = Agent(
        name="Expert_Maintainability",
        role=AgentRole.SPECIALIST,
        system_prompt="""You are a code maintainability specialist.
Always prioritize and argue in favor of:
- Readable and clear code
- Adequate documentation
- Design patterns
Defend your positions with technical arguments."""
    )

    expert_security = Agent(
        name="Expert_Security",
        role=AgentRole.SPECIALIST,
        system_prompt="""You are a software security specialist.
Always prioritize and argue in favor of:
- Input validation
- Vulnerability protection
- Principle of least privilege
Defend your positions with technical arguments."""
    )

    moderator = Agent(
        name="Moderator",
        role=AgentRole.ORCHESTRATOR,
        system_prompt="""You are a moderator of technical discussions.
Your role is to:
- Synthesize different perspectives
- Identify points of consensus
- Propose balanced solutions
- Facilitate decision making"""
    )

    system.add_agent(expert_performance)
    system.add_agent(expert_maintainability)
    system.add_agent(expert_security)
    system.add_agent(moderator)

    question = "What is the best way to implement user authentication?"

    print(f"\nQuestion for debate: {question}")
    print("-" * 40)

    # Each specialist gives their opinion
    print("\n1. Performance Expert:")
    perf_opinion = system.agent_respond("Expert_Performance", question)
    print(f"   {perf_opinion[:400]}...")

    print("\n2. Maintainability Expert:")
    maint_opinion = system.agent_respond("Expert_Maintainability", question)
    print(f"   {maint_opinion[:400]}...")

    print("\n3. Security Expert:")
    sec_opinion = system.agent_respond("Expert_Security", question)
    print(f"   {sec_opinion[:400]}...")

    # Moderator synthesizes
    print("\n4. MODERATOR synthesizes:")
    synthesis_prompt = f"""
Synthesize the following opinions on: {question}

Performance: {perf_opinion}

Maintainability: {maint_opinion}

Security: {sec_opinion}

Propose a balanced solution that considers all perspectives.
"""
    synthesis = system.agent_respond("Moderator", synthesis_prompt)
    print(f"   {synthesis[:600]}...")

    return system


def demonstrate_hierarchical_pattern():
    """Demonstrates the Hierarchical multi-agent pattern."""

    print("\n" + "=" * 60)
    print("HIERARCHICAL PATTERN")
    print("=" * 60)

    print("""
    In the Hierarchical pattern, agents are organized in levels:

                    ┌──────────────┐
                    │   Director   │
                    │  (Level 1)   │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Backend   │  │  Frontend  │  │   DevOps   │
    │  Manager   │  │  Manager   │  │  Manager   │
    │ (Level 2)  │  │ (Level 2)  │  │ (Level 2)  │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
       ┌──┴──┐         ┌──┴──┐         ┌──┴──┐
       ▼     ▼         ▼     ▼         ▼     ▼
    Dev1   Dev2     Dev3   Dev4     Dev5   Dev6
    (L3)   (L3)     (L3)   (L3)     (L3)   (L3)

    Top-down delegation, bottom-up reporting.
    """)

    system = MultiAgentSystem(name="hierarchical-system")

    # Director (Level 1)
    director = Agent(
        name="Director",
        role=AgentRole.ORCHESTRATOR,
        system_prompt="""You are the technical director of a company.
Your role is to:
- Define the overall technical vision
- Delegate tasks to managers
- Make strategic decisions
- Resolve conflicts between areas

Always delegate specific tasks to each area."""
    )

    # Managers (Level 2)
    backend_manager = Agent(
        name="Backend_Manager",
        role=AgentRole.PLANNER,
        system_prompt="""You are the backend manager.
Responsible for:
- APIs and services
- Databases
- Integrations
- Server performance

Receive tasks from the director and distribute to your team."""
    )

    frontend_manager = Agent(
        name="Frontend_Manager",
        role=AgentRole.PLANNER,
        system_prompt="""You are the frontend manager.
Responsible for:
- User interface
- User experience
- Client performance
- Accessibility

Receive tasks from the director and distribute to your team."""
    )

    system.add_agent(director)
    system.add_agent(backend_manager)
    system.add_agent(frontend_manager)

    project = "Develop an e-commerce system"

    print(f"\nProject: {project}")
    print("-" * 40)

    # Director defines strategy
    print("\n1. DIRECTOR defines strategy:")
    strategy = system.agent_respond(
        "Director",
        f"Define strategy and delegate tasks for: {project}"
    )
    print(f"   {strategy[:500]}...")

    # Managers receive and plan
    print("\n2. BACKEND MANAGER plans:")
    backend_plan = system.agent_respond(
        "Backend_Manager",
        f"Based on the director's strategy, plan backend tasks:\n{strategy}"
    )
    print(f"   {backend_plan[:400]}...")

    print("\n3. FRONTEND MANAGER plans:")
    frontend_plan = system.agent_respond(
        "Frontend_Manager",
        f"Based on the director's strategy, plan frontend tasks:\n{strategy}"
    )
    print(f"   {frontend_plan[:400]}...")

    return system


def main():
    print("=" * 60)
    print("MULTI-AGENT APPLICATIONS WITH LLMs")
    print("=" * 60)

    print("""
    Multi-agent systems allow multiple AI agents to collaborate
    to solve complex tasks.

    Main patterns:

    1. PIPELINE - Sequential processing
       Input → Agent1 → Agent2 → Agent3 → Output

    2. DEBATE - Discussion for consensus
       Agents with different perspectives debate

    3. HIERARCHICAL - Level-based organization
       Top-down delegation, bottom-up reporting

    4. ORCHESTRATOR - Central coordination
       A central agent coordinates the others
    """)

    # Demonstrate each pattern
    print("\n" + "=" * 60)
    print("CREATING DEVELOPMENT TEAM")
    print("=" * 60)

    system = create_software_development_team()

    # Demonstrate patterns (each makes API calls)
    demonstrate_pipeline_pattern()
    demonstrate_debate_pattern()
    demonstrate_hierarchical_pattern()

    print("\n" + "=" * 60)
    print("BENEFITS OF MULTI-AGENTS")
    print("=" * 60)

    print("""
    1. Specialization
       - Each agent focuses on a specific area
       - More targeted and effective prompts

    2. Scalability
       - Add new agents as needed
       - Parallelize independent tasks

    3. Quality
       - Cross-review between agents
       - Multiple perspectives on the problem

    4. Maintainability
       - Modular and independent agents
       - Easy to update or replace

    5. Traceability
       - Communication history between agents
       - Decision audit trail
    """)

    print("\n" + "=" * 60)
    print("POPULAR FRAMEWORKS")
    print("=" * 60)

    print("""
    For implementing multi-agents in production:

    1. LangGraph (LangChain)
       - Agent graphs with states
       - LangChain integration
       - pip install langgraph

    2. AutoGen (Microsoft)
       - Multi-agent conversation
       - Code execution
       - pip install pyautogen

    3. CrewAI
       - Agent teams with roles
       - Tasks and processes
       - pip install crewai

    4. Swarm (OpenAI)
       - Experimental framework
       - Handoffs between agents
       - Lightweight and educational
    """)

    print("\nEnd of Multi-Agent demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
