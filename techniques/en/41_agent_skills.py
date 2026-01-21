"""
Technique 41: Agent Skills System

Demonstrates on-demand knowledge loading for AI agents through a skills system.
Skills are loaded only when needed, saving tokens and improving precision.

Key concepts from FullCycle AI Tech Week (Aula 2):
- Skills are NOT loaded automatically - they're on-demand
- Skill descriptions are like SEO - they must be optimized for the AI to find them
- "Invisible skills" = skills that are never used because of poor descriptions
- Skills can include: guidelines, reference material, scripts, hooks

Use cases:
- Development agents that need context-specific knowledge
- Specialized agents that only load relevant skills
- Token optimization through lazy loading
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class Skill:
    """
    A skill represents on-demand knowledge for an AI agent.

    The description is CRITICAL - it's like SEO for AI.
    If the description is poor, the skill becomes "invisible" and never gets used.
    """
    name: str
    description: str  # CRITICAL: This is like SEO - must be optimized!
    keywords: list[str]  # Additional keywords for matching
    content: str  # The actual knowledge/guidelines
    scripts: dict[str, str] = field(default_factory=dict)  # Optional scripts

    def to_context(self) -> str:
        """Convert skill to context string for the AI."""
        context = f"# Skill: {self.name}\n\n{self.content}"
        if self.scripts:
            context += "\n\n## Available Scripts:\n"
            for script_name, script_desc in self.scripts.items():
                context += f"- `{script_name}`: {script_desc}\n"
        return context


class SkillRegistry:
    """
    Registry of available skills with semantic matching.

    Key insight: The AI decides which skills to load based on descriptions,
    so descriptions must be optimized like SEO keywords.
    """

    def __init__(self):
        self.skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """Register a new skill."""
        self.skills[skill.name] = skill

    def get_skill_catalog(self) -> str:
        """Get a catalog of all skills (names and descriptions only)."""
        catalog = "# Available Skills\n\n"
        for name, skill in self.skills.items():
            catalog += f"## {name}\n"
            catalog += f"**Description**: {skill.description}\n"
            catalog += f"**Keywords**: {', '.join(skill.keywords)}\n\n"
        return catalog

    def load_skill(self, name: str) -> Optional[Skill]:
        """Load a specific skill by name."""
        return self.skills.get(name)

    def search_skills(self, query: str) -> list[Skill]:
        """Search skills by keyword matching."""
        query_lower = query.lower()
        matches = []
        for skill in self.skills.values():
            # Check name, description, and keywords
            if (query_lower in skill.name.lower() or
                query_lower in skill.description.lower() or
                any(query_lower in kw.lower() for kw in skill.keywords)):
                matches.append(skill)
        return matches


def create_sample_skills() -> SkillRegistry:
    """Create sample skills demonstrating the concept."""
    registry = SkillRegistry()

    # Skill 1: Python Testing
    # Note: Description is optimized like SEO - includes key terms the AI might search for
    testing_skill = Skill(
        name="python-testing",
        description=(
            "Python testing guidelines with pytest, testify patterns, unit tests, "
            "integration tests, mocking, fixtures, TDD, test coverage, assertions, "
            "parametrized tests, test organization, CI/CD testing integration"
        ),
        keywords=[
            "test", "pytest", "unittest", "mock", "fixture", "TDD",
            "coverage", "assert", "integration", "unit test"
        ],
        content="""
## Python Testing Guidelines

### Test Organization
- Use `tests/` directory at project root
- Mirror source structure: `tests/unit/`, `tests/integration/`
- Name test files: `test_<module>.py`
- Name test functions: `test_<what_it_tests>`

### Pytest Best Practices
1. Use fixtures for setup/teardown
2. Parametrize tests for multiple inputs
3. Use markers for categorization (@pytest.mark.slow)
4. Keep tests isolated and independent

### Example Structure
```python
# tests/unit/test_calculator.py
import pytest
from calculator import add

@pytest.fixture
def calculator():
    return Calculator()

@pytest.mark.parametrize("a,b,expected", [(1,2,3), (0,0,0), (-1,1,0)])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for critical paths
- Run: `pytest --cov=src --cov-report=html`
""",
        scripts={
            "run_tests.sh": "Run all tests with coverage",
            "validate_tests.sh": "Validate test naming conventions",
        }
    )
    registry.register(testing_skill)

    # Skill 2: API Development
    api_skill = Skill(
        name="api-development",
        description=(
            "REST API development with FastAPI, endpoints, routes, HTTP methods, "
            "request/response models, authentication, JWT, rate limiting, "
            "OpenAPI documentation, Pydantic models, dependency injection"
        ),
        keywords=[
            "api", "rest", "fastapi", "endpoint", "route", "http",
            "jwt", "authentication", "pydantic", "openapi", "swagger"
        ],
        content="""
## API Development Guidelines

### FastAPI Structure
```
app/
  ├── main.py           # Application entry
  ├── api/
  │   ├── routes/       # Route handlers
  │   └── deps.py       # Dependencies
  ├── models/           # Pydantic models
  ├── services/         # Business logic
  └── core/             # Config, security
```

### Endpoint Conventions
- Use plural nouns: `/users`, `/products`
- Use HTTP verbs correctly: GET, POST, PUT, DELETE
- Return appropriate status codes

### Authentication
- Use JWT for stateless auth
- Implement refresh tokens
- Rate limit sensitive endpoints

### Example Endpoint
```python
from fastapi import APIRouter, Depends, HTTPException
from models import User, UserCreate

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=User, status_code=201)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    return await user_service.create(db, user)
```
""",
        scripts={
            "generate_openapi.sh": "Generate OpenAPI spec",
            "validate_api.sh": "Validate API against spec",
        }
    )
    registry.register(api_skill)

    # Skill 3: Error Handling
    error_skill = Skill(
        name="error-handling",
        description=(
            "Error handling patterns, exception hierarchy, try/except blocks, "
            "custom exceptions, error logging, graceful degradation, "
            "error recovery, validation errors, business logic errors"
        ),
        keywords=[
            "error", "exception", "try", "except", "catch", "throw",
            "logging", "validation", "recovery", "graceful"
        ],
        content="""
## Error Handling Guidelines

### Exception Hierarchy
```python
class AppError(Exception):
    '''Base application error'''
    pass

class ValidationError(AppError):
    '''Input validation failed'''
    pass

class NotFoundError(AppError):
    '''Resource not found'''
    pass

class BusinessRuleError(AppError):
    '''Business rule violation'''
    pass
```

### Best Practices
1. Be specific - catch specific exceptions
2. Log before handling - preserve context
3. Fail fast - validate early
4. Provide context - include relevant data

### Logging Pattern
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except ValidationError as e:
    logger.warning(f"Validation failed: {e}", extra={"input": data})
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise AppError("An unexpected error occurred") from e
```
""",
        scripts={
            "check_error_handling.sh": "Analyze error handling coverage",
        }
    )
    registry.register(error_skill)

    return registry


class SkillAwareAgent:
    """
    An AI agent that can discover and load skills on-demand.

    Key insight: The agent first sees the skill catalog (descriptions only),
    then decides which skills to load based on the task.
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.loaded_skills: list[str] = []

    def process_task(self, task: str) -> str:
        """
        Process a task with intelligent skill loading.

        Flow:
        1. Show task + skill catalog to AI
        2. AI decides which skills are relevant
        3. Load only those skills
        4. Execute task with loaded skills
        """
        # Step 1: Ask AI which skills are needed
        skill_catalog = self.registry.get_skill_catalog()

        selection_prompt = f"""You are an AI agent with access to specialized skills.

Given a task, analyze which skills would be helpful and select them.

## Available Skills (descriptions only - content not loaded yet):
{skill_catalog}

## Task:
{task}

## Instructions:
1. Analyze what the task requires
2. Select only the skills that are DIRECTLY relevant
3. Return a JSON object with: {{"skills": ["skill-name-1", "skill-name-2"]}}

Only select skills that will genuinely help. Don't load unnecessary skills (wastes tokens).
"""

        selection_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": selection_prompt}],
            response_format={"type": "json_object"}
        )

        selected = json.loads(selection_response.choices[0].message.content)
        skill_names = selected.get("skills", [])

        # Step 2: Load selected skills
        loaded_content = []
        for name in skill_names:
            skill = self.registry.load_skill(name)
            if skill:
                loaded_content.append(skill.to_context())
                self.loaded_skills.append(name)

        # Step 3: Execute task with loaded skills
        skills_context = "\n\n---\n\n".join(loaded_content) if loaded_content else "No skills loaded."

        execution_prompt = f"""You are an AI development agent.

## Loaded Skills:
{skills_context}

## Task:
{task}

## Instructions:
Execute the task using the guidelines from the loaded skills.
Be specific and follow the patterns/conventions defined in the skills.
"""

        execution_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": execution_prompt}]
        )

        return execution_response.choices[0].message.content


def main():
    print("=" * 60)
    print("Agent Skills System - On-Demand Knowledge Loading")
    print("=" * 60)

    # Create skill registry
    registry = create_sample_skills()

    print("\n📚 Registered Skills:")
    for name in registry.skills:
        print(f"  - {name}")

    # Create skill-aware agent
    agent = SkillAwareAgent(registry)

    # Example 1: Task that needs testing skill
    print("\n" + "=" * 60)
    print("Example 1: Testing-related task")
    print("=" * 60)

    task1 = "Write unit tests for a function that calculates fibonacci numbers"
    print(f"\nTask: {task1}")

    response1 = agent.process_task(task1)
    print(f"\nLoaded skills: {agent.loaded_skills}")
    print(f"\nResponse:\n{response1[:1000]}...")

    # Example 2: Task that needs API skill
    print("\n" + "=" * 60)
    print("Example 2: API-related task")
    print("=" * 60)

    agent.loaded_skills = []  # Reset
    task2 = "Create a REST endpoint to manage user profiles with authentication"
    print(f"\nTask: {task2}")

    response2 = agent.process_task(task2)
    print(f"\nLoaded skills: {agent.loaded_skills}")
    print(f"\nResponse:\n{response2[:1000]}...")

    # Example 3: Task that needs multiple skills
    print("\n" + "=" * 60)
    print("Example 3: Task requiring multiple skills")
    print("=" * 60)

    agent.loaded_skills = []  # Reset
    task3 = "Create an API endpoint with proper error handling and tests"
    print(f"\nTask: {task3}")

    response3 = agent.process_task(task3)
    print(f"\nLoaded skills: {agent.loaded_skills}")
    print(f"\nResponse:\n{response3[:1000]}...")

    print("\n" + "=" * 60)
    print("Key Insights from FullCycle AI Tech Week:")
    print("=" * 60)
    print("""
1. Skills are NOT loaded automatically - they're on-demand
2. Skill descriptions are like SEO - optimize them!
3. "Invisible skills" = skills never used due to poor descriptions
4. Skills save tokens by loading only what's needed
5. Skills can include: guidelines, reference material, scripts
    """)


if __name__ == "__main__":
    main()
