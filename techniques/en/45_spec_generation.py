"""
Technique 45: Spec-Driven Development

Demonstrates generating comprehensive technical specifications before writing code.
This ensures alignment between requirements and implementation.

Key concepts from FullCycle AI Tech Week (Aula 2):
- Generate specification BEFORE code
- Spec includes: architecture, interfaces, use cases, constraints
- Validate spec consistency before implementation
- Break down into implementable tasks

Use cases:
- New feature development
- System redesign
- API design
- Architecture decisions
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class UseCase:
    """A use case in the specification."""
    name: str
    actor: str
    description: str
    preconditions: list[str]
    postconditions: list[str]
    main_flow: list[str]
    alternative_flows: list[str] = field(default_factory=list)


@dataclass
class Interface:
    """An interface/contract in the specification."""
    name: str
    type: str  # "api", "event", "function", "class"
    description: str
    inputs: dict
    outputs: dict
    constraints: list[str] = field(default_factory=list)


@dataclass
class TechnicalSpec:
    """Complete technical specification for a feature."""
    title: str
    version: str
    created_at: datetime
    summary: str
    goals: list[str]
    non_goals: list[str]
    architecture: dict
    use_cases: list[UseCase]
    interfaces: list[Interface]
    data_models: list[dict]
    constraints: list[str]
    dependencies: list[str]
    risks: list[dict]
    implementation_phases: list[dict]
    success_metrics: list[str]

    def to_markdown(self) -> str:
        """Convert specification to markdown format."""
        md = f"# Technical Specification: {self.title}\n\n"
        md += f"**Version**: {self.version}\n"
        md += f"**Created**: {self.created_at.strftime('%Y-%m-%d')}\n\n"

        md += f"## Summary\n{self.summary}\n\n"

        md += "## Goals\n"
        for goal in self.goals:
            md += f"- {goal}\n"
        md += "\n"

        md += "## Non-Goals\n"
        for ng in self.non_goals:
            md += f"- {ng}\n"
        md += "\n"

        md += "## Architecture\n"
        md += f"**Pattern**: {self.architecture.get('pattern', 'N/A')}\n"
        md += f"**Components**:\n"
        for comp in self.architecture.get('components', []):
            md += f"- {comp}\n"
        md += "\n"

        md += "## Use Cases\n"
        for uc in self.use_cases:
            md += f"### {uc.name}\n"
            md += f"**Actor**: {uc.actor}\n"
            md += f"**Description**: {uc.description}\n"
            md += f"**Preconditions**: {', '.join(uc.preconditions)}\n"
            md += f"**Flow**: {' → '.join(uc.main_flow)}\n\n"

        md += "## Interfaces\n"
        for iface in self.interfaces:
            md += f"### {iface.name} ({iface.type})\n"
            md += f"{iface.description}\n"
            md += f"**Inputs**: {json.dumps(iface.inputs, indent=2)}\n"
            md += f"**Outputs**: {json.dumps(iface.outputs, indent=2)}\n\n"

        md += "## Data Models\n"
        for model in self.data_models:
            md += f"### {model.get('name', 'Unknown')}\n"
            md += f"**Fields**: {json.dumps(model.get('fields', {}), indent=2)}\n\n"

        md += "## Constraints\n"
        for constraint in self.constraints:
            md += f"- {constraint}\n"
        md += "\n"

        md += "## Implementation Phases\n"
        for i, phase in enumerate(self.implementation_phases, 1):
            md += f"### Phase {i}: {phase.get('name', 'Unknown')}\n"
            md += f"**Tasks**:\n"
            for task in phase.get('tasks', []):
                md += f"- {task}\n"
            md += "\n"

        md += "## Success Metrics\n"
        for metric in self.success_metrics:
            md += f"- {metric}\n"

        return md


class SpecGenerator:
    """
    Generates comprehensive technical specifications from requirements.

    Key insight from FullCycle: Spec before code prevents wasted effort
    and ensures all stakeholders are aligned.
    """

    def __init__(self):
        self.specs: dict[str, TechnicalSpec] = {}

    def generate_spec(self, requirements: str, project_context: str = "") -> TechnicalSpec:
        """
        Generate a complete technical specification from requirements.

        Flow:
        1. Analyze requirements
        2. Define goals and non-goals
        3. Design architecture
        4. Define use cases
        5. Specify interfaces
        6. Plan implementation
        """
        print("Generating technical specification...")

        # Step 1: Initial analysis
        analysis = self._analyze_requirements(requirements, project_context)

        # Step 2: Generate architecture
        architecture = self._design_architecture(requirements, analysis)

        # Step 3: Define use cases
        use_cases = self._define_use_cases(requirements, analysis)

        # Step 4: Specify interfaces
        interfaces = self._specify_interfaces(requirements, architecture)

        # Step 5: Define data models
        data_models = self._define_data_models(requirements, interfaces)

        # Step 6: Plan implementation
        phases = self._plan_implementation(use_cases, interfaces)

        # Build complete spec
        spec = TechnicalSpec(
            title=analysis.get("title", "Untitled Feature"),
            version="1.0.0",
            created_at=datetime.now(),
            summary=analysis.get("summary", ""),
            goals=analysis.get("goals", []),
            non_goals=analysis.get("non_goals", []),
            architecture=architecture,
            use_cases=[UseCase(**uc) for uc in use_cases],
            interfaces=[Interface(**iface) for iface in interfaces],
            data_models=data_models,
            constraints=analysis.get("constraints", []),
            dependencies=analysis.get("dependencies", []),
            risks=analysis.get("risks", []),
            implementation_phases=phases,
            success_metrics=analysis.get("success_metrics", [])
        )

        self.specs[spec.title] = spec
        return spec

    def _analyze_requirements(self, requirements: str, context: str) -> dict:
        """Analyze requirements to extract key information."""
        prompt = f"""Analyze these requirements and extract key information.

Requirements:
{requirements}

Context:
{context if context else "No additional context provided."}

Return JSON with:
{{
    "title": "Feature title",
    "summary": "One paragraph summary",
    "goals": ["goal 1", "goal 2"],
    "non_goals": ["explicitly out of scope 1"],
    "constraints": ["technical constraint 1"],
    "dependencies": ["external dependency 1"],
    "risks": [{{"risk": "...", "mitigation": "..."}}],
    "success_metrics": ["measurable metric 1"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _design_architecture(self, requirements: str, analysis: dict) -> dict:
        """Design the system architecture."""
        prompt = f"""Design the architecture for this feature.

Requirements: {requirements}
Goals: {analysis.get('goals', [])}
Constraints: {analysis.get('constraints', [])}

Return JSON with:
{{
    "pattern": "architecture pattern (e.g., Clean Architecture, MVC)",
    "components": ["component 1", "component 2"],
    "layers": ["layer 1", "layer 2"],
    "data_flow": "description of data flow",
    "integration_points": ["integration 1"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _define_use_cases(self, requirements: str, analysis: dict) -> list[dict]:
        """Define detailed use cases."""
        prompt = f"""Define use cases for this feature.

Requirements: {requirements}
Goals: {analysis.get('goals', [])}

Return JSON with:
{{
    "use_cases": [
        {{
            "name": "UC001: Name",
            "actor": "User/System",
            "description": "...",
            "preconditions": ["condition 1"],
            "postconditions": ["condition 1"],
            "main_flow": ["step 1", "step 2"],
            "alternative_flows": ["alt flow 1"]
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("use_cases", [])

    def _specify_interfaces(self, requirements: str, architecture: dict) -> list[dict]:
        """Specify all interfaces/contracts."""
        prompt = f"""Specify the interfaces for this feature.

Requirements: {requirements}
Architecture: {json.dumps(architecture)}

Return JSON with:
{{
    "interfaces": [
        {{
            "name": "InterfaceName",
            "type": "api/event/function/class",
            "description": "...",
            "inputs": {{"param": "type"}},
            "outputs": {{"field": "type"}},
            "constraints": ["constraint 1"]
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("interfaces", [])

    def _define_data_models(self, requirements: str, interfaces: list) -> list[dict]:
        """Define data models."""
        prompt = f"""Define data models for this feature.

Requirements: {requirements}
Interfaces: {json.dumps(interfaces)}

Return JSON with:
{{
    "models": [
        {{
            "name": "ModelName",
            "description": "...",
            "fields": {{"field_name": "type"}},
            "validations": ["validation 1"],
            "relationships": ["relationship 1"]
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("models", [])

    def _plan_implementation(self, use_cases: list, interfaces: list) -> list[dict]:
        """Plan implementation phases."""
        prompt = f"""Plan the implementation phases.

Use Cases: {json.dumps(use_cases)}
Interfaces: {json.dumps(interfaces)}

Return JSON with:
{{
    "phases": [
        {{
            "name": "Phase name",
            "description": "...",
            "tasks": ["task 1", "task 2"],
            "deliverables": ["deliverable 1"],
            "dependencies": ["dependency on phase X"]
        }}
    ]
}}

Order phases by dependencies (foundation first, then features)."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("phases", [])


class SpecValidator:
    """
    Validates technical specifications for consistency and completeness.

    Key insight: Validate BEFORE coding to catch issues early.
    """

    def validate(self, spec: TechnicalSpec) -> dict:
        """
        Validate a specification for issues.

        Returns validation results with any issues found.
        """
        issues = []
        warnings = []

        # Check completeness
        if not spec.goals:
            issues.append("No goals defined")
        if not spec.use_cases:
            issues.append("No use cases defined")
        if not spec.interfaces:
            warnings.append("No interfaces defined")
        if not spec.implementation_phases:
            issues.append("No implementation phases defined")

        # Check consistency
        if spec.goals and spec.use_cases:
            # Verify each goal has at least one use case
            for goal in spec.goals:
                has_uc = any(
                    goal.lower() in uc.description.lower()
                    for uc in spec.use_cases
                )
                if not has_uc:
                    warnings.append(f"Goal '{goal[:50]}...' may not have a covering use case")

        # Check interfaces have proper definitions
        for iface in spec.interfaces:
            if not iface.inputs and not iface.outputs:
                warnings.append(f"Interface '{iface.name}' has no inputs or outputs")

        # Use LLM for deeper validation
        spec_summary = f"""
Title: {spec.title}
Goals: {spec.goals}
Use Cases: {[uc.name for uc in spec.use_cases]}
Interfaces: {[i.name for i in spec.interfaces]}
Phases: {[p.get('name') for p in spec.implementation_phases]}
"""

        validation_prompt = f"""Validate this technical specification for issues.

{spec_summary}

Check for:
1. Missing critical components
2. Logical inconsistencies
3. Unclear requirements
4. Missing error handling
5. Security considerations

Return JSON:
{{
    "is_valid": true/false,
    "issues": ["critical issue 1"],
    "warnings": ["warning 1"],
    "suggestions": ["improvement 1"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": validation_prompt}],
            response_format={"type": "json_object"}
        )

        llm_validation = json.loads(response.choices[0].message.content)

        return {
            "is_valid": len(issues) == 0 and llm_validation.get("is_valid", False),
            "issues": issues + llm_validation.get("issues", []),
            "warnings": warnings + llm_validation.get("warnings", []),
            "suggestions": llm_validation.get("suggestions", [])
        }


class TaskBreakdown:
    """
    Breaks down specifications into implementable tasks.

    Key insight: Tasks should be atomic and independently verifiable.
    """

    def breakdown(self, spec: TechnicalSpec) -> list[dict]:
        """Break down a specification into tasks."""
        all_tasks = []

        for i, phase in enumerate(spec.implementation_phases):
            phase_tasks = self._breakdown_phase(phase, i + 1, spec)
            all_tasks.extend(phase_tasks)

        return all_tasks

    def _breakdown_phase(self, phase: dict, phase_num: int, spec: TechnicalSpec) -> list[dict]:
        """Break down a single phase into tasks."""
        prompt = f"""Break down this implementation phase into atomic tasks.

Phase: {phase.get('name')}
Tasks outline: {phase.get('tasks', [])}
Full spec context: {spec.title}

Return JSON:
{{
    "tasks": [
        {{
            "id": "T{phase_num}.1",
            "name": "Task name",
            "description": "Detailed description",
            "type": "code/test/config/docs",
            "acceptance_criteria": ["criterion 1"],
            "dependencies": ["T1.1"],
            "estimated_complexity": "low/medium/high"
        }}
    ]
}}

Tasks should be:
1. Atomic (one clear deliverable)
2. Independently testable
3. Clear acceptance criteria"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("tasks", [])


def demonstrate_spec_generation():
    """Demonstrate the spec-driven development workflow."""
    print("=" * 60)
    print("Spec-Driven Development")
    print("=" * 60)

    print("""
Workflow:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Requirements                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────┐                                   │
│  │ Spec Generation     │                                   │
│  │  - Goals            │                                   │
│  │  - Architecture     │                                   │
│  │  - Use Cases        │                                   │
│  │  - Interfaces       │                                   │
│  │  - Data Models      │                                   │
│  └──────────┬──────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                   │
│  │ Spec Validation     │  ◄─── Fix issues                  │
│  │  - Completeness     │                                   │
│  │  - Consistency      │                                   │
│  │  - Security         │                                   │
│  └──────────┬──────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                   │
│  │ Task Breakdown      │                                   │
│  │  - Atomic tasks     │                                   │
│  │  - Dependencies     │                                   │
│  │  - Acceptance       │                                   │
│  └──────────┬──────────┘                                   │
│             │                                               │
│             ▼                                               │
│  Implementation (with clear guidance!)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

    # Example requirements
    requirements = """
    Build a user authentication system with the following features:
    - User registration with email verification
    - Login with email/password
    - Password reset functionality
    - JWT-based session management
    - Rate limiting for security
    - Audit logging for compliance
    """

    project_context = """
    Tech stack: Python/FastAPI, PostgreSQL, Redis
    Existing patterns: Clean Architecture
    Compliance: GDPR, SOC2
    """

    # Generate specification
    generator = SpecGenerator()
    print("\n📋 Generating specification...")
    spec = generator.generate_spec(requirements, project_context)

    # Validate specification
    validator = SpecValidator()
    print("\n✅ Validating specification...")
    validation = validator.validate(spec)

    print(f"\nValidation Result: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")
    if validation['suggestions']:
        print("Suggestions:")
        for suggestion in validation['suggestions']:
            print(f"  💡 {suggestion}")

    # Break down into tasks
    print("\n📝 Breaking down into tasks...")
    breakdown = TaskBreakdown()
    tasks = breakdown.breakdown(spec)

    print(f"\nGenerated {len(tasks)} tasks:")
    for task in tasks[:10]:  # Show first 10
        print(f"\n  [{task.get('id')}] {task.get('name')}")
        print(f"      Type: {task.get('type')}")
        print(f"      Complexity: {task.get('estimated_complexity')}")
        if task.get('dependencies'):
            print(f"      Depends on: {task.get('dependencies')}")

    # Generate markdown spec
    print("\n" + "=" * 60)
    print("Generated Specification (Markdown)")
    print("=" * 60)
    print(spec.to_markdown()[:2000] + "...")


def main():
    print("=" * 60)
    print("Spec-Driven Development")
    print("Key Concepts from FullCycle AI Tech Week")
    print("=" * 60)

    print("""
Why Spec-Driven Development?

1. Alignment BEFORE Code
   - Everyone agrees on what to build
   - Prevents wasted implementation effort
   - Catches issues when cheap to fix

2. Complete Specification Includes:
   - Goals and Non-Goals
   - Architecture decisions
   - Use cases with flows
   - Interface contracts
   - Data models
   - Implementation phases

3. Validation Before Implementation
   - Check completeness
   - Verify consistency
   - Identify risks early

4. Task Breakdown
   - Atomic, testable tasks
   - Clear dependencies
   - Acceptance criteria

"The spec IS the code's blueprint - don't build without it"
- FullCycle AI Tech Week
""")

    # Run demonstration
    demonstrate_spec_generation()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. Generate spec BEFORE code
2. Validate spec for completeness and consistency
3. Break down into atomic, testable tasks
4. Use spec as implementation guide
5. Update spec as requirements evolve

Spec-driven development prevents:
- Misalignment between stakeholders
- Wasted implementation effort
- Missing edge cases
- Incomplete features
""")


if __name__ == "__main__":
    main()
