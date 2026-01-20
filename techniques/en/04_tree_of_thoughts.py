"""
Tree of Thoughts (ToT)

Technique that explores multiple reasoning paths in parallel,
evaluates each one and selects the most promising. Useful for problems
that require exploration and backtracking.

Use cases:
- Strategic planning
- Problems with multiple possible solutions
- Games and puzzles
- Decision optimization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker for this script
token_tracker = TokenUsage()


def generate_thoughts(problem: str, num_thoughts: int = 3) -> list[str]:
    """Generates multiple initial reasoning paths."""
    llm = get_llm(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative problem solver.
Given a problem, generate {num_thoughts} DIFFERENT and INDEPENDENT approaches to solve it.

Each approach should:
- Be unique and distinct from the others
- Have clear logic
- Be a viable first step toward the solution

Output format (use exactly this format):
APPROACH 1: [description of first approach]
APPROACH 2: [description of second approach]
APPROACH 3: [description of third approach]"""),
        ("user", "{problem}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problem": problem, "num_thoughts": num_thoughts})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "generate_thoughts")

    result = response.content

    # Parse approaches
    lines = result.strip().split("\n")
    approaches = []
    current_approach = ""

    for line in lines:
        if line.startswith("APPROACH"):
            if current_approach:
                approaches.append(current_approach.strip())
            current_approach = line.split(":", 1)[1] if ":" in line else line
        else:
            current_approach += " " + line

    if current_approach:
        approaches.append(current_approach.strip())

    return approaches[:num_thoughts]


def evaluate_thought(problem: str, thought: str) -> dict:
    """Evaluates the quality and viability of a reasoning path."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critical solution evaluator.
Evaluate the proposed approach to solve the problem.

Evaluation criteria (0-10 each):
1. VIABILITY: Is the approach feasible?
2. EFFICIENCY: Is the approach efficient?
3. COMPLETENESS: Does the approach solve the problem completely?
4. CREATIVITY: Is the approach innovative?

Response format:
VIABILITY: [0-10]
EFFICIENCY: [0-10]
COMPLETENESS: [0-10]
CREATIVITY: [0-10]
TOTAL: [sum/40]
JUSTIFICATION: [brief explanation]
NEXT_STEP: [suggestion on how to continue this approach]"""),
        ("user", "PROBLEM: {problem}\n\nPROPOSED APPROACH: {thought}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problem": problem, "thought": thought})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "evaluate_thought")

    result = response.content

    # Basic parsing to extract total score
    lines = result.strip().split("\n")
    evaluation = {
        "full_text": result,
        "score": 0,
        "next_step": ""
    }

    for line in lines:
        if line.startswith("TOTAL:"):
            try:
                # Extract number from format "XX/40" or similar
                value = line.split(":")[1].strip().split("/")[0]
                evaluation["score"] = float(value)
            except (ValueError, IndexError):
                evaluation["score"] = 0
        elif line.startswith("NEXT_STEP:"):
            evaluation["next_step"] = line.split(":", 1)[1].strip()

    return evaluation


def expand_thought(problem: str, thought: str, next_step: str) -> str:
    """Expands a promising thought to the next level."""
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a methodical problem solver.
Continue developing the proposed approach, following the suggested next step.

Develop the reasoning in detail, showing:
1. How to implement the next step
2. Possible obstacles and how to overcome them
3. Expected results from this stage"""),
        ("user", """PROBLEM: {problem}

CURRENT APPROACH: {thought}

SUGGESTED NEXT STEP: {next_step}

Continue the development:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "problem": problem,
        "thought": thought,
        "next_step": next_step
    })

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "expand_thought")

    return response.content


def synthesize_solution(problem: str, best_path: list[str]) -> str:
    """Synthesizes the final solution from the best path found."""
    llm = get_llm(temperature=0.3)

    path_text = "\n\n".join([f"STAGE {i+1}:\n{stage}" for i, stage in enumerate(best_path)])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in solution synthesis.
Based on the developed reasoning path, present the final solution in a clear and structured way.

The solution should include:
1. SUMMARY: Overview of the solution
2. IMPLEMENTATION STEPS: Ordered list of actions
3. REQUIRED RESOURCES: What is needed to implement
4. EXPECTED OUTCOME: What will be achieved
5. RISKS AND MITIGATIONS: Possible problems and how to avoid them"""),
        ("user", "PROBLEM: {problem}\n\nREASONING PATH:\n{path}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problem": problem, "path": path_text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "synthesize_solution")

    return response.content


def tree_of_thoughts(problem: str, depth: int = 2) -> str:
    """
    Implements the complete Tree of Thoughts algorithm.

    Args:
        problem: The problem to be solved
        depth: How many expansion levels to explore

    Returns:
        The synthesized solution
    """
    print(f"\n🌳 Starting Tree of Thoughts (depth={depth})")
    print("=" * 50)

    # Phase 1: Generate initial thoughts
    print("\n📝 Phase 1: Generating initial approaches...")
    thoughts = generate_thoughts(problem, num_thoughts=3)

    for i, t in enumerate(thoughts, 1):
        print(f"\n  Approach {i}: {t[:100]}...")

    # Phase 2: Evaluate and select the best
    print("\n⚖️ Phase 2: Evaluating approaches...")
    evaluations = []

    for i, thought in enumerate(thoughts, 1):
        evaluation = evaluate_thought(problem, thought)
        evaluations.append({
            "thought": thought,
            "evaluation": evaluation
        })
        print(f"\n  Approach {i}: Score = {evaluation['score']}/40")

    # Sort by score and select the best
    evaluations.sort(key=lambda x: x["evaluation"]["score"], reverse=True)
    best = evaluations[0]

    print(f"\n✅ Best approach selected (score: {best['evaluation']['score']}/40)")

    # Phase 3: Expand the best path
    path = [best["thought"]]
    current_thought = best["thought"]
    next_step = best["evaluation"]["next_step"]

    for level in range(depth):
        print(f"\n🔍 Phase 3.{level+1}: Expanding level {level+1}...")

        expansion = expand_thought(problem, current_thought, next_step)
        path.append(expansion)

        # Evaluate the expansion to get next step
        evaluation = evaluate_thought(problem, expansion)
        next_step = evaluation["next_step"]
        current_thought = expansion

        print(f"  Expansion complete (score: {evaluation['score']}/40)")

    # Phase 4: Synthesize solution
    print("\n📋 Phase 4: Synthesizing final solution...")
    solution = synthesize_solution(problem, path)

    return solution


def main():
    print("=" * 60)
    print("TREE OF THOUGHTS (ToT) - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Complex problem for demonstration
    problem = """
    A technology startup with 15 employees needs to decide
    how to expand its operations. Currently operates only in the US,
    has a SaaS product with 500 paying customers and revenue of
    $200k/month. The goal is to triple revenue in 18 months.

    Constraints:
    - Available investment budget: $1 million
    - Technical team is already at capacity
    - Product still has significant technical debt

    What is the best growth strategy?
    """

    print(f"\n📌 PROBLEM:\n{problem.strip()}")

    # Execute Tree of Thoughts
    solution = tree_of_thoughts(problem, depth=2)

    print("\n" + "=" * 60)
    print("🎯 FINAL SOLUTION")
    print("=" * 60)
    print(solution)

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Tree of Thoughts")

    print("\nEnd of Tree of Thoughts demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
