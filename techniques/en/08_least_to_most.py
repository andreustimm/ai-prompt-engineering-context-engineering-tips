"""
Least-to-Most Prompting

Technique that decomposes complex problems into simpler sub-problems,
solving them progressively from easiest to hardest, using previous
results to inform subsequent solutions.

Use cases:
- Multi-step math problems
- Complex reasoning tasks
- Planning and scheduling
- Educational explanations
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


def decompose_problem(problem: str) -> list[str]:
    """Decompose a complex problem into simpler sub-problems."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at breaking down complex problems into simpler sub-problems.
Given a problem, identify the component sub-problems that need to be solved.
Order them from simplest to most complex.
Return ONLY a numbered list of sub-problems, nothing else.

Format:
1. [simplest sub-problem]
2. [next sub-problem]
3. [next sub-problem]
... and so on"""),
        ("user", "Problem: {problem}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problem": problem})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Decomposition")

    # Parse the numbered list
    lines = response.content.strip().split('\n')
    sub_problems = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Remove the number and period
            parts = line.split('.', 1)
            if len(parts) > 1:
                sub_problems.append(parts[1].strip())

    return sub_problems


def solve_sub_problem(sub_problem: str, previous_context: str = "") -> str:
    """Solve a single sub-problem, optionally using previous context."""
    llm = get_llm(temperature=0.2)

    if previous_context:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a problem-solving expert.
Solve the given sub-problem using the provided context from previous solutions.
Be concise but thorough. Show your work."""),
            ("user", """Previous solutions:
{context}

Current sub-problem to solve: {sub_problem}

Solution:""")
        ])
        inputs = {"sub_problem": sub_problem, "context": previous_context}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a problem-solving expert.
Solve the given sub-problem. Be concise but thorough. Show your work."""),
            ("user", """Sub-problem: {sub_problem}

Solution:""")
        ])
        inputs = {"sub_problem": sub_problem}

    chain = prompt | llm
    response = chain.invoke(inputs)

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Sub-problem")

    return response.content


def synthesize_final_answer(problem: str, solutions: list[tuple[str, str]]) -> str:
    """Synthesize all sub-problem solutions into a final answer."""
    llm = get_llm(temperature=0.2)

    solutions_text = "\n\n".join([
        f"Sub-problem: {sp}\nSolution: {sol}"
        for sp, sol in solutions
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at synthesizing solutions.
Given the original problem and solutions to its sub-problems,
provide a clear, comprehensive final answer."""),
        ("user", """Original Problem: {problem}

Sub-problem Solutions:
{solutions}

Please provide the final, complete answer to the original problem:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"problem": problem, "solutions": solutions_text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Synthesis")

    return response.content


def least_to_most_solve(problem: str) -> dict:
    """
    Solve a complex problem using least-to-most prompting.

    Returns:
        Dictionary with decomposition, sub-solutions, and final answer
    """
    print("\n   Step 1: Decomposing problem...")
    sub_problems = decompose_problem(problem)

    print(f"\n   Found {len(sub_problems)} sub-problems:")
    for i, sp in enumerate(sub_problems, 1):
        print(f"      {i}. {sp}")

    print("\n   Step 2: Solving sub-problems progressively...")
    solutions = []
    context = ""

    for i, sub_problem in enumerate(sub_problems, 1):
        print(f"\n      Solving sub-problem {i}/{len(sub_problems)}...")
        solution = solve_sub_problem(sub_problem, context)
        solutions.append((sub_problem, solution))

        # Update context with this solution
        context += f"\nSub-problem: {sub_problem}\nSolution: {solution}\n"

    print("\n   Step 3: Synthesizing final answer...")
    final_answer = synthesize_final_answer(problem, solutions)

    return {
        "sub_problems": sub_problems,
        "solutions": solutions,
        "final_answer": final_answer
    }


def solve_math_problem(problem: str) -> dict:
    """Solve a math problem using least-to-most approach."""
    return least_to_most_solve(problem)


def create_learning_path(topic: str, current_level: str) -> dict:
    """Create a progressive learning path for a topic."""
    problem = f"Create a learning path for someone at {current_level} level to master {topic}"
    return least_to_most_solve(problem)


def plan_project(project_description: str) -> dict:
    """Plan a project by breaking it into progressive milestones."""
    problem = f"Plan the implementation of this project: {project_description}"
    return least_to_most_solve(problem)


def explain_concept(concept: str) -> dict:
    """Explain a complex concept by building from fundamentals."""
    problem = f"Explain {concept} starting from basic concepts and building up to full understanding"
    return least_to_most_solve(problem)


def main():
    print("=" * 60)
    print("LEAST-TO-MOST PROMPTING - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Complex Math Problem
    print("\n🔢 COMPLEX MATH PROBLEM")
    print("-" * 40)

    math_problem = """
    A train leaves City A at 9:00 AM traveling at 60 mph toward City B.
    Another train leaves City B at 10:00 AM traveling at 80 mph toward City A.
    The cities are 280 miles apart.
    At what time will the trains meet, and how far from City A?
    """

    print(f"\nProblem: {math_problem.strip()}")
    result = solve_math_problem(math_problem)

    print(f"\n📋 FINAL ANSWER:")
    print("-" * 40)
    print(result["final_answer"])

    # Example 2: Learning Path
    print("\n\n📚 LEARNING PATH CREATION")
    print("-" * 40)

    topic = "Machine Learning"
    level = "beginner with basic Python knowledge"

    print(f"\nTopic: {topic}")
    print(f"Current Level: {level}")

    result = create_learning_path(topic, level)

    print(f"\n📋 LEARNING PATH:")
    print("-" * 40)
    print(result["final_answer"])

    # Example 3: Project Planning
    print("\n\n🏗️ PROJECT PLANNING")
    print("-" * 40)

    project = "Build a REST API for a todo application with user authentication, CRUD operations, and database persistence"

    print(f"\nProject: {project}")

    result = plan_project(project)

    print(f"\n📋 PROJECT PLAN:")
    print("-" * 40)
    print(result["final_answer"])

    # Example 4: Concept Explanation
    print("\n\n💡 CONCEPT EXPLANATION")
    print("-" * 40)

    concept = "how neural networks learn through backpropagation"

    print(f"\nConcept: {concept}")

    result = explain_concept(concept)

    print(f"\n📋 EXPLANATION:")
    print("-" * 40)
    print(result["final_answer"])

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Least-to-Most Prompting")

    print("\nEnd of Least-to-Most demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
