"""
Chain of Thought (CoT) Prompting

Technique that instructs the model to "think step by step" before
reaching the final answer. Significantly improves performance
on reasoning tasks.

Use cases:
- Mathematical problems
- Logical reasoning
- Complex problem analysis
- Decision making
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


def solve_math_problem(problem: str) -> str:
    """Solves mathematical problems using step-by-step reasoning."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced math teacher.
Solve the following problem showing your step-by-step reasoning.

IMPORTANT:
1. First, identify what the problem is asking
2. List the given data
3. Show each calculation step with explanation
4. Arrive at the final answer clearly

Use the format:
UNDERSTANDING: [what the problem asks]
DATA: [provided information]
STEP 1: [first calculation/reasoning]
STEP 2: [second calculation/reasoning]
...
FINAL ANSWER: [result]"""),
        ("user", "{problem}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problem": problem})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def logical_reasoning(puzzle: str) -> str:
    """Solves logical puzzles with structured reasoning."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in logic and puzzle solving.
Let's think step by step to solve this problem.

For each reasoning step:
1. Identify the given premises
2. Make logical deductions from the premises
3. Eliminate impossible possibilities
4. Reach the conclusion

Clearly show your thought process before giving the final answer."""),
        ("user", "{puzzle}")
    ])

    chain = prompt | llm
    response = chain.invoke({"puzzle": puzzle})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def analyze_decision(situation: str) -> str:
    """Analyzes a situation and helps with decision making."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced strategic consultant.
Analyze the presented situation thinking step by step:

1. CONTEXT: Summarize the current situation
2. STAKEHOLDERS: Identify who is affected
3. OPTIONS: List possible alternatives
4. PROS AND CONS: Analyze each option
5. RISKS: Identify potential problems
6. RECOMMENDATION: Suggest the best decision with justification

Be analytical and objective at each step."""),
        ("user", "{situation}")
    ])

    chain = prompt | llm
    response = chain.invoke({"situation": situation})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def debug_code(code: str, error: str) -> str:
    """Analyzes code and error to find the solution."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior developer specializing in debugging.
Analyze the code and reported error, thinking step by step:

1. CODE UNDERSTANDING: What should the code do?
2. ERROR ANALYSIS: What does the error message indicate?
3. LOCATION: Where is the problem in the code?
4. ROOT CAUSE: Why does the error occur?
5. SOLUTION: How to fix the problem?
6. CORRECTED CODE: Show the correct version

Explain each step of your reasoning."""),
        ("user", "CODE:\n```\n{code}\n```\n\nERROR:\n{error}")
    ])

    chain = prompt | llm
    response = chain.invoke({"code": code, "error": error})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def main():
    print("=" * 60)
    print("CHAIN OF THOUGHT (CoT) - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Math Problem
    print("\n🔢 MATH PROBLEM")
    print("-" * 40)

    math_problem = """
    A store sells t-shirts for $45 each. If a customer buys
    3 or more t-shirts, they get a 15% discount on the total. John bought
    5 t-shirts and paid with a $200 bill. How much change did he receive?
    """

    print(f"Problem: {math_problem.strip()}")
    print("\nSolution:")
    print(solve_math_problem(math_problem))

    # Example 2: Logical Reasoning
    print("\n\n🧩 LOGIC PUZZLE")
    print("-" * 40)

    puzzle = """
    Three friends (Anna, Bob, and Carol) have different professions:
    doctor, engineer, and lawyer. We know that:
    1. Anna is not a doctor
    2. Bob is not a lawyer
    3. The doctor is friends with Carol, but not with Anna

    What is each person's profession?
    """

    print(f"Puzzle: {puzzle.strip()}")
    print("\nSolution:")
    print(logical_reasoning(puzzle))

    # Example 3: Decision Making
    print("\n\n💼 DECISION ANALYSIS")
    print("-" * 40)

    situation = """
    I'm the manager of a tech startup with 20 employees.
    We received two proposals:
    A) $2 million investment from a venture capital fund
       that wants 30% of the company and a board seat.
    B) $1.5 million bank loan with 12% annual interest,
       without giving up equity.

    Our current revenue is $500k/month and we're growing
    15% per month. Which option should I choose?
    """

    print(f"Situation: {situation.strip()}")
    print("\nAnalysis:")
    print(analyze_decision(situation))

    # Example 4: Code Debugging
    print("\n\n🐛 CODE DEBUGGING")
    print("-" * 40)

    code = """def calculate_average(grades):
    total = 0
    for grade in grades:
        total += grade
    return total / len(grades)

result = calculate_average([])
print(f"Average: {result}")"""

    error = "ZeroDivisionError: division by zero"

    print(f"Code:\n{code}")
    print(f"\nError: {error}")
    print("\nAnalysis and Solution:")
    print(debug_code(code, error))

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Chain of Thought")

    print("\nEnd of Chain of Thought demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
