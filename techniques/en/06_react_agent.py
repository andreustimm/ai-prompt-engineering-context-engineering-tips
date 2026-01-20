"""
ReAct (Reasoning + Acting) Agent

Technique that combines reasoning (Thought) with actions (Action) and
observations (Observation) in an iterative loop. The agent thinks,
acts, observes the result and repeats until the problem is solved.

Use cases:
- Information research and analysis
- Tasks requiring external tools
- Problems needing multiple steps
- Integration with APIs and databases
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain import hub
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Attempt to import Wikipedia
try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("Warning: Wikipedia not available. Install with: pip install wikipedia")

# Global token tracker for this script
token_tracker = TokenUsage()


class TokenCounterCallback(BaseCallbackHandler):
    """Callback to count tokens during agent execution."""

    def __init__(self, tracker: TokenUsage):
        self.tracker = tracker

    def on_llm_end(self, response, **kwargs):
        """Called when the LLM finishes generating."""
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            self.tracker.add(input_tokens, output_tokens)
            print_token_usage(input_tokens, output_tokens, "agent_step")


def create_calculator_tool() -> Tool:
    """Creates a simple calculation tool."""

    def calculate(expression: str) -> str:
        """Evaluates a simple mathematical expression."""
        try:
            # Only safe basic operations
            clean_expression = expression.replace("^", "**")
            # Allow only safe characters
            allowed_chars = set("0123456789+-*/().** ")
            if not all(c in allowed_chars for c in clean_expression):
                return "Error: Expression contains disallowed characters"

            result = eval(clean_expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"

    return Tool(
        name="calculator",
        description="Useful for doing mathematical calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5'.",
        func=calculate
    )


def create_search_tool() -> Tool:
    """Creates a web search tool using DuckDuckGo."""
    search = DuckDuckGoSearchRun()

    return Tool(
        name="web_search",
        description="Useful for searching current information on the internet. Use to find recent data, news, or information you don't know.",
        func=search.run
    )


def create_wikipedia_tool() -> Tool:
    """Creates a Wikipedia search tool."""
    if not WIKIPEDIA_AVAILABLE:
        def wikipedia_fallback(query: str) -> str:
            return "Wikipedia is not available. Use web search as an alternative."

        return Tool(
            name="wikipedia",
            description="Searches information on Wikipedia (currently unavailable).",
            func=wikipedia_fallback
        )

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    return Tool(
        name="wikipedia",
        description="Useful for searching encyclopedic information and historical facts on Wikipedia. Good for concepts, biographies, and historical events.",
        func=wikipedia.run
    )


def create_react_agent_instance():
    """
    Creates a ReAct agent with configured tools.

    Returns:
        Configured AgentExecutor
    """
    llm = get_llm(temperature=0)

    # Create tools
    tools = [
        create_search_tool(),
        create_wikipedia_tool(),
        create_calculator_tool()
    ]

    # Standard ReAct template from LangChain Hub
    # You can also create a custom one
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        # Fallback to manual template if hub is not available
        prompt = ChatPromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

    # Create agent
    agent = create_react_agent(llm, tools, prompt)

    # Callback to count tokens
    token_callback = TokenCounterCallback(token_tracker)

    # Create executor with configurations
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        callbacks=[token_callback]
    )

    return agent_executor


def execute_agent(question: str) -> str:
    """
    Executes the ReAct agent to answer a question.

    Args:
        question: The question to be answered

    Returns:
        Agent's response
    """
    agent = create_react_agent_instance()

    try:
        result = agent.invoke({"input": question})
        return result.get("output", "Could not get a response.")
    except Exception as e:
        return f"Error executing agent: {str(e)}"


def demonstrate_react_manual():
    """
    Demonstrates the ReAct pattern manually for educational purposes.
    Explicitly shows the Thought-Action-Observation cycle.
    """
    print("\n🔄 Manual ReAct Pattern Demonstration")
    print("=" * 50)

    llm = get_llm(temperature=0)

    question = "What is the current population of the United States and what percentage of the world population does it represent?"

    print(f"\n❓ Question: {question}")

    # Template that forces the ReAct pattern
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that uses the ReAct pattern to answer questions.
For each question, follow this process:

1. THOUGHT: Analyze what you need to do
2. ACTION: Decide which action to take and with what input
3. OBSERVATION: Analyze the result
4. Repeat until you have the final answer

Available actions:
- SEARCH: search the internet (input: search term)
- CALCULATE: do calculation (input: mathematical expression)
- ANSWER: give the final answer (input: answer)

Response format:
THOUGHT: [your reasoning]
ACTION: [SEARCH/CALCULATE/ANSWER]
ACTION_INPUT: [input for the action]"""),
        ("user", "{question}")
    ])

    chain = prompt | llm

    # ReAct loop simulation
    context = ""
    iteration = 1
    max_iterations = 4

    while iteration <= max_iterations:
        print(f"\n--- Iteration {iteration} ---")

        if context:
            message = f"{question}\n\nPrevious context:\n{context}\n\nContinue the reasoning:"
        else:
            message = question

        response = chain.invoke({"question": message})

        # Extract and record tokens
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, f"iteration_{iteration}")

        answer = response.content
        print(answer)

        # Check if reached final answer
        if "ACTION: ANSWER" in answer:
            print("\n✅ Agent reached the final answer!")
            break

        # Simulate action execution
        if "ACTION: SEARCH" in answer:
            # In production, this would execute real search
            observation = "[Simulation] US Population: ~335 million. World population: ~8 billion."
        elif "ACTION: CALCULATE" in answer:
            # In production, this would execute real calculation
            observation = "[Simulation] 335/8000 * 100 = 4.19%"
        else:
            observation = "[Unrecognized action]"

        print(f"\nOBSERVATION: {observation}")
        context += f"\n{answer}\nOBSERVATION: {observation}"

        iteration += 1

    return "Demonstration complete"


def main():
    print("=" * 60)
    print("ReAct AGENT - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Demo 1: Manual ReAct Pattern (educational)
    demonstrate_react_manual()

    # Demo 2: Complete ReAct Agent
    print("\n" + "=" * 60)
    print("🤖 ReAct AGENT WITH REAL TOOLS")
    print("=" * 60)

    questions = [
        "Who won the last FIFA World Cup and in which country was it held?",
        "Calculate what is 15% of 3500 and then multiply by 12.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 Question {i}: {question}")
        print("=" * 60)

        try:
            response = execute_agent(question)
            print(f"\n🎯 Final Answer: {response}")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

        print()

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - ReAct Agent")

    print("\nEnd of ReAct Agent demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
