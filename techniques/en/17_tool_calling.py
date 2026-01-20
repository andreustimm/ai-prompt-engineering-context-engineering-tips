"""
Tool Calling (Function Calling)

Technique that allows LLMs to invoke external tools/functions to
perform actions or retrieve information beyond their training data.

Features:
- Custom tool definitions
- Automatic argument parsing
- Tool execution and response handling
- Multi-tool workflows

Use cases:
- Calculator and math operations
- Database queries
- API integrations
- System operations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


# Define custom tools using the @tool decorator

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3")

    Returns:
        The result of the calculation
    """
    try:
        # Using eval with restricted builtins for safety
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """
    Get the current date and time.

    Args:
        timezone: Optional timezone (not implemented, uses local time)

    Returns:
        Current date and time as string
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def get_weather(city: str) -> str:
    """
    Get weather information for a city (simulated).

    Args:
        city: Name of the city

    Returns:
        Weather information for the city
    """
    # Simulated weather data
    weather_data = {
        "new york": {"temp": 72, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 59, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 68, "condition": "Sunny", "humidity": 55},
        "paris": {"temp": 64, "condition": "Cloudy", "humidity": 70},
        "sydney": {"temp": 75, "condition": "Sunny", "humidity": 60},
        "são paulo": {"temp": 78, "condition": "Partly cloudy", "humidity": 72},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"Weather in {city}: {data['temp']}°F, {data['condition']}, Humidity: {data['humidity']}%"
    else:
        return f"Weather data not available for {city}. Try: New York, London, Tokyo, Paris, Sydney, São Paulo"


@tool
def search_database(query: str, table: str = "products") -> str:
    """
    Search a simulated database (demonstration only).

    Args:
        query: Search query
        table: Table to search in (products, users, orders)

    Returns:
        Simulated search results
    """
    # Simulated database
    databases = {
        "products": [
            {"id": 1, "name": "Laptop Pro", "price": 1299, "category": "Electronics"},
            {"id": 2, "name": "Wireless Mouse", "price": 49, "category": "Electronics"},
            {"id": 3, "name": "Office Chair", "price": 299, "category": "Furniture"},
        ],
        "users": [
            {"id": 1, "name": "John Doe", "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "product_id": 1, "status": "shipped"},
            {"id": 102, "user_id": 2, "product_id": 3, "status": "processing"},
        ],
    }

    if table not in databases:
        return f"Table '{table}' not found. Available: products, users, orders"

    results = databases[table]
    # Simple text search
    query_lower = query.lower()
    matching = [r for r in results if query_lower in str(r).lower()]

    if matching:
        return f"Found {len(matching)} results in {table}:\n" + json.dumps(matching, indent=2)
    else:
        return f"No results found for '{query}' in {table}"


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.

    Args:
        value: The numeric value to convert
        from_unit: Source unit (km, miles, kg, lb, c, f)
        to_unit: Target unit (km, miles, kg, lb, c, f)

    Returns:
        Converted value with units
    """
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("c", "f"): lambda x: x * 9/5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported"


def run_tool_calling_agent(query: str, tools: list, max_iterations: int = 5) -> str:
    """
    Run an agent that can call tools to answer queries.

    Args:
        query: User's question or request
        tools: List of tools available to the agent
        max_iterations: Maximum number of tool calls

    Returns:
        Final response from the agent
    """
    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(content=query)]

    for i in range(max_iterations):
        print(f"\n   Iteration {i+1}...")

        # Get LLM response
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Extract tokens
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        # Check if there are tool calls
        if not response.tool_calls:
            print(f"   No more tool calls needed")
            return response.content

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"   Calling tool: {tool_name}({tool_args})")

            # Find and execute the tool
            for t in tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    print(f"   Tool result: {result[:100]}...")

                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
                    break

    return "Max iterations reached without final answer"


def main():
    print("=" * 60)
    print("TOOL CALLING (Function Calling) - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Define available tools
    tools = [calculate, get_current_time, get_weather, search_database, convert_units]

    print("\n📋 Available Tools:")
    for t in tools:
        print(f"   - {t.name}: {t.description.split('.')[0]}")

    # Example 1: Calculator
    print("\n\n🔢 CALCULATOR EXAMPLE")
    print("-" * 40)

    query1 = "What is 15% of 250 plus 100?"
    print(f"\nQuery: {query1}")
    print("\nExecuting...")
    result1 = run_tool_calling_agent(query1, tools)
    print(f"\n📋 Final Answer: {result1}")

    # Example 2: Weather
    print("\n\n🌤️ WEATHER EXAMPLE")
    print("-" * 40)

    query2 = "What's the weather like in Tokyo and London?"
    print(f"\nQuery: {query2}")
    print("\nExecuting...")
    result2 = run_tool_calling_agent(query2, tools)
    print(f"\n📋 Final Answer: {result2}")

    # Example 3: Database Search
    print("\n\n🔍 DATABASE SEARCH EXAMPLE")
    print("-" * 40)

    query3 = "Find all electronics products in the database"
    print(f"\nQuery: {query3}")
    print("\nExecuting...")
    result3 = run_tool_calling_agent(query3, tools)
    print(f"\n📋 Final Answer: {result3}")

    # Example 4: Unit Conversion
    print("\n\n📐 UNIT CONVERSION EXAMPLE")
    print("-" * 40)

    query4 = "Convert 100 kilometers to miles and 30 degrees Celsius to Fahrenheit"
    print(f"\nQuery: {query4}")
    print("\nExecuting...")
    result4 = run_tool_calling_agent(query4, tools)
    print(f"\n📋 Final Answer: {result4}")

    # Example 5: Multi-tool Query
    print("\n\n🔄 MULTI-TOOL QUERY EXAMPLE")
    print("-" * 40)

    query5 = "What time is it, what's the weather in Paris, and calculate 20% tip on a $85 dinner bill?"
    print(f"\nQuery: {query5}")
    print("\nExecuting...")
    result5 = run_tool_calling_agent(query5, tools)
    print(f"\n📋 Final Answer: {result5}")

    print_total_usage(token_tracker, "TOTAL - Tool Calling")

    print("\nEnd of Tool Calling demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
