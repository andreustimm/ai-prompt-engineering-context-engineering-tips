"""
Few-Shot Prompting (One-Shot / Few-Shot)

Technique where we provide examples to the model before asking the question.
This helps the model understand the format and type of expected response.

Use cases:
- Format conversion
- Code generation following patterns
- Classification with specific categories
- Tasks with specific output format
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker for this script
token_tracker = TokenUsage()


def classify_support_ticket(ticket: str) -> str:
    """Classifies support tickets using examples."""
    llm = get_llm(temperature=0)

    # Few-shot examples
    examples = [
        {
            "ticket": "I can't login to my account, the password is correct but an error appears",
            "classification": "CATEGORY: Authentication\nPRIORITY: High\nACTION: Check account lockout and access logs"
        },
        {
            "ticket": "I'd like to know how to export reports to PDF",
            "classification": "CATEGORY: Usage Question\nPRIORITY: Low\nACTION: Forward documentation and tutorial"
        },
        {
            "ticket": "The system has been very slow since yesterday, it takes 30 seconds to load each page",
            "classification": "CATEGORY: Performance\nPRIORITY: Critical\nACTION: Escalate to infrastructure team"
        },
        {
            "ticket": "I need to add 5 more users to my business account",
            "classification": "CATEGORY: Commercial\nPRIORITY: Medium\nACTION: Forward to sales team"
        }
    ]

    # Template for each example
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{ticket}"),
        ("assistant", "{classification}")
    ])

    # Few-shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    # Final prompt
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a support ticket classification system. Classify each ticket with category, priority, and recommended action."),
        few_shot_prompt,
        ("user", "{ticket}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"ticket": ticket})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def convert_to_sql(description: str) -> str:
    """Converts natural language descriptions to SQL."""
    llm = get_llm(temperature=0)

    examples = [
        {
            "description": "List all customers from Brazil",
            "sql": "SELECT * FROM customers WHERE country = 'Brazil';"
        },
        {
            "description": "Count how many orders were placed in January 2024",
            "sql": "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31';"
        },
        {
            "description": "Show the top 10 best-selling products with name and quantity",
            "sql": "SELECT p.name, SUM(oi.quantity) as total_sold\nFROM products p\nJOIN order_items oi ON p.id = oi.product_id\nGROUP BY p.id, p.name\nORDER BY total_sold DESC\nLIMIT 10;"
        },
        {
            "description": "Update the price of all products in the 'Electronics' category by increasing 10%",
            "sql": "UPDATE products SET price = price * 1.10 WHERE category = 'Electronics';"
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{description}"),
        ("assistant", "```sql\n{sql}\n```")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Convert the natural language description into a valid SQL query.
Consider the following available tables:
- customers (id, name, email, country, registration_date)
- products (id, name, price, category, stock)
- orders (id, customer_id, order_date, status, total)
- order_items (id, order_id, product_id, quantity, unit_price)"""),
        few_shot_prompt,
        ("user", "{description}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"description": description})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def generate_docstring(code: str) -> str:
    """Generates Google-style docstrings for Python functions."""
    llm = get_llm(temperature=0.3)

    examples = [
        {
            "code": "def add(a, b):\n    return a + b",
            "documented": '''def add(a, b):
    """Adds two numbers.

    Args:
        a: First number to be added.
        b: Second number to be added.

    Returns:
        The sum of a and b.
    """
    return a + b'''
        },
        {
            "code": "def find_user(user_id, include_inactive=False):\n    users = db.query(User).filter(User.id == user_id)\n    if not include_inactive:\n        users = users.filter(User.active == True)\n    return users.first()",
            "documented": '''def find_user(user_id, include_inactive=False):
    """Finds a user by ID in the database.

    Args:
        user_id: Unique ID of the user to find.
        include_inactive: If True, includes inactive users in the search.
            Defaults to False.

    Returns:
        The User object if found, None otherwise.
    """
    users = db.query(User).filter(User.id == user_id)
    if not include_inactive:
        users = users.filter(User.active == True)
    return users.first()'''
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{code}"),
        ("assistant", "{documented}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Python developer. Add Google Style docstrings to the provided code. Keep the original code, just add the documentation."),
        few_shot_prompt,
        ("user", "{code}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"code": code})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def extract_structured_data(text: str) -> str:
    """Extracts structured data from free text in JSON format."""
    llm = get_llm(temperature=0)

    examples = [
        {
            "text": "John Smith, 35 years old, lives in New York and works as a software engineer at TechCorp. His email is john.smith@email.com",
            "json": '{\n  "name": "John Smith",\n  "age": 35,\n  "city": "New York",\n  "profession": "software engineer",\n  "company": "TechCorp",\n  "email": "john.smith@email.com"\n}'
        },
        {
            "text": "Product: Dell XPS 15 Laptop, price $1,500.00, in stock (23 units), category: Electronics/Computers",
            "json": '{\n  "product": "Dell XPS 15 Laptop",\n  "brand": "Dell",\n  "model": "XPS 15",\n  "price": 1500.00,\n  "currency": "USD",\n  "stock": 23,\n  "available": true,\n  "category": ["Electronics", "Computers"]\n}'
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{text}"),
        ("assistant", "```json\n{json}\n```")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data extraction system. Extract information from the text and return in structured JSON format. Infer additional fields when appropriate."),
        few_shot_prompt,
        ("user", "{text}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"text": text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def main():
    print("=" * 60)
    print("FEW-SHOT PROMPTING - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Ticket Classification
    print("\n🎫 SUPPORT TICKET CLASSIFICATION")
    print("-" * 40)

    ticket = "My card was charged twice for the same order #12345, I need an urgent refund"

    print(f"\nTicket: {ticket}")
    print(f"\nClassification:\n{classify_support_ticket(ticket)}")

    # Example 2: SQL Conversion
    print("\n\n💾 SQL CONVERSION")
    print("-" * 40)

    descriptions = [
        "Show total sales by month in 2024",
        "Find customers who haven't placed orders in the last 6 months"
    ]

    for desc in descriptions:
        print(f"\nDescription: {desc}")
        print(f"SQL: {convert_to_sql(desc)}")

    # Example 3: Docstring Generation
    print("\n\n📝 DOCSTRING GENERATION")
    print("-" * 40)

    code = """def calculate_discount(total_amount, coupon=None, vip_customer=False):
    discount = 0
    if coupon and coupon in VALID_COUPONS:
        discount += VALID_COUPONS[coupon]
    if vip_customer:
        discount += 0.1
    return total_amount * (1 - min(discount, 0.5))"""

    print(f"\nOriginal code:\n{code}")
    print(f"\nWith docstring:\n{generate_docstring(code)}")

    # Example 4: Structured Data Extraction
    print("\n\n📊 STRUCTURED DATA EXTRACTION")
    print("-" * 40)

    text = """
    Reservation confirmed: Marriott Hotel New York, check-in March 15, 2024 at 2pm,
    check-out March 18, 2024 at 12pm. Superior double room, 3 nights, total amount
    $890.00 (breakfast included). Guest: Mary Johnson, SSN: 123-45-6789
    """

    print(f"\nText: {text.strip()}")
    print(f"\nExtracted data:\n{extract_structured_data(text)}")

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Few-Shot Prompting")

    print("\nEnd of Few-Shot demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
