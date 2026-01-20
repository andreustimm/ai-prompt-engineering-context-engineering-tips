"""
Zero-Shot Prompting

Technique where the model receives a task without prior examples.
The model uses only its pre-trained knowledge to respond.

Use cases:
- Sentiment classification
- Text translation
- Entity extraction
- Simple Q&A
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


def classify_sentiment(text: str) -> str:
    """Classifies the sentiment of a text without examples."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analyzer. Classify the sentiment of the text as: POSITIVE, NEGATIVE, or NEUTRAL. Respond only with the classification."),
        ("user", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def translate_text(text: str, target_language: str) -> str:
    """Translates text to the specified language."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate the text to {language}. Return only the translation, without explanations."),
        ("user", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text, "language": target_language})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def extract_entities(text: str) -> str:
    """Extracts named entities from a text."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a named entity extractor.
Extract and list the following entities from the text:
- PERSON: people's names
- LOCATION: places, cities, countries
- ORGANIZATION: companies, institutions
- DATE: dates and time periods

Output format:
PERSON: [list]
LOCATION: [list]
ORGANIZATION: [list]
DATE: [list]"""),
        ("user", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def summarize_text(text: str) -> str:
    """Summarizes a text in a few sentences."""
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization expert. Summarize the following text in no more than 3 sentences, keeping the most important information."),
        ("user", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def main():
    print("=" * 60)
    print("ZERO-SHOT PROMPTING - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Sentiment Classification
    print("\n📊 SENTIMENT CLASSIFICATION")
    print("-" * 40)

    sentiment_texts = [
        "This product is amazing! It exceeded all my expectations.",
        "Terrible customer service, I'm never going back to that store.",
        "The package arrived today at 2pm as expected."
    ]

    for text in sentiment_texts:
        print(f"\nText: {text[:50]}...")
        result = classify_sentiment(text)
        print(f"Sentiment: {result}")

    # Example 2: Translation
    print("\n\n🌍 TRANSLATION")
    print("-" * 40)

    original_text = "A inteligência artificial está transformando como trabalhamos e vivemos."
    print(f"\nOriginal: {original_text}")
    translation = translate_text(original_text, "English")
    print(f"Translation: {translation}")

    # Example 3: Entity Extraction
    print("\n\n🏷️ ENTITY EXTRACTION")
    print("-" * 40)

    entity_text = """
    In March 2024, Elon Musk announced that Tesla will open a new
    factory in Austin, Texas. The partnership with the State Government
    provides for investments of $5 billion over the next 3 years.
    """

    print(f"\nText: {entity_text.strip()}")
    entities = extract_entities(entity_text)
    print(f"\nExtracted entities:\n{entities}")

    # Example 4: Summary
    print("\n\n📝 TEXT SUMMARY")
    print("-" * 40)

    long_text = """
    Generative artificial intelligence is revolutionizing various sectors
    of the global economy. Technology companies invest billions in research
    and development of increasingly sophisticated language models.
    Tools like ChatGPT, Claude, and Gemini allow regular users
    to perform complex writing, programming, and data analysis tasks.
    However, experts warn of the risks associated with irresponsible
    use of these technologies, including the spread of misinformation
    and data privacy concerns.
    """

    print(f"\nOriginal text: {long_text.strip()}")
    summary = summarize_text(long_text)
    print(f"\nSummary:\n{summary}")

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Zero-Shot Prompting")

    print("\nEnd of Zero-Shot demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
