"""
Skeleton of Thought (SoT)

Technique that first generates a "skeleton" of the response (structure/topics)
and then expands each part. Allows parallelization and more
organized responses.

Use cases:
- Long content generation
- Structured responses
- Articles and documentation
- Detailed analyses
"""

import sys
import asyncio
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


def generate_skeleton(topic: str, context: str = "") -> list[str]:
    """
    Phase 1: Generates the skeleton (list of topics) for a theme.

    Args:
        topic: The main topic to be developed
        context: Optional additional context

    Returns:
        List of topics that form the skeleton
    """
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in content structuring.
Given a topic, create a skeleton (outline) with the main points to be covered.

Rules:
- List between 4 and 7 main topics
- Each topic should be clear and focused
- Topics should follow a logical order
- Use short and descriptive titles

Output format (one topic per line):
1. [First topic]
2. [Second topic]
3. [Third topic]
..."""),
        ("user", "TOPIC: {topic}\n{context_text}")
    ])

    context_text = f"CONTEXT: {context}" if context else ""

    chain = prompt | llm
    response = chain.invoke({"topic": topic, "context_text": context_text})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "generate_skeleton")

    result = response.content

    # Parse topics
    lines = result.strip().split("\n")
    topics = []

    for line in lines:
        # Remove numbering and clean
        line = line.strip()
        if line and line[0].isdigit():
            # Remove "1.", "2.", etc.
            parts = line.split(".", 1)
            if len(parts) > 1:
                topic_text = parts[1].strip()
                if topic_text:
                    topics.append(topic_text)
        elif line and not line[0].isdigit():
            # Line without number, but may be a topic
            if line.startswith("-"):
                topics.append(line[1:].strip())

    return topics


def expand_topic(main_topic: str, topic: str, context: str = "") -> str:
    """
    Phase 2: Expands a specific topic from the skeleton.

    Args:
        main_topic: The main topic (for context)
        topic: The topic to be expanded
        context: Optional additional context

    Returns:
        Expanded text of the topic
    """
    llm = get_llm(temperature=0.6)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a writer specialized in creating informative content.
Expand the provided topic in a clear and detailed way.

Rules:
- Write 2-3 paragraphs about the topic
- Be informative and precise
- Use examples when appropriate
- Keep focus on the main theme
- Don't repeat information from the topic title"""),
        ("user", "MAIN TOPIC: {main_topic}\n\nTOPIC TO EXPAND: {topic}\n{context_text}")
    ])

    context_text = f"\nADDITIONAL CONTEXT: {context}" if context else ""

    chain = prompt | llm
    response = chain.invoke({
        "main_topic": main_topic,
        "topic": topic,
        "context_text": context_text
    })

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, f"expand: {topic[:20]}...")

    return response.content


async def expand_topic_async(llm, main_topic: str, topic: str, context: str = "") -> dict:
    """Async version of topic expansion for parallelization."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a writer specialized in creating informative content.
Expand the provided topic in a clear and detailed way.

Rules:
- Write 2-3 paragraphs about the topic
- Be informative and precise
- Use examples when appropriate
- Keep focus on the main theme"""),
        ("user", "MAIN TOPIC: {main_topic}\n\nTOPIC TO EXPAND: {topic}")
    ])

    chain = prompt | llm

    # Execute in thread pool to not block
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: chain.invoke({"main_topic": main_topic, "topic": topic})
    )

    # Extract tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)

    return {
        "topic": topic,
        "content": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


async def skeleton_of_thought_async(topic: str, context: str = "") -> str:
    """
    Async implementation of Skeleton of Thought with parallel expansion.

    Args:
        topic: The topic to be developed
        context: Optional additional context

    Returns:
        Complete generated text
    """
    print("\n🦴 Starting Skeleton of Thought (Async)")
    print("=" * 50)

    # Phase 1: Generate skeleton
    print("\n📋 Phase 1: Generating skeleton...")
    topics = generate_skeleton(topic, context)

    print(f"  Skeleton generated with {len(topics)} topics:")
    for i, t in enumerate(topics, 1):
        print(f"    {i}. {t}")

    # Phase 2: Expand topics in parallel
    print("\n✍️ Phase 2: Expanding topics in parallel...")

    llm = get_llm(temperature=0.6)

    # Create async tasks for each topic
    tasks = [
        expand_topic_async(llm, topic, t, context)
        for t in topics
    ]

    # Execute all in parallel
    results = await asyncio.gather(*tasks)

    # Show tokens per topic
    for result in results:
        print_token_usage(
            result["input_tokens"],
            result["output_tokens"],
            f"expand: {result['topic'][:20]}..."
        )

    # Sort results in original topic order
    sorted_results = sorted(
        results,
        key=lambda x: topics.index(x["topic"])
    )

    print(f"  {len(results)} sections expanded!")

    # Phase 3: Assemble final document
    print("\n📄 Phase 3: Assembling final document...")

    document = f"# {topic}\n\n"

    for i, result in enumerate(sorted_results, 1):
        document += f"## {i}. {result['topic']}\n\n"
        document += f"{result['content']}\n\n"

    return document


def skeleton_of_thought_sync(topic: str, context: str = "") -> str:
    """
    Synchronous implementation of Skeleton of Thought.

    Args:
        topic: The topic to be developed
        context: Optional additional context

    Returns:
        Complete generated text
    """
    print("\n🦴 Starting Skeleton of Thought (Sync)")
    print("=" * 50)

    # Phase 1: Generate skeleton
    print("\n📋 Phase 1: Generating skeleton...")
    topics = generate_skeleton(topic, context)

    print(f"  Skeleton generated with {len(topics)} topics:")
    for i, t in enumerate(topics, 1):
        print(f"    {i}. {t}")

    # Phase 2: Expand each topic sequentially
    print("\n✍️ Phase 2: Expanding topics...")

    sections = []
    for i, t in enumerate(topics, 1):
        print(f"  Expanding topic {i}/{len(topics)}: {t[:30]}...")
        content = expand_topic(topic, t, context)
        sections.append({"topic": t, "content": content})

    # Phase 3: Assemble final document
    print("\n📄 Phase 3: Assembling final document...")

    document = f"# {topic}\n\n"

    for i, section in enumerate(sections, 1):
        document += f"## {i}. {section['topic']}\n\n"
        document += f"{section['content']}\n\n"

    return document


def generate_with_review(topic: str, context: str = "") -> str:
    """
    Skeleton of Thought with additional review step.

    Args:
        topic: The topic to be developed
        context: Optional additional context

    Returns:
        Final reviewed text
    """
    # Generate base document
    document = skeleton_of_thought_sync(topic, context)

    # Review phase
    print("\n🔍 Extra Phase: Reviewing document...")

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional editor.
Review the provided document and make improvements in:
- Cohesion between sections
- Text clarity
- Repetition correction
- Adding transitions between topics

Return the complete reviewed document, keeping the title structure."""),
        ("user", "{document}")
    ])

    chain = prompt | llm
    response = chain.invoke({"document": document})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "review")

    return response.content


def main():
    print("=" * 60)
    print("SKELETON OF THOUGHT (SoT) - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Topic for demonstration
    topic = "Artificial Intelligence in Medicine: Applications and Challenges"
    context = "Focus on practical applications already in use and ethical challenges"

    print(f"\n📌 TOPIC: {topic}")
    print(f"📝 CONTEXT: {context}")

    # Demo 1: Synchronous Version
    print("\n" + "-" * 60)
    print("DEMO 1: Synchronous Version")
    print("-" * 60)

    document_sync = skeleton_of_thought_sync(topic, context)

    print("\n" + "=" * 60)
    print("📄 GENERATED DOCUMENT (Synchronous Version)")
    print("=" * 60)
    print(document_sync)

    # Demo 2: Async Version (if supported)
    print("\n" + "-" * 60)
    print("DEMO 2: Asynchronous Version (Parallel)")
    print("-" * 60)

    topic2 = "Web Application Security Best Practices"

    try:
        document_async = asyncio.run(skeleton_of_thought_async(topic2, ""))

        print("\n" + "=" * 60)
        print("📄 GENERATED DOCUMENT (Asynchronous Version)")
        print("=" * 60)
        print(document_async)
    except Exception as e:
        print(f"  Error in async version: {e}")
        print("  Using sync version as fallback...")
        document_async = skeleton_of_thought_sync(topic2, "")
        print(document_async)

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Skeleton of Thought")

    print("\nEnd of Skeleton of Thought demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
