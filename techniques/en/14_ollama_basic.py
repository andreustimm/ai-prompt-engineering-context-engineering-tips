"""
Ollama Basic - Local LLM

Using locally running LLM models via Ollama for privacy-focused
and offline AI applications.

Models available:
- llama3.2: General purpose (default)
- mistral: Fast and efficient
- codellama: Code generation
- phi3: Small but capable

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Start server: ollama serve
3. Pull model: ollama pull llama3.2

Use cases:
- Privacy-sensitive applications
- Offline operation
- Cost-effective development/testing
- Local experimentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_ollama_llm,
    is_ollama_available,
    TokenUsage,
    print_total_usage
)

# Global token tracker (Ollama doesn't provide token counts)
token_tracker = TokenUsage()


def check_ollama_status():
    """Check if Ollama is running and available."""
    if is_ollama_available():
        print("   ✓ Ollama is running and accessible")
        return True
    else:
        print("   ✗ Ollama is not available")
        print("   Please ensure Ollama is installed and running:")
        print("   1. Install from https://ollama.ai")
        print("   2. Run: ollama serve")
        print("   3. Pull a model: ollama pull llama3.2")
        return False


def basic_completion(prompt: str, model: str = None) -> str:
    """Simple text completion with Ollama."""
    llm = get_ollama_llm(model=model, temperature=0.7)

    response = llm.invoke(prompt)
    return response.content


def chat_with_system_prompt(user_message: str, system_prompt: str, model: str = None) -> str:
    """Chat completion with system prompt."""
    llm = get_ollama_llm(model=model, temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{message}")
    ])

    chain = prompt | llm
    response = chain.invoke({"message": user_message})
    return response.content


def summarize_text(text: str, model: str = None) -> str:
    """Summarize text using local LLM."""
    llm = get_ollama_llm(model=model, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization expert. Provide concise summaries."),
        ("user", "Summarize the following text in 2-3 sentences:\n\n{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content


def translate_text(text: str, target_language: str, model: str = None) -> str:
    """Translate text using local LLM."""
    llm = get_ollama_llm(model=model, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate accurately while maintaining the original meaning."),
        ("user", "Translate the following text to {language}:\n\n{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text, "language": target_language})
    return response.content


def generate_code(description: str, language: str = "Python", model: str = None) -> str:
    """Generate code from description using local LLM."""
    # Use codellama if available and no model specified
    if model is None:
        model = "codellama"  # Will fallback to default if not available

    try:
        llm = get_ollama_llm(model=model, temperature=0.2)
    except Exception:
        llm = get_ollama_llm(temperature=0.2)  # Use default model

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert {language} programmer. Write clean, well-documented code."),
        ("user", """Write {language} code for the following:

{description}

Provide only the code with comments, no explanations outside the code.""")
    ])

    chain = prompt | llm
    response = chain.invoke({"description": description, "language": language})
    return response.content


def analyze_sentiment(text: str, model: str = None) -> str:
    """Analyze sentiment of text."""
    llm = get_ollama_llm(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the sentiment of the text and respond with:
SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Brief explanation]"""),
        ("user", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content


def answer_question(question: str, context: str = None, model: str = None) -> str:
    """Answer a question, optionally with context."""
    llm = get_ollama_llm(model=model, temperature=0.5)

    if context:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer questions based on the provided context. If the answer isn't in the context, say so."),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])
        inputs = {"context": context, "question": question}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Provide accurate and informative answers."),
            ("user", "{question}")
        ])
        inputs = {"question": question}

    chain = prompt | llm
    response = chain.invoke(inputs)
    return response.content


def compare_models(prompt: str, models: list[str] = None) -> dict:
    """Compare responses from different models."""
    if models is None:
        models = ["llama3.2", "mistral"]

    results = {}

    for model in models:
        try:
            print(f"\n   Testing {model}...")
            llm = get_ollama_llm(model=model, temperature=0.7)
            response = llm.invoke(prompt)
            results[model] = {
                "success": True,
                "response": response.content
            }
        except Exception as e:
            results[model] = {
                "success": False,
                "error": str(e)
            }

    return results


def main():
    print("=" * 60)
    print("OLLAMA BASIC - Local LLM Demo")
    print("=" * 60)

    # Check Ollama availability
    print("\n🔍 CHECKING OLLAMA STATUS")
    print("-" * 40)

    if not check_ollama_status():
        print("\nDemo cannot continue without Ollama running.")
        print("Please start Ollama and try again.")
        return

    # Reset tracker
    token_tracker.reset()

    # Example 1: Basic Completion
    print("\n\n📝 BASIC COMPLETION")
    print("-" * 40)

    prompt = "Explain what artificial intelligence is in one paragraph."
    print(f"\nPrompt: {prompt}")
    print("\nResponse:")
    print(basic_completion(prompt))

    # Example 2: Chat with System Prompt
    print("\n\n💬 CHAT WITH SYSTEM PROMPT")
    print("-" * 40)

    system = "You are a helpful cooking assistant. Provide practical cooking tips."
    message = "How do I make fluffy scrambled eggs?"

    print(f"\nSystem: {system}")
    print(f"User: {message}")
    print("\nResponse:")
    print(chat_with_system_prompt(message, system))

    # Example 3: Summarization
    print("\n\n📋 TEXT SUMMARIZATION")
    print("-" * 40)

    long_text = """
    Machine learning is a subset of artificial intelligence that enables computers
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can access data, learn from it,
    and make predictions or decisions. Machine learning algorithms are used in
    a wide variety of applications, such as email filtering, computer vision,
    and speech recognition. The three main types are supervised learning,
    unsupervised learning, and reinforcement learning, each suited for different
    types of problems and data.
    """

    print(f"\nOriginal ({len(long_text)} chars)")
    print("\nSummary:")
    print(summarize_text(long_text))

    # Example 4: Translation
    print("\n\n🌐 TRANSLATION")
    print("-" * 40)

    text_to_translate = "The quick brown fox jumps over the lazy dog."
    print(f"\nOriginal: {text_to_translate}")
    print("\nPortuguese:")
    print(translate_text(text_to_translate, "Portuguese"))

    # Example 5: Sentiment Analysis
    print("\n\n😊 SENTIMENT ANALYSIS")
    print("-" * 40)

    reviews = [
        "This product exceeded all my expectations! Absolutely love it!",
        "Terrible experience. The item arrived broken and support was unhelpful.",
        "It's okay. Does what it's supposed to do, nothing special."
    ]

    for review in reviews:
        print(f"\nReview: {review[:50]}...")
        print(analyze_sentiment(review))

    # Example 6: Code Generation
    print("\n\n💻 CODE GENERATION")
    print("-" * 40)

    code_request = "a function that calculates the Fibonacci sequence up to n numbers"
    print(f"\nRequest: {code_request}")
    print("\nGenerated Code:")
    print(generate_code(code_request, "Python"))

    # Example 7: Q&A
    print("\n\n❓ QUESTION ANSWERING")
    print("-" * 40)

    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
    It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.
    The tower is 330 meters tall and was the tallest man-made structure in the world
    until 1930. It is named after the engineer Gustave Eiffel, whose company designed
    and built the tower.
    """

    question = "When was the Eiffel Tower built and how tall is it?"
    print(f"\nContext provided: Yes")
    print(f"Question: {question}")
    print("\nAnswer:")
    print(answer_question(question, context))

    print("\n\n" + "=" * 60)
    print("Note: Ollama doesn't provide token counts like OpenAI.")
    print("For cost tracking, monitor Ollama server logs or use timing.")
    print("=" * 60)

    print("\nEnd of Ollama Basic demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
