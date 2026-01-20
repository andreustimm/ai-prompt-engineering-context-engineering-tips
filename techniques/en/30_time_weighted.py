"""
Time-Weighted Retrieval

Incorporates temporal relevance into retrieval, giving preference
to more recent documents. Useful when freshness matters.

Components:
- Timestamp metadata: Documents have creation/update times
- Decay function: Older documents get lower scores
- Combined scoring: Semantic similarity * time weight

Use cases:
- News and current events
- Chat history (recent messages more relevant)
- Documentation (prefer latest versions)
- Log analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

token_tracker = TokenUsage()


def create_time_weighted_documents() -> list[Document]:
    """Create sample documents with timestamps."""
    now = datetime.now()

    documents = [
        Document(
            page_content="Breaking: AI company announces major breakthrough in reasoning capabilities. The new model shows unprecedented performance on complex tasks.",
            metadata={"source": "news", "timestamp": (now - timedelta(hours=2)).isoformat(), "title": "AI Breakthrough"}
        ),
        Document(
            page_content="Weather forecast: Sunny conditions expected throughout the week with temperatures around 75°F.",
            metadata={"source": "weather", "timestamp": (now - timedelta(hours=6)).isoformat(), "title": "Weather Update"}
        ),
        Document(
            page_content="Historical overview of artificial intelligence development from 1950s to 2020.",
            metadata={"source": "article", "timestamp": (now - timedelta(days=180)).isoformat(), "title": "AI History"}
        ),
        Document(
            page_content="Machine learning best practices updated for 2024: Use RAG for knowledge-intensive tasks.",
            metadata={"source": "guide", "timestamp": (now - timedelta(days=30)).isoformat(), "title": "ML Best Practices 2024"}
        ),
        Document(
            page_content="Company policy update: Remote work guidelines have been revised effective immediately.",
            metadata={"source": "policy", "timestamp": (now - timedelta(hours=12)).isoformat(), "title": "Policy Update"}
        ),
        Document(
            page_content="Legacy documentation for Python 2.7 migration strategies.",
            metadata={"source": "docs", "timestamp": (now - timedelta(days=365)).isoformat(), "title": "Python 2.7 Migration"}
        ),
        Document(
            page_content="New research paper on transformer architectures shows improved efficiency.",
            metadata={"source": "research", "timestamp": (now - timedelta(days=7)).isoformat(), "title": "Transformer Research"}
        ),
        Document(
            page_content="Quarterly financial report shows strong growth in AI sector investments.",
            metadata={"source": "finance", "timestamp": (now - timedelta(days=45)).isoformat(), "title": "Q3 Financial Report"}
        ),
    ]

    return documents


def calculate_time_weight(timestamp_str: str, decay_rate: float = 0.01, time_unit: str = "hours") -> float:
    """
    Calculate time-based weight using exponential decay.

    weight = exp(-decay_rate * time_elapsed)

    Args:
        timestamp_str: ISO format timestamp
        decay_rate: How fast relevance decays (higher = faster decay)
        time_unit: "hours", "days", or "minutes"

    Returns:
        Weight between 0 and 1 (1 = most recent)
    """
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        elapsed = datetime.now() - timestamp

        if time_unit == "hours":
            time_value = elapsed.total_seconds() / 3600
        elif time_unit == "days":
            time_value = elapsed.total_seconds() / 86400
        else:  # minutes
            time_value = elapsed.total_seconds() / 60

        weight = math.exp(-decay_rate * time_value)
        return max(0.01, min(1.0, weight))  # Clamp between 0.01 and 1
    except:
        return 0.5  # Default weight if parsing fails


class TimeWeightedRetriever:
    """Retriever that combines semantic similarity with time decay."""

    def __init__(self, documents: list[Document], decay_rate: float = 0.01, time_unit: str = "hours", collection_name: str = "time_weighted"):
        """
        Initialize time-weighted retriever.

        Args:
            documents: Documents with timestamp metadata
            decay_rate: Decay rate for time weighting
            time_unit: Time unit for decay calculation
        """
        self.documents = documents
        self.decay_rate = decay_rate
        self.time_unit = time_unit
        self.embeddings = get_embeddings()

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name
        )

    def retrieve(self, query: str, k: int = 5, time_weight_factor: float = 0.5) -> list[tuple[Document, float, float, float]]:
        """
        Retrieve with time-weighted scoring.

        Args:
            query: Search query
            k: Number of results
            time_weight_factor: Balance between semantic (0) and time (1)

        Returns:
            List of (doc, combined_score, semantic_score, time_weight)
        """
        # Get more results than needed for reranking
        results = self.vectorstore.similarity_search_with_score(query, k=k * 2)

        # Calculate combined scores
        scored_results = []
        for doc, distance in results:
            semantic_score = 1 / (1 + distance)  # Convert distance to similarity
            timestamp = doc.metadata.get("timestamp", datetime.now().isoformat())
            time_weight = calculate_time_weight(timestamp, self.decay_rate, self.time_unit)

            # Combined score
            combined = (1 - time_weight_factor) * semantic_score + time_weight_factor * time_weight
            scored_results.append((doc, combined, semantic_score, time_weight))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:k]


def compare_with_without_time_weighting(documents: list[Document], query: str):
    """Compare results with and without time weighting."""

    print(f"\n   Query: '{query}'")
    print("   " + "=" * 50)

    # Standard retrieval (no time weighting)
    print("\n   📚 STANDARD RETRIEVAL (Semantic Only)")
    print("-" * 30)

    vectorstore = Chroma.from_documents(documents, get_embeddings(), collection_name="standard_compare")
    results = vectorstore.similarity_search_with_score(query, k=4)

    for i, (doc, score) in enumerate(results, 1):
        timestamp = doc.metadata.get("timestamp", "Unknown")
        age = get_age_string(timestamp)
        print(f"\n   {i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"      Age: {age}")
        print(f"      Score: {1/(1+score):.4f}")
        print(f"      Preview: {doc.page_content[:60]}...")

    # Time-weighted retrieval
    print("\n   ⏰ TIME-WEIGHTED RETRIEVAL")
    print("-" * 30)

    retriever = TimeWeightedRetriever(documents, decay_rate=0.05, time_unit="days", collection_name="time_compare")
    results = retriever.retrieve(query, k=4, time_weight_factor=0.4)

    for i, (doc, combined, semantic, time_w) in enumerate(results, 1):
        timestamp = doc.metadata.get("timestamp", "Unknown")
        age = get_age_string(timestamp)
        print(f"\n   {i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"      Age: {age}")
        print(f"      Semantic: {semantic:.4f} | Time: {time_w:.4f} | Combined: {combined:.4f}")
        print(f"      Preview: {doc.page_content[:60]}...")


def get_age_string(timestamp_str: str) -> str:
    """Convert timestamp to human-readable age."""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        elapsed = datetime.now() - timestamp

        if elapsed.days > 0:
            return f"{elapsed.days} days ago"
        elif elapsed.seconds > 3600:
            return f"{elapsed.seconds // 3600} hours ago"
        else:
            return f"{elapsed.seconds // 60} minutes ago"
    except:
        return "Unknown"


def generate_answer(query: str, documents: list[Document]) -> str:
    """Generate answer from retrieved documents."""
    llm = get_llm(temperature=0.3)

    context = "\n\n".join([
        f"[{doc.metadata.get('title', 'Untitled')} - {get_age_string(doc.metadata.get('timestamp', ''))}]\n{doc.page_content}"
        for doc in documents
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context. Prioritize recent information when relevant."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    response = (prompt | llm).invoke({"context": context, "question": query})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer")
    return response.content


def main():
    print("=" * 60)
    print("TIME-WEIGHTED RETRIEVAL")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb required. Install: pip install chromadb")
        return

    token_tracker.reset()

    print("\n📚 CREATING TIME-STAMPED DOCUMENTS")
    print("-" * 40)
    documents = create_time_weighted_documents()
    print(f"   Created {len(documents)} documents with timestamps")

    for doc in documents:
        age = get_age_string(doc.metadata.get("timestamp", ""))
        print(f"   - {doc.metadata.get('title')}: {age}")

    print("\n🔧 CREATING TIME-WEIGHTED RETRIEVER")
    print("-" * 40)
    retriever = TimeWeightedRetriever(
        documents,
        decay_rate=0.05,  # Decay per day
        time_unit="days",
        collection_name="time_weighted_demo"
    )
    print("   Decay rate: 0.05 per day")
    print("   Recent documents will score higher!")

    queries = [
        "What's the latest news about AI?",
        "Tell me about machine learning best practices",
    ]

    print("\n\n❓ TIME-WEIGHTED QUERIES")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 40)

        results = retriever.retrieve(query, k=3, time_weight_factor=0.4)

        print("\n   Results (with time weighting):")
        for i, (doc, combined, semantic, time_w) in enumerate(results, 1):
            age = get_age_string(doc.metadata.get("timestamp", ""))
            print(f"\n   {i}. {doc.metadata.get('title')}")
            print(f"      Age: {age}")
            print(f"      Scores: semantic={semantic:.3f}, time={time_w:.3f}, combined={combined:.3f}")

        docs_for_answer = [r[0] for r in results]
        answer = generate_answer(query, docs_for_answer)
        print(f"\n   Answer: {answer[:250]}...")

    print("\n\n📊 COMPARISON: WITH vs WITHOUT TIME WEIGHTING")
    print("=" * 60)
    compare_with_without_time_weighting(documents, "What are the latest developments in AI?")

    print("\n\n💡 TIME WEIGHTING CONFIGURATION")
    print("-" * 40)
    print("""
   | Use Case          | Decay Rate | Time Unit | Weight Factor |
   |-------------------|------------|-----------|---------------|
   | News/Current      | 0.1-0.5    | hours     | 0.5-0.7       |
   | Chat history      | 0.05-0.1   | hours     | 0.3-0.5       |
   | Documentation     | 0.01-0.05  | days      | 0.2-0.4       |
   | Research papers   | 0.001-0.01 | days      | 0.1-0.3       |

   Higher decay_rate = faster aging of documents
   Higher weight_factor = more preference for recent docs
    """)

    print_total_usage(token_tracker, "TOTAL - Time-Weighted Retrieval")
    print("\nEnd of Time-Weighted Retrieval demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
