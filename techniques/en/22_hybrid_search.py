"""
Hybrid Search (BM25 + Vector)

Combines keyword-based search (BM25) with semantic search (vectors)
for improved retrieval accuracy. This approach leverages the strengths
of both methods.

Components:
- BM25 Retriever: Traditional keyword matching (TF-IDF based)
- Vector Retriever: Semantic similarity search
- Ensemble Retriever: Combines results using Reciprocal Rank Fusion

Use cases:
- Technical documentation with specific terminology
- Legal documents with precise keyword requirements
- Product search combining exact matches with semantic understanding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Conditional imports
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Run: pip install chromadb")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not installed. Run: pip install rank-bm25")

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


def load_sample_documents() -> list[Document]:
    """Load sample documents for demonstration."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    documents = []

    for file_path in sample_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                ))
        except Exception as e:
            print(f"   Warning: Could not load {file_path}: {e}")

    for file_path in sample_dir.glob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                ))
        except Exception as e:
            print(f"   Warning: Could not load {file_path}: {e}")

    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = i
        chunks.extend(doc_chunks)

    return chunks


class BM25Retriever:
    """BM25-based keyword retriever."""

    def __init__(self, documents: list[Document]):
        """Initialize BM25 retriever with documents."""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")

        self.documents = documents
        # Tokenize documents for BM25
        self.tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Retrieve top-k documents for query."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], scores[idx]))

        return results


class VectorRetriever:
    """Vector-based semantic retriever."""

    def __init__(self, documents: list[Document], collection_name: str = "hybrid_search"):
        """Initialize vector retriever with documents."""
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        self.embeddings = get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name
        )

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Retrieve top-k documents with similarity scores."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Note: Chroma returns distance, lower is better. Convert to similarity.
        return [(doc, 1 / (1 + score)) for doc, score in results]


def reciprocal_rank_fusion(
    results_list: list[list[tuple[Document, float]]],
    weights: list[float] = None,
    k: int = 60
) -> list[tuple[Document, float]]:
    """
    Combine results from multiple retrievers using Reciprocal Rank Fusion.

    RRF score = sum(weight_i / (k + rank_i)) for each retriever

    Args:
        results_list: List of results from each retriever
        weights: Weights for each retriever (default: equal weights)
        k: RRF parameter (typically 60)

    Returns:
        Combined and reranked results
    """
    if weights is None:
        weights = [1.0] * len(results_list)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Calculate RRF scores
    doc_scores = {}

    for retriever_idx, results in enumerate(results_list):
        weight = weights[retriever_idx]

        for rank, (doc, _) in enumerate(results, start=1):
            # Use document content as key (simplified)
            doc_key = doc.page_content[:100]

            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": doc, "score": 0.0}

            # RRF formula
            doc_scores[doc_key]["score"] += weight / (k + rank)

    # Sort by combined score
    sorted_results = sorted(
        doc_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [(item["doc"], item["score"]) for item in sorted_results]


class HybridRetriever:
    """Combines BM25 and Vector retrievers."""

    def __init__(
        self,
        documents: list[Document],
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        collection_name: str = "hybrid_search"
    ):
        """
        Initialize hybrid retriever.

        Args:
            documents: Documents to index
            bm25_weight: Weight for BM25 results
            vector_weight: Weight for vector results
        """
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        print("   Initializing BM25 retriever...")
        self.bm25_retriever = BM25Retriever(documents)

        print("   Initializing Vector retriever...")
        self.vector_retriever = VectorRetriever(documents, collection_name)

    def retrieve(self, query: str, k: int = 4, initial_k: int = 10) -> list[tuple[Document, float]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query
            k: Number of final results
            initial_k: Number of candidates from each retriever

        Returns:
            Combined results with RRF scores
        """
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, k=initial_k)
        vector_results = self.vector_retriever.retrieve(query, k=initial_k)

        # Combine using RRF
        combined = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            weights=[self.bm25_weight, self.vector_weight]
        )

        return combined[:k]


def compare_retrieval_methods(chunks: list[Document], query: str, k: int = 3):
    """Compare BM25, Vector, and Hybrid retrieval methods."""

    print(f"\n   Query: '{query}'")
    print("   " + "=" * 50)

    # BM25 only
    print("\n   📚 BM25 (Keyword) Results:")
    print("   " + "-" * 30)

    bm25_retriever = BM25Retriever(chunks)
    bm25_results = bm25_retriever.retrieve(query, k=k)

    for i, (doc, score) in enumerate(bm25_results, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. Score: {score:.4f}")
        print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"      Preview: {preview}...")
        print()

    # Vector only
    print("\n   🔮 Vector (Semantic) Results:")
    print("   " + "-" * 30)

    vector_retriever = VectorRetriever(chunks, collection_name="compare_vector")
    vector_results = vector_retriever.retrieve(query, k=k)

    for i, (doc, score) in enumerate(vector_results, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. Score: {score:.4f}")
        print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"      Preview: {preview}...")
        print()

    # Hybrid
    print("\n   🔄 Hybrid (BM25 + Vector) Results:")
    print("   " + "-" * 30)

    combined = reciprocal_rank_fusion(
        [bm25_results, vector_results],
        weights=[0.4, 0.6]
    )

    for i, (doc, score) in enumerate(combined[:k], 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. RRF Score: {score:.4f}")
        print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"      Preview: {preview}...")
        print()


def generate_answer_with_context(query: str, documents: list[tuple[Document, float]]) -> str:
    """Generate answer using retrieved context."""

    llm = get_llm(temperature=0.3)

    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc, _ in documents
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the answer is not in the context, say so.
Be concise but thorough."""),
        ("user", """Context:
{context}

Question: {question}

Answer:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer Generation")

    return response.content


def demonstrate_hybrid_advantage():
    """Show cases where hybrid search outperforms individual methods."""

    print("\n   Demonstrating Hybrid Search Advantages...")
    print("   " + "=" * 50)

    # Create synthetic documents that highlight differences
    documents = [
        Document(
            page_content="The Python programming language was created by Guido van Rossum. It emphasizes code readability and uses significant whitespace.",
            metadata={"source": "python_overview.txt"}
        ),
        Document(
            page_content="Machine learning algorithms can be implemented in Python using libraries like scikit-learn, TensorFlow, and PyTorch.",
            metadata={"source": "ml_tools.txt"}
        ),
        Document(
            page_content="A python is a large snake found in tropical regions. These reptiles are constrictors, meaning they squeeze their prey.",
            metadata={"source": "animals.txt"}
        ),
        Document(
            page_content="Web development with Django and Flask frameworks makes Python an excellent choice for building scalable applications.",
            metadata={"source": "web_dev.txt"}
        ),
        Document(
            page_content="Programming languages like Java, C++, and Python are widely used in software development.",
            metadata={"source": "languages.txt"}
        ),
    ]

    # Query that benefits from hybrid approach
    query = "Python programming web development"

    print(f"\n   Query: '{query}'")
    print("\n   This query benefits from hybrid search because:")
    print("   - BM25 catches exact 'Python' keyword matches")
    print("   - Vector search understands semantic relationship to web dev")
    print("   - Combined approach filters out the snake document")

    compare_retrieval_methods(documents, query, k=3)


def main():
    print("=" * 60)
    print("HYBRID SEARCH (BM25 + Vector)")
    print("=" * 60)

    if not BM25_AVAILABLE:
        print("\nError: rank-bm25 is required for this demo.")
        print("Install with: pip install rank-bm25")
        return

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb is required for this demo.")
        print("Install with: pip install chromadb")
        return

    token_tracker.reset()

    # Load and prepare documents
    print("\n📚 LOADING AND PREPARING DOCUMENTS")
    print("-" * 40)

    documents = load_sample_documents()

    if not documents:
        print("   No documents found. Using sample documents.")
        documents = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                metadata={"source": "sample.txt"}
            )
        ]

    print(f"   Loaded {len(documents)} documents")

    chunks = chunk_documents(documents, chunk_size=400, chunk_overlap=80)
    print(f"   Created {len(chunks)} chunks")

    # Initialize hybrid retriever
    print("\n🔧 INITIALIZING HYBRID RETRIEVER")
    print("-" * 40)

    hybrid_retriever = HybridRetriever(
        chunks,
        bm25_weight=0.4,
        vector_weight=0.6,
        collection_name="hybrid_demo"
    )

    # Demo queries
    queries = [
        "What is machine learning?",
        "neural networks deep learning",
        "How to build AI applications?"
    ]

    print("\n\n❓ HYBRID SEARCH QUERIES")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: {query}")
        print("-" * 40)

        results = hybrid_retriever.retrieve(query, k=3)

        print("\n   Retrieved Documents:")
        for i, (doc, score) in enumerate(results, 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {i}. RRF Score: {score:.4f}")
            print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"      Preview: {preview}...")
            print()

        print("\n   Generating answer...")
        answer = generate_answer_with_context(query, results)
        print(f"\n   Answer: {answer}")

    # Compare methods
    print("\n\n📊 COMPARING RETRIEVAL METHODS")
    print("=" * 60)

    compare_retrieval_methods(
        chunks,
        "artificial intelligence applications",
        k=3
    )

    # Demonstrate hybrid advantage
    print("\n\n🎯 HYBRID SEARCH ADVANTAGE")
    print("=" * 60)

    demonstrate_hybrid_advantage()

    # Weight tuning guide
    print("\n\n💡 WEIGHT TUNING GUIDE")
    print("-" * 40)
    print("""
   Adjust weights based on your use case:

   | Use Case                  | BM25 Weight | Vector Weight |
   |---------------------------|-------------|---------------|
   | Technical docs (precise)  | 0.6         | 0.4           |
   | General knowledge         | 0.3         | 0.7           |
   | Code search               | 0.5         | 0.5           |
   | FAQ/Support               | 0.4         | 0.6           |

   Tips:
   - Higher BM25 weight for exact term matching (IDs, codes, names)
   - Higher Vector weight for conceptual/semantic queries
   - Start with 0.4/0.6 and tune based on results
    """)

    print_total_usage(token_tracker, "TOTAL - Hybrid Search")

    print("\nEnd of Hybrid Search demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
