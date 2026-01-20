"""
Ensemble Retrieval

Combines multiple retrievers using Reciprocal Rank Fusion (RRF)
to leverage the strengths of different retrieval strategies.

Components:
- Multiple Retrievers: Different search strategies
- RRF Algorithm: Combines ranked results
- Weighted Fusion: Adjustable retriever weights

Use cases:
- Combining semantic and keyword search
- Using multiple embedding models
- Ensemble of specialized retrievers
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

token_tracker = TokenUsage()


def load_documents() -> list[Document]:
    """Load sample documents."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    docs = []
    for fp in sample_dir.glob("*.txt"):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                docs.append(Document(page_content=f.read(), metadata={"source": fp.name}))
        except Exception:
            pass
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents for retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_documents([doc]))
    return chunks


class BM25Retriever:
    """BM25 keyword retriever."""
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.tokenized = [d.page_content.lower().split() for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], scores[i]) for i in top_idx]


class VectorRetriever:
    """Vector similarity retriever."""
    def __init__(self, docs: list[Document], collection: str):
        self.vectorstore = Chroma.from_documents(docs, get_embeddings(), collection_name=collection)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc, 1/(1+score)) for doc, score in results]


def reciprocal_rank_fusion(results_list: list[list[tuple[Document, float]]], weights: list[float] = None, k: int = 60) -> list[tuple[Document, float]]:
    """Combine results using RRF."""
    if weights is None:
        weights = [1.0] * len(results_list)
    weights = [w/sum(weights) for w in weights]

    doc_scores = {}
    for idx, results in enumerate(results_list):
        for rank, (doc, _) in enumerate(results, 1):
            key = doc.page_content[:100]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0}
            doc_scores[key]["score"] += weights[idx] / (k + rank)

    sorted_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [(r["doc"], r["score"]) for r in sorted_results]


class EnsembleRetriever:
    """Combines multiple retrievers with RRF."""

    def __init__(self, retrievers: list, weights: list[float] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        all_results = []
        for retriever in self.retrievers:
            results = retriever.retrieve(query, k=k*2)
            all_results.append(results)
        return reciprocal_rank_fusion(all_results, self.weights)[:k]


def generate_answer(query: str, docs: list[Document]) -> str:
    """Generate answer from documents."""
    llm = get_llm(temperature=0.3)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context. Be concise."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    response = (prompt | llm).invoke({"context": context, "question": query})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer")
    return response.content


def main():
    print("=" * 60)
    print("ENSEMBLE RETRIEVAL")
    print("=" * 60)

    if not CHROMA_AVAILABLE or not BM25_AVAILABLE:
        print("\nError: chromadb and rank-bm25 required")
        return

    token_tracker.reset()

    print("\n📚 LOADING DOCUMENTS")
    print("-" * 40)
    docs = load_documents()
    if not docs:
        docs = [Document(page_content="Machine learning is AI that learns from data.")]
    chunks = chunk_documents(docs)
    print(f"   Created {len(chunks)} chunks")

    print("\n🔧 CREATING ENSEMBLE RETRIEVER")
    print("-" * 40)
    bm25_retriever = BM25Retriever(chunks)
    vector_retriever = VectorRetriever(chunks, "ensemble_demo")

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    print("   BM25 weight: 0.4, Vector weight: 0.6")

    queries = ["What is machine learning?", "neural network architectures"]

    print("\n\n❓ ENSEMBLE QUERIES")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 40)

        # Compare individual vs ensemble
        bm25_results = bm25_retriever.retrieve(query, k=3)
        vector_results = vector_retriever.retrieve(query, k=3)
        ensemble_results = ensemble.retrieve(query, k=3)

        print("\n   BM25 Top Result:")
        print(f"      {bm25_results[0][0].page_content[:80]}...")

        print("\n   Vector Top Result:")
        print(f"      {vector_results[0][0].page_content[:80]}...")

        print("\n   Ensemble Top Result:")
        print(f"      {ensemble_results[0][0].page_content[:80]}...")

        docs_for_answer = [r[0] for r in ensemble_results]
        answer = generate_answer(query, docs_for_answer)
        print(f"\n   Answer: {answer[:200]}...")

    print("\n\n💡 ENSEMBLE TIPS")
    print("-" * 40)
    print("""
   Weight Tuning:
   - More BM25 for exact term matching
   - More Vector for semantic understanding
   - Start with 0.4/0.6 and adjust based on results

   Combine different strategies for best results!
    """)

    print_total_usage(token_tracker, "TOTAL - Ensemble Retrieval")
    print("\nEnd of Ensemble Retrieval demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
