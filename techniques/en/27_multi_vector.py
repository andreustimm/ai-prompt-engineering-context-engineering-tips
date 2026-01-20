"""
Multi-Vector Retrieval

Stores multiple vector representations for each document:
- Summaries: For high-level matching
- Hypothetical questions: For question-matching
- Original content: For full context

This enables matching based on different representations
while returning the original document for context.

Use cases:
- Matching user questions to document content
- Improving retrieval for diverse query types
- Combining different semantic representations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def load_sample_documents() -> list[Document]:
    """Load sample documents."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    documents = []
    for file_path in sample_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(Document(page_content=f.read(), metadata={"source": file_path.name}))
        except Exception:
            pass
    return documents


class InMemoryStore:
    """Simple in-memory document store."""
    def __init__(self):
        self.store = {}

    def add(self, doc_id: str, doc: Document):
        self.store[doc_id] = doc

    def get(self, doc_id: str) -> Document | None:
        return self.store.get(doc_id)


def generate_summary(text: str) -> str:
    """Generate a summary of the text."""
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text in 2-3 sentences. Focus on key concepts."),
        ("user", "{text}")
    ])
    response = (prompt | llm).invoke({"text": text[:3000]})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    return response.content


def generate_questions(text: str, num_questions: int = 3) -> list[str]:
    """Generate hypothetical questions that this text could answer."""
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate {num} questions that the following text could answer.
Return ONLY the questions, one per line, numbered."""),
        ("user", "{text}")
    ])
    response = (prompt | llm).invoke({"text": text[:3000], "num": num_questions})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)

    lines = response.content.strip().split('\n')
    questions = []
    for line in lines:
        cleaned = line.strip()
        if cleaned and cleaned[0].isdigit():
            cleaned = cleaned.split('.', 1)[-1].strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:num_questions]


class MultiVectorRetriever:
    """Retriever with multiple vector representations per document."""

    def __init__(self, collection_name: str = "multi_vector"):
        self.embeddings = get_embeddings()
        self.docstore = InMemoryStore()
        self.vectorstore = None
        self.collection_name = collection_name
        self.all_vectors = []

    def add_documents(self, documents: list[Document], generate_summaries: bool = True, generate_hypothetical: bool = True):
        """Add documents with multiple representations."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

        for doc in documents:
            chunks = text_splitter.split_documents([doc])

            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                self.docstore.add(doc_id, chunk)

                # Vector 1: Original content (always)
                self.all_vectors.append(Document(
                    page_content=chunk.page_content,
                    metadata={"doc_id": doc_id, "type": "original", "source": doc.metadata.get("source")}
                ))

                # Vector 2: Summary
                if generate_summaries:
                    print(f"   Generating summary for chunk...")
                    summary = generate_summary(chunk.page_content)
                    self.all_vectors.append(Document(
                        page_content=summary,
                        metadata={"doc_id": doc_id, "type": "summary", "source": doc.metadata.get("source")}
                    ))

                # Vector 3: Hypothetical questions
                if generate_hypothetical:
                    print(f"   Generating questions for chunk...")
                    questions = generate_questions(chunk.page_content, num_questions=2)
                    for q in questions:
                        self.all_vectors.append(Document(
                            page_content=q,
                            metadata={"doc_id": doc_id, "type": "question", "source": doc.metadata.get("source")}
                        ))

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.all_vectors,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )
        return len(self.all_vectors)

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Retrieve original documents via multi-vector matching."""
        if not self.vectorstore:
            raise ValueError("No documents added.")

        # Search across all vector types
        results = self.vectorstore.similarity_search(query, k=k * 3)

        # Get unique original documents
        seen_ids = set()
        original_docs = []

        for result in results:
            doc_id = result.metadata.get("doc_id")
            if doc_id and doc_id not in seen_ids:
                original = self.docstore.get(doc_id)
                if original:
                    seen_ids.add(doc_id)
                    original_docs.append(original)
                    if len(original_docs) >= k:
                        break

        return original_docs

    def retrieve_with_match_info(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve with information about what matched."""
        results = self.vectorstore.similarity_search_with_score(query, k=k * 5)

        doc_info = {}
        for result, score in results:
            doc_id = result.metadata.get("doc_id")
            match_type = result.metadata.get("type")

            if doc_id not in doc_info:
                original = self.docstore.get(doc_id)
                if original:
                    doc_info[doc_id] = {
                        "original": original,
                        "matches": [],
                        "best_score": score
                    }

            if doc_id in doc_info:
                doc_info[doc_id]["matches"].append({
                    "type": match_type,
                    "content": result.page_content[:100],
                    "score": score
                })

        sorted_results = sorted(doc_info.values(), key=lambda x: x["best_score"])[:k]
        return sorted_results


def generate_answer(query: str, documents: list[Document]) -> str:
    """Generate answer from retrieved documents."""
    llm = get_llm(temperature=0.3)
    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context. Be thorough but concise."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    response = (prompt | llm).invoke({"context": context, "question": query})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer Generation")
    return response.content


def main():
    print("=" * 60)
    print("MULTI-VECTOR RETRIEVAL")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb required. Install: pip install chromadb")
        return

    token_tracker.reset()

    print("\n📚 LOADING DOCUMENTS")
    print("-" * 40)
    documents = load_sample_documents()[:2]  # Limit for demo
    if not documents:
        documents = [Document(page_content="Machine learning enables systems to learn from data." * 20)]
    print(f"   Loaded {len(documents)} documents")

    print("\n🔧 CREATING MULTI-VECTOR INDEX")
    print("-" * 40)
    retriever = MultiVectorRetriever(collection_name="multi_vector_demo")
    num_vectors = retriever.add_documents(documents, generate_summaries=True, generate_hypothetical=True)
    print(f"   Created {num_vectors} vector representations")

    queries = ["What is machine learning?", "How do AI systems learn?"]

    print("\n\n❓ MULTI-VECTOR QUERIES")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 40)

        results = retriever.retrieve_with_match_info(query, k=2)

        for i, info in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"      Matched via: {[m['type'] for m in info['matches'][:3]]}")
            print(f"      Original preview: {info['original'].page_content[:150]}...")

        docs = [r["original"] for r in results]
        answer = generate_answer(query, docs)
        print(f"\n   Answer: {answer[:250]}...")

    print("\n\n💡 MULTI-VECTOR BENEFITS")
    print("-" * 40)
    print("""
   - Summaries: Match high-level concepts
   - Questions: Match user query phrasing
   - Original: Provide full context for answers

   Best for: FAQ systems, documentation search, knowledge bases
    """)

    print_total_usage(token_tracker, "TOTAL - Multi-Vector Retrieval")
    print("\nEnd of Multi-Vector Retrieval demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
