"""
Parent-Document Retrieval

Uses small chunks for precise retrieval but returns the larger
parent document for context. This combines the benefits of
precise matching with comprehensive context.

Components:
- Child Splitter: Creates small chunks for embedding/search
- Parent Splitter: Creates larger chunks for context
- Document Store: Maps child chunks to parent documents

Use cases:
- When precise matching is important but context is needed
- Long documents where small chunks may lose meaning
- Improving answer quality with fuller context
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
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

    return documents


class InMemoryDocStore:
    """Simple in-memory document store."""

    def __init__(self):
        self.store = {}

    def add(self, doc_id: str, document: Document):
        """Add document to store."""
        self.store[doc_id] = document

    def get(self, doc_id: str) -> Document | None:
        """Get document by ID."""
        return self.store.get(doc_id)

    def mget(self, doc_ids: list[str]) -> list[Document | None]:
        """Get multiple documents by IDs."""
        return [self.store.get(doc_id) for doc_id in doc_ids]


class ParentDocumentRetriever:
    """
    Retriever that uses small chunks for search but returns parent documents.

    The key insight is that small chunks enable precise matching,
    while returning the parent document provides sufficient context
    for the LLM to generate good answers.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 400,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 100,
        collection_name: str = "parent_doc_retriever"
    ):
        """
        Initialize the parent document retriever.

        Args:
            parent_chunk_size: Size of parent chunks (returned for context)
            parent_chunk_overlap: Overlap between parent chunks
            child_chunk_size: Size of child chunks (used for search)
            child_chunk_overlap: Overlap between child chunks
            collection_name: Name for the vector store collection
        """
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.docstore = InMemoryDocStore()
        self.vectorstore = None
        self.embeddings = get_embeddings()
        self.collection_name = collection_name

    def add_documents(self, documents: list[Document]):
        """
        Process and add documents to the retriever.

        1. Split into parent chunks
        2. Split each parent into child chunks
        3. Store parent chunks in docstore
        4. Embed and store child chunks in vectorstore
        """
        all_child_chunks = []

        for doc in documents:
            # Split into parent chunks
            parent_chunks = self.parent_splitter.split_documents([doc])

            for parent in parent_chunks:
                # Generate unique ID for parent
                parent_id = str(uuid.uuid4())

                # Store parent in docstore
                self.docstore.add(parent_id, parent)

                # Split parent into child chunks
                child_chunks = self.child_splitter.split_documents([parent])

                # Add parent_id to each child's metadata
                for child in child_chunks:
                    child.metadata["parent_id"] = parent_id
                    child.metadata["source"] = doc.metadata.get("source", "Unknown")
                    all_child_chunks.append(child)

        # Create vector store with child chunks
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is required")

        self.vectorstore = Chroma.from_documents(
            documents=all_child_chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )

        return len(all_child_chunks)

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """
        Retrieve parent documents based on child chunk matching.

        Args:
            query: Search query
            k: Number of parent documents to return

        Returns:
            List of parent documents
        """
        if not self.vectorstore:
            raise ValueError("No documents added. Call add_documents first.")

        # Retrieve more child chunks than needed (some may share parents)
        child_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k * 3})
        child_chunks = child_retriever.invoke(query)

        # Get unique parent documents
        seen_parents = set()
        parent_docs = []

        for child in child_chunks:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                parent = self.docstore.get(parent_id)
                if parent:
                    seen_parents.add(parent_id)
                    parent_docs.append(parent)

                    if len(parent_docs) >= k:
                        break

        return parent_docs

    def retrieve_with_child_info(self, query: str, k: int = 3) -> list[dict]:
        """
        Retrieve with information about matching child chunks.

        Returns both parent documents and the child chunks that matched.
        """
        if not self.vectorstore:
            raise ValueError("No documents added. Call add_documents first.")

        # Retrieve child chunks
        results = self.vectorstore.similarity_search_with_score(query, k=k * 3)

        # Group by parent
        parent_info = {}

        for child, score in results:
            parent_id = child.metadata.get("parent_id")
            if parent_id:
                if parent_id not in parent_info:
                    parent = self.docstore.get(parent_id)
                    if parent:
                        parent_info[parent_id] = {
                            "parent": parent,
                            "matching_children": [],
                            "best_score": score
                        }

                if parent_id in parent_info:
                    parent_info[parent_id]["matching_children"].append({
                        "content": child.page_content[:100] + "...",
                        "score": score
                    })

        # Sort by best score and return top k
        sorted_results = sorted(
            parent_info.values(),
            key=lambda x: x["best_score"]
        )[:k]

        return sorted_results


def compare_chunk_sizes(documents: list[Document], query: str):
    """Compare retrieval results with different chunk sizes."""

    print(f"\n   Query: '{query}'")
    print("   " + "=" * 50)

    # Small chunks only (no parent retrieval)
    print("\n   📄 SMALL CHUNKS ONLY (400 chars)")
    print("-" * 30)

    small_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    small_chunks = []
    for doc in documents:
        small_chunks.extend(small_splitter.split_documents([doc]))

    small_vectorstore = Chroma.from_documents(
        documents=small_chunks,
        embedding=get_embeddings(),
        collection_name="small_chunks_compare"
    )

    small_results = small_vectorstore.similarity_search(query, k=3)

    total_context = 0
    for i, doc in enumerate(small_results, 1):
        print(f"\n   {i}. Length: {len(doc.page_content)} chars")
        print(f"      Preview: {doc.page_content[:100]}...")
        total_context += len(doc.page_content)

    print(f"\n   Total context: {total_context} chars")

    # Parent document retrieval
    print("\n   📚 PARENT DOCUMENT RETRIEVAL (400 -> 2000 chars)")
    print("-" * 30)

    parent_retriever = ParentDocumentRetriever(
        parent_chunk_size=2000,
        child_chunk_size=400,
        collection_name="parent_compare"
    )

    parent_retriever.add_documents(documents)
    parent_results = parent_retriever.retrieve(query, k=3)

    total_context = 0
    for i, doc in enumerate(parent_results, 1):
        print(f"\n   {i}. Length: {len(doc.page_content)} chars")
        print(f"      Preview: {doc.page_content[:100]}...")
        total_context += len(doc.page_content)

    print(f"\n   Total context: {total_context} chars")
    print("\n   Note: Parent retrieval provides ~5x more context!")


def generate_answer(query: str, documents: list[Document]) -> str:
    """Generate answer using retrieved context."""
    llm = get_llm(temperature=0.3)

    context = "\n\n---\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based on the provided context.
The context contains comprehensive information - use it to provide detailed, accurate answers."""),
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


def main():
    print("=" * 60)
    print("PARENT-DOCUMENT RETRIEVAL")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb is required for this demo.")
        print("Install with: pip install chromadb")
        return

    token_tracker.reset()

    # Load documents
    print("\n📚 LOADING DOCUMENTS")
    print("-" * 40)

    documents = load_sample_documents()
    if not documents:
        print("   No documents found. Using sample documents.")
        documents = [
            Document(
                page_content="Machine learning is a field of AI... " * 50,
                metadata={"source": "sample.txt"}
            )
        ]

    print(f"   Loaded {len(documents)} documents")
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"   Total characters: {total_chars:,}")

    # Create parent document retriever
    print("\n\n🔧 CREATING PARENT DOCUMENT RETRIEVER")
    print("-" * 40)

    retriever = ParentDocumentRetriever(
        parent_chunk_size=2000,
        parent_chunk_overlap=400,
        child_chunk_size=400,
        child_chunk_overlap=100,
        collection_name="parent_doc_demo"
    )

    num_children = retriever.add_documents(documents)
    print(f"   Created {num_children} child chunks")
    print(f"   Parent chunk size: 2000 chars")
    print(f"   Child chunk size: 400 chars")

    # Demo queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the applications of AI?"
    ]

    print("\n\n❓ QUERYING WITH PARENT DOCUMENT RETRIEVAL")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 40)

        # Retrieve with detailed info
        results = retriever.retrieve_with_child_info(query, k=2)

        print(f"\n   Retrieved {len(results)} parent documents:")
        for i, info in enumerate(results, 1):
            parent = info["parent"]
            children = info["matching_children"]

            print(f"\n   Parent {i}:")
            print(f"      Length: {len(parent.page_content)} chars")
            print(f"      Source: {parent.metadata.get('source', 'Unknown')}")
            print(f"      Matching children: {len(children)}")
            print(f"      Best match score: {info['best_score']:.4f}")
            print(f"      Preview: {parent.page_content[:150]}...")

        # Generate answer
        parent_docs = [r["parent"] for r in results]
        print("\n   Generating answer...")
        answer = generate_answer(query, parent_docs)
        print(f"\n   Answer: {answer[:300]}...")

    # Compare with regular chunking
    print("\n\n📊 COMPARISON: PARENT vs SMALL CHUNKS")
    print("=" * 60)

    compare_chunk_sizes(documents, "What are the ethical considerations in AI?")

    # Best practices
    print("\n\n💡 PARENT DOCUMENT BEST PRACTICES")
    print("-" * 40)
    print("""
   | Parameter          | Recommended Range    | Notes                    |
   |--------------------|----------------------|--------------------------|
   | Parent chunk size  | 1500-3000 chars      | Full context for answers |
   | Child chunk size   | 300-500 chars        | Precise matching         |
   | Parent overlap     | 200-400 chars        | Continuity between chunks|
   | Child overlap      | 50-100 chars         | Coverage without waste   |

   When to use Parent Document Retrieval:
   - Long documents with interconnected information
   - When small chunks lose important context
   - When answer quality is more important than token cost
   - Technical or educational content

   Trade-offs:
   + More context leads to better answers
   + Precise matching with small chunks
   - Higher token usage per query
   - More complex implementation
    """)

    print_total_usage(token_tracker, "TOTAL - Parent Document Retrieval")

    print("\nEnd of Parent Document Retrieval demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
