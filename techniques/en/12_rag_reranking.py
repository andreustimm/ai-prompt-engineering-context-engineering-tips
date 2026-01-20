"""
RAG with Reranking

Enhanced RAG that uses a reranking step to improve retrieval quality.
After initial retrieval, a reranker model scores and reorders documents
by relevance to the query.

Components:
- Initial Retrieval: Get k candidates (more than needed)
- Reranking: Score and reorder by relevance
- Final Selection: Use top n reranked documents

Use cases:
- Improved accuracy for complex queries
- Better handling of semantic similarity
- Production RAG systems requiring high precision
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

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)
import os

# Global token tracker
token_tracker = TokenUsage()


def load_documents(path: str) -> list[Document]:
    """Load documents from path."""
    documents = []
    path_obj = Path(path)

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.glob("**/*"))

    for file_path in files:
        ext = file_path.suffix.lower()
        try:
            if ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
            elif ext == '.pdf':
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                except ImportError:
                    pass
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

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


class LLMReranker:
    """Reranker using LLM to score document relevance."""

    def __init__(self):
        self.llm = get_llm(temperature=0)

    def rerank(self, query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
        """
        Rerank documents using LLM scoring.

        Args:
            query: The search query
            documents: List of candidate documents
            top_n: Number of top documents to return

        Returns:
            Reranked list of top_n documents
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance scoring expert. Given a query and a document,
score the document's relevance to the query on a scale of 0-10.
Return ONLY a single number (the score), nothing else."""),
            ("user", """Query: {query}

Document:
{document}

Relevance score (0-10):""")
        ])

        chain = prompt | self.llm
        scored_docs = []

        for doc in documents:
            response = chain.invoke({
                "query": query,
                "document": doc.page_content[:1000]  # Limit context
            })

            # Extract tokens
            input_tokens, output_tokens = extract_tokens_from_response(response)
            token_tracker.add(input_tokens, output_tokens)

            # Parse score
            try:
                score = float(response.content.strip())
            except ValueError:
                score = 5.0  # Default if parsing fails

            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_n documents
        return [doc for doc, score in scored_docs[:top_n]]


class CohereReranker:
    """Reranker using Cohere's reranking API."""

    def __init__(self, api_key: str = None):
        if not COHERE_AVAILABLE:
            raise ImportError("cohere package required. Install with: pip install cohere")

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found. Set environment variable or pass api_key.")

        self.client = cohere.Client(self.api_key)

    def rerank(self, query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
        """
        Rerank documents using Cohere's reranking model.

        Args:
            query: The search query
            documents: List of candidate documents
            top_n: Number of top documents to return

        Returns:
            Reranked list of top_n documents
        """
        # Prepare documents for Cohere
        doc_texts = [doc.page_content for doc in documents]

        # Call Cohere rerank
        results = self.client.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=doc_texts,
            top_n=top_n
        )

        # Map back to Document objects
        reranked = []
        for result in results.results:
            idx = result.index
            reranked.append(documents[idx])

        return reranked


class CrossEncoderReranker:
    """Reranker using sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

    def rerank(self, query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
        """
        Rerank documents using CrossEncoder model.

        Args:
            query: The search query
            documents: List of candidate documents
            top_n: Number of top documents to return

        Returns:
            Reranked list of top_n documents
        """
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine with documents and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_n]]


class RAGWithReranking:
    """RAG system with reranking capability."""

    def __init__(self, collection_name: str = "rag_rerank", reranker_type: str = "llm"):
        self.collection_name = collection_name
        self.vectorstore = None
        self.chunks = []

        # Initialize reranker
        if reranker_type == "cohere":
            try:
                self.reranker = CohereReranker()
            except (ImportError, ValueError) as e:
                print(f"   Cohere reranker not available: {e}")
                print("   Falling back to LLM reranker")
                self.reranker = LLMReranker()
        elif reranker_type == "cross-encoder":
            try:
                self.reranker = CrossEncoderReranker()
            except ImportError as e:
                print(f"   CrossEncoder not available: {e}")
                print("   Falling back to LLM reranker")
                self.reranker = LLMReranker()
        else:
            self.reranker = LLMReranker()

    def load_and_index(self, path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Load documents and create vector index."""
        print(f"\n   Loading documents from: {path}")
        documents = load_documents(path)
        print(f"   Loaded {len(documents)} document(s)")

        print(f"\n   Chunking documents...")
        self.chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        print(f"   Created {len(self.chunks)} chunks")

        print(f"\n   Creating vector store...")
        embeddings = get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.collection_name
        )
        print(f"   Vector store ready!")

    def query(self, question: str, initial_k: int = 10, final_k: int = 3) -> dict:
        """
        Query with reranking.

        Args:
            question: User's question
            initial_k: Number of initial candidates to retrieve
            final_k: Number of documents after reranking

        Returns:
            Dictionary with answer and metadata
        """
        if not self.vectorstore:
            raise ValueError("No documents indexed.")

        # Step 1: Initial retrieval (get more than we need)
        print(f"\n   Step 1: Retrieving {initial_k} initial candidates...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": initial_k})
        initial_docs = retriever.invoke(question)

        # Step 2: Reranking
        print(f"\n   Step 2: Reranking to top {final_k}...")
        reranked_docs = self.reranker.rerank(question, initial_docs, top_n=final_k)

        # Step 3: Generate answer
        print(f"\n   Step 3: Generating answer...")
        answer = self._generate_answer(question, reranked_docs)

        return {
            "question": question,
            "answer": answer,
            "initial_candidates": len(initial_docs),
            "reranked_docs": reranked_docs
        }

    def _generate_answer(self, query: str, context_docs: list[Document]) -> str:
        """Generate answer using reranked context."""
        llm = get_llm(temperature=0.3)

        context = "\n\n---\n\n".join([
            f"[Chunk {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
            for doc in context_docs
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the context to provide accurate, detailed answers. Cite chunk numbers when possible."""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": query})

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Generation")

        return response.content


def compare_with_without_reranking(rag_system: RAGWithReranking, question: str):
    """Compare RAG results with and without reranking."""

    print("\n" + "=" * 50)
    print("COMPARISON: With vs Without Reranking")
    print("=" * 50)

    # Without reranking (just use top 3 from initial retrieval)
    print("\n📌 WITHOUT RERANKING (top 3 from similarity search):")
    print("-" * 40)

    retriever = rag_system.vectorstore.as_retriever(search_kwargs={"k": 3})
    basic_docs = retriever.invoke(question)

    llm = get_llm(temperature=0.3)
    context = "\n\n".join([doc.page_content[:500] for doc in basic_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context provided."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])

    response = (prompt | llm).invoke({"context": context, "question": question})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)

    print(f"Answer: {response.content}")

    # With reranking
    print("\n📌 WITH RERANKING (top 3 after reranking 10 candidates):")
    print("-" * 40)

    result = rag_system.query(question, initial_k=10, final_k=3)
    print(f"Answer: {result['answer']}")


def main():
    print("=" * 60)
    print("RAG WITH RERANKING - Demo")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb required. Install with: pip install chromadb")
        return

    token_tracker.reset()

    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not sample_dir.exists():
        print(f"\nError: Sample directory not found at {sample_dir}")
        return

    # Initialize RAG with LLM reranking (most available option)
    print("\n📚 INITIALIZING RAG WITH RERANKING")
    print("-" * 40)

    rag = RAGWithReranking(collection_name="rag_rerank_demo", reranker_type="llm")

    rag.load_and_index(
        str(sample_dir),
        chunk_size=500,
        chunk_overlap=100
    )

    # Test queries
    queries = [
        "What are the main components of a neural network?",
        "How does reinforcement learning differ from supervised learning?",
        "What are the ethical considerations in AI development?"
    ]

    print("\n\n❓ QUERYING WITH RERANKING")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n📌 Question {i}: {query}")
        print("-" * 40)

        result = rag.query(query, initial_k=10, final_k=3)

        print(f"\n📋 Answer:")
        print(result["answer"])

        print(f"\n📊 Retrieval Stats:")
        print(f"   Initial candidates: {result['initial_candidates']}")
        print(f"   After reranking: {len(result['reranked_docs'])}")

    # Compare with and without reranking
    print("\n\n🔄 COMPARING WITH AND WITHOUT RERANKING")
    compare_with_without_reranking(rag, "What is the difference between CNN and RNN?")

    print_total_usage(token_tracker, "TOTAL - RAG with Reranking")

    print("\nEnd of RAG with Reranking demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
