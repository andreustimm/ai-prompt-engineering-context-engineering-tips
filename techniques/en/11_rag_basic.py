"""
RAG (Retrieval-Augmented Generation) - Basic

Technique that enhances LLM responses by retrieving relevant information
from a knowledge base before generating answers.

Components:
- Document Loader: Load documents (PDF, TXT, MD)
- Text Splitter: Chunk documents for embedding
- Embeddings: Convert text to vectors
- Vector Store: ChromaDB for similarity search
- Retriever: Find relevant chunks
- LLM: Generate answers using retrieved context

Use cases:
- Question answering over documents
- Knowledge base chatbots
- Document search and summarization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Conditional imports for RAG components
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

# Global token tracker for this script
token_tracker = TokenUsage()


def load_text_file(file_path: str) -> list[Document]:
    """Load a text file and return as Document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]


def load_pdf_file(file_path: str) -> list[Document]:
    """Load a PDF file and return as Documents (one per page)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        return loader.load()
    except ImportError:
        print("Warning: pypdf not installed. Run: pip install pypdf")
        return []


def load_markdown_file(file_path: str) -> list[Document]:
    """Load a markdown file and return as Document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]


def load_documents(path: str) -> list[Document]:
    """
    Load documents from a file or directory.
    Supports: .txt, .pdf, .md files
    """
    documents = []
    path_obj = Path(path)

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.glob("**/*"))

    for file_path in files:
        ext = file_path.suffix.lower()
        try:
            if ext == '.txt':
                documents.extend(load_text_file(str(file_path)))
            elif ext == '.pdf':
                documents.extend(load_pdf_file(str(file_path)))
            elif ext == '.md':
                documents.extend(load_markdown_file(str(file_path)))
        except Exception as e:
            print(f"   Warning: Could not load {file_path}: {e}")

    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def create_vector_store(chunks: list[Document], collection_name: str = "rag_collection"):
    """
    Create a ChromaDB vector store from document chunks.

    Args:
        chunks: List of document chunks
        collection_name: Name for the ChromaDB collection

    Returns:
        ChromaDB vector store
    """
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb is required. Install with: pip install chromadb")

    embeddings = get_embeddings()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    return vectorstore


def retrieve_relevant_chunks(vectorstore, query: str, k: int = 4) -> list[Document]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        vectorstore: ChromaDB vector store
        query: Search query
        k: Number of chunks to retrieve

    Returns:
        List of relevant document chunks
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def generate_answer(query: str, context_docs: list[Document]) -> str:
    """
    Generate an answer using the LLM with retrieved context.

    Args:
        query: User's question
        context_docs: Retrieved relevant documents

    Returns:
        Generated answer
    """
    llm = get_llm(temperature=0.3)

    # Format context from retrieved documents
    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Chunk: {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
        for doc in context_docs
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the answer is not in the context, say so.
Always cite which source/chunk the information comes from when possible."""),
        ("user", """Context:
{context}

Question: {question}

Answer:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Generation")

    return response.content


class SimpleRAG:
    """
    Simple RAG system that combines document loading, chunking,
    vector storage, retrieval, and generation.
    """

    def __init__(self, collection_name: str = "simple_rag"):
        self.collection_name = collection_name
        self.vectorstore = None
        self.documents = []
        self.chunks = []

    def load_and_index(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Load documents from path and create vector index."""
        print(f"\n   Loading documents from: {path}")
        self.documents = load_documents(path)
        print(f"   Loaded {len(self.documents)} document(s)")

        print(f"\n   Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
        self.chunks = chunk_documents(self.documents, chunk_size, chunk_overlap)
        print(f"   Created {len(self.chunks)} chunks")

        print(f"\n   Creating vector store...")
        self.vectorstore = create_vector_store(self.chunks, self.collection_name)
        print(f"   Vector store ready!")

    def query(self, question: str, k: int = 4) -> dict:
        """
        Query the RAG system.

        Args:
            question: User's question
            k: Number of chunks to retrieve

        Returns:
            Dictionary with answer and retrieved chunks
        """
        if not self.vectorstore:
            raise ValueError("No documents indexed. Call load_and_index first.")

        print(f"\n   Retrieving {k} relevant chunks...")
        relevant_chunks = retrieve_relevant_chunks(self.vectorstore, question, k)

        print(f"   Generating answer...")
        answer = generate_answer(question, relevant_chunks)

        return {
            "question": question,
            "answer": answer,
            "source_chunks": relevant_chunks
        }

    def show_chunks(self, n: int = 5):
        """Display first n chunks for inspection."""
        print(f"\n   First {min(n, len(self.chunks))} chunks:")
        for i, chunk in enumerate(self.chunks[:n]):
            print(f"\n   --- Chunk {i} ---")
            print(f"   Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"   Content preview: {chunk.page_content[:200]}...")


def main():
    print("=" * 60)
    print("RAG (Retrieval-Augmented Generation) - Basic Demo")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb is required for this demo.")
        print("Install with: pip install chromadb")
        return

    # Reset tracker
    token_tracker.reset()

    # Path to sample documents
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not sample_dir.exists():
        print(f"\nError: Sample directory not found at {sample_dir}")
        print("Please ensure sample_data/documents directory exists with sample files.")
        return

    # Initialize RAG system
    print("\n📚 INITIALIZING RAG SYSTEM")
    print("-" * 40)

    rag = SimpleRAG(collection_name="ai_handbook_rag")

    # Load and index documents
    rag.load_and_index(
        str(sample_dir),
        chunk_size=800,
        chunk_overlap=150
    )

    # Show some chunks
    rag.show_chunks(n=3)

    # Example queries
    queries = [
        "What are the different types of machine learning?",
        "Explain what neural networks are and how they work.",
        "What is the history of artificial intelligence?",
        "What are the ethical considerations in AI development?"
    ]

    print("\n\n❓ QUERYING THE RAG SYSTEM")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n📌 Question {i}: {query}")
        print("-" * 40)

        result = rag.query(query, k=3)

        print(f"\n📋 Answer:")
        print(result["answer"])

        print(f"\n📎 Sources used:")
        for chunk in result["source_chunks"]:
            source = Path(chunk.metadata.get('source', 'Unknown')).name
            idx = chunk.metadata.get('chunk_index', '?')
            print(f"   - {source} (chunk {idx})")

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - RAG Basic")

    print("\nEnd of RAG Basic demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
