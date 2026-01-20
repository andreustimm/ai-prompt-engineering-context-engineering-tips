"""
Self-Query Retrieval

Uses an LLM to automatically generate metadata filters from natural
language queries. This enables structured filtering without requiring
users to specify filters explicitly.

Components:
- Query Parser: LLM parses query into semantic search + filters
- Metadata Schema: Defines filterable fields and their types
- Filtered Retrieval: Applies both semantic search and filters

Use cases:
- Product search with price/category filters
- Document search with date/author filters
- Any search requiring structured filtering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from langchain_core.prompts import ChatPromptTemplate
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


def load_product_catalog() -> list[Document]:
    """Load product catalog with metadata."""
    catalog_path = Path(__file__).parent.parent.parent / "sample_data" / "documents" / "products_catalog.json"

    if catalog_path.exists():
        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for product in data.get("products", []):
            doc = Document(
                page_content=f"{product['name']}: {product['description']}",
                metadata={
                    "id": product["id"],
                    "name": product["name"],
                    "category": product["category"],
                    "subcategory": product["subcategory"],
                    "price": product["price"],
                    "brand": product["brand"],
                    "rating": product["rating"],
                    "in_stock": product["in_stock"],
                    "release_date": product["release_date"],
                    "color": product.get("color", "Unknown")
                }
            )
            documents.append(doc)

        return documents
    else:
        # Fallback sample products
        return [
            Document(
                page_content="ProBook X15 Laptop: High-performance laptop with 15.6 inch 4K display",
                metadata={"name": "ProBook X15", "category": "Electronics", "subcategory": "Computers",
                         "price": 1299.99, "brand": "TechPro", "rating": 4.5, "in_stock": True}
            ),
            Document(
                page_content="BudgetBook 14: Affordable laptop for everyday computing",
                metadata={"name": "BudgetBook 14", "category": "Electronics", "subcategory": "Computers",
                         "price": 449.99, "brand": "ValueTech", "rating": 4.2, "in_stock": True}
            ),
            Document(
                page_content="SmartPhone Pro Max: Flagship smartphone with 108MP camera",
                metadata={"name": "SmartPhone Pro Max", "category": "Electronics", "subcategory": "Smartphones",
                         "price": 999.99, "brand": "TechPro", "rating": 4.7, "in_stock": True}
            ),
        ]


# Define metadata field info for self-query
METADATA_FIELD_INFO = [
    {
        "name": "category",
        "description": "The main category of the product (e.g., 'Electronics')",
        "type": "string"
    },
    {
        "name": "subcategory",
        "description": "The subcategory (e.g., 'Computers', 'Smartphones', 'Audio', 'Tablets', 'Wearables', 'Cameras')",
        "type": "string"
    },
    {
        "name": "price",
        "description": "The price of the product in USD",
        "type": "float"
    },
    {
        "name": "brand",
        "description": "The brand/manufacturer of the product",
        "type": "string"
    },
    {
        "name": "rating",
        "description": "Customer rating from 1.0 to 5.0",
        "type": "float"
    },
    {
        "name": "in_stock",
        "description": "Whether the product is currently in stock",
        "type": "boolean"
    }
]


class SelfQueryRetriever:
    """Retriever that parses queries into semantic search + metadata filters."""

    def __init__(self, vectorstore, metadata_fields: list[dict]):
        """
        Initialize the self-query retriever.

        Args:
            vectorstore: Vector store with documents
            metadata_fields: List of metadata field definitions
        """
        self.vectorstore = vectorstore
        self.metadata_fields = metadata_fields
        self.llm = get_llm(temperature=0)

    def parse_query(self, query: str) -> dict:
        """
        Parse natural language query into search query and filters.

        Args:
            query: Natural language query

        Returns:
            Dictionary with 'search_query' and 'filters'
        """
        fields_description = "\n".join([
            f"- {f['name']} ({f['type']}): {f['description']}"
            for f in self.metadata_fields
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query parser for a product search system.
Given a natural language query, extract:
1. The semantic search query (what the user is looking for)
2. Any metadata filters implied by the query

Available metadata fields:
{fields}

Return a JSON object with this exact structure:
{{
    "search_query": "the semantic search text",
    "filters": {{
        "field_name": {{"operator": "op", "value": value}}
    }}
}}

Supported operators:
- "eq": equals (for strings, numbers, booleans)
- "gt": greater than (for numbers)
- "gte": greater than or equal (for numbers)
- "lt": less than (for numbers)
- "lte": less than or equal (for numbers)
- "contains": contains substring (for strings)

If no filters are implied, return an empty filters object: {{}}
Return ONLY the JSON object, no other text."""),
            ("user", "Query: {query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "fields": fields_description
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Query Parsing")

        # Parse JSON response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
            else:
                result = {"search_query": query, "filters": {}}

        return result

    def apply_filters(self, documents: list[Document], filters: dict) -> list[Document]:
        """
        Apply metadata filters to documents.

        Args:
            documents: Documents to filter
            filters: Filter specifications

        Returns:
            Filtered documents
        """
        if not filters:
            return documents

        filtered = []

        for doc in documents:
            passes = True

            for field, condition in filters.items():
                if field not in doc.metadata:
                    passes = False
                    break

                value = doc.metadata[field]
                op = condition.get("operator", "eq")
                filter_value = condition.get("value")

                if op == "eq":
                    passes = value == filter_value
                elif op == "gt":
                    passes = value > filter_value
                elif op == "gte":
                    passes = value >= filter_value
                elif op == "lt":
                    passes = value < filter_value
                elif op == "lte":
                    passes = value <= filter_value
                elif op == "contains":
                    passes = str(filter_value).lower() in str(value).lower()

                if not passes:
                    break

            if passes:
                filtered.append(doc)

        return filtered

    def retrieve(self, query: str, k: int = 5) -> tuple[list[Document], dict]:
        """
        Retrieve documents using self-query.

        Args:
            query: Natural language query
            k: Number of documents to retrieve

        Returns:
            Tuple of (filtered documents, parsed query info)
        """
        # Parse the query
        parsed = self.parse_query(query)
        search_query = parsed.get("search_query", query)
        filters = parsed.get("filters", {})

        print(f"\n   Parsed Query:")
        print(f"      Search: '{search_query}'")
        print(f"      Filters: {json.dumps(filters, indent=2) if filters else 'None'}")

        # Retrieve more documents than needed (will filter)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k * 3})
        documents = retriever.invoke(search_query)

        # Apply filters
        filtered = self.apply_filters(documents, filters)

        return filtered[:k], parsed


def create_vectorstore(documents: list[Document], collection_name: str = "self_query_demo"):
    """Create vector store from documents."""
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb is required")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )

    return vectorstore


def generate_answer(query: str, documents: list[Document]) -> str:
    """Generate answer using retrieved products."""
    llm = get_llm(temperature=0.3)

    context = "\n\n".join([
        f"Product: {doc.metadata.get('name', 'Unknown')}\n"
        f"Price: ${doc.metadata.get('price', 'N/A')}\n"
        f"Brand: {doc.metadata.get('brand', 'N/A')}\n"
        f"Rating: {doc.metadata.get('rating', 'N/A')}/5\n"
        f"In Stock: {'Yes' if doc.metadata.get('in_stock') else 'No'}\n"
        f"Description: {doc.page_content}"
        for doc in documents
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful shopping assistant. Based on the available products,
help the user find what they're looking for. Mention specific products, prices, and features."""),
        ("user", """Available Products:
{context}

Customer Query: {query}

Response:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer Generation")

    return response.content


def demonstrate_self_query():
    """Demonstrate self-query retrieval with various queries."""

    print("\n   Loading product catalog...")
    documents = load_product_catalog()
    print(f"   Loaded {len(documents)} products")

    print("\n   Creating vector store...")
    vectorstore = create_vectorstore(documents, "self_query_products")
    print("   Vector store ready!")

    # Create self-query retriever
    retriever = SelfQueryRetriever(vectorstore, METADATA_FIELD_INFO)

    # Test queries
    queries = [
        "cheap laptops under $500",
        "best rated smartphones",
        "TechPro products in stock",
        "wireless headphones with good ratings",
        "cameras for professional photography"
    ]

    print("\n" + "=" * 60)
    print("SELF-QUERY RETRIEVAL DEMONSTRATIONS")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 40)

        documents, parsed = retriever.retrieve(query, k=3)

        print(f"\n   Retrieved {len(documents)} products:")
        for i, doc in enumerate(documents, 1):
            print(f"\n   {i}. {doc.metadata.get('name', 'Unknown')}")
            print(f"      Price: ${doc.metadata.get('price', 'N/A')}")
            print(f"      Rating: {doc.metadata.get('rating', 'N/A')}/5")
            print(f"      Brand: {doc.metadata.get('brand', 'N/A')}")
            print(f"      In Stock: {'Yes' if doc.metadata.get('in_stock') else 'No'}")

        print("\n   Generating recommendation...")
        answer = generate_answer(query, documents)
        print(f"\n   Response: {answer[:300]}...")


def compare_with_without_filters():
    """Compare results with and without automatic filtering."""

    print("\n   Loading product catalog...")
    documents = load_product_catalog()

    print("\n   Creating vector store...")
    vectorstore = create_vectorstore(documents, "filter_comparison")

    query = "affordable laptops under $600 with good reviews"

    print(f"\n   Query: '{query}'")
    print("=" * 50)

    # Without self-query (plain semantic search)
    print("\n   📚 WITHOUT SELF-QUERY (Semantic Only):")
    print("-" * 30)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    plain_results = retriever.invoke(query)

    for i, doc in enumerate(plain_results[:3], 1):
        print(f"\n   {i}. {doc.metadata.get('name', 'Unknown')}")
        print(f"      Price: ${doc.metadata.get('price', 'N/A')}")
        print(f"      Rating: {doc.metadata.get('rating', 'N/A')}/5")

    # With self-query
    print("\n   🔍 WITH SELF-QUERY (Semantic + Filters):")
    print("-" * 30)

    self_query_retriever = SelfQueryRetriever(vectorstore, METADATA_FIELD_INFO)
    filtered_results, parsed = self_query_retriever.retrieve(query, k=5)

    for i, doc in enumerate(filtered_results[:3], 1):
        print(f"\n   {i}. {doc.metadata.get('name', 'Unknown')}")
        print(f"      Price: ${doc.metadata.get('price', 'N/A')}")
        print(f"      Rating: {doc.metadata.get('rating', 'N/A')}/5")

    print("\n   Note: Self-query correctly filters by price < $600")


def main():
    print("=" * 60)
    print("SELF-QUERY RETRIEVAL")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb is required for this demo.")
        print("Install with: pip install chromadb")
        return

    token_tracker.reset()

    # Demo 1: Basic self-query demonstration
    print("\n\n🛒 SELF-QUERY PRODUCT SEARCH")
    print("=" * 60)

    demonstrate_self_query()

    # Demo 2: Comparison
    print("\n\n📊 COMPARISON: WITH vs WITHOUT FILTERS")
    print("=" * 60)

    compare_with_without_filters()

    # Best practices
    print("\n\n💡 SELF-QUERY BEST PRACTICES")
    print("-" * 40)
    print("""
   | Consideration         | Recommendation                          |
   |-----------------------|-----------------------------------------|
   | Metadata Schema       | Define clear, well-documented fields    |
   | Field Types           | Use appropriate types (string, float)   |
   | Value Ranges          | Document valid values for each field    |
   | Fallback              | Handle cases where parsing fails        |

   Tips:
   - Keep metadata fields simple and unambiguous
   - Provide good field descriptions for the LLM
   - Test with various query phrasings
   - Consider combining with other retrieval methods
   - Cache parsed queries for repeated searches

   Common Use Cases:
   - E-commerce product search
   - Document management with metadata
   - Job/candidate search systems
   - Real estate property search
    """)

    print_total_usage(token_tracker, "TOTAL - Self-Query Retrieval")

    print("\nEnd of Self-Query Retrieval demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
