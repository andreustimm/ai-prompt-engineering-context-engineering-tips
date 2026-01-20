"""
Structured Output

Technique for extracting structured data from LLM responses using
Pydantic models, JSON mode, and schema validation.

Features:
- Pydantic model validation
- JSON mode output
- Type-safe extraction
- Schema enforcement

Use cases:
- Data extraction from text
- Form filling automation
- API response generation
- Document parsing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


# Define Pydantic models for structured output

class Person(BaseModel):
    """Model for extracting person information."""
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age in years")
    occupation: Optional[str] = Field(default=None, description="Job or profession")
    location: Optional[str] = Field(default=None, description="City or country")


class Product(BaseModel):
    """Model for product information."""
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")
    features: list[str] = Field(default_factory=list, description="Key features")
    rating: Optional[float] = Field(default=None, ge=0, le=5, description="Rating out of 5")


class ContactInfo(BaseModel):
    """Model for contact information."""
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    website: Optional[str] = Field(default=None, description="Website URL")
    address: Optional[str] = Field(default=None, description="Physical address")


class Company(BaseModel):
    """Model for company information."""
    name: str = Field(description="Company name")
    industry: str = Field(description="Business industry")
    founded: Optional[int] = Field(default=None, description="Year founded")
    employees: Optional[str] = Field(default=None, description="Number of employees range")
    description: Optional[str] = Field(default=None, description="Brief company description")
    contact: Optional[ContactInfo] = Field(default=None, description="Contact information")


class SentimentAnalysis(BaseModel):
    """Model for sentiment analysis results."""
    sentiment: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    key_phrases: list[str] = Field(default_factory=list, description="Key phrases that indicate sentiment")
    summary: str = Field(description="Brief summary of the sentiment analysis")


class MeetingNotes(BaseModel):
    """Model for meeting notes extraction."""
    title: str = Field(description="Meeting title or topic")
    date: Optional[str] = Field(default=None, description="Meeting date")
    attendees: list[str] = Field(default_factory=list, description="List of attendees")
    agenda_items: list[str] = Field(default_factory=list, description="Agenda items discussed")
    action_items: list[str] = Field(default_factory=list, description="Action items and tasks")
    decisions: list[str] = Field(default_factory=list, description="Decisions made")
    next_steps: Optional[str] = Field(default=None, description="Next steps or follow-up")


def extract_with_structured_output(text: str, model: type[BaseModel], instructions: str = "") -> BaseModel:
    """
    Extract structured data from text using Pydantic model.

    Args:
        text: Input text to extract from
        model: Pydantic model class defining the structure
        instructions: Additional extraction instructions

    Returns:
        Populated Pydantic model instance
    """
    llm = get_llm(temperature=0)

    # Use with_structured_output for guaranteed schema compliance
    structured_llm = llm.with_structured_output(model)

    system_prompt = f"""You are an expert at extracting structured information from text.
Extract the requested information and return it in the specified format.
{instructions}
If information is not available, use null/None for optional fields."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{text}")
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"text": text})

    return result


def extract_person(text: str) -> Person:
    """Extract person information from text."""
    return extract_with_structured_output(
        text,
        Person,
        "Focus on extracting name, age, occupation, and location."
    )


def extract_product(text: str) -> Product:
    """Extract product information from text."""
    return extract_with_structured_output(
        text,
        Product,
        "Extract product details including name, price, category, features, and rating."
    )


def extract_company(text: str) -> Company:
    """Extract company information from text."""
    return extract_with_structured_output(
        text,
        Company,
        "Extract company details including name, industry, founding year, employee count, and contact info."
    )


def analyze_sentiment_structured(text: str) -> SentimentAnalysis:
    """Analyze sentiment and return structured result."""
    return extract_with_structured_output(
        text,
        SentimentAnalysis,
        "Analyze the sentiment of the text and provide confidence score and key phrases."
    )


def extract_meeting_notes(text: str) -> MeetingNotes:
    """Extract meeting notes from transcript or summary."""
    return extract_with_structured_output(
        text,
        MeetingNotes,
        "Extract meeting information including attendees, agenda, action items, and decisions."
    )


def main():
    print("=" * 60)
    print("STRUCTURED OUTPUT - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Example 1: Person Extraction
    print("\n👤 PERSON EXTRACTION")
    print("-" * 40)

    person_text = """
    John Smith is a 35-year-old software engineer living in San Francisco.
    He has been working in the tech industry for over 10 years and currently
    leads a team at a major tech company.
    """

    print(f"\nInput text:\n{person_text.strip()}")
    person = extract_person(person_text)

    print(f"\n📋 Extracted Person:")
    print(f"   Name: {person.name}")
    print(f"   Age: {person.age}")
    print(f"   Occupation: {person.occupation}")
    print(f"   Location: {person.location}")

    # Example 2: Product Extraction
    print("\n\n📦 PRODUCT EXTRACTION")
    print("-" * 40)

    product_text = """
    The new iPhone 15 Pro is priced at $999 and falls under the smartphone category.
    It features a titanium design, A17 Pro chip, 48MP camera system, and USB-C port.
    Customers have given it an average rating of 4.7 out of 5 stars.
    """

    print(f"\nInput text:\n{product_text.strip()}")
    product = extract_product(product_text)

    print(f"\n📋 Extracted Product:")
    print(f"   Name: {product.name}")
    print(f"   Price: ${product.price}")
    print(f"   Category: {product.category}")
    print(f"   Features: {', '.join(product.features)}")
    print(f"   Rating: {product.rating}/5")

    # Example 3: Company Extraction
    print("\n\n🏢 COMPANY EXTRACTION")
    print("-" * 40)

    company_text = """
    OpenAI is an artificial intelligence research company founded in 2015.
    Based in San Francisco, the company has around 500-1000 employees and focuses
    on developing safe and beneficial AI. They can be reached at openai.com
    and their email is support@openai.com.
    """

    print(f"\nInput text:\n{company_text.strip()}")
    company = extract_company(company_text)

    print(f"\n📋 Extracted Company:")
    print(f"   Name: {company.name}")
    print(f"   Industry: {company.industry}")
    print(f"   Founded: {company.founded}")
    print(f"   Employees: {company.employees}")
    if company.contact:
        print(f"   Email: {company.contact.email}")
        print(f"   Website: {company.contact.website}")

    # Example 4: Sentiment Analysis
    print("\n\n😊 SENTIMENT ANALYSIS (Structured)")
    print("-" * 40)

    sentiment_text = """
    I absolutely love this new restaurant! The food was incredible, the service
    was outstanding, and the atmosphere was perfect for a date night. The only
    minor issue was the wait time, but it was totally worth it. Highly recommend!
    """

    print(f"\nInput text:\n{sentiment_text.strip()}")
    sentiment = analyze_sentiment_structured(sentiment_text)

    print(f"\n📋 Sentiment Analysis:")
    print(f"   Sentiment: {sentiment.sentiment}")
    print(f"   Confidence: {sentiment.confidence:.1%}")
    print(f"   Key Phrases: {', '.join(sentiment.key_phrases)}")
    print(f"   Summary: {sentiment.summary}")

    # Example 5: Meeting Notes Extraction
    print("\n\n📝 MEETING NOTES EXTRACTION")
    print("-" * 40)

    meeting_text = """
    Q3 Planning Meeting - October 15, 2024

    Attendees: Sarah (PM), John (Dev Lead), Emily (Design), Mike (QA)

    We discussed the roadmap for Q3 and decided to prioritize the mobile app redesign.
    The team agreed to launch the beta by December 1st.

    Action Items:
    - John to create technical spec by Oct 20
    - Emily to finalize mockups by Oct 25
    - Mike to set up testing environment
    - Sarah to coordinate with marketing

    Decision: We will use React Native for the mobile app to share code with web.

    Next meeting scheduled for October 22nd to review progress.
    """

    print(f"\nInput text:\n{meeting_text.strip()}")
    meeting = extract_meeting_notes(meeting_text)

    print(f"\n📋 Extracted Meeting Notes:")
    print(f"   Title: {meeting.title}")
    print(f"   Date: {meeting.date}")
    print(f"   Attendees: {', '.join(meeting.attendees)}")
    print(f"   Action Items:")
    for item in meeting.action_items:
        print(f"      - {item}")
    print(f"   Decisions:")
    for decision in meeting.decisions:
        print(f"      - {decision}")
    print(f"   Next Steps: {meeting.next_steps}")

    print("\n\n" + "=" * 60)
    print("Note: Structured output uses with_structured_output() for")
    print("guaranteed schema compliance with Pydantic models.")
    print("=" * 60)

    print("\nEnd of Structured Output demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
