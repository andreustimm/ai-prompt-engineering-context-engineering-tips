"""
Memory/Conversation

Techniques for maintaining conversation context and memory across
multiple interactions with an LLM.

Memory Types:
- Buffer Memory: Stores complete conversation history
- Window Memory: Stores last N exchanges
- Summary Memory: Stores summarized history
- Entity Memory: Tracks entities mentioned

Use cases:
- Chatbots and virtual assistants
- Multi-turn dialogues
- Personalized interactions
- Context-aware responses
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


class BufferMemory:
    """
    Simple buffer memory that stores complete conversation history.
    Good for short conversations where full context is important.
    """

    def __init__(self, max_messages: int = 100):
        self.messages: list = []
        self.max_messages = max_messages

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(HumanMessage(content=content))
        self._trim_if_needed()

    def add_ai_message(self, content: str):
        """Add an AI message to history."""
        self.messages.append(AIMessage(content=content))
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Keep only the most recent messages if limit exceeded."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> list:
        """Get all messages."""
        return self.messages.copy()

    def clear(self):
        """Clear all messages."""
        self.messages = []


class WindowMemory:
    """
    Window memory that stores only the last K exchanges.
    Good for long conversations where recent context matters most.
    """

    def __init__(self, window_size: int = 5):
        self.messages: list = []
        self.window_size = window_size  # Number of exchanges (user + AI pairs)

    def add_exchange(self, user_message: str, ai_message: str):
        """Add a complete exchange."""
        self.messages.append(HumanMessage(content=user_message))
        self.messages.append(AIMessage(content=ai_message))
        self._trim_window()

    def _trim_window(self):
        """Keep only messages within the window."""
        max_messages = self.window_size * 2  # 2 messages per exchange
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> list:
        return self.messages.copy()

    def clear(self):
        self.messages = []


class SummaryMemory:
    """
    Memory that maintains a running summary of the conversation.
    Good for very long conversations to prevent token overflow.
    """

    def __init__(self):
        self.summary: str = ""
        self.recent_messages: list = []
        self.max_recent = 4  # Keep last 4 messages for context

    def add_exchange(self, user_message: str, ai_message: str):
        """Add an exchange and potentially update summary."""
        self.recent_messages.append(HumanMessage(content=user_message))
        self.recent_messages.append(AIMessage(content=ai_message))

        # If too many recent messages, incorporate into summary
        if len(self.recent_messages) > self.max_recent:
            self._update_summary()

    def _update_summary(self):
        """Update the summary with older messages."""
        llm = get_llm(temperature=0)

        # Take oldest messages to summarize
        to_summarize = self.recent_messages[:-self.max_recent]

        if not to_summarize:
            return

        context = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in to_summarize
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize the following conversation, keeping key information
and any important details or decisions. Be concise but comprehensive."""),
            ("user", """Previous summary (if any):
{previous_summary}

New conversation to incorporate:
{conversation}

Updated summary:""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "previous_summary": self.summary or "None",
            "conversation": context
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        self.summary = response.content
        self.recent_messages = self.recent_messages[-self.max_recent:]

    def get_context(self) -> dict:
        """Get summary and recent messages."""
        return {
            "summary": self.summary,
            "recent_messages": self.recent_messages.copy()
        }

    def clear(self):
        self.summary = ""
        self.recent_messages = []


class EntityMemory:
    """
    Memory that tracks entities mentioned in conversation.
    Good for maintaining knowledge about people, places, things discussed.
    """

    def __init__(self):
        self.entities: dict = {}  # entity_name -> description/info

    def extract_and_update_entities(self, text: str):
        """Extract entities from text and update memory."""
        llm = get_llm(temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract named entities from the text. For each entity,
provide a brief description based on what was mentioned.
Return as JSON: {"entity_name": "description", ...}
Only include clearly named entities (people, places, organizations, products)."""),
            ("user", "{text}")
        ])

        chain = prompt | llm
        response = chain.invoke({"text": text})

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        # Parse response
        try:
            import json
            # Try to extract JSON from response
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                entities = json.loads(content[start:end])
                self.entities.update(entities)
        except Exception:
            pass  # If parsing fails, skip

    def get_entity_info(self, entity_name: str) -> Optional[str]:
        """Get information about a specific entity."""
        return self.entities.get(entity_name.lower())

    def get_all_entities(self) -> dict:
        """Get all tracked entities."""
        return self.entities.copy()

    def clear(self):
        self.entities = {}


class ConversationChain:
    """
    Conversation chain with configurable memory.
    """

    def __init__(self, memory_type: str = "buffer", system_prompt: str = None):
        self.memory_type = memory_type
        self.system_prompt = system_prompt or "You are a helpful assistant."

        # Initialize appropriate memory
        if memory_type == "buffer":
            self.memory = BufferMemory()
        elif memory_type == "window":
            self.memory = WindowMemory(window_size=5)
        elif memory_type == "summary":
            self.memory = SummaryMemory()
        else:
            self.memory = BufferMemory()

    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        llm = get_llm(temperature=0.7)

        # Build messages list
        messages = [SystemMessage(content=self.system_prompt)]

        if self.memory_type == "summary":
            context = self.memory.get_context()
            if context["summary"]:
                messages.append(SystemMessage(content=f"Conversation summary: {context['summary']}"))
            messages.extend(context["recent_messages"])
        else:
            messages.extend(self.memory.get_messages())

        messages.append(HumanMessage(content=user_input))

        # Get response
        response = llm.invoke(messages)

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens)

        ai_response = response.content

        # Update memory
        if self.memory_type == "buffer":
            self.memory.add_user_message(user_input)
            self.memory.add_ai_message(ai_response)
        elif self.memory_type == "window":
            self.memory.add_exchange(user_input, ai_response)
        elif self.memory_type == "summary":
            self.memory.add_exchange(user_input, ai_response)

        return ai_response

    def reset(self):
        """Reset conversation memory."""
        self.memory.clear()


def demo_buffer_memory():
    """Demonstrate buffer memory."""
    print("\n📚 BUFFER MEMORY DEMO")
    print("-" * 40)

    chain = ConversationChain(
        memory_type="buffer",
        system_prompt="You are a helpful travel advisor."
    )

    exchanges = [
        "Hi! I'm planning a trip to Japan.",
        "What's the best time to visit?",
        "What are the must-see places in Tokyo?",
        "What was the first thing I mentioned?"
    ]

    for user_msg in exchanges:
        print(f"\n👤 User: {user_msg}")
        response = chain.chat(user_msg)
        print(f"🤖 Assistant: {response[:300]}...")


def demo_window_memory():
    """Demonstrate window memory."""
    print("\n📚 WINDOW MEMORY DEMO (window=3)")
    print("-" * 40)

    chain = ConversationChain(
        memory_type="window",
        system_prompt="You are a helpful cooking assistant."
    )

    exchanges = [
        "I want to make pasta.",
        "What ingredients do I need for carbonara?",
        "How do I cook the pasta perfectly?",
        "What about the sauce?",
        "What did I originally want to make?"  # Tests if early context is lost
    ]

    for user_msg in exchanges:
        print(f"\n👤 User: {user_msg}")
        response = chain.chat(user_msg)
        print(f"🤖 Assistant: {response[:300]}...")


def demo_entity_memory():
    """Demonstrate entity memory."""
    print("\n📚 ENTITY MEMORY DEMO")
    print("-" * 40)

    entity_memory = EntityMemory()

    texts = [
        "I just started working at Google in San Francisco. My manager is Sarah Chen.",
        "Sarah mentioned that our team is working on a new AI project called Aurora.",
        "We're partnering with Stanford University for the research."
    ]

    for text in texts:
        print(f"\n📝 Processing: {text[:50]}...")
        entity_memory.extract_and_update_entities(text)

    print("\n📋 Tracked Entities:")
    for entity, info in entity_memory.get_all_entities().items():
        print(f"   - {entity}: {info}")


def main():
    print("=" * 60)
    print("MEMORY/CONVERSATION - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Demo 1: Buffer Memory
    demo_buffer_memory()

    # Demo 2: Window Memory
    demo_window_memory()

    # Demo 3: Entity Memory
    demo_entity_memory()

    print_total_usage(token_tracker, "TOTAL - Memory/Conversation")

    print("\n\n" + "=" * 60)
    print("Memory Types Summary:")
    print("  - Buffer: Full history, good for short conversations")
    print("  - Window: Last N exchanges, good for long conversations")
    print("  - Summary: Compressed history, prevents token overflow")
    print("  - Entity: Tracks named entities for knowledge persistence")
    print("=" * 60)

    print("\nEnd of Memory/Conversation demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
