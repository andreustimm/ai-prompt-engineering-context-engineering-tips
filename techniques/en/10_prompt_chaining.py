"""
Prompt Chaining

Technique that connects multiple prompts in a pipeline, where the output
of one prompt becomes the input for the next, creating a workflow.

Use cases:
- Content creation pipelines
- Research and analysis workflows
- Data processing chains
- Multi-step document generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker for this script
token_tracker = TokenUsage()


def execute_chain_step(prompt_template: ChatPromptTemplate, inputs: dict, step_name: str, temperature: float = 0.7) -> str:
    """Execute a single step in the prompt chain."""
    llm = get_llm(temperature=temperature)
    chain = prompt_template | llm
    response = chain.invoke(inputs)

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, step_name)

    return response.content


def research_analyze_summarize(topic: str) -> dict:
    """
    A three-step chain: Research -> Analyze -> Summarize

    Step 1: Gather key facts about the topic
    Step 2: Analyze implications and connections
    Step 3: Create an executive summary
    """
    results = {}

    # Step 1: Research
    print("\n   Step 1: Researching topic...")
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research expert. Gather and list key facts about the given topic.
Include: definitions, key statistics, main players/components, historical context, and current state.
Format as organized bullet points."""),
        ("user", "Research topic: {topic}")
    ])

    results["research"] = execute_chain_step(
        research_prompt, {"topic": topic}, "Research", temperature=0.3
    )

    # Step 2: Analyze
    print("\n   Step 2: Analyzing findings...")
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an analytical expert. Based on the research findings provided,
identify patterns, implications, opportunities, and potential challenges.
Provide deep insights that go beyond the surface-level facts."""),
        ("user", """Research findings:
{research}

Provide your analysis:""")
    ])

    results["analysis"] = execute_chain_step(
        analysis_prompt, {"research": results["research"]}, "Analysis", temperature=0.5
    )

    # Step 3: Summarize
    print("\n   Step 3: Creating executive summary...")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a business communication expert. Create a concise executive summary
that combines the research and analysis into actionable insights.
Keep it under 200 words while covering the most important points."""),
        ("user", """Research:
{research}

Analysis:
{analysis}

Create the executive summary:""")
    ])

    results["summary"] = execute_chain_step(
        summary_prompt,
        {"research": results["research"], "analysis": results["analysis"]},
        "Summary",
        temperature=0.4
    )

    return results


def content_creation_pipeline(topic: str, target_audience: str, content_type: str = "blog post") -> dict:
    """
    Content creation pipeline: Outline -> Draft -> Edit -> Polish

    Step 1: Create detailed outline
    Step 2: Write first draft
    Step 3: Edit for clarity and flow
    Step 4: Polish and finalize
    """
    results = {}

    # Step 1: Outline
    print("\n   Step 1: Creating outline...")
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a content strategist. Create a detailed outline for a {content_type}
about the given topic, optimized for the target audience.
Include: hook/intro, main sections, key points for each section, and conclusion."""),
        ("user", """Topic: {topic}
Target audience: {audience}

Create the outline:""")
    ])

    results["outline"] = execute_chain_step(
        outline_prompt,
        {"topic": topic, "audience": target_audience, "content_type": content_type},
        "Outline",
        temperature=0.5
    )

    # Step 2: Draft
    print("\n   Step 2: Writing first draft...")
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a skilled content writer. Write the first draft based on the outline provided.
Focus on getting all the ideas down with good flow, but don't worry about perfection yet.
Write in a style appropriate for a {content_type}."""),
        ("user", """Outline:
{outline}

Target audience: {audience}

Write the first draft:""")
    ])

    results["draft"] = execute_chain_step(
        draft_prompt,
        {"outline": results["outline"], "audience": target_audience, "content_type": content_type},
        "Draft",
        temperature=0.7
    )

    # Step 3: Edit
    print("\n   Step 3: Editing for clarity...")
    edit_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional editor. Review and edit the draft for:
- Clarity and readability
- Logical flow between paragraphs
- Removal of redundancies
- Stronger word choices
- Consistent tone

Provide the edited version."""),
        ("user", """Draft to edit:
{draft}

Target audience: {audience}

Edited version:""")
    ])

    results["edited"] = execute_chain_step(
        edit_prompt,
        {"draft": results["draft"], "audience": target_audience},
        "Edit",
        temperature=0.3
    )

    # Step 4: Polish
    print("\n   Step 4: Final polish...")
    polish_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a publishing expert. Apply final polish to the content:
- Ensure perfect grammar and punctuation
- Add compelling opening and closing
- Optimize for engagement
- Add any missing transitions
- Ensure it's ready for publication

Provide the final polished version."""),
        ("user", """Content to polish:
{edited}

Final polished version:""")
    ])

    results["final"] = execute_chain_step(
        polish_prompt,
        {"edited": results["edited"]},
        "Polish",
        temperature=0.3
    )

    return results


def data_insight_chain(data_description: str, business_context: str) -> dict:
    """
    Data analysis chain: Extract -> Interpret -> Recommend

    Step 1: Extract key data points
    Step 2: Interpret patterns and trends
    Step 3: Generate business recommendations
    """
    results = {}

    # Step 1: Extract
    print("\n   Step 1: Extracting key data points...")
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analyst. From the data description provided,
identify and list the key metrics, trends, and notable data points.
Be specific with numbers and percentages where available."""),
        ("user", """Data description:
{data}

Business context: {context}

Extract key data points:""")
    ])

    results["extraction"] = execute_chain_step(
        extract_prompt,
        {"data": data_description, "context": business_context},
        "Extract",
        temperature=0.2
    )

    # Step 2: Interpret
    print("\n   Step 2: Interpreting patterns...")
    interpret_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a business intelligence expert. Based on the extracted data points,
identify patterns, correlations, and what the data is telling us.
Consider the business context in your interpretation."""),
        ("user", """Extracted data points:
{extraction}

Business context: {context}

Interpretation:""")
    ])

    results["interpretation"] = execute_chain_step(
        interpret_prompt,
        {"extraction": results["extraction"], "context": business_context},
        "Interpret",
        temperature=0.4
    )

    # Step 3: Recommend
    print("\n   Step 3: Generating recommendations...")
    recommend_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strategic business advisor. Based on the data analysis,
provide specific, actionable recommendations.
Prioritize by impact and feasibility. Include both quick wins and long-term strategies."""),
        ("user", """Data extraction:
{extraction}

Interpretation:
{interpretation}

Business context: {context}

Recommendations:""")
    ])

    results["recommendations"] = execute_chain_step(
        recommend_prompt,
        {
            "extraction": results["extraction"],
            "interpretation": results["interpretation"],
            "context": business_context
        },
        "Recommend",
        temperature=0.5
    )

    return results


def translation_localization_chain(text: str, source_lang: str, target_lang: str, target_culture: str) -> dict:
    """
    Translation and localization chain: Translate -> Adapt -> Verify

    Step 1: Direct translation
    Step 2: Cultural adaptation
    Step 3: Quality verification
    """
    results = {}

    # Step 1: Translate
    print("\n   Step 1: Translating...")
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional translator. Translate the text accurately
from {source_lang} to {target_lang}. Maintain the original meaning and tone."""),
        ("user", """Text to translate:
{text}

Translation:""")
    ])

    results["translation"] = execute_chain_step(
        translate_prompt,
        {"text": text, "source_lang": source_lang, "target_lang": target_lang},
        "Translate",
        temperature=0.3
    )

    # Step 2: Adapt
    print("\n   Step 2: Cultural adaptation...")
    adapt_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a localization expert for {culture}.
Adapt the translation to be culturally appropriate:
- Adjust idioms and expressions
- Consider local customs and sensitivities
- Adapt references that may not translate well
- Maintain natural language flow for native speakers"""),
        ("user", """Translation to adapt:
{translation}

Culturally adapted version:""")
    ])

    results["adaptation"] = execute_chain_step(
        adapt_prompt,
        {"translation": results["translation"], "culture": target_culture},
        "Adapt",
        temperature=0.4
    )

    # Step 3: Verify
    print("\n   Step 3: Quality verification...")
    verify_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality assurance specialist for {target_lang} content.
Review the localized text and:
1. Check for any remaining translation issues
2. Verify cultural appropriateness
3. Ensure natural language flow
4. Make any final corrections

Provide the final verified version and a brief quality report."""),
        ("user", """Original ({source_lang}):
{original}

Localized version:
{adaptation}

Final verified version and quality notes:""")
    ])

    results["verified"] = execute_chain_step(
        verify_prompt,
        {
            "original": text,
            "adaptation": results["adaptation"],
            "source_lang": source_lang,
            "target_lang": target_lang
        },
        "Verify",
        temperature=0.2
    )

    return results


def main():
    print("=" * 60)
    print("PROMPT CHAINING - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Research-Analyze-Summarize
    print("\n🔍 RESEARCH -> ANALYZE -> SUMMARIZE")
    print("-" * 40)

    topic = "The impact of artificial intelligence on healthcare diagnostics"
    print(f"\nTopic: {topic}")

    results = research_analyze_summarize(topic)

    print(f"\n📋 EXECUTIVE SUMMARY:")
    print("-" * 40)
    print(results["summary"])

    # Example 2: Content Creation Pipeline
    print("\n\n✍️ CONTENT CREATION PIPELINE")
    print("-" * 40)

    content_topic = "5 Productivity Tips for Remote Workers"
    audience = "professional knowledge workers aged 25-45"

    print(f"\nTopic: {content_topic}")
    print(f"Audience: {audience}")

    results = content_creation_pipeline(content_topic, audience, "blog post")

    print(f"\n📋 FINAL CONTENT:")
    print("-" * 40)
    print(results["final"][:1500] + "..." if len(results["final"]) > 1500 else results["final"])

    # Example 3: Data Insight Chain
    print("\n\n📊 DATA INSIGHT CHAIN")
    print("-" * 40)

    data_desc = """
    Q3 Sales Report:
    - Total revenue: $2.4M (up 15% from Q2)
    - New customers: 340 (down 8% from Q2)
    - Customer retention rate: 92%
    - Average deal size: $7,058 (up 25%)
    - Sales cycle length: 45 days (up from 38 days)
    - Top performing region: West Coast (35% of revenue)
    - Underperforming region: Midwest (8% of revenue)
    """
    business_ctx = "B2B SaaS company targeting mid-market companies, goal is 30% YoY growth"

    print(f"\nData: {data_desc.strip()}")
    print(f"\nContext: {business_ctx}")

    results = data_insight_chain(data_desc, business_ctx)

    print(f"\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    print(results["recommendations"])

    # Example 4: Translation and Localization
    print("\n\n🌐 TRANSLATION & LOCALIZATION")
    print("-" * 40)

    original_text = """
    Our Black Friday sale is a slam dunk! Get up to 50% off everything in the store.
    Don't miss out - these deals are going like hotcakes! Sale runs through Cyber Monday.
    """

    print(f"\nOriginal (English): {original_text.strip()}")

    results = translation_localization_chain(
        original_text,
        source_lang="English",
        target_lang="Brazilian Portuguese",
        target_culture="Brazilian consumers"
    )

    print(f"\n📋 LOCALIZED VERSION:")
    print("-" * 40)
    print(results["verified"])

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Prompt Chaining")

    print("\nEnd of Prompt Chaining demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
