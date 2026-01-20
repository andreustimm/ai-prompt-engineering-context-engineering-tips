"""
Vision/Multimodal

Technique for analyzing images using vision-capable LLMs like GPT-4o.
Combines text and image understanding for comprehensive analysis.

Features:
- Image description and analysis
- Object detection and identification
- Text extraction from images (OCR)
- Chart and diagram interpretation

Requirements:
- Vision-capable model (gpt-4o, gpt-4o-mini)
- Images in supported formats (PNG, JPEG, GIF, WebP)

Use cases:
- Image captioning
- Visual question answering
- Document analysis
- Chart interpretation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import base64
from langchain_core.messages import HumanMessage
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the media type based on file extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


def analyze_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """
    Analyze an image using a vision-capable LLM.

    Args:
        image_path: Path to the image file
        prompt: Question or instruction about the image

    Returns:
        LLM's analysis of the image
    """
    llm = get_llm(temperature=0.3)

    # Encode image
    base64_image = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    # Create message with image
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}"
                }
            }
        ]
    )

    response = llm.invoke([message])

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def describe_image(image_path: str) -> str:
    """Generate a detailed description of an image."""
    return analyze_image(
        image_path,
        "Describe this image in detail. Include information about objects, colors, composition, and any notable features."
    )


def extract_text_from_image(image_path: str) -> str:
    """Extract text visible in an image (OCR-like functionality)."""
    return analyze_image(
        image_path,
        "Extract and list all text visible in this image. Include any labels, captions, or written content."
    )


def analyze_chart(image_path: str) -> str:
    """Analyze a chart or graph image."""
    return analyze_image(
        image_path,
        """Analyze this chart or graph. Provide:
1. Type of chart (bar, line, pie, etc.)
2. What data it represents
3. Key insights or trends
4. Any notable values or outliers"""
    )


def identify_objects(image_path: str) -> str:
    """Identify and list objects in an image."""
    return analyze_image(
        image_path,
        "Identify and list all distinct objects, people, or elements visible in this image. Provide a brief description of each."
    )


def answer_about_image(image_path: str, question: str) -> str:
    """Answer a specific question about an image."""
    return analyze_image(image_path, question)


def compare_images(image_path1: str, image_path2: str, aspect: str = "general") -> str:
    """
    Compare two images.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        aspect: What aspect to compare (general, colors, objects, etc.)

    Returns:
        Comparison analysis
    """
    llm = get_llm(temperature=0.3)

    # Encode both images
    base64_image1 = encode_image_to_base64(image_path1)
    base64_image2 = encode_image_to_base64(image_path2)
    media_type1 = get_image_media_type(image_path1)
    media_type2 = get_image_media_type(image_path2)

    prompt = f"""Compare these two images with focus on {aspect}.
Describe:
1. Similarities between the images
2. Differences between the images
3. Overall assessment"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type1};base64,{base64_image1}"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type2};base64,{base64_image2}"}
            }
        ]
    )

    response = llm.invoke([message])

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def analyze_diagram(image_path: str) -> str:
    """Analyze a technical diagram or flowchart."""
    return analyze_image(
        image_path,
        """Analyze this diagram or flowchart. Describe:
1. The type of diagram
2. The main components or elements
3. The relationships or flow between elements
4. The overall purpose or message of the diagram"""
    )


def main():
    print("=" * 60)
    print("VISION/MULTIMODAL - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Path to sample images
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "images"

    if not sample_dir.exists():
        print(f"\nError: Sample images directory not found at {sample_dir}")
        print("Please ensure sample_data/images directory exists with sample images.")
        return

    # Check for available images
    available_images = list(sample_dir.glob("*.*"))
    if not available_images:
        print(f"\nNo images found in {sample_dir}")
        return

    print(f"\n📷 Found {len(available_images)} image(s):")
    for img in available_images:
        print(f"   - {img.name}")

    # Example 1: Chart Analysis
    chart_path = sample_dir / "chart.png"
    if chart_path.exists():
        print("\n\n📊 CHART ANALYSIS")
        print("-" * 40)
        print(f"\nAnalyzing: {chart_path.name}")
        result = analyze_chart(str(chart_path))
        print(f"\n📋 Analysis:\n{result}")

    # Example 2: Diagram Analysis
    diagram_path = sample_dir / "diagram.png"
    if diagram_path.exists():
        print("\n\n📐 DIAGRAM ANALYSIS")
        print("-" * 40)
        print(f"\nAnalyzing: {diagram_path.name}")
        result = analyze_diagram(str(diagram_path))
        print(f"\n📋 Analysis:\n{result}")

    # Example 3: Photo Description
    photo_path = sample_dir / "photo.jpg"
    if photo_path.exists():
        print("\n\n🖼️ PHOTO DESCRIPTION")
        print("-" * 40)
        print(f"\nAnalyzing: {photo_path.name}")
        result = describe_image(str(photo_path))
        print(f"\n📋 Description:\n{result}")

    # Example 4: Object Identification
    if photo_path.exists():
        print("\n\n🔍 OBJECT IDENTIFICATION")
        print("-" * 40)
        print(f"\nIdentifying objects in: {photo_path.name}")
        result = identify_objects(str(photo_path))
        print(f"\n📋 Objects Found:\n{result}")

    # Example 5: Visual Question Answering
    if chart_path.exists():
        print("\n\n❓ VISUAL Q&A")
        print("-" * 40)
        question = "What is the highest value shown in this chart?"
        print(f"\nImage: {chart_path.name}")
        print(f"Question: {question}")
        result = answer_about_image(str(chart_path), question)
        print(f"\n📋 Answer:\n{result}")

    # Example 6: Image Comparison
    if chart_path.exists() and diagram_path.exists():
        print("\n\n🔄 IMAGE COMPARISON")
        print("-" * 40)
        print(f"\nComparing: {chart_path.name} vs {diagram_path.name}")
        result = compare_images(str(chart_path), str(diagram_path), "visualization type and purpose")
        print(f"\n📋 Comparison:\n{result}")

    print_total_usage(token_tracker, "TOTAL - Vision/Multimodal")

    print("\n\n" + "=" * 60)
    print("Note: Vision capabilities require a vision-enabled model")
    print("like gpt-4o or gpt-4o-mini. Ensure OPENAI_MODEL is set correctly.")
    print("=" * 60)

    print("\nEnd of Vision/Multimodal demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
