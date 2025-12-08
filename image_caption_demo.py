#!/usr/bin/env python3
"""
Demonstration script for image captioning functionality.

This script shows how the two-round image captioning works:
1. First round: Get a basic description of the image
2. Second round: Get detailed analysis based on user query

Note: This requires a valid OPENROUTER_API_KEY environment variable.
"""

import os
from react_agent import two_round_image_caption

def demo_image_captioning():
    """Demonstrate the image captioning functionality."""
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("This demo requires an OpenRouter API key to function.")
        print("\nTo get an API key:")
        print("1. Go to https://openrouter.ai/")
        print("2. Sign up and get your API key")
        print("3. Set the environment variable: export OPENROUTER_API_KEY=your_key")
        return
    
    print("Image Captioning Demonstration")
    print("="*60)
    print()
    print("This demo shows how the Discord bot handles image attachments:")
    print()
    print("1. User sends a message with an image attachment")
    print("2. Bot detects the image and registers the caption_image tool")
    print("3. Bot's ReAct agent can choose to use the tool")
    print("4. Tool performs two-round captioning:")
    print("   - Round 1: Basic description of the image")
    print("   - Round 2: Detailed analysis based on user query")
    print()
    print("="*60)
    print()
    
    # Example image URL (you can replace with any public image URL)
    example_image_url = "https://example.com/test-image.jpg"
    user_query = "What colors are in this image?"
    
    print(f"Example Usage:")
    print(f"  Image URL: {example_image_url}")
    print(f"  User Query: {user_query}")
    print()
    print("Note: This is a demonstration. In actual use:")
    print("- The bot automatically detects image attachments from Discord messages")
    print("- Images are downloaded from Discord's CDN")
    print("- The caption_image tool is registered dynamically")
    print("- The ReAct agent decides when to use the tool")
    print()
    print("="*60)
    print()
    
    # Show the function signature
    print("Function signature:")
    print()
    print("two_round_image_caption(")
    print("    image_url: str,")
    print("    api_key: str,")
    print("    user_query: str = None,")
    print("    model: str = 'nvidia/nemotron-nano-12b-v2-vl:free'")
    print(")")
    print()
    print("="*60)
    print()
    
    print("Features:")
    print("✓ Downloads images from Discord attachments")
    print("✓ Converts images to base64 for API submission")
    print("✓ Uses nemotron-nano-12b-v2-vl:free (vision model)")
    print("✓ Two-round captioning for detailed analysis")
    print("✓ Context-aware captions based on user query")
    print("✓ Automatic cleanup of temporary image files")
    print()
    print("="*60)
    print()
    
    print("Integration with Discord Bot:")
    print()
    print("When a user sends a message like:")
    print("  '@Bot What's in this image?' [with image attached]")
    print()
    print("The bot will:")
    print("1. Detect the image attachment")
    print("2. Register the caption_image tool")
    print("3. Include image context in the prompt")
    print("4. Let the ReAct agent analyze the image")
    print("5. Return a detailed description")
    print()
    print("="*60)
    

if __name__ == "__main__":
    demo_image_captioning()
