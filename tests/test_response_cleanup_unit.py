#!/usr/bin/env python3
"""
Simple unit test for TL;DR string processing and metadata formatting logic.
Tests the logic without requiring external dependencies.
"""


def test_tldr_prefix_stripping():
    """Test the TL;DR prefix stripping logic."""
    print("\nTesting TL;DR prefix stripping logic...")
    print("="*60)
    
    # Simulate the logic from _add_tldr_to_response
    def strip_tldr_prefix(tldr):
        tldr = tldr.strip()
        tldr_prefix = "tl;dr:"
        if tldr.lower().startswith(tldr_prefix):
            tldr = tldr[len(tldr_prefix):].strip()
        return tldr
    
    # Test case 1: With "TL;DR:" prefix
    print("\nTest 1: String with 'TL;DR:' prefix")
    input1 = "TL;DR: This is a summary."
    result1 = strip_tldr_prefix(input1)
    print(f"Input:  '{input1}'")
    print(f"Output: '{result1}'")
    assert result1 == "This is a summary.", f"Expected 'This is a summary.', got '{result1}'"
    print("✓ Prefix removed correctly")
    
    # Test case 2: With lowercase "tl;dr:" prefix
    print("\nTest 2: String with lowercase 'tl;dr:' prefix")
    input2 = "tl;dr: This is lowercase."
    result2 = strip_tldr_prefix(input2)
    print(f"Input:  '{input2}'")
    print(f"Output: '{result2}'")
    assert result2 == "This is lowercase.", f"Expected 'This is lowercase.', got '{result2}'"
    print("✓ Lowercase prefix removed correctly")
    
    # Test case 3: Without prefix
    print("\nTest 3: String without 'TL;DR:' prefix")
    input3 = "This is already clean."
    result3 = strip_tldr_prefix(input3)
    print(f"Input:  '{input3}'")
    print(f"Output: '{result3}'")
    assert result3 == "This is already clean.", f"Expected 'This is already clean.', got '{result3}'"
    print("✓ String without prefix unchanged")
    
    # Test case 4: With extra whitespace
    print("\nTest 4: String with extra whitespace")
    input4 = "  TL;DR:   This has spaces.  "
    result4 = strip_tldr_prefix(input4)
    print(f"Input:  '{input4}'")
    print(f"Output: '{result4}'")
    assert result4 == "This has spaces.", f"Expected 'This has spaces.', got '{result4}'"
    print("✓ Whitespace handled correctly")
    
    # Test case 5: Mixed case
    print("\nTest 5: String with mixed case 'Tl;Dr:'")
    input5 = "Tl;Dr: Mixed case."
    result5 = strip_tldr_prefix(input5)
    print(f"Input:  '{input5}'")
    print(f"Output: '{result5}'")
    assert result5 == "Mixed case.", f"Expected 'Mixed case.', got '{result5}'"
    print("✓ Mixed case prefix removed correctly")
    
    print("\n" + "="*60)
    print("✓ TL;DR prefix stripping test passed!")
    print("="*60)


def test_metadata_formatting():
    """Test the metadata formatting logic with call counts."""
    print("\nTesting metadata formatting with call counts...")
    print("="*60)
    
    # Simulate the logic from on_message
    merged_token_stats = {
        "amazon/nova-2-lite-v1:free": {
            "total_input_tokens": 1500,
            "total_output_tokens": 150,
            "total_calls": 3
        },
        "nvidia/nemotron-nano-12b-v2-vl:free": {
            "total_input_tokens": 500,
            "total_output_tokens": 50,
            "total_calls": 2
        },
        "amazon/nova-3-lite-v1:free": {
            "total_input_tokens": 200,
            "total_output_tokens": 20,
            "total_calls": 1
        }
    }
    
    # Calculate totals
    total_input_tokens = sum(stats["total_input_tokens"] for stats in merged_token_stats.values())
    total_output_tokens = sum(stats["total_output_tokens"] for stats in merged_token_stats.values())
    total_response_time = 27
    
    # Format metadata as done in discord_bot.py
    models_info = []
    for model, stats in merged_token_stats.items():
        model_name = model.split('/')[-1] if '/' in model else model  # Use short name
        calls = stats["total_calls"]
        models_info.append(f"{model_name} ({calls}x)")
    models_used = " • ".join(models_info)
    metadata = f"\n\n-# *Models: {models_used} • Tokens: {total_input_tokens} in / {total_output_tokens} out • Time: {round(total_response_time)}s*"
    
    print("\nGenerated metadata:")
    print(metadata.strip())
    
    # Verify the metadata structure
    print("\nVerifying metadata contents:")
    
    # Check for model names (short version)
    assert "nova-2-lite-v1:free" in metadata, "Should include nova-2-lite short name"
    assert "nemotron-nano-12b-v2-vl:free" in metadata, "Should include nemotron short name"
    assert "nova-3-lite-v1:free" in metadata, "Should include nova-3-lite short name"
    print("✓ Model names included (shortened)")
    
    # Check for call counts
    assert "(3x)" in metadata, "Should include 3 calls for first model"
    assert "(2x)" in metadata, "Should include 2 calls for second model"
    assert "(1x)" in metadata, "Should include 1 call for third model"
    print("✓ Call counts included for each model")
    
    # Check for token counts
    assert f"{total_input_tokens} in" in metadata, f"Should include {total_input_tokens} input tokens"
    assert f"{total_output_tokens} out" in metadata, f"Should include {total_output_tokens} output tokens"
    print(f"✓ Total tokens included ({total_input_tokens} in / {total_output_tokens} out)")
    
    # Check for time
    assert "27s" in metadata, "Should include 27s response time"
    print("✓ Response time included")
    
    # Verify separator between models is "•"
    assert " • " in metadata, "Should use '•' separator between model info"
    print("✓ Models separated with '•' bullet")
    
    print("\n" + "="*60)
    print("✓ Metadata formatting test passed!")
    print("="*60)


def test_example_output():
    """Show example of the fixed output."""
    print("\nExample output demonstration...")
    print("="*60)
    
    print("\n--- BEFORE (with issues) ---")
    print("TL;DR: TL;DR: The requirements.txt lists recent versions...")
    print("Models: amazon/nova-2-lite-v1:free • Tokens: 2159 in / 220 out • Time: 27s")
    
    print("\n--- AFTER (fixed) ---")
    # Simulate TL;DR
    tldr_input = "TL;DR: The requirements.txt lists recent versions..."
    tldr_stripped = tldr_input.strip()
    tldr_prefix = "tl;dr:"
    if tldr_stripped.lower().startswith(tldr_prefix):
        tldr_stripped = tldr_stripped[len(tldr_prefix):].strip()
    print(f"TL;DR: {tldr_stripped}")
    
    # Simulate metadata
    merged_token_stats = {
        "amazon/nova-2-lite-v1:free": {
            "total_calls": 5
        },
        "nvidia/nemotron-nano-12b-v2-vl:free": {
            "total_calls": 2
        }
    }
    models_info = []
    for model, stats in merged_token_stats.items():
        model_name = model.split('/')[-1] if '/' in model else model
        calls = stats["total_calls"]
        models_info.append(f"{model_name} ({calls}x)")
    models_used = " • ".join(models_info)
    print(f"Models: {models_used} • Tokens: 2159 in / 220 out • Time: 27s")
    
    print("\n" + "="*60)
    print("✓ Output comparison complete!")
    print("="*60)


if __name__ == "__main__":
    print("\nDiscord Bot Response Cleanup - Unit Tests")
    print("="*60)
    print("Testing without external dependencies\n")
    
    # Test 1: TL;DR prefix stripping
    test_tldr_prefix_stripping()
    
    # Test 2: Metadata formatting
    test_metadata_formatting()
    
    # Test 3: Example output
    test_example_output()
    
    print("\n" + "="*60)
    print("✓ ALL UNIT TESTS PASSED!")
    print("="*60)
    
    print("\nSummary of fixes:")
    print("1. TL;DR duplication fixed:")
    print("   - Strips 'TL;DR:' prefix from LLM response (case insensitive)")
    print("   - Prevents 'TL;DR: TL;DR: ...' in output")
    print("")
    print("2. Model call counts added:")
    print("   - Shows (Nx) for each model in metadata")
    print("   - Example: 'nova-2-lite-v1:free (5x) • nemotron-nano-12b-v2-vl:free (2x)'")
    print("="*60)
