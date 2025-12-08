#!/usr/bin/env python3
"""
Test script for the read_file tool functionality.
This script tests the file reading tool to ensure it works correctly
and has proper security measures in place.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from react_agent import read_file
from pathlib import Path
import tempfile


def test_read_existing_file():
    """Test reading a file that exists."""
    print("Testing read_file with existing file...")
    print("="*60)
    
    result = read_file('config.yaml')
    
    assert 'Error' not in result, "Should successfully read config.yaml"
    assert 'base_url' in result, "Should contain base_url configuration"
    assert len(result) > 0, "Should return non-empty content"
    
    print(f"✓ Successfully read file ({len(result)} characters)")
    print(f"✓ First 100 chars: {result[:100]}...")
    print()


def test_read_nonexistent_file():
    """Test reading a file that doesn't exist."""
    print("Testing read_file with non-existent file...")
    print("="*60)
    
    result = read_file('nonexistent_file_12345.txt')
    
    assert 'Error' in result, "Should return error for non-existent file"
    assert 'not found' in result, "Error should mention file not found"
    
    print(f"✓ Correctly returned error: {result}")
    print()


def test_directory_traversal_prevention():
    """Test that directory traversal is prevented."""
    print("Testing directory traversal prevention...")
    print("="*60)
    
    # Try to access a file outside cwd
    result = read_file('../../../etc/passwd')
    
    assert 'Error' in result, "Should return error for directory traversal"
    assert 'Access denied' in result or 'outside' in result, "Error should mention access denial"
    
    print(f"✓ Directory traversal blocked: {result}")
    print()


def test_read_from_subdirectory():
    """Test reading a file from a subdirectory."""
    print("Testing read_file from subdirectory...")
    print("="*60)
    
    # Assuming tests directory exists
    result = read_file('tests/test_react_agent.py')
    
    if 'Error' not in result:
        assert 'MockReActAgent' in result, "Should contain test content"
        print(f"✓ Successfully read file from subdirectory ({len(result)} characters)")
    else:
        print(f"Note: {result}")
    print()


def test_file_size_limit():
    """Test that files larger than 1MB are rejected."""
    print("Testing file size limit...")
    print("="*60)
    
    # Create a temporary large file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.txt') as f:
        temp_filename = f.name
        # Write more than 1MB
        f.write('x' * (1024 * 1024 + 1))
    
    try:
        result = read_file(os.path.basename(temp_filename))
        
        assert 'Error' in result, "Should return error for large file"
        assert 'too large' in result, "Error should mention file size"
        
        print(f"✓ Large file rejected: {result[:100]}...")
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    print()


def test_binary_file_handling():
    """Test that binary files are properly handled."""
    print("Testing binary file handling...")
    print("="*60)
    
    # Create a temporary binary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir='.', suffix='.bin') as f:
        temp_filename = f.name
        # Write some binary data
        f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')
    
    try:
        result = read_file(os.path.basename(temp_filename))
        
        assert 'Error' in result, "Should return error for binary file"
        assert 'binary' in result, "Error should mention binary file"
        
        print(f"✓ Binary file rejected: {result}")
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    print()


def test_directory_not_file():
    """Test that directories cannot be read as files."""
    print("Testing directory rejection...")
    print("="*60)
    
    result = read_file('tests')
    
    assert 'Error' in result, "Should return error for directory"
    assert 'not a file' in result, "Error should mention it's not a file"
    
    print(f"✓ Directory rejected: {result}")
    print()


def test_agent_has_read_file_tool():
    """Test that the ReActAgent has the read_file tool registered."""
    print("Testing agent tool registration...")
    print("="*60)
    
    from react_agent import ReActAgent
    
    agent = ReActAgent('test_key')
    
    assert 'read_file' in agent.tools, "Agent should have read_file tool"
    assert 'function' in agent.tools['read_file'], "read_file should have a function"
    assert 'description' in agent.tools['read_file'], "read_file should have a description"
    
    tool_desc = agent.tools['read_file']['description']
    assert 'current working directory' in tool_desc, "Description should mention cwd"
    
    print(f"✓ read_file tool registered")
    print(f"✓ Description: {tool_desc}")
    print()


def test_configurable_base_url():
    """Test that base_url can be configured."""
    print("Testing configurable base_url...")
    print("="*60)
    
    from react_agent import ReActAgent, MODEL_CONFIG
    
    # Test default base_url
    agent1 = ReActAgent('test_key')
    assert agent1.api_url == MODEL_CONFIG.get('base_url', 'https://openrouter.ai/api/v1/chat/completions')
    print(f"✓ Default base_url: {agent1.api_url}")
    
    # Test custom base_url
    custom_url = 'http://localhost:8080/v1/chat/completions'
    agent2 = ReActAgent('test_key', base_url=custom_url)
    assert agent2.api_url == custom_url
    print(f"✓ Custom base_url: {agent2.api_url}")
    
    # Test that config has base_url
    assert 'base_url' in MODEL_CONFIG, "Config should have base_url"
    print(f"✓ Config base_url: {MODEL_CONFIG.get('base_url')}")
    print()


if __name__ == "__main__":
    print("Read File Tool Tests")
    print("="*60)
    
    test_read_existing_file()
    test_read_nonexistent_file()
    test_directory_traversal_prevention()
    test_read_from_subdirectory()
    test_file_size_limit()
    test_binary_file_handling()
    test_directory_not_file()
    test_agent_has_read_file_tool()
    test_configurable_base_url()
    
    print("="*60)
    print("✓ ALL READ FILE TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Summary:")
    print("The read_file tool:")
    print("1. Reads files from the current working directory")
    print("2. Prevents directory traversal attacks")
    print("3. Enforces a 1MB file size limit")
    print("4. Handles binary files gracefully")
    print("5. Rejects directories")
    print("6. Is properly registered with the ReAct agent")
    print("\nThe base_url configuration:")
    print("1. Can be set in config.yaml")
    print("2. Can be passed to ReActAgent constructor")
    print("3. Defaults to OpenRouter API")
    print("4. Allows using local LLMs or other OpenAI-compatible endpoints")
    print("="*60)
