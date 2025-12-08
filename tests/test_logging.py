#!/usr/bin/env python3
"""
Test script for logging functionality in discord_bot and react_agent.
This script verifies that logging is working correctly for user queries, 
tool usage, and LLM calls.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from discord_bot import ReActDiscordBot
from react_agent import ReActAgent


class LogCapture:
    """Helper class to capture log messages."""
    
    def __init__(self):
        self.handler = logging.StreamHandler(StringIO())
        self.handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
    
    def __enter__(self):
        # Add handler to both loggers
        discord_logger = logging.getLogger('discord_bot')
        react_logger = logging.getLogger('react_agent')
        discord_logger.addHandler(self.handler)
        react_logger.addHandler(self.handler)
        return self
    
    def __exit__(self, *args):
        # Remove handler from both loggers
        discord_logger = logging.getLogger('discord_bot')
        react_logger = logging.getLogger('react_agent')
        discord_logger.removeHandler(self.handler)
        react_logger.removeHandler(self.handler)
    
    def get_logs(self):
        """Get captured log messages."""
        return self.handler.stream.getvalue()


def test_user_query_logging():
    """Test that user queries are logged in discord_bot."""
    print("Testing user query logging...")
    print("="*60)
    
    # We can't fully test the async message handler without a full Discord mock,
    # but we can verify the logging exists in the code
    discord_bot_path = os.path.join(os.path.dirname(__file__), '..', 'discord_bot.py')
    with open(discord_bot_path, 'r') as f:
        content = f.read()
    
    # Check that user query logging exists
    assert 'logger.info(f"User query received' in content, \
        "Should have logging for user queries"
    
    print("✓ User query logging code found in discord_bot.py")
    print("  Pattern: logger.info(f\"User query received from {message.author.display_name}: {question}\")")
    
    print("\n" + "="*60)
    print("✓ User query logging test passed!")
    print("="*60)


def test_tool_usage_logging():
    """Test that tool usage and arguments are logged in react_agent."""
    print("\nTesting tool usage logging...")
    print("="*60)
    
    # Create a mock agent
    agent = ReActAgent("mock_key", "mock_model")
    
    # Override tools with mock functions
    agent.tools = {
        "duckduckgo_search": {
            "function": lambda q: [{"title": "Test", "href": "http://test.com", "body": "Test"}],
            "description": "Search tool",
            "parameters": ["query"]
        }
    }
    
    # Capture logs
    with LogCapture() as log_capture:
        # Execute a tool action
        result = agent._execute_action("duckduckgo_search", "test query")
        logs = log_capture.get_logs()
        
        print("\nCaptured logs:")
        print(logs)
        
        # Verify tool usage was logged
        assert "Tool used: duckduckgo_search" in logs, \
            "Should log tool name"
        assert "Arguments: test query" in logs, \
            "Should log tool arguments"
        assert "completed successfully" in logs, \
            "Should log tool completion"
        
        print("✓ Tool usage logged correctly")
        print("  - Tool name: duckduckgo_search")
        print("  - Arguments: test query")
        print("  - Completion status: logged")
    
    print("\n" + "="*60)
    print("✓ Tool usage logging test passed!")
    print("="*60)


def test_llm_logging_in_react_agent():
    """Test that LLM calls are logged with model, response time, and tokens."""
    print("\nTesting LLM logging in react_agent...")
    print("="*60)
    
    agent = ReActAgent("mock_key", "mock_model")
    
    # Mock the requests.post call
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response from LLM"}}]
    }
    
    with patch('requests.post', return_value=mock_response), \
         LogCapture() as log_capture:
        
        # Call LLM
        start = time.time()
        result = agent._call_llm("Test prompt for LLM")
        elapsed = time.time() - start
        
        logs = log_capture.get_logs()
        
        print("\nCaptured logs:")
        print(logs)
        
        # Verify LLM logging
        assert "LLM call started" in logs, "Should log LLM call start"
        assert "Model: mock_model" in logs, "Should log model name"
        assert "Input tokens:" in logs, "Should log input tokens"
        assert "LLM call completed" in logs, "Should log LLM call completion"
        assert "Response time:" in logs, "Should log response time"
        assert "Output tokens:" in logs, "Should log output tokens"
        
        # Verify token calculation (character len / 4.5)
        input_tokens = int(len("Test prompt for LLM") / 4.5)
        output_tokens = int(len("Test response from LLM") / 4.5)
        
        assert f"Input tokens: {input_tokens}" in logs, \
            f"Should calculate input tokens as {input_tokens}"
        assert f"Output tokens: {output_tokens}" in logs, \
            f"Should calculate output tokens as {output_tokens}"
        
        print("✓ LLM call logged correctly")
        print(f"  - Model: mock_model")
        print(f"  - Input tokens: {input_tokens} (calculated from prompt length)")
        print(f"  - Output tokens: {output_tokens} (calculated from response length)")
        print(f"  - Response time: logged")
    
    print("\n" + "="*60)
    print("✓ LLM logging test passed!")
    print("="*60)


def test_llm_logging_in_discord_bot():
    """Test that LLM calls are logged in discord_bot."""
    print("\nTesting LLM logging in discord_bot...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.ReActAgent') as MockAgent:
        
        bot = ReActDiscordBot("test_token", "test_api_key")
        
        # Mock the requests.post call
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        
        with patch('requests.post', return_value=mock_response), \
             LogCapture() as log_capture:
            
            # Call LLM
            result = bot._call_llm("Test prompt", model="test-model")
            
            logs = log_capture.get_logs()
            
            print("\nCaptured logs:")
            print(logs)
            
            # Verify LLM logging
            assert "LLM call started" in logs, "Should log LLM call start"
            assert "Model: test-model" in logs, "Should log model name"
            assert "Input tokens:" in logs, "Should log input tokens"
            assert "LLM call completed" in logs, "Should log LLM call completion"
            assert "Response time:" in logs, "Should log response time"
            assert "Output tokens:" in logs, "Should log output tokens"
            
            print("✓ LLM call logged correctly in discord_bot")
            print(f"  - Model: test-model")
            print(f"  - Input tokens: logged")
            print(f"  - Output tokens: logged")
            print(f"  - Response time: logged")
    
    print("\n" + "="*60)
    print("✓ Discord bot LLM logging test passed!")
    print("="*60)


def test_llm_error_logging():
    """Test that LLM errors are logged properly."""
    print("\nTesting LLM error logging...")
    print("="*60)
    
    agent = ReActAgent("mock_key", "mock_model")
    
    # Mock the requests.post call to raise an exception
    with patch('requests.post', side_effect=Exception("API Error")), \
         LogCapture() as log_capture:
        
        # Call LLM (should fail)
        result = agent._call_llm("Test prompt")
        
        logs = log_capture.get_logs()
        
        print("\nCaptured logs:")
        print(logs)
        
        # Verify error logging
        assert "LLM call failed" in logs, "Should log LLM call failure"
        assert "Model: mock_model" in logs, "Should log model name"
        assert "Error: API Error" in logs, "Should log error message"
        
        print("✓ LLM error logged correctly")
        print(f"  - Error message: logged")
        print(f"  - Model: mock_model")
    
    print("\n" + "="*60)
    print("✓ LLM error logging test passed!")
    print("="*60)


def test_tool_error_logging():
    """Test that tool errors are logged properly."""
    print("\nTesting tool error logging...")
    print("="*60)
    
    agent = ReActAgent("mock_key", "mock_model")
    
    # Override tools with a function that raises an exception
    def failing_tool_function(input_arg):
        """A tool function that always fails."""
        raise Exception("Tool error")
    
    agent.tools = {
        "failing_tool": {
            "function": failing_tool_function,
            "description": "A tool that fails",
            "parameters": ["input"]
        }
    }
    
    # Capture logs
    with LogCapture() as log_capture:
        # Execute a tool action that will fail
        result = agent._execute_action("failing_tool", "test input")
        logs = log_capture.get_logs()
        
        print("\nCaptured logs:")
        print(logs)
        
        # Verify tool error was logged
        assert "Tool used: failing_tool" in logs, \
            "Should log tool name"
        assert "Arguments: test input" in logs, \
            "Should log tool arguments"
        assert "Tool failing_tool failed with exception" in logs, \
            "Should log tool failure"
        
        print("✓ Tool error logged correctly")
        print("  - Tool name: failing_tool")
        print("  - Arguments: test input")
        print("  - Error: logged")
    
    print("\n" + "="*60)
    print("✓ Tool error logging test passed!")
    print("="*60)


if __name__ == "__main__":
    print("Logging Tests")
    print("="*60)
    
    # Test user query logging
    test_user_query_logging()
    
    # Test tool usage logging
    test_tool_usage_logging()
    
    # Test LLM logging in react_agent
    test_llm_logging_in_react_agent()
    
    # Test LLM logging in discord_bot
    test_llm_logging_in_discord_bot()
    
    # Test LLM error logging
    test_llm_error_logging()
    
    # Test tool error logging
    test_tool_error_logging()
    
    print("\n" + "="*60)
    print("✓ ALL LOGGING TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Summary:")
    print("The logging implementation covers:")
    print("1. User queries in discord_bot")
    print("2. Tool usage and arguments in react_agent")
    print("3. LLM model, response time, and token counts (input/output)")
    print("4. Error logging for both LLM calls and tool execution")
    print("="*60)
