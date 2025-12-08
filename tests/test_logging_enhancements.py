#!/usr/bin/env python3
"""
Test script for Discord bot logging enhancements.
This script verifies that LLM calls and tool calls are tracked properly
and that query logs are saved correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from react_agent import ReActAgent
from discord_bot import ReActDiscordBot


def test_react_agent_tracking():
    """
    Test that the ReAct agent properly tracks LLM calls and tool calls.
    """
    print("Testing ReAct agent tracking...")
    print("="*60)
    
    # Create a mock API key
    api_key = "test_api_key"
    
    # Create agent
    agent = ReActAgent(api_key)
    
    # Test 1: Verify initialization
    print("\nTest 1: Verify tracking initialization")
    assert hasattr(agent, 'call_sequence'), "Agent should have call_sequence"
    assert hasattr(agent, 'token_stats'), "Agent should have token_stats"
    assert len(agent.call_sequence) == 0, "call_sequence should be empty initially"
    assert len(agent.token_stats) == 0, "token_stats should be empty initially"
    print("✓ Tracking initialized correctly")
    
    # Test 2: Test reset_tracking method
    print("\nTest 2: Test reset_tracking method")
    agent.call_sequence.append({"test": "data"})
    agent.token_stats["test_model"] = {"total_calls": 1}
    agent.reset_tracking()
    assert len(agent.call_sequence) == 0, "call_sequence should be empty after reset"
    assert len(agent.token_stats) == 0, "token_stats should be empty after reset"
    print("✓ reset_tracking works correctly")
    
    # Test 3: Test get_tracking_data method
    print("\nTest 3: Test get_tracking_data method")
    agent.call_sequence.append({"type": "test"})
    agent.token_stats["model1"] = {"total_calls": 1}
    tracking_data = agent.get_tracking_data()
    assert "call_sequence" in tracking_data, "Should have call_sequence"
    assert "token_stats" in tracking_data, "Should have token_stats"
    assert len(tracking_data["call_sequence"]) == 1, "Should have one call in sequence"
    assert "model1" in tracking_data["token_stats"], "Should have model1 in stats"
    print("✓ get_tracking_data works correctly")
    
    print("\n" + "="*60)
    print("✓ ReAct agent tracking test passed!")
    print("="*60)


def test_llm_call_tracking():
    """
    Test that LLM calls are tracked with tokens/sec and token counts.
    """
    print("\nTesting LLM call tracking...")
    print("="*60)
    
    with patch('react_agent.requests.post') as mock_post:
        # Create a mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test response"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create agent and make a call
        agent = ReActAgent("test_api_key")
        result = agent._call_llm("Test prompt")
        
        # Test 1: Verify call was tracked
        print("\nTest 1: Verify LLM call was tracked")
        assert len(agent.call_sequence) == 1, "Should have one call tracked"
        call_entry = agent.call_sequence[0]
        assert call_entry["type"] == "llm_call", "Should be llm_call type"
        assert "model" in call_entry, "Should have model"
        assert "input_tokens" in call_entry, "Should have input_tokens"
        assert "output_tokens" in call_entry, "Should have output_tokens"
        assert "input_tokens_per_sec" in call_entry, "Should have input_tokens_per_sec"
        assert "output_tokens_per_sec" in call_entry, "Should have output_tokens_per_sec"
        assert "response_time_seconds" in call_entry, "Should have response_time_seconds"
        print("✓ LLM call tracked with all required fields")
        
        # Test 2: Verify token stats are aggregated
        print("\nTest 2: Verify token stats aggregation")
        assert len(agent.token_stats) == 1, "Should have stats for one model"
        model_stats = agent.token_stats[agent.model]
        assert "total_input_tokens" in model_stats, "Should have total_input_tokens"
        assert "total_output_tokens" in model_stats, "Should have total_output_tokens"
        assert "total_calls" in model_stats, "Should have total_calls"
        assert model_stats["total_calls"] == 1, "Should have 1 call"
        print("✓ Token stats aggregated correctly")
        
        # Test 3: Make another call and verify aggregation
        print("\nTest 3: Test multiple calls aggregation")
        result = agent._call_llm("Another test prompt")
        assert len(agent.call_sequence) == 2, "Should have two calls tracked"
        assert model_stats["total_calls"] == 2, "Should have 2 calls"
        assert model_stats["total_input_tokens"] > 0, "Should have accumulated input tokens"
        assert model_stats["total_output_tokens"] > 0, "Should have accumulated output tokens"
        print("✓ Multiple calls aggregated correctly")
    
    print("\n" + "="*60)
    print("✓ LLM call tracking test passed!")
    print("="*60)


def test_tool_call_tracking():
    """
    Test that tool calls are tracked with inputs and outputs.
    """
    print("\nTesting tool call tracking...")
    print("="*60)
    
    with patch('react_agent.duckduckgo_search') as mock_search:
        # Mock search results
        mock_search.return_value = [
            {"title": "Test Result", "href": "http://test.com", "body": "Test body"}
        ]
        
        # Create agent
        agent = ReActAgent("test_api_key")
        
        # Execute an action
        result = agent._execute_action("duckduckgo_search", "test query")
        
        # Test 1: Verify tool call was tracked
        print("\nTest 1: Verify tool call was tracked")
        assert len(agent.call_sequence) == 1, "Should have one call tracked"
        call_entry = agent.call_sequence[0]
        assert call_entry["type"] == "tool_call", "Should be tool_call type"
        assert call_entry["tool_name"] == "duckduckgo_search", "Should have correct tool name"
        assert "input" in call_entry, "Should have input"
        assert "output" in call_entry, "Should have output"
        assert "execution_time_seconds" in call_entry, "Should have execution_time_seconds"
        print("✓ Tool call tracked with all required fields")
        
        # Test 2: Test error tracking
        print("\nTest 2: Test tool error tracking")
        agent.reset_tracking()
        result = agent._execute_action("nonexistent_tool", "test input")
        assert len(agent.call_sequence) == 1, "Should have one call tracked"
        call_entry = agent.call_sequence[0]
        assert "error" in call_entry, "Should have error field"
        assert "Unknown action" in call_entry["error"], "Should have error message"
        print("✓ Tool errors tracked correctly")
    
    print("\n" + "="*60)
    print("✓ Tool call tracking test passed!")
    print("="*60)


def test_discord_bot_query_logging():
    """
    Test that Discord bot saves query logs correctly.
    """
    print("\nTesting Discord bot query logging...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.ReActAgent') as MockAgent, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        # Create mock agent
        mock_agent_instance = Mock()
        mock_agent_instance.get_tracking_data.return_value = {
            "call_sequence": [
                {
                    "type": "llm_call",
                    "model": "test-model",
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            ],
            "token_stats": {
                "test-model": {
                    "total_input_tokens": 10,
                    "total_output_tokens": 20,
                    "total_calls": 1
                }
            }
        }
        mock_agent_instance.reset_tracking = Mock()
        MockAgent.return_value = mock_agent_instance
        
        # Create bot instance with temporary directory
        bot = ReActDiscordBot("test_token", "test_api_key")
        
        # Override the QUERY_LOGS_DIR to use temp directory
        bot.QUERY_LOGS_DIR = Path(tmpdir) / "query_logs"
        bot.QUERY_LOGS_DIR.mkdir(exist_ok=True)
        
        # Test 1: Test reset tracking
        print("\nTest 1: Test reset tracking")
        bot._reset_query_tracking()
        assert len(bot.current_query_log) == 0, "Should reset query log"
        assert len(bot.current_query_token_stats) == 0, "Should reset token stats"
        mock_agent_instance.reset_tracking.assert_called_once()
        print("✓ Reset tracking works correctly")
        
        # Test 2: Test save query log
        print("\nTest 2: Test save query log")
        bot.current_query_log = [
            {
                "type": "llm_call",
                "model": "intent-model",
                "input_tokens": 5,
                "output_tokens": 3
            }
        ]
        bot.current_query_token_stats = {
            "intent-model": {
                "total_input_tokens": 5,
                "total_output_tokens": 3,
                "total_calls": 1
            }
        }
        
        bot._save_query_log("123456", "Test question?", "Test answer", "TestUser")
        
        # Verify log file was created with new naming format (username_timestamp.json)
        log_files = list(bot.QUERY_LOGS_DIR.glob("*.json"))
        assert len(log_files) == 1, "Should create one log file"
        # Check that filename starts with username (TestUser)
        assert log_files[0].name.startswith("TestUser_"), f"Filename should start with username, got: {log_files[0].name}"
        print(f"✓ Log file created with username-based naming: {log_files[0].name}")
        
        # Test 3: Verify log file contents
        print("\nTest 3: Verify log file contents")
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert "message_id" in log_data, "Should have message_id"
        assert log_data["message_id"] == "123456", "Should have correct message_id"
        assert "username" in log_data, "Should have username"
        assert log_data["username"] == "TestUser", "Should have correct username"
        assert "timestamp" in log_data, "Should have timestamp"
        assert "user_query" in log_data, "Should have user_query"
        assert "final_response" in log_data, "Should have final_response"
        assert "call_sequence" in log_data, "Should have call_sequence"
        assert "token_stats_by_model" in log_data, "Should have token_stats_by_model"
        
        # Verify call sequence includes both bot and agent calls
        assert len(log_data["call_sequence"]) == 2, "Should have 2 calls (bot + agent)"
        
        # Verify token stats are merged
        assert "intent-model" in log_data["token_stats_by_model"], "Should have intent-model stats"
        assert "test-model" in log_data["token_stats_by_model"], "Should have test-model stats"
        
        intent_stats = log_data["token_stats_by_model"]["intent-model"]
        assert intent_stats["total_input_tokens"] == 5, "Should have correct input tokens"
        assert intent_stats["total_output_tokens"] == 3, "Should have correct output tokens"
        
        print("✓ Log file contents are correct")
        print(f"  - Message ID: {log_data['message_id']}")
        print(f"  - Username: {log_data['username']}")
        print(f"  - Call sequence length: {len(log_data['call_sequence'])}")
        print(f"  - Models tracked: {list(log_data['token_stats_by_model'].keys())}")
    
    print("\n" + "="*60)
    print("✓ Discord bot query logging test passed!")
    print("="*60)


def test_llm_call_tracking_in_discord_bot():
    """
    Test that Discord bot _call_llm method tracks calls properly.
    """
    print("\nTesting Discord bot _call_llm tracking...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.ReActAgent') as MockAgent, \
         patch('discord_bot.requests.post') as mock_post:
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create bot
        bot = ReActDiscordBot("test_token", "test_api_key")
        
        # Make LLM call
        result = bot._call_llm("Test prompt", model="test-model")
        
        # Test 1: Verify tracking
        print("\nTest 1: Verify LLM call was tracked")
        assert len(bot.current_query_log) == 1, "Should have one call tracked"
        call_entry = bot.current_query_log[0]
        assert call_entry["type"] == "llm_call", "Should be llm_call type"
        assert call_entry["model"] == "test-model", "Should have correct model"
        assert "input_tokens" in call_entry, "Should have input_tokens"
        assert "output_tokens" in call_entry, "Should have output_tokens"
        assert "input_tokens_per_sec" in call_entry, "Should have input_tokens_per_sec"
        assert "output_tokens_per_sec" in call_entry, "Should have output_tokens_per_sec"
        print("✓ Discord bot LLM call tracked correctly")
        
        # Test 2: Verify token stats
        print("\nTest 2: Verify token stats aggregation")
        assert "test-model" in bot.current_query_token_stats, "Should have stats for test-model"
        stats = bot.current_query_token_stats["test-model"]
        assert stats["total_calls"] == 1, "Should have 1 call"
        assert stats["total_input_tokens"] > 0, "Should have input tokens"
        assert stats["total_output_tokens"] > 0, "Should have output tokens"
        print("✓ Token stats aggregated correctly")
    
    print("\n" + "="*60)
    print("✓ Discord bot _call_llm tracking test passed!")
    print("="*60)


def test_query_log_filename_edge_cases():
    """
    Test that query log filenames handle edge cases properly.
    """
    print("\nTesting query log filename edge cases...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.ReActAgent') as MockAgent, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        # Create mock agent
        mock_agent_instance = Mock()
        mock_agent_instance.get_tracking_data.return_value = {
            "call_sequence": [],
            "token_stats": {}
        }
        mock_agent_instance.reset_tracking = Mock()
        MockAgent.return_value = mock_agent_instance
        
        # Create bot instance with temporary directory
        bot = ReActDiscordBot("test_token", "test_api_key")
        bot.QUERY_LOGS_DIR = Path(tmpdir) / "query_logs"
        bot.QUERY_LOGS_DIR.mkdir(exist_ok=True)
        
        # Test 1: Empty username
        print("\nTest 1: Test empty username handling")
        bot._save_query_log("123", "Test?", "Answer", "")
        log_files = list(bot.QUERY_LOGS_DIR.glob("*.json"))
        assert len(log_files) == 1, "Should create one log file"
        assert log_files[0].name.startswith("unknown_user_"), f"Empty username should use 'unknown_user', got: {log_files[0].name}"
        print(f"✓ Empty username handled correctly: {log_files[0].name}")
        
        # Clean up
        log_files[0].unlink()
        
        # Test 2: Username with special characters
        print("\nTest 2: Test username with special characters")
        bot._save_query_log("124", "Test?", "Answer", "John@Doe#123!")
        log_files = list(bot.QUERY_LOGS_DIR.glob("*.json"))
        assert len(log_files) == 1, "Should create one log file"
        # Should sanitize special chars to underscores
        assert "John_Doe_123_" in log_files[0].name, f"Should sanitize special chars, got: {log_files[0].name}"
        print(f"✓ Special characters sanitized correctly: {log_files[0].name}")
        
        # Clean up
        log_files[0].unlink()
        
        # Test 3: Very long username
        print("\nTest 3: Test very long username (truncation)")
        long_username = "A" * 100  # 100 characters
        bot._save_query_log("125", "Test?", "Answer", long_username)
        log_files = list(bot.QUERY_LOGS_DIR.glob("*.json"))
        assert len(log_files) == 1, "Should create one log file"
        # Username should be truncated to 50 chars
        filename_parts = log_files[0].stem.split('_')
        username_part = filename_parts[0]
        assert len(username_part) <= 50, f"Username should be truncated to 50 chars, got {len(username_part)}"
        print(f"✓ Long username truncated correctly: {len(username_part)} characters")
        
        # Clean up
        log_files[0].unlink()
        
        # Test 4: Username with only special characters
        print("\nTest 4: Test username with only special characters")
        bot._save_query_log("126", "Test?", "Answer", "!@#$%^&*()")
        log_files = list(bot.QUERY_LOGS_DIR.glob("*.json"))
        assert len(log_files) == 1, "Should create one log file"
        assert log_files[0].name.startswith("unknown_user_"), f"Special-only username should use 'unknown_user', got: {log_files[0].name}"
        print(f"✓ Special-only username handled correctly: {log_files[0].name}")
    
    print("\n" + "="*60)
    print("✓ Query log filename edge cases test passed!")
    print("="*60)


if __name__ == "__main__":
    print("Discord Bot Logging Enhancements Tests")
    print("="*60)
    
    # Test 1: ReAct agent tracking
    test_react_agent_tracking()
    
    # Test 2: LLM call tracking
    test_llm_call_tracking()
    
    # Test 3: Tool call tracking
    test_tool_call_tracking()
    
    # Test 4: Discord bot query logging
    test_discord_bot_query_logging()
    
    # Test 5: Discord bot LLM call tracking
    test_llm_call_tracking_in_discord_bot()
    
    # Test 6: Query log filename edge cases
    test_query_log_filename_edge_cases()
    
    print("\n" + "="*60)
    print("✓ ALL LOGGING ENHANCEMENT TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Summary:")
    print("The logging enhancements now:")
    print("1. Track LLM calls with input/output tokens and tokens/sec")
    print("2. Track tool calls with inputs and outputs")
    print("3. Maintain ordered sequence of all calls")
    print("4. Aggregate token statistics by model")
    print("5. Save query logs to query_logs directory as JSON")
    print("6. Include complete call sequence in logs")
    print("7. Properly reset tracking for each new query")
    print("="*60)
