#!/usr/bin/env python3
"""
Test script for the ReAct agent functionality.
This script tests the agent's parsing and reasoning logic without requiring API access.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from react_agent import ReActAgent

# Mock LLM response for testing
class MockReActAgent(ReActAgent):
    """Mock agent for testing without API access."""
    
    def __init__(self):
        # Initialize without API key
        self.api_key = "mock_key"
        self.model = "mock_model"
        self.api_url = "mock_url"
        self.tools = {
            "duckduckgo_search": {
                "function": lambda q: [{"title": "Test Result", "href": "http://test.com", "body": "Test content"}],
                "description": "Search the web using DuckDuckGo. Input should be a search query string.",
                "parameters": ["query"]
            },
            "scrape_url": {
                "function": lambda url: "# Test Article\n\nThis is test content from the URL.",
                "description": "Scrape and parse HTML content from a URL into markdown format. Input should be a URL.",
                "parameters": ["url"]
            }
        }
        self.test_responses = []
        self.response_index = 0
    
    def set_test_responses(self, responses):
        """Set predefined responses for testing."""
        self.test_responses = responses
        self.response_index = 0
    
    def _call_llm(self, prompt):
        """Return mock response instead of calling actual LLM."""
        if self.response_index < len(self.test_responses):
            response = self.test_responses[self.response_index]
            self.response_index += 1
            return response
        return "Final Answer: Test completed"


def test_parse_response():
    """Test the response parsing logic."""
    agent = MockReActAgent()
    
    print("Testing response parsing...")
    print("="*60)
    
    # Test 1: Parse thought, action, and action input
    response1 = """Thought: I need to search for information about Python.
Action: duckduckgo_search
Action Input: Python programming language"""
    
    parsed = agent._parse_response(response1)
    print("\nTest 1 - Parse complete ReAct response:")
    print(f"Response: {response1}")
    print(f"Parsed: {parsed}")
    assert parsed['thought'] == "I need to search for information about Python."
    assert parsed['action'] == "duckduckgo_search"
    assert parsed['action_input'] == "Python programming language"
    print("✓ Test 1 passed")
    
    # Test 2: Parse final answer
    response2 = """Final Answer: Python is a high-level programming language known for its simplicity and readability."""
    
    parsed = agent._parse_response(response2)
    print("\nTest 2 - Parse final answer:")
    print(f"Response: {response2}")
    print(f"Parsed: {parsed}")
    assert parsed['final_answer'] is not None
    assert "Python is a high-level" in parsed['final_answer']
    print("✓ Test 2 passed")
    
    print("\n✓ All parsing tests passed!")


def test_tool_execution():
    """Test the tool execution logic."""
    agent = MockReActAgent()
    
    print("\nTesting tool execution...")
    print("="*60)
    
    # Test search tool
    result = agent._execute_action("duckduckgo_search", "test query")
    print("\nTest 3 - Execute search tool:")
    print(f"Result: {result[:100]}...")
    assert "Test Result" in result
    print("✓ Test 3 passed")
    
    # Test scrape tool
    result = agent._execute_action("scrape_url", "http://example.com")
    print("\nTest 4 - Execute scrape tool:")
    print(f"Result: {result[:100]}...")
    assert "Test Article" in result
    print("✓ Test 4 passed")
    
    # Test unknown tool
    result = agent._execute_action("unknown_tool", "input")
    print("\nTest 5 - Unknown tool error handling:")
    print(f"Result: {result}")
    assert "Error" in result
    print("✓ Test 5 passed")
    
    print("\n✓ All tool execution tests passed!")


def test_agent_loop():
    """Test the full agent loop."""
    agent = MockReActAgent()
    
    print("\nTesting agent loop...")
    print("="*60)
    
    # Set up mock responses
    agent.set_test_responses([
        """Thought: I need to search for information.
Action: duckduckgo_search
Action Input: test query""",
        """Thought: Based on the search results, I can now provide an answer.
Final Answer: Here is the answer based on the search results."""
    ])
    
    result = agent.run("What is a test?", max_iterations=3, verbose=False)
    print("\nTest 6 - Full agent loop:")
    print(f"Result: {result}")
    assert "answer based on the search results" in result
    print("✓ Test 6 passed")
    
    print("\n✓ All agent loop tests passed!")


if __name__ == "__main__":
    print("ReAct Agent Tests")
    print("="*60)
    
    test_parse_response()
    test_tool_execution()
    test_agent_loop()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Note: To use the actual agent with real API calls:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Run: python react_agent.py")
    print("="*60)
