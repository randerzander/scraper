#!/usr/bin/env python3
"""
Test case for retriever tool with read_url integration.
Tests fetching content from a URL and indexing it in the vector database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.read_url import read_url
from tools import retriever


def test_read_url_and_index():
    """Test reading a URL and indexing its content in the vector database."""
    
    # Use a simple, stable URL for testing
    test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    
    print(f"\n{'='*60}")
    print(f"Testing read_url and retriever integration")
    print(f"{'='*60}\n")
    
    print(f"Fetching content from: {test_url}")
    
    # Read URL - this should automatically write to vector DB
    content = read_url(test_url)
    
    print(f"\nContent fetched: {len(content)} characters")
    print(f"First 200 chars: {content[:200]}...")
    
    # Test search functionality
    print(f"\n{'='*60}")
    print("Testing vector database search")
    print(f"{'='*60}\n")
    
    test_queries = [
        "What is Python?",
        "Python programming language features",
        "Who created Python?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        results = retriever.search(query, limit=2)
        print(results)
    
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_read_url_and_index()
