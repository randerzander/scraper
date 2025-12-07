#!/usr/bin/env python3
"""
Example usage of the ReAct agent.
This demonstrates how to use the agent in your own code.
"""

import os
from react_agent import ReActAgent


def main():
    """Example usage of the ReAct agent."""
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("="*80)
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("="*80)
        print("\nTo use this agent, you need an OpenRouter API key:")
        print("1. Sign up at https://openrouter.ai/")
        print("2. Get your API key from https://openrouter.ai/keys")
        print("3. Set the environment variable:")
        print("   export OPENROUTER_API_KEY=your_api_key_here")
        print("\nOr create a .env file:")
        print("   cp .env.example .env")
        print("   # Edit .env and add your API key")
        print("="*80)
        return
    
    # Create the agent
    print("Initializing ReAct Agent...")
    agent = ReActAgent(api_key)
    
    # Example 1: Simple search-based question
    print("\n" + "="*80)
    print("EXAMPLE 1: Search-based question")
    print("="*80)
    question1 = "What are the latest developments in quantum computing?"
    print(f"\nQuestion: {question1}")
    print("\n" + "-"*80)
    
    answer1 = agent.run(question1, max_iterations=5, verbose=True)
    
    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print("="*80)
    print(answer1)
    
    # Example 2: Question requiring URL scraping
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Question requiring web scraping")
    print("="*80)
    question2 = "Find the latest Python release notes and summarize the main new features."
    print(f"\nQuestion: {question2}")
    print("\n" + "-"*80)
    
    answer2 = agent.run(question2, max_iterations=7, verbose=True)
    
    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print("="*80)
    print(answer2)


if __name__ == "__main__":
    main()
