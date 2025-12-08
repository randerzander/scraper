#!/usr/bin/env python3
"""
Example usage of the create_github_issue tool.
This demonstrates how the ReAct agent can be used to create GitHub issues.
"""

import os
from react_agent import ReActAgent

def main():
    """Example of using the ReAct agent with the GitHub issue creation tool."""
    
    # Check for required API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not openrouter_key:
        print("="*80)
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("="*80)
        print("\nTo use this example, you need:")
        print("1. OpenRouter API key from https://openrouter.ai/keys")
        print("2. GitHub token from https://github.com/settings/tokens")
        print("\nSet them with:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        print("  export GITHUB_TOKEN=your_token_here")
        print("="*80)
        return
    
    if not github_token:
        print("="*80)
        print("WARNING: GITHUB_TOKEN environment variable not set")
        print("="*80)
        print("\nThe GitHub issue creation tool will not work without it.")
        print("Get a token from: https://github.com/settings/tokens")
        print("The token needs 'repo' scope for creating issues.")
        print("\nSet it with:")
        print("  export GITHUB_TOKEN=your_token_here")
        print("="*80)
        print("\nContinuing with other tools...\n")
    
    # Create the agent
    print("Initializing ReAct Agent...")
    agent = ReActAgent(openrouter_key)
    
    # Example: Using the agent to create a GitHub issue
    print("\n" + "="*80)
    print("EXAMPLE: Creating a GitHub issue using the ReAct agent")
    print("="*80)
    
    question = """Create a GitHub issue on the randerzander/scraper repository with the title 
    "Add support for custom rate limiting" and description "Implement configurable rate limiting 
    for API calls to avoid hitting rate limits on free tier services." """
    
    print(f"\nQuestion: {question}")
    print("\n" + "-"*80)
    
    if github_token:
        answer = agent.run(question, max_iterations=3, verbose=True)
        print("\n" + "="*80)
        print("FINAL ANSWER:")
        print("="*80)
        print(answer)
    else:
        print("\nSkipping GitHub issue creation due to missing token.")
        print("Example JSON input for the tool:")
        print("""
{
    "title": "Add support for custom rate limiting",
    "description": "Implement configurable rate limiting for API calls to avoid hitting rate limits on free tier services."
}
        """)


if __name__ == "__main__":
    main()
