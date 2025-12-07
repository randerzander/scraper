# scraper

A simple ReAct (Reasoning + Acting) agent that can search the web and scrape URLs to answer questions.

## Features

- **ReAct Agent**: Uses the ReAct pattern for reasoning and tool selection
- **DuckDuckGo Search**: Search the web for information
- **URL Scraping**: Parse HTML content into readable markdown using [pyreadability](https://github.com/randerzander/pyreadability)
- **OpenRouter Integration**: Uses OpenRouter API with tngtech/deepseek-r1t2-chimera:free model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/randerzander/scraper.git
cd scraper
```

2. Install dependencies:

   **Option A: Using uv (recommended - faster installation)**
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Create a virtual environment named "scraper"
   uv venv scraper
   
   # Activate the virtual environment
   source scraper/bin/activate  # On macOS/Linux
   # Or on Windows: scraper\Scripts\activate
   
   # Install dependencies with uv
   uv pip install -r requirements.txt
   ```

   **Option B: Using pip (traditional method)**
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Or create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Basic Usage

Run the example script:
```bash
python example.py
```

Or run the default demo:
```bash
python react_agent.py
```

### Using in Your Own Code

```python
from react_agent import ReActAgent
import os

# Initialize the agent
api_key = os.getenv("OPENROUTER_API_KEY")
agent = ReActAgent(api_key)

# Ask a question
answer = agent.run("What is the latest news about artificial intelligence?", verbose=True)
print(answer)
```

### Discord Bot

Run the agent as a Discord bot that responds to mentions:

```bash
python discord_bot.py
```

**Requirements:**
1. Create a `token.txt` file in the current directory with your Discord bot token
2. Set the `OPENROUTER_API_KEY` environment variable

**How to get a Discord bot token:**
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application (or select an existing one)
3. Go to the "Bot" section and click "Reset Token"
4. Save the token to `token.txt` in the project directory
5. Enable "Message Content Intent" in the Bot settings
6. Use the OAuth2 URL Generator to create an invite link with:
   - Scopes: `bot`
   - Bot Permissions: `Read Messages/View Channels`, `Send Messages`, `Read Message History`

**Usage:**
- Mention the bot in a Discord message followed by your question
- Example: `@YourBot What is the latest news about artificial intelligence?`
- The bot will use the ReAct agent to search and scrape information to answer your question

### Testing

Run the test suite to verify the implementation:
```bash
python test_react_agent.py
```

This will test the agent's parsing logic, tool execution, and reasoning loop without requiring API access.

### Testing uv Virtual Environment Setup

To verify that the uv virtual environment setup works correctly:
```bash
python test_uv_venv.py
```

This test script will:
1. Verify uv is installed
2. Create a virtual environment named "scraper"
3. Install dependencies using uv pip
4. Run the test suite in the virtual environment
5. Clean up the test environment

## How It Works

The ReAct agent follows a thought-action-observation loop:

1. **Thought**: The agent reasons about what information it needs
2. **Action**: The agent selects a tool to use (search or scrape)
3. **Action Input**: The agent provides input for the selected tool
4. **Observation**: The agent receives results from the tool
5. Repeat until the agent has enough information to provide a **Final Answer**

## Available Tools

- `duckduckgo_search`: Search the web using DuckDuckGo
  - Input: A search query string
  - Output: List of search results with titles, URLs, and snippets

- `scrape_url`: Scrape and parse HTML content from a URL
  - Input: A URL to scrape
  - Output: Markdown-formatted content extracted from the page

## Example

```
Question: What is the latest news about artificial intelligence?

Iteration 1:
Thought: I need to search for the latest news about artificial intelligence.
Action: duckduckgo_search
Action Input: latest artificial intelligence news
Observation: [Search results with recent AI news articles]

Iteration 2:
Thought: Based on the search results, I can see recent developments in AI...
Final Answer: [Comprehensive answer based on search results]
```

## Repository Structure

- `react_agent.py` - Main implementation of the ReAct agent
- `example.py` - Example usage with multiple scenarios
- `discord_bot.py` - Discord bot wrapper for the ReAct agent
- `test_react_agent.py` - Test suite for the agent
- `test_uv_venv.py` - Test script for uv virtual environment setup
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment configuration
- `README.md` - This file

## Requirements

- Python 3.8+
- OpenRouter API key (free tier available)
- Internet connection for web search and URL scraping

## Implementation Details

The ReAct agent uses a simple but effective approach:

1. **Prompt Engineering**: Uses clear instructions to guide the LLM through the ReAct pattern
2. **Regex Parsing**: Extracts structured information from LLM responses
3. **Tool Integration**: Seamlessly integrates DuckDuckGo search and URL scraping
4. **Iteration Control**: Prevents infinite loops with configurable max iterations
5. **Error Handling**: Gracefully handles network errors and parsing failures

The agent does not use any heavyweight frameworks like LangChain or LlamaIndex, making it:
- Easy to understand and modify
- Minimal dependencies
- Transparent reasoning process
- Fully customizable

## License

MIT