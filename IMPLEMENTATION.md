# ReAct Agent Implementation Summary

## Overview

This repository implements a simple ReAct (Reasoning + Acting) agent using plain prompting without any frameworks like LangChain or LlamaIndex. The agent can search the web and scrape URLs to answer questions intelligently.

## What is ReAct?

ReAct is a prompting paradigm that combines **Reasoning** and **Acting**:

1. **Thought**: The agent reasons about what information it needs
2. **Action**: The agent selects and uses a tool
3. **Observation**: The agent receives and analyzes the results
4. Repeat until the agent has enough information to provide a **Final Answer**

This creates a transparent reasoning process where you can see the agent's "thinking" at each step.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     ReAct Agent                         │
│                                                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │   Prompt     │───▶│  LLM (via    │                   │
│  │   Builder    │    │  OpenRouter) │                   │
│  └──────────────┘    └──────┬───────┘                   │
│                             │                            │
│                             ▼                            │
│                      ┌──────────────┐                    │
│                      │   Response   │                    │
│                      │   Parser     │                    │
│                      └──────┬───────┘                    │
│                             │                            │
│                    ┌────────┴────────┐                   │
│                    │                 │                   │
│              ┌─────▼─────┐    ┌─────▼──────┐            │
│              │DuckDuckGo │    │  Scrape    │            │
│              │  Search   │    │    URL     │            │
│              └───────────┘    └────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. No Framework Dependencies
- Direct API calls to OpenRouter
- Simple regex-based parsing
- No complex abstractions
- Easy to understand and modify

### 2. Two Powerful Tools

#### DuckDuckGo Search
- Searches the web for information
- Returns formatted results with titles, URLs, and snippets
- Handles errors gracefully

#### URL Scraper
- Uses pyreadability to extract main content
- Converts HTML to clean markdown
- Filters out navigation, ads, and other noise
- Includes proper User-Agent handling

### 3. Transparent Reasoning
- See every thought and action the agent takes
- Understand why it chose specific tools
- Debug issues easily with verbose output

### 4. Robust Error Handling
- Network timeouts
- Malformed responses
- Missing tools
- API errors

## Implementation Details

### Prompt Engineering

The agent uses a carefully crafted prompt that:
1. Clearly explains the ReAct format
2. Lists available tools and their purposes
3. Provides examples of the expected format
4. Guides the LLM to produce structured output

### Response Parsing

Uses regular expressions to extract:
- `Thought:` - The agent's reasoning
- `Action:` - The tool to use
- `Action Input:` - The parameters for the tool
- `Final Answer:` - The final response

### Tool Execution

Tools are stored in a dictionary with:
- Function reference
- Description for the LLM
- Parameter names
- Execution wrapper for error handling

### Iteration Control

- Configurable max iterations (default: 5)
- Prevents infinite loops
- Allows complex multi-step reasoning

## Usage Examples

### Simple Search

```python
from react_agent import ReActAgent
import os

agent = ReActAgent(os.getenv("OPENROUTER_API_KEY"))
answer = agent.run("What is the weather like in San Francisco?")
print(answer)
```

### Search + Scrape

```python
# The agent will automatically decide to search first,
# then scrape relevant URLs
answer = agent.run(
    "What are the new features in Python 3.12? "
    "Find and read the official release notes."
)
print(answer)
```

### Custom Configuration

```python
agent = ReActAgent(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="tngtech/deepseek-r1t2-chimera:free"
)

# More iterations for complex tasks
answer = agent.run(
    question="Compare the features of Python and JavaScript",
    max_iterations=10,
    verbose=True
)
```

## Testing

The repository includes a comprehensive test suite that validates:
- Response parsing logic
- Tool execution
- Agent loop
- Error handling

Run tests with:
```bash
python test_react_agent.py
```

All tests are designed to work without API access or internet connectivity.

## Model Choice

The agent uses `tngtech/deepseek-r1t2-chimera:free` by default because:
- Free tier available on OpenRouter
- Good reasoning capabilities
- Fast response times
- Supports the ReAct format well

You can easily switch to other models by passing a different model name to the constructor.

## Limitations and Future Improvements

### Current Limitations
1. No memory between runs
2. Fixed set of tools
3. No parallel tool execution
4. Limited context window awareness

### Possible Improvements
1. Add conversation memory
2. Support for custom tools
3. Streaming responses
4. Better context management
5. Tool result caching
6. Multi-agent collaboration

## Contributing

This is a minimal implementation designed for clarity and simplicity. If you want to extend it:

1. Add new tools by updating the `self.tools` dictionary
2. Improve parsing by modifying `_parse_response()`
3. Add new prompting strategies in `_create_prompt()`
4. Implement better error recovery in `_execute_action()`

## License

MIT

## Credits

- ReAct pattern: [Yao et al., 2022](https://arxiv.org/abs/2210.03629)
- pyreadability: https://github.com/randerzander/pyreadability
- OpenRouter: https://openrouter.ai/
