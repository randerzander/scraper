# Discord Bot Logging Enhancements

This document describes the logging enhancements added to the Discord bot for tracking LLM calls and tool usage.

## Overview

The logging system tracks all LLM calls and tool invocations during query processing, providing detailed metrics and saving complete query logs to JSON files.

## Features

### 1. LLM Call Tracking

Each LLM call is tracked with the following information:
- **Model**: The model used (e.g., `x-ai/grok-4.1-fast`)
- **Input/Output Tokens**: Number of tokens in the prompt and response
- **Tokens per Second**: Speed metrics for both input and output processing
- **Response Time**: Total time taken for the API call
- **Content Preview**: First 500 characters of input and output

### 2. Tool Call Tracking

Each tool invocation is tracked with:
- **Tool Name**: The tool that was called (e.g., `duckduckgo_search`)
- **Input**: The input provided to the tool
- **Output**: The result from the tool execution
- **Execution Time**: Time taken to execute the tool

### 3. Aggregated Token Statistics

Token usage is aggregated by model across all calls in a query:
- **Total Input Tokens**: Sum of all input tokens
- **Total Output Tokens**: Sum of all output tokens
- **Total Calls**: Number of calls to each model

### 4. Query Logs

After each query is processed, a JSON log file is saved to `data/query_logs/` with:
- **Message ID**: Discord message ID
- **Timestamp**: When the query was processed
- **User Query**: The user's question
- **Final Response**: The bot's answer
- **Call Sequence**: Ordered list of all LLM and tool calls
- **Token Stats by Model**: Aggregated statistics for each model used

## Log File Structure

```json
{
  "message_id": "123456789",
  "timestamp": "2025-12-08T04:00:00",
  "user_query": "What is the weather today?",
  "final_response": "Based on current data, the weather is...",
  "call_sequence": [
    {
      "type": "llm_call",
      "model": "nvidia/nemotron-nano-12b-v2-vl:free",
      "timestamp": 1733630400.0,
      "input": "Analyze the following message...",
      "output": "{\"is_sarcastic\": false, \"confidence\": \"high\"}",
      "input_tokens": 25,
      "output_tokens": 10,
      "response_time_seconds": 0.5,
      "input_tokens_per_sec": 50.0,
      "output_tokens_per_sec": 20.0
    },
    {
      "type": "tool_call",
      "tool_name": "duckduckgo_search",
      "timestamp": 1733630401.0,
      "input": "weather today",
      "output": "1. Weather.com - Current weather...",
      "execution_time_seconds": 1.2
    },
    {
      "type": "llm_call",
      "model": "x-ai/grok-4.1-fast",
      "timestamp": 1733630402.0,
      "input": "[Current date and time: 2025-12-08 04:00:00]...",
      "output": "Based on the search results...",
      "input_tokens": 150,
      "output_tokens": 85,
      "response_time_seconds": 2.3,
      "input_tokens_per_sec": 65.2,
      "output_tokens_per_sec": 37.0
    }
  ],
  "token_stats_by_model": {
    "nvidia/nemotron-nano-12b-v2-vl:free": {
      "total_input_tokens": 25,
      "total_output_tokens": 10,
      "total_calls": 1
    },
    "x-ai/grok-4.1-fast": {
      "total_input_tokens": 150,
      "total_output_tokens": 85,
      "total_calls": 1
    }
  }
}
```

## Implementation Details

### ReActAgent Changes

The `ReActAgent` class now includes:
- `call_sequence`: List to store all calls in order
- `token_stats`: Dictionary to aggregate token usage by model
- `reset_tracking()`: Method to clear tracking for a new query
- `get_tracking_data()`: Method to retrieve all tracking data

The `_call_llm()` method automatically tracks each call with timing and token information.

The `_execute_action()` method tracks tool calls with inputs, outputs, and execution time.

### Discord Bot Changes

The `ReActDiscordBot` class now includes:
- `current_query_log`: List to track bot-level LLM calls
- `current_query_token_stats`: Dictionary for bot-level token aggregation
- `_save_query_log()`: Method to save the complete query log
- `_reset_query_tracking()`: Method to reset tracking for a new query

The `on_message` handler:
1. Resets tracking at the start of each query
2. Processes the query normally
3. Saves the complete log after sending the response

## Console Output

During query processing, the bot displays:
- User queries in green
- Tool calls in yellow
- Final responses in red
- Query log save confirmation in cyan
- Token statistics summary by model

Example:
```
[USER QUERY] John: What's the weather?
[TOOL CALL] duckduckgo_search: weather today
[FINAL RESPONSE] Based on current data...
[QUERY LOG] Saved to query_123456_20251208_040000.json
[TOKEN STATS]
  Model: nvidia/nemotron-nano-12b-v2-vl:free
    Calls: 1
    Input tokens: 25
    Output tokens: 10
    Total tokens: 35
  Model: x-ai/grok-4.1-fast
    Calls: 2
    Input tokens: 200
    Output tokens: 120
    Total tokens: 320
```

## Testing

The logging functionality is tested in:
- `tests/test_logging_enhancements.py`: Unit tests for tracking features
- `tests/test_integration_logging.py`: Integration test demonstrating complete workflow

Run tests with:
```bash
python tests/test_logging_enhancements.py
python tests/test_integration_logging.py
```

## Notes

- Log files are automatically saved to `data/query_logs/`
- Log files are ignored by git (configured in `.gitignore`)
- Content is truncated to 500 characters in logs to keep file sizes manageable
- Token counts are estimated using `CHARS_PER_TOKEN = 4.5` constant
- Actual token usage may vary from estimates
