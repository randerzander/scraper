# scraper

A lean ReAct (Reasoning + Acting) agent that can search the web and scrape URLs. Optional Discord bot wrapper included.

## Quick Start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Set your OpenRouter API key
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

3) Run
```bash
python react_agent.py    # demo
python example.py        # examples
```

## Discord Bot

Requirements:
- `.bot_token` file with your Discord bot token
- `OPENROUTER_API_KEY` environment variable

Run:
```bash
python discord_bot.py
```

Configuration (`config.yaml`):
```yaml
auto_restart: true  # auto-restart on file changes
base_url: "https://openrouter.ai/api/v1/chat/completions"
default_model: "amazon/nova-2-lite-v1:free"
image_caption_model: "nvidia/nemotron-nano-12b-v2-vl:free"
```

Features:
- Auto-restart on code changes (configurable)
- Reply chain context support
- Image captioning with vision models
- Reaction-based eval logging (🧪 to log question, ✅ to mark accepted answer)

## Usage in Code

```python
from react_agent import ReActAgent
import os

agent = ReActAgent(os.getenv("OPENROUTER_API_KEY"))
answer = agent.run("Latest AI news?", verbose=True)
print(answer)
```

## Available Tools

- `duckduckgo_search(query)` - Search the web
- `scrape_url(url)` - Scrape and parse HTML to markdown
- `read_file(filepath)` - Read files from current directory (1MB limit, path traversal protected)

## Requirements

- Python 3.8+
- OpenRouter API key (free tier available)
- Internet connection for search/scrape

## Tests

```bash
python tests/test_react_agent.py
python tests/test_discord_bot.py
```

## License

MIT
