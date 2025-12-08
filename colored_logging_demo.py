#!/usr/bin/env python3
"""
Demo script to showcase the colored logging feature.
This script demonstrates how the colored output appears when the bot is running.
"""

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

print("\n" + "="*80)
print("Discord Bot Colored Logging Demo")
print("="*80)

print("\nThis is how the logs will appear when the bot is running:")
print()

# User Query (Green)
print(f"{Fore.GREEN}[USER QUERY] Alice: What is the weather in San Francisco today?{Style.RESET_ALL}")
print("2025-12-08 10:30:15 - discord_bot - INFO - User query received from Alice: What is the weather in San Francisco today?")
print()

# Tool Calls (Yellow)
print(f"{Fore.YELLOW}[TOOL CALL] duckduckgo_search: weather San Francisco today{Style.RESET_ALL}")
print("2025-12-08 10:30:16 - react_agent - INFO - Tool used: duckduckgo_search, Arguments: weather San Francisco today")
print("2025-12-08 10:30:17 - react_agent - INFO - Tool duckduckgo_search completed successfully, returned 5 results")
print()

print(f"{Fore.YELLOW}[TOOL CALL] scrape_url: https://weather.com/weather/today/l/San+Francisco+CA{Style.RESET_ALL}")
print("2025-12-08 10:30:18 - react_agent - INFO - Tool used: scrape_url, Arguments: https://weather.com/weather/today/l/San+Francisco+CA")
print("2025-12-08 10:30:19 - react_agent - INFO - Tool scrape_url completed successfully, scraped 2350 characters")
print()

# Final Response (Red)
print(f"{Fore.RED}[FINAL RESPONSE] Based on current weather data, San Francisco has partly cloudy skies with a temperature of 62°F...{Style.RESET_ALL}")
print()

# Evaluation Logging (Cyan)
print(f"{Fore.CYAN}[EVAL] Question logged: What is the weather in San Francisco today?...{Style.RESET_ALL}")
print("2025-12-08 10:30:25 - discord_bot - INFO - Eval question logged from Alice: What is the weather in San Francisco today?")
print()

print(f"{Fore.CYAN}[EVAL] Answer accepted for question: What is the weather in San Francisco today?...{Style.RESET_ALL}")
print("2025-12-08 10:30:30 - discord_bot - INFO - Accepted answer logged for message 123456789")
print()

print("="*80)
print("Color Legend:")
print(f"  {Fore.GREEN}GREEN{Style.RESET_ALL}  = User queries from Discord")
print(f"  {Fore.YELLOW}YELLOW{Style.RESET_ALL} = Tool calls (search, scrape, etc.)")
print(f"  {Fore.RED}RED{Style.RESET_ALL}    = Final responses sent to user")
print(f"  {Fore.CYAN}CYAN{Style.RESET_ALL}   = Evaluation logging (reactions)")
print("="*80)

print("\nReaction-based Evaluation:")
print("  🧪 (test tube)  = React to a user message to log it as an eval question")
print("  ✅ (check mark) = React to a bot response to mark it as the accepted answer")
print()
print("Eval data is stored in: data/eval_qs.jsonl")
print("="*80)
