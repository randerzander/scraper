#!/usr/bin/env python3
"""
Discord bot wrapper for the ReAct agent.
This bot reads questions from Discord messages and uses the ReAct agent to answer them.

The bot's token should be in a file named '.bot_token' in the current working directory.
"""

import os
import sys
import asyncio
import discord
import requests
import json
import logging
import time
import yaml
import subprocess
import threading
import fcntl
from datetime import datetime
from pathlib import Path
from react_agent import ReActAgent, two_round_image_caption
from colorama import Fore, Style
from utils import setup_logging, CHARS_PER_TOKEN

# Configure logging with colored formatter
setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "auto_restart": True,
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "default_model": "amazon/nova-2-lite-v1:free",
    "intent_detection_model": "amazon/nova-2-lite-v1:free",
    "image_caption_model": "nvidia/nemotron-nano-12b-v2-vl:free",
    "conciseness_model": "amazon/nova-2-lite-v1:free",
    "tldr_model": "amazon/nova-2-lite-v1:free"
}

def load_config():
    """Load bot configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Config loading failed: {e}, using defaults")
        return DEFAULT_CONFIG.copy()

CONFIG = load_config()
MODEL_CONFIG = CONFIG


class ReActDiscordBot:
    """Discord bot that wraps the ReAct agent."""
    
    # Data directory for evaluation logging
    DATA_DIR = Path("data")
    EVAL_FILE = DATA_DIR / "eval_qs.jsonl"
    QUERY_LOGS_DIR = DATA_DIR / "query_logs"
    
    async def _add_reaction(self, message, emoji: str):
        """Add a reaction to a message."""
        await message.add_reaction(emoji)
    
    async def _remove_reaction(self, message, emoji: str):
        """Remove a reaction from a message."""
        await message.remove_reaction(emoji, self.client.user)
    
    def _extract_image_urls(self, message) -> list[str]:
        """Extract image URLs from message attachments."""
        return [att.url for att in message.attachments 
                if att.content_type and att.content_type.startswith('image/')]
    
    def _remove_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        text = text.replace(f"<@{self.client.user.id}>", "").strip()
        return text.replace(f"<@!{self.client.user.id}>", "").strip()
    
    def __init__(self, token: str, api_key: str):
        """
        Initialize the Discord bot with ReAct agent.
        
        Args:
            token: Discord bot token
            api_key: OpenRouter API key for the ReAct agent
        """
        self.token = token
        self.api_key = api_key
        # Get base_url from config, default to OpenRouter
        base_url = CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        self.agent = ReActAgent(api_key, base_url=base_url)
        
        # Ensure data directory exists
        self.DATA_DIR.mkdir(exist_ok=True)
        self.QUERY_LOGS_DIR.mkdir(exist_ok=True)
        
        # Initialize tracking for current query
        self.current_query_log = []
        self.current_query_token_stats = {}
        
        # Set up Discord intents
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.reactions = True  # Required to read reactions
        
        self.client = discord.Client(intents=intents)
        
        # Register event handlers
        @self.client.event
        async def on_ready():
            print(f"Bot logged in as {self.client.user}")
            print(f"Bot is ready to answer questions!")
            print(f"Invite link: https://discord.com/api/oauth2/authorize?client_id={self.client.user.id}&permissions=2048&scope=bot")
        
        async def get_reply_chain(message) -> tuple[str, list[str]]:
            """
            Get the full reply chain context for a message.
            Follows the chain of replied-to messages up to the root.
            Also collects image attachments from the reply chain.
            
            Args:
                message: The Discord message to get reply chain for
                
            Returns:
                A tuple of (formatted string with reply chain context, list of image URLs from replies)
            """
            chain = []
            reply_image_urls = []
            current_msg = message
            max_chain_depth = 10  # Limit chain depth to prevent performance issues
            depth = 0
            
            # Follow the reply chain backwards
            while current_msg.reference and depth < max_chain_depth:
                # Get the referenced message (may need to fetch if not cached)
                ref_msg = getattr(current_msg, 'referenced_message', None)
                if not ref_msg and current_msg.reference.message_id:
                    # Fetch the message if it's not in cache
                    ref_msg = await current_msg.channel.fetch_message(current_msg.reference.message_id)
                
                if not ref_msg:
                    break
                
                # Check for image attachments in the replied message
                img_urls = self._extract_image_urls(ref_msg)
                reply_image_urls.extend(img_urls)
                for url in img_urls:
                    logger.info(f"Found image attachment in reply chain: {url}")
                
                # Format the message content
                author_name = ref_msg.author.display_name
                content = self._remove_bot_mention(ref_msg.content)
                
                # Add to chain (we're building it backwards, will reverse later)
                if content:
                    chain.append(f"{author_name}: {content}")
                
                # Move to the next message in the chain
                current_msg = ref_msg
                depth += 1
            
            # Reverse to get chronological order (oldest first)
            chain.reverse()
            # Also reverse image URLs to match chronological order (oldest first)
            reply_image_urls.reverse()
            
            reply_text = ""
            if chain:
                reply_text = "Previous conversation context:\n" + "\n".join(chain) + "\n\n"
            
            return reply_text, reply_image_urls
        
        @self.client.event
        async def on_message(message):
            # Don't respond to the bot's own messages
            if message.author == self.client.user:
                return
            
            # Only respond to messages that mention the bot
            if self.client.user.mentioned_in(message):
                question = self._remove_bot_mention(message.content)
                
                # Check for image attachments
                image_urls = self._extract_image_urls(message)
                for url in image_urls:
                    logger.info(f"Found image attachment: {url}")
                
                # If there are images but no question, provide a default question
                if not question and image_urls:
                    question = "What do you see in this image?"
                
                await self._add_reaction(message, "⏳")
                
                # Log the user query in green
                print(f"{Fore.GREEN}[USER QUERY] {message.author.display_name}: {question}{Style.RESET_ALL}")
                logger.info(f"User query received from {message.author.display_name}: {question}")
                
                # Reset tracking for new query
                self._reset_query_tracking()
                
                # Track start time for total response time
                query_start_time = time.time()
                
                # Get reply chain context if this is a reply (also gets images from reply chain)
                reply_context, reply_image_urls = await get_reply_chain(message)
                
                # Combine images from current message and reply chain
                if reply_image_urls:
                    image_urls.extend(reply_image_urls)
                    logger.info(f"Added {len(reply_image_urls)} image(s) from reply chain")
                
                try:
                    # Build image context if images are present
                    image_context = ""
                    if image_urls:
                        details = "\n".join([f"Image {i} URL: {u}" for i, u in enumerate(image_urls, 1)])
                        image_context = f"\n\n[Images attached: {len(image_urls)} image(s)]\n{details}\n\nYou can use the 'caption_image' tool to analyze these images.\n"
                    
                    # Build question with context
                    question_with_context = f"""[You are a Discord bot named Usefool]
{image_context}
{reply_context}User question: {question}"""
                    
                    # Register channel history tool for this message processing
                    self._register_channel_history_tool(message.channel, message.id)
                    
                    # Register image caption tool if images are present
                    if image_urls:
                        self._register_image_caption_tool(question)
                    
                    # Create iteration callback to alternate hourglass reactions
                    async def update_hourglass(iteration_num):
                        if iteration_num > 0:
                            prev_emoji = "⏳" if iteration_num % 2 == 1 else "⌛"
                            await self._remove_reaction(message, prev_emoji)
                        new_emoji = "⌛" if iteration_num % 2 == 1 else "⏳"
                        await self._add_reaction(message, new_emoji)
                    
                    # Wrapper to make callback thread-safe for asyncio.to_thread
                    def iteration_callback(iteration_num):
                        # Schedule the coroutine in the event loop
                        asyncio.run_coroutine_threadsafe(update_hourglass(iteration_num), self.client.loop)
                    
                    # Use the ReAct agent to answer the question (verbose=False to reduce log noise)
                    # Run in a thread pool to avoid blocking the Discord event loop and heartbeat
                    answer = await asyncio.to_thread(
                        self.agent.run, question_with_context, max_iterations=10, verbose=False, iteration_callback=iteration_callback
                    )
                    
                    # Unregister channel history tool to avoid memory leaks
                    self._unregister_channel_history_tool()
                    
                    # Unregister image caption tool if it was registered
                    if image_urls:
                        self._unregister_image_caption_tool()
                    
                    # Log the final response in red
                    print(f"{Fore.RED}[FINAL RESPONSE] {answer[:100]}...{Style.RESET_ALL}" if len(answer) > 100 else f"{Fore.RED}[FINAL RESPONSE] {answer}{Style.RESET_ALL}")
                    
                    # Remove both hourglass emoji reactions
                    await self._remove_reaction(message, "⏳")
                    await self._remove_reaction(message, "⌛")
                    
                    # Calculate total response time
                    total_response_time = time.time() - query_start_time
                    
                    # Get tracking data from agent and merge with discord bot stats
                    agent_tracking = self.agent.get_tracking_data()
                    merged_token_stats = dict(self.current_query_token_stats)
                    self._merge_token_stats(agent_tracking["token_stats"], merged_token_stats)
                    
                    # Count tool calls by type from agent tracking
                    tool_call_counts = {}
                    for entry in agent_tracking["call_sequence"]:
                        if entry["type"] == "tool_call":
                            tool_name = entry["tool_name"]
                            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                    
                    # Calculate totals across all models
                    total_input_tokens = sum(stats["total_input_tokens"] for stats in merged_token_stats.values())
                    total_output_tokens = sum(stats["total_output_tokens"] for stats in merged_token_stats.values())
                    
                    # Format metadata in small font with call counts per model
                    models_info = [f"{m.split('/')[-1]} ({s['total_calls']}x)" 
                                   for m, s in merged_token_stats.items()]
                    models_used = " • ".join(models_info)
                    
                    # Format tool calls breakdown
                    tool_calls_info = ""
                    if tool_call_counts:
                        tool_calls_info = f" • Tools: {', '.join(f'{t}: {c}' for t, c in tool_call_counts.items())}"
                    
                    metadata = f"\n\n-# *Models: {models_used} • Tokens: {total_input_tokens} in / {total_output_tokens} out{tool_calls_info} • Time: {round(total_response_time)}s*"
                    complete_answer = answer + metadata
                    
                    # Discord has a 2000 character limit for messages - split if needed
                    for chunk in self._split_long_message(complete_answer, 1900):
                        await message.channel.send(chunk)
                    
                    # Save query log after successful response
                    self._save_query_log(str(message.id), question, complete_answer, message.author.display_name)
                
                except Exception as e:
                    # Unregister channel history tool in case of error
                    self._unregister_channel_history_tool()
                    
                    # Unregister image caption tool in case of error
                    self._unregister_image_caption_tool()
                    
                    await self._remove_reaction(message, "⏳")
                    await message.channel.send(f"❌ Error: {str(e)}")
                    print(f"Error processing question: {e}")
        
        @self.client.event
        async def on_reaction_add(reaction, user):
            """Handle reactions added to messages."""
            # Don't process reactions from the bot itself
            if user == self.client.user:
                return
            
            # Handle 🧪 (test tube) reaction - log question to eval file
            if str(reaction.emoji) == "🧪":
                message = reaction.message
                # Only log user messages (not bot responses)
                if message.author != self.client.user:
                    await self._log_eval_question(message, user)
            
            # Handle ✅ (check mark) reaction - log accepted answer
            elif str(reaction.emoji) == "✅":
                message = reaction.message
                # Only log bot responses
                if message.author == self.client.user:
                    await self._log_accepted_answer(message, user)
    
    async def _read_channel_history_async(self, channel, current_message_id, count=10):
        """
        Async helper to read channel history.
        
        Args:
            channel: The Discord channel object
            current_message_id: The ID of the current message to exclude
            count: Number of messages to retrieve
            
        Returns:
            Formatted string with recent channel messages
        """
        messages = []
        async for msg in channel.history(limit=count + 10):  # Fetch extra to account for filtering
            # Skip the current message that triggered the bot
            if msg.id == current_message_id:
                continue
            # Skip bot's own messages to avoid self-referential context
            if msg.author == self.client.user:
                continue
            messages.append(msg)
            if len(messages) >= count:
                break
        
        # Format messages (oldest first)
        messages.reverse()
        formatted_messages = []
        for msg in messages:
            content = self._remove_bot_mention(msg.content)
            if content:
                timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
                formatted_messages.append(f"[{timestamp}] {msg.author.display_name}: {content}")
        
        if formatted_messages:
            return f"Recent channel history ({len(formatted_messages)} messages):\n" + "\n".join(formatted_messages)
        else:
            return "No recent messages found in channel history."
    
    def _create_channel_history_tool(self, channel, current_message_id):
        """
        Create a channel history reading tool for the current Discord channel.
        
        Args:
            channel: The Discord channel object
            current_message_id: The ID of the current message to exclude from history
            
        Returns:
            A function that reads channel history synchronously
        """
        def read_channel_history(count: str = "10") -> str:
            """
            Read the last N messages from the Discord channel history.
            This tool helps the bot understand recent conversation context.
            
            Args:
                count: Number of messages to retrieve (default: 10)
                
            Returns:
                Formatted string with recent channel messages
            """
            try:
                # Parse count, default to 10 if invalid
                try:
                    limit = int(count)
                    if limit < 1 or limit > 50:  # Cap at 50 to avoid overwhelming context
                        limit = 10
                except (ValueError, TypeError):
                    limit = 10
                
                # This function runs in a separate thread via asyncio.to_thread()
                # So we need to create a new event loop for this thread
                # We can't use asyncio.run() directly because it's not available in all contexts
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self._read_channel_history_async(channel, current_message_id, limit)
                    )
                    return result
                finally:
                    loop.close()
                    # Clear the event loop for this thread to avoid leaks
                    asyncio.set_event_loop(None)
                    
            except Exception as e:
                return f"Error reading channel history: {str(e)}"
        
        return read_channel_history
    
    def _register_channel_history_tool(self, channel, current_message_id):
        """
        Register the channel history tool with the agent.
        
        Args:
            channel: The Discord channel object
            current_message_id: The ID of the current message to exclude from history
        """
        tool_function = self._create_channel_history_tool(channel, current_message_id)
        
        self.agent.tools["read_channel_history"] = {
            "function": tool_function,
            "description": "Read the last N messages from the Discord channel history to understand recent conversation context. Input should be a number (default: 10, max: 50).",
            "parameters": ["count"]
        }
    
    def _unregister_channel_history_tool(self):
        """
        Remove the channel history tool from the agent.
        This should be called after processing each message to avoid memory leaks.
        """
        if "read_channel_history" in self.agent.tools:
            del self.agent.tools["read_channel_history"]
    
    def _create_image_caption_tool(self, user_query: str):
        """
        Create an image caption tool for the current message.
        
        Args:
            user_query: The user's query/question about the image
            
        Returns:
            A function that captions images using two-round captioning
        """
        def caption_image(image_url: str) -> str:
            """
            Caption an image using two-round captioning with nemotron VLM.
            First round gets a basic caption, second round provides detailed analysis
            based on the user's query.
            
            Args:
                image_url: URL of the image to caption
                
            Returns:
                Detailed caption from two-round analysis
            """
            try:
                result = two_round_image_caption(
                    image_url=image_url,
                    api_key=self.api_key,
                    user_query=user_query,
                    model=MODEL_CONFIG.get("image_caption_model", "nvidia/nemotron-nano-12b-v2-vl:free"),
                    base_url=MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
                )
                return result
            except Exception as e:
                return f"Error captioning image: {str(e)}"
        
        return caption_image
    
    def _register_image_caption_tool(self, user_query: str):
        """
        Register the image caption tool with the agent.
        
        Args:
            user_query: The user's query/question about the image
        """
        tool_function = self._create_image_caption_tool(user_query)
        
        self.agent.tools["caption_image"] = {
            "function": tool_function,
            "description": "Caption an image using two-round analysis with a Vision Language Model. First round provides a basic description, second round gives detailed analysis based on the user's query. Input should be an image URL.",
            "parameters": ["image_url"]
        }
    
    def _unregister_image_caption_tool(self):
        """
        Remove the image caption tool from the agent.
        This should be called after processing each message to avoid memory leaks.
        """
        if "caption_image" in self.agent.tools:
            del self.agent.tools["caption_image"]
    
    def _split_long_message(self, text: str, limit: int = 1900):
        chunks = []
        remaining = text
        while len(remaining) > limit:
            split_pos = remaining.rfind(' ', 0, limit)
            split_pos = limit if split_pos == -1 else split_pos
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:].strip()
        chunks.append(remaining)
        return chunks

    async def _remove_hourglasses(self, message):
        await self._remove_reaction(message, "\u23f3")
        await self._remove_reaction(message, "\u231b")

    def _call_llm(self, prompt: str, timeout: int = 10, model: str = None) -> str:
        """
        Helper method to call the LLM API.
        
        Args:
            prompt: The prompt to send to the LLM
            timeout: Request timeout in seconds
            model: Model to use. If None, uses config default
            
        Returns:
            The LLM's response content
            
        Raises:
            Exception: If the API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use specified model or default to the main reasoning model from config
        model_to_use = model if model is not None else MODEL_CONFIG.get("default_model", "amazon/nova-2-lite-v1:free")
        
        data = {
            "model": model_to_use,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Calculate input tokens
        input_tokens = int(len(prompt) / CHARS_PER_TOKEN)
        
        # Get base_url from config
        base_url = MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        
        # Log LLM call
        logger.info(f"LLM call started - Model: {model_to_use}, Input tokens: {input_tokens}")
        start_time = time.time()
        
        response = requests.post(
            base_url,
            headers=headers,
            json=data,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Calculate output tokens and response time
        output_tokens = int(len(content) / CHARS_PER_TOKEN)
        response_time = time.time() - start_time
        
        # Calculate tokens/sec
        input_tokens_per_sec = input_tokens / response_time if response_time > 0 else 0
        output_tokens_per_sec = output_tokens / response_time if response_time > 0 else 0
        
        # Log LLM response
        logger.info(f"LLM call completed - Model: {model_to_use}, Response time: {response_time:.2f}s, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
        # Track call in query log
        call_entry = {
            "type": "llm_call",
            "model": model_to_use,
            "timestamp": time.time(),
            "input": prompt,
            "output": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_time_seconds": round(response_time, 2),
            "input_tokens_per_sec": round(input_tokens_per_sec, 2),
            "output_tokens_per_sec": round(output_tokens_per_sec, 2)
        }
        self.current_query_log.append(call_entry)
        
        # Aggregate token stats by model
        if model_to_use not in self.current_query_token_stats:
            self.current_query_token_stats[model_to_use] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0
            }
        self.current_query_token_stats[model_to_use]["total_input_tokens"] += input_tokens
        self.current_query_token_stats[model_to_use]["total_output_tokens"] += output_tokens
        self.current_query_token_stats[model_to_use]["total_calls"] += 1
        
        return content
    
    from contextlib import contextmanager
    
    @contextmanager
    def _file_locked(self, f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _merge_token_stats(self, src, dst):
        for m, s in src.items():
            d = dst.setdefault(m, {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0})
            d["total_input_tokens"] += s.get("total_input_tokens", 0)
            d["total_output_tokens"] += s.get("total_output_tokens", 0)
            d["total_calls"] += s.get("total_calls", 0)
    
    async def _log_eval_question(self, message: discord.Message, user: discord.User):
        """
        Log a user question to the evaluation file when tagged with 🧪.
        
        Args:
            message: The Discord message to log
            user: The user who added the reaction
        """
        try:
            # Extract the question text
            question = self._remove_bot_mention(message.content)
            
            if not question:
                return
            
            # Create eval entry
            eval_entry = {
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "author": message.author.display_name,
                "author_id": str(message.author.id),
                "question": question,
                "timestamp": message.created_at.isoformat(),
                "tagged_by": user.display_name,
                "tagged_by_id": str(user.id),
                "accepted_answer": None
            }
            
            # Append to eval file with file locking for thread safety
            with open(self.EVAL_FILE, 'a', encoding='utf-8') as f:
                with self._file_locked(f):
                    f.write(json.dumps(eval_entry) + '\n')
            
            print(f"{Fore.CYAN}[EVAL] Question logged: {question[:50]}...{Style.RESET_ALL}")
            logger.info(f"Eval question logged from {message.author.display_name}: {question}")
            
            await self._add_reaction(message, "📝")
        
        except Exception as e:
            logger.error(f"Failed to log eval question: {str(e)}")
    
    async def _log_accepted_answer(self, message: discord.Message, user: discord.User):
        """
        Log an accepted answer when a bot response is tagged with ✅.
        Updates the corresponding eval entry if it exists.
        
        Args:
            message: The bot's response message
            user: The user who added the reaction
        """
        try:
            answer = message.content
            
            # Check if this is a reply to a user question
            if not message.reference or not message.reference.message_id:
                return
            
            # Fetch the original message
            try:
                original_msg = await message.channel.fetch_message(message.reference.message_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                return
            
            original_msg_id = str(original_msg.id)
            
            # Read existing eval entries with file locking
            if not self.EVAL_FILE.exists():
                return
            
            entries = []
            updated = False
            
            # Read and update entries with shared lock
            with open(self.EVAL_FILE, 'r+', encoding='utf-8') as f:
                with self._file_locked(f):
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            if entry.get("message_id") == original_msg_id:
                                entry.update({
                                    "accepted_answer": answer,
                                    "accepted_by": user.display_name,
                                    "accepted_by_id": str(user.id),
                                    "accepted_at": datetime.now().isoformat()
                                })
                                updated = True
                                print(f"{Fore.CYAN}[EVAL] Answer accepted for question: {entry['question'][:50]}...{Style.RESET_ALL}")
                                logger.info(f"Accepted answer logged for message {original_msg_id}")
                            entries.append(entry)
                    if updated:
                        f.seek(0)
                        f.truncate()
                        for entry in entries:
                            f.write(json.dumps(entry) + '\n')
            
            # Add confirmation reaction if update was successful
            if updated:
                await self._add_reaction(message, "💚")
        
        except Exception as e:
            logger.error(f"Failed to log accepted answer: {str(e)}")
    
    def _save_query_log(self, message_id: str, user_query: str, final_response: str, username: str):
        """
        Save the query log to a JSON file in the query_logs directory.
        
        Args:
            message_id: Discord message ID
            user_query: The user's query
            final_response: The final response from the bot
            username: Username of the person who submitted the query
        """
        try:
            # Get tracking data from agent
            agent_tracking = self.agent.get_tracking_data()
            
            # Combine all logs (discord bot + agent) and sort by timestamp for chronological order
            all_logs = self.current_query_log + agent_tracking["call_sequence"]
            all_logs.sort(key=lambda x: x.get("timestamp", 0))
            
            # Merge token stats from agent and discord bot
            merged_token_stats = dict(self.current_query_token_stats)
            self._merge_token_stats(agent_tracking["token_stats"], merged_token_stats)
            
            # Create the log entry
            log_entry = {
                "message_id": message_id,
                "username": username,
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "final_response": final_response[:1000] + "..." if len(final_response) > 1000 else final_response,
                "call_sequence": all_logs,
                "token_stats_by_model": merged_token_stats
            }
            
            # Save to file with username and timestamp in filename
            # Format: username_YYYYMMDD_HHMMSS.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize username to be filesystem-safe (replace spaces and special chars with underscore)
            safe_username = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in username)
            # Handle empty username and limit length to prevent filesystem issues
            if not safe_username or safe_username.replace('_', '') == '':
                safe_username = "unknown_user"
            # Limit username to 50 characters to prevent excessively long filenames
            safe_username = safe_username[:50]
            filename = f"{safe_username}_{timestamp}.json"
            filepath = self.QUERY_LOGS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Query log saved to {filepath}")
            print(f"{Fore.CYAN}[QUERY LOG] Saved to {filename}{Style.RESET_ALL}")
            
            # Print token statistics summary
            for model, stats in merged_token_stats.items():
                print(f"{Fore.CYAN}[TOKEN STATS] Model: {model} | Calls: {stats['total_calls']} | Input: {stats['total_input_tokens']} | Output: {stats['total_output_tokens']} | Total: {stats['total_input_tokens'] + stats['total_output_tokens']}{Style.RESET_ALL}")
        
        except Exception as e:
            logger.error(f"Failed to save query log: {str(e)}")
    
    def _reset_query_tracking(self):
        """
        Reset tracking for a new query.
        Should be called at the start of processing each user message.
        """
        self.current_query_log = []
        self.current_query_token_stats = {}
        self.agent.reset_tracking()
    
    def run(self):
        """Start the Discord bot."""
        print("Starting Discord bot...")
        self.client.run(self.token)


# Auto-restart functionality classes
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class BotRestartHandler(FileSystemEventHandler):
    """Handler that restarts the bot when Python or YAML files change."""
    
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_restart = 0
        self.debounce_seconds = 2  # Wait 2 seconds before restarting to avoid multiple restarts
        
    def on_modified(self, event):
        """Called when a file is modified."""
        if event.is_directory:
            return
            
        # Only restart for .py and .yaml files
        if event.src_path.endswith(('.py', '.yaml', '.yml')):
            current_time = time.time()
            # Debounce: only restart if it's been at least debounce_seconds since last restart
            if current_time - self.last_restart >= self.debounce_seconds:
                print(f"\n🔄 Detected change in: {event.src_path}")
                print("🔄 Restarting bot...")
                self.last_restart = current_time
                self.restart_callback()

class BotRunner:
    """Manages running and restarting the Discord bot process."""
    
    def __init__(self):
        self.process = None
        self.should_run = True
        self.output_thread = None
        self.restart_count = 0
        self.last_restart_time = 0
        self.max_consecutive_restarts = 5
        self.restart_reset_time = 60  # Reset restart count after 60 seconds of successful running
        
    def _read_output(self):
        """Read and print output from the bot process in a separate thread."""
        if self.process and self.process.stdout:
            try:
                for line in iter(self.process.stdout.readline, ''):
                    if line and self.should_run:
                        print(line, end='')
                    if self.process.poll() is not None:
                        # Process ended
                        break
            except Exception as e:
                print(f"Error reading output: {e}")
        
    def start_bot(self, is_restart=False):
        """Start the Discord bot process.
        
        Args:
            is_restart: True if this is a restart, False for initial start
        """
        if self.process is not None:
            self.stop_bot()
        
        # Check if we're restarting too frequently (only for restarts, not initial start)
        if is_restart:
            current_time = time.time()
            if current_time - self.last_restart_time > self.restart_reset_time:
                # Reset restart count if enough time has passed
                self.restart_count = 0
            
            if self.restart_count >= self.max_consecutive_restarts:
                print(f"\n⚠️  Bot has restarted {self.restart_count} times in quick succession.")
                print("⚠️  There may be a persistent issue preventing the bot from starting.")
                print("⚠️  Please check the error messages above and fix the issue.")
                print("⚠️  The bot will pause for 30 seconds before trying again...")
                time.sleep(30)
                self.restart_count = 0
            
            self.last_restart_time = current_time
            self.restart_count += 1
        
        print("🚀 Starting Discord bot...")
        # Create environment with DISCORD_BOT_SUBPROCESS flag to prevent nested auto-restart
        env = os.environ.copy()
        env["DISCORD_BOT_SUBPROCESS"] = "1"
        
        # Use absolute path to ensure script can be found from any directory
        script_path = Path(__file__).absolute()
        
        self.process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Start output reading in a separate thread to avoid blocking
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()
    
    def stop_bot(self):
        """Stop the Discord bot process."""
        if self.process is not None and self.process.poll() is None:
            print("\n🛑 Stopping bot...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️  Bot didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
            self.process = None
    
    def restart_bot(self):
        """Restart the Discord bot process."""
        self.stop_bot()
        time.sleep(1)  # Brief pause before restart
        if self.should_run:
            self.start_bot(is_restart=True)


def run_with_auto_restart():
    """Run the bot with auto-restart functionality."""
    print("="*80)
    print("🤖 Discord Bot with Auto-Restart Enabled")
    print("="*80)
    print("\nThe bot will automatically restart when .py or .yaml files change.")
    print("Press Ctrl+C to stop.\n")
    
    # Get the current directory
    watch_path = Path.cwd()
    
    # Create bot runner
    runner = BotRunner()
    
    # Create file system event handler
    event_handler = BotRestartHandler(runner.restart_bot)
    
    # Create observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=True)
    observer.start()
    
    try:
        # Start the bot
        runner.start_bot()
        
        # Keep the script running
        while runner.should_run:
            time.sleep(1)
            
            # Check if bot process died unexpectedly
            if runner.process is not None:
                exit_code = runner.process.poll()
                if exit_code is not None:
                    # Process ended
                    if exit_code != 0:
                        print(f"\n⚠️  Bot process ended with exit code {exit_code}. Restarting...")
                        time.sleep(2)
                        runner.restart_bot()
                    else:
                        # Clean exit, don't restart
                        print("\n✅ Bot process ended cleanly.")
                        runner.should_run = False
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        runner.should_run = False
        runner.stop_bot()
        observer.stop()
    
    observer.join()
    print("👋 Goodbye!")


def main():
    """
    Main function to run the Discord bot.
    Reads the token from .bot_token in the current working directory.
    """
    # Read Discord token from .bot_token
    token_file = ".bot_token"
    if not os.path.exists(token_file):
        print("ERROR: .bot_token file not found")
        return
    
    with open(token_file, 'r') as f:
        token = f.read().strip()
    
    if not token:
        print("ERROR: .bot_token is empty")
        return
    
    # Get OpenRouter API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        return
    
    # Check if auto-restart is enabled in config
    auto_restart = CONFIG.get("auto_restart", True)
    
    # If running as a subprocess (from auto-restart wrapper), run normally
    # Otherwise check if auto-restart should be enabled
    if auto_restart and not os.environ.get("DISCORD_BOT_SUBPROCESS"):
        # Run with auto-restart wrapper
        run_with_auto_restart()
    else:
        # Create and run the bot normally
        bot = ReActDiscordBot(token, api_key)
        bot.run()


if __name__ == "__main__":
    main()
