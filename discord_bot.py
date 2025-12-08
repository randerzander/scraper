#!/usr/bin/env python3
"""
Discord bot wrapper for the ReAct agent.
This bot reads questions from Discord messages and uses the ReAct agent to answer them.

The bot's token should be in a file named 'token.txt' in the current working directory.
"""

import os
import asyncio
import discord
import requests
import json
import logging
import time
import yaml
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # fcntl is not available on Windows
    HAS_FCNTL = False
from datetime import datetime
from pathlib import Path
from react_agent import ReActAgent, two_round_image_caption
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Token calculation constant
CHARS_PER_TOKEN = 4.5


# Load model configuration
def load_model_config():
    """Load model configuration from model_config.yaml."""
    config_path = Path(__file__).parent / "model_config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Model config file not found at {config_path}, using defaults")
        return {
            "default_model": "amazon/nova-2-lite-v1:free",
            "intent_detection_model": "amazon/nova-2-lite-v1:free",
            "image_caption_model": "nvidia/nemotron-nano-12b-v2-vl:free",
            "conciseness_model": "amazon/nova-2-lite-v1:free",
            "tldr_model": "amazon/nova-2-lite-v1:free"
        }
    except Exception as e:
        logger.error(f"Error loading model config: {e}, using defaults")
        return {
            "default_model": "amazon/nova-2-lite-v1:free",
            "intent_detection_model": "amazon/nova-2-lite-v1:free",
            "image_caption_model": "nvidia/nemotron-nano-12b-v2-vl:free",
            "conciseness_model": "amazon/nova-2-lite-v1:free",
            "tldr_model": "amazon/nova-2-lite-v1:free"
        }

MODEL_CONFIG = load_model_config()


class ReActDiscordBot:
    """Discord bot that wraps the ReAct agent."""
    
    # Data directory for evaluation logging
    DATA_DIR = Path("data")
    EVAL_FILE = DATA_DIR / "eval_qs.jsonl"
    QUERY_LOGS_DIR = DATA_DIR / "query_logs"
    
    def __init__(self, token: str, api_key: str):
        """
        Initialize the Discord bot with ReAct agent.
        
        Args:
            token: Discord bot token
            api_key: OpenRouter API key for the ReAct agent
        """
        self.token = token
        self.api_key = api_key
        self.agent = ReActAgent(api_key)
        
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
                try:
                    # Get the referenced message (may need to fetch if not cached)
                    ref_msg = getattr(current_msg, 'referenced_message', None)
                    if not ref_msg and current_msg.reference.message_id:
                        # Fetch the message if it's not in cache
                        ref_msg = await current_msg.channel.fetch_message(current_msg.reference.message_id)
                    
                    if not ref_msg:
                        break
                    
                    # Check for image attachments in the replied message
                    if ref_msg.attachments:
                        for attachment in ref_msg.attachments:
                            # Check if attachment is an image
                            if attachment.content_type and attachment.content_type.startswith('image/'):
                                reply_image_urls.append(attachment.url)
                                logger.info(f"Found image attachment in reply chain: {attachment.url}")
                    
                    # Format the message content
                    author_name = ref_msg.author.display_name
                    content = ref_msg.content
                    
                    # Remove bot mentions from content for clarity
                    content = content.replace(f"<@{self.client.user.id}>", "").strip()
                    content = content.replace(f"<@!{self.client.user.id}>", "").strip()
                    
                    # Add to chain (we're building it backwards, will reverse later)
                    if content:
                        chain.append(f"{author_name}: {content}")
                    
                    # Move to the next message in the chain
                    current_msg = ref_msg
                    depth += 1
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    # If we can't fetch the message, stop here
                    break
            
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
                # Extract the question by removing only the bot's mention
                question = message.content
                question = question.replace(f"<@{self.client.user.id}>", "").strip()
                question = question.replace(f"<@!{self.client.user.id}>", "").strip()
                
                # Check for image attachments
                image_urls = []
                if message.attachments:
                    for attachment in message.attachments:
                        # Check if attachment is an image
                        if attachment.content_type and attachment.content_type.startswith('image/'):
                            image_urls.append(attachment.url)
                            logger.info(f"Found image attachment: {attachment.url}")
                
                if not question and not image_urls:
                    await message.channel.send("Please ask me a question after mentioning me!")
                    return
                
                # If there are images but no question, provide a default question
                if not question and image_urls:
                    question = "What do you see in this image?"
                
                # Log the user query in green
                print(f"{Fore.GREEN}[USER QUERY] {message.author.display_name}: {question}{Style.RESET_ALL}")
                logger.info(f"User query received from {message.author.display_name}: {question}")
                
                # Reset tracking for new query
                self._reset_query_tracking()
                
                # Get reply chain context if this is a reply (also gets images from reply chain)
                reply_context, reply_image_urls = await get_reply_chain(message)
                
                # Combine images from current message and reply chain
                if reply_image_urls:
                    image_urls.extend(reply_image_urls)
                    logger.info(f"Added {len(reply_image_urls)} image(s) from reply chain")
                
                # Add a thinking emoji reaction to the original message
                try:
                    await message.add_reaction("⏳")
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    # If we can't add a reaction, continue anyway
                    pass
                
                try:
                    # Detect user intent (sarcastic vs serious)
                    intent = await asyncio.to_thread(self._detect_intent, question)
                    is_sarcastic = intent.get("is_sarcastic", False)
                    
                    # Add current datetime to the query so the bot can consider current time
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Build image context if images are present
                    image_context = ""
                    if image_urls:
                        image_context = f"\n\n[Images attached: {len(image_urls)} image(s)]\n"
                        for i, img_url in enumerate(image_urls, 1):
                            image_context += f"Image {i} URL: {img_url}\n"
                        image_context += "\nYou can use the 'caption_image' tool to analyze these images.\n"
                    
                    # Modify the question based on intent
                    if is_sarcastic:
                        # For sarcastic queries, instruct agent to be concise and sarcastic
                        question_with_context = f"""[Current date and time: {current_time}]
[User Intent: Sarcastic/Humorous - Respond with wit, sarcasm, and keep it VERY concise (2-3 sentences max)]
{image_context}
{reply_context}User question: {question}"""
                    else:
                        # For serious queries, instruct agent to be thorough
                        question_with_context = f"""[Current date and time: {current_time}]
[User Intent: Serious - Provide a thorough, informative response]
{image_context}
{reply_context}User question: {question}"""
                    
                    # Register channel history tool for this message processing
                    self._register_channel_history_tool(message.channel, message.id)
                    
                    # Register image caption tool if images are present
                    if image_urls:
                        self._register_image_caption_tool(question)
                    
                    # Use the ReAct agent to answer the question (verbose=False to reduce log noise)
                    # Run in a thread pool to avoid blocking the Discord event loop and heartbeat
                    answer = await asyncio.to_thread(
                        self.agent.run, question_with_context, max_iterations=5, verbose=False
                    )
                    
                    # Unregister channel history tool to avoid memory leaks
                    self._unregister_channel_history_tool()
                    
                    # Unregister image caption tool if it was registered
                    if image_urls:
                        self._unregister_image_caption_tool()
                    
                    # Log the final response in red
                    print(f"{Fore.RED}[FINAL RESPONSE] {answer[:100]}...{Style.RESET_ALL}" if len(answer) > 100 else f"{Fore.RED}[FINAL RESPONSE] {answer}{Style.RESET_ALL}")
                    
                    # If response is longer than 1000 characters, make it more concise
                    if len(answer) > 1000:
                        logger.info(f"Response length {len(answer)} exceeds 1000 characters, making it more concise...")
                        answer = await asyncio.to_thread(self._make_response_concise, answer)
                        logger.info(f"Concise response length: {len(answer)}")
                    
                    # For serious queries with long responses, add TL;DR
                    if not is_sarcastic:
                        answer = await asyncio.to_thread(self._add_tldr_to_response, answer)
                    
                    # Remove the thinking emoji reaction
                    try:
                        await message.remove_reaction("⏳", self.client.user)
                    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                        # If we can't remove the reaction, continue anyway
                        pass
                    
                    # Save the complete answer before sending it
                    complete_answer = answer
                    
                    # Discord has a 2000 character limit for messages
                    if len(answer) > 1900:
                        # Split the answer into multiple messages at word boundaries
                        chunks = []
                        while len(answer) > 1900:
                            # Find the last space before the 1900 character limit
                            split_pos = answer.rfind(' ', 0, 1900)
                            if split_pos == -1:  # No space found, split at limit
                                split_pos = 1900
                            chunks.append(answer[:split_pos])
                            answer = answer[split_pos:].strip()
                        if answer:  # Add remaining text
                            chunks.append(answer)
                        
                        for i, chunk in enumerate(chunks):
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send(answer)
                    
                    # Save query log after successful response
                    self._save_query_log(str(message.id), question, complete_answer)
                
                except Exception as e:
                    # Unregister channel history tool in case of error
                    self._unregister_channel_history_tool()
                    
                    # Unregister image caption tool in case of error
                    self._unregister_image_caption_tool()
                    
                    # Remove the thinking emoji reaction if there was an error
                    try:
                        await message.remove_reaction("⏳", self.client.user)
                    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                        # If we can't remove the reaction, continue anyway
                        pass
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
            author_name = msg.author.display_name
            content = msg.content
            
            # Remove bot mentions from content for clarity
            content = content.replace(f"<@{self.client.user.id}>", "").strip()
            content = content.replace(f"<@!{self.client.user.id}>", "").strip()
            
            # Format timestamp
            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
            
            if content:
                formatted_messages.append(f"[{timestamp}] {author_name}: {content}")
        
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
                    model=MODEL_CONFIG.get("image_caption_model", "nvidia/nemotron-nano-12b-v2-vl:free")
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
        
        # Log LLM call
        logger.info(f"LLM call started - Model: {model_to_use}, Input tokens: {input_tokens}")
        start_time = time.time()
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
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
            "input": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "output": content[:500] + "..." if len(content) > 500 else content,
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
    
    def _detect_intent(self, message: str) -> dict:
        """
        Detect the intent of a user message (sarcastic vs serious).
        Uses a faster model for quick intent classification.
        
        Args:
            message: The user's message
            
        Returns:
            Dictionary with 'is_sarcastic' (bool) and 'confidence' (str)
        """
        prompt = f"""Analyze the following message and determine if the user is being sarcastic or serious.
Respond with ONLY a JSON object in this exact format:
{{"is_sarcastic": true/false, "confidence": "high/medium/low"}}

Message: "{message}"

JSON Response:"""
        
        try:
            # Use intent detection model from config
            content = self._call_llm(prompt, model=MODEL_CONFIG.get("intent_detection_model", "amazon/nova-2-lite-v1:free"))
            
            # Extract JSON from response (handle cases with markdown code blocks)
            if "```json" in content:
                parts = content.split("```json")
                if len(parts) > 1 and "```" in parts[1]:
                    content = parts[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) > 1:
                    content = parts[1].strip()
            
            # Try to parse JSON
            intent_data = json.loads(content)
            return intent_data
        except Exception as e:
            # Default to serious if detection fails
            print(f"Intent detection failed: {e}")
            return {"is_sarcastic": False, "confidence": "low"}
    
    def _make_response_concise(self, response: str) -> str:
        """
        Make a response more concise if it's too long.
        
        Args:
            response: The full response text
            
        Returns:
            A more concise version of the response
        """
        prompt = f"""Make the following response MUCH more concise while preserving the key information and main points. 
Aim to reduce it significantly in length while keeping it informative and coherent.

Original response:
{response}

Concise version:"""
        
        try:
            concise_response = self._call_llm(prompt, model=MODEL_CONFIG.get("conciseness_model", "amazon/nova-2-lite-v1:free"))
            return concise_response
        except Exception as e:
            print(f"Failed to make response concise: {e}")
            # Return original response if conciseness reduction fails
            return response
    
    def _add_tldr_to_response(self, response: str) -> str:
        """
        Add a TL;DR to long responses.
        
        Args:
            response: The full response text
            
        Returns:
            Response with TL;DR appended at the end if applicable
        """
        # Add TL;DR for responses longer than 300 characters
        if len(response) > 300:
            prompt = f"""Provide a concise TL;DR (1-2 sentences max) for this response:

{response}

TL;DR:"""
            
            try:
                tldr = self._call_llm(prompt, model=MODEL_CONFIG.get("tldr_model", "amazon/nova-2-lite-v1:free"))
                # Format the response with TL;DR at the end
                return f"{response}\n\n---\n\n**TL;DR:** {tldr}"
            except Exception as e:
                print(f"TL;DR generation failed: {e}")
                return response
        
        return response
    
    def _lock_file(self, f):
        """Lock a file for exclusive access (cross-platform)."""
        if HAS_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    
    def _unlock_file(self, f):
        """Unlock a file (cross-platform)."""
        if HAS_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    async def _log_eval_question(self, message: discord.Message, user: discord.User):
        """
        Log a user question to the evaluation file when tagged with 🧪.
        
        Args:
            message: The Discord message to log
            user: The user who added the reaction
        """
        try:
            # Extract the question text
            question = message.content
            # Remove bot mentions
            question = question.replace(f"<@{self.client.user.id}>", "").strip()
            question = question.replace(f"<@!{self.client.user.id}>", "").strip()
            
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
                try:
                    # Acquire exclusive lock
                    self._lock_file(f)
                    f.write(json.dumps(eval_entry) + '\n')
                finally:
                    # Release lock
                    self._unlock_file(f)
            
            print(f"{Fore.CYAN}[EVAL] Question logged: {question[:50]}...{Style.RESET_ALL}")
            logger.info(f"Eval question logged from {message.author.display_name}: {question}")
            
            # Add confirmation reaction
            try:
                await message.add_reaction("📝")
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                pass
        
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
                try:
                    # Acquire exclusive lock for read-modify-write
                    self._lock_file(f)
                    
                    # Read all entries
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            # Update the entry if it matches the original message
                            if entry.get("message_id") == original_msg_id:
                                entry["accepted_answer"] = answer
                                entry["accepted_by"] = user.display_name
                                entry["accepted_by_id"] = str(user.id)
                                entry["accepted_at"] = datetime.now().isoformat()
                                updated = True
                                print(f"{Fore.CYAN}[EVAL] Answer accepted for question: {entry['question'][:50]}...{Style.RESET_ALL}")
                                logger.info(f"Accepted answer logged for message {original_msg_id}")
                            entries.append(entry)
                    
                    # Write back all entries if we updated one
                    if updated:
                        # Truncate and rewrite
                        f.seek(0)
                        f.truncate()
                        for entry in entries:
                            f.write(json.dumps(entry) + '\n')
                finally:
                    # Release lock
                    self._unlock_file(f)
            
            # Add confirmation reaction if update was successful
            if updated:
                try:
                    await message.add_reaction("💚")
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    pass
        
        except Exception as e:
            logger.error(f"Failed to log accepted answer: {str(e)}")
    
    def _save_query_log(self, message_id: str, user_query: str, final_response: str):
        """
        Save the query log to a JSON file in the query_logs directory.
        
        Args:
            message_id: Discord message ID
            user_query: The user's query
            final_response: The final response from the bot
        """
        try:
            # Get tracking data from agent
            agent_tracking = self.agent.get_tracking_data()
            
            # Combine all logs (discord bot + agent) and sort by timestamp for chronological order
            all_logs = self.current_query_log + agent_tracking["call_sequence"]
            all_logs.sort(key=lambda x: x.get("timestamp", 0))
            
            # Merge token stats from agent and discord bot
            merged_token_stats = dict(self.current_query_token_stats)
            for model, stats in agent_tracking["token_stats"].items():
                if model not in merged_token_stats:
                    merged_token_stats[model] = {
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_calls": 0
                    }
                merged_token_stats[model]["total_input_tokens"] += stats["total_input_tokens"]
                merged_token_stats[model]["total_output_tokens"] += stats["total_output_tokens"]
                merged_token_stats[model]["total_calls"] += stats["total_calls"]
            
            # Create the log entry
            log_entry = {
                "message_id": message_id,
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "final_response": final_response[:1000] + "..." if len(final_response) > 1000 else final_response,
                "call_sequence": all_logs,
                "token_stats_by_model": merged_token_stats
            }
            
            # Save to file with timestamp in filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_{message_id}_{timestamp}.json"
            filepath = self.QUERY_LOGS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Query log saved to {filepath}")
            print(f"{Fore.CYAN}[QUERY LOG] Saved to {filename}{Style.RESET_ALL}")
            
            # Print token statistics summary
            print(f"{Fore.CYAN}[TOKEN STATS]{Style.RESET_ALL}")
            for model, stats in merged_token_stats.items():
                print(f"  Model: {model}")
                print(f"    Calls: {stats['total_calls']}")
                print(f"    Input tokens: {stats['total_input_tokens']}")
                print(f"    Output tokens: {stats['total_output_tokens']}")
                print(f"    Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}")
        
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


def main():
    """
    Main function to run the Discord bot.
    Reads the token from token.txt in the current working directory.
    """
    # Read Discord token from token.txt
    token_file = "token.txt"
    if not os.path.exists(token_file):
        print("="*80)
        print("ERROR: token.txt file not found in the current directory")
        print("="*80)
        print("\nTo use this Discord bot, you need:")
        print("1. A Discord bot token")
        print("2. Create a file named 'token.txt' in the current directory")
        print("3. Put your Discord bot token in that file")
        print("\nHow to get a Discord bot token:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Create a new application (or select an existing one)")
        print("3. Go to the 'Bot' section")
        print("4. Click 'Reset Token' to get your token")
        print("5. Save the token to token.txt")
        print("\nBot Permissions Required:")
        print("- Read Messages/View Channels")
        print("- Send Messages")
        print("- Read Message History")
        print("="*80)
        return
    
    with open(token_file, 'r') as f:
        token = f.read().strip()
    
    if not token:
        print("ERROR: token.txt is empty")
        return
    
    # Get OpenRouter API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("="*80)
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("="*80)
        print("\nTo use this bot, you need an OpenRouter API key:")
        print("1. Sign up at https://openrouter.ai/")
        print("2. Get your API key from https://openrouter.ai/keys")
        print("3. Set the environment variable:")
        print("   export OPENROUTER_API_KEY=your_api_key_here")
        print("\nOr create a .env file:")
        print("   cp .env.example .env")
        print("   # Edit .env and add your API key")
        print("="*80)
        return
    
    # Create and run the bot
    bot = ReActDiscordBot(token, api_key)
    bot.run()


if __name__ == "__main__":
    main()
