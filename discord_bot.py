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
from datetime import datetime
from react_agent import ReActAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Token calculation constant
CHARS_PER_TOKEN = 4.5


class ReActDiscordBot:
    """Discord bot that wraps the ReAct agent."""
    
    # Model configuration
    DEFAULT_MODEL = "x-ai/grok-4.1-fast"
    INTENT_DETECTION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
    
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
        
        # Set up Discord intents
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        
        self.client = discord.Client(intents=intents)
        
        # Register event handlers
        @self.client.event
        async def on_ready():
            print(f"Bot logged in as {self.client.user}")
            print(f"Bot is ready to answer questions!")
            print(f"Invite link: https://discord.com/api/oauth2/authorize?client_id={self.client.user.id}&permissions=2048&scope=bot")
        
        async def get_reply_chain(message) -> str:
            """
            Get the full reply chain context for a message.
            Follows the chain of replied-to messages up to the root.
            
            Args:
                message: The Discord message to get reply chain for
                
            Returns:
                A formatted string with the reply chain context, or empty string if no replies
            """
            chain = []
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
            
            if chain:
                return "Previous conversation context:\n" + "\n".join(chain) + "\n\n"
            return ""
        
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
                
                if not question:
                    await message.channel.send("Please ask me a question after mentioning me!")
                    return
                
                # Log the user query
                logger.info(f"User query received from {message.author.display_name}: {question}")
                
                # Get reply chain context if this is a reply
                reply_context = await get_reply_chain(message)
                
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
                    
                    # Modify the question based on intent
                    if is_sarcastic:
                        # For sarcastic queries, instruct agent to be concise and sarcastic
                        question_with_context = f"""[Current date and time: {current_time}]
[User Intent: Sarcastic/Humorous - Respond with wit, sarcasm, and keep it VERY concise (2-3 sentences max)]

{reply_context}User question: {question}"""
                    else:
                        # For serious queries, instruct agent to be thorough
                        question_with_context = f"""[Current date and time: {current_time}]
[User Intent: Serious - Provide a thorough, informative response]

{reply_context}User question: {question}"""
                    
                    # Register channel history tool for this message processing
                    self._register_channel_history_tool(message.channel, message.id)
                    
                    # Use the ReAct agent to answer the question (verbose=False to reduce log noise)
                    # Run in a thread pool to avoid blocking the Discord event loop and heartbeat
                    answer = await asyncio.to_thread(
                        self.agent.run, question_with_context, max_iterations=5, verbose=False
                    )
                    
                    # Unregister channel history tool to avoid memory leaks
                    self._unregister_channel_history_tool()
                    
                    # For serious queries with long responses, add TL;DR
                    if not is_sarcastic:
                        answer = await asyncio.to_thread(self._add_tldr_to_response, answer)
                    
                    # Remove the thinking emoji reaction
                    try:
                        await message.remove_reaction("⏳", self.client.user)
                    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                        # If we can't remove the reaction, continue anyway
                        pass
                    
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
                
                except Exception as e:
                    # Unregister channel history tool in case of error
                    self._unregister_channel_history_tool()
                    
                    # Remove the thinking emoji reaction if there was an error
                    try:
                        await message.remove_reaction("⏳", self.client.user)
                    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                        # If we can't remove the reaction, continue anyway
                        pass
                    await message.channel.send(f"❌ Error: {str(e)}")
                    print(f"Error processing question: {e}")
    
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
    
    def _call_llm(self, prompt: str, timeout: int = 10, model: str = None) -> str:
        """
        Helper method to call the LLM API.
        
        Args:
            prompt: The prompt to send to the LLM
            timeout: Request timeout in seconds
            model: Model to use. If None, uses DEFAULT_MODEL
            
        Returns:
            The LLM's response content
            
        Raises:
            Exception: If the API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use specified model or default to the main reasoning model
        model_to_use = model if model is not None else self.DEFAULT_MODEL
        
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
        
        # Log LLM response
        logger.info(f"LLM call completed - Model: {model_to_use}, Response time: {response_time:.2f}s, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
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
            # Use faster model for intent detection
            content = self._call_llm(prompt, model=self.INTENT_DETECTION_MODEL)
            
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
    
    def _add_tldr_to_response(self, response: str) -> str:
        """
        Add a TL;DR to long responses.
        
        Args:
            response: The full response text
            
        Returns:
            Response with TL;DR prepended if applicable
        """
        # Add TL;DR for responses longer than 300 characters
        if len(response) > 300:
            prompt = f"""Provide a concise TL;DR (1-2 sentences max) for this response:

{response}

TL;DR:"""
            
            try:
                tldr = self._call_llm(prompt)
                # Format the response with TL;DR
                return f"**TL;DR:** {tldr}\n\n---\n\n{response}"
            except Exception as e:
                print(f"TL;DR generation failed: {e}")
                return response
        
        return response
    
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
