#!/usr/bin/env python3
"""
Discord bot wrapper for the ReAct agent.
This bot reads questions from Discord messages and uses the ReAct agent to answer them.

The bot's token should be in a file named 'token.txt' in the current working directory.
"""

import os
import discord
from react_agent import ReActAgent


class ReActDiscordBot:
    """Discord bot that wraps the ReAct agent."""
    
    def __init__(self, token: str, api_key: str):
        """
        Initialize the Discord bot with ReAct agent.
        
        Args:
            token: Discord bot token
            api_key: OpenRouter API key for the ReAct agent
        """
        self.token = token
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
        
        @self.client.event
        async def on_message(message):
            # Don't respond to the bot's own messages
            if message.author == self.client.user:
                return
            
            # Only respond to messages that mention the bot
            if self.client.user.mentioned_in(message):
                # Extract the question by removing the bot mention
                question = message.content
                for mention in message.mentions:
                    question = question.replace(f"<@{mention.id}>", "").strip()
                    question = question.replace(f"<@!{mention.id}>", "").strip()
                
                if not question:
                    await message.channel.send("Please ask me a question after mentioning me!")
                    return
                
                # Send a "thinking" message
                thinking_msg = await message.channel.send("🤔 Let me think about that...")
                
                try:
                    # Use the ReAct agent to answer the question
                    answer = self.agent.run(question, max_iterations=5, verbose=True)
                    
                    # Discord has a 2000 character limit for messages
                    if len(answer) > 1900:
                        # Split the answer into multiple messages
                        chunks = [answer[i:i+1900] for i in range(0, len(answer), 1900)]
                        await thinking_msg.delete()
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                await message.channel.send(f"**Answer:**\n{chunk}")
                            else:
                                await message.channel.send(chunk)
                    else:
                        await thinking_msg.edit(content=f"**Answer:**\n{answer}")
                
                except Exception as e:
                    await thinking_msg.edit(content=f"❌ Error: {str(e)}")
                    print(f"Error processing question: {e}")
    
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
