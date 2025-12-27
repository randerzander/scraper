#!/usr/bin/env python3
"""
User info tool for the agent.
Allows users to tell the bot information about themselves.
"""

import logging
from pathlib import Path
from utils import create_tool_spec


logger = logging.getLogger(__name__)


# Tool specifications for agent registration
ADD_USERINFO_SPEC = create_tool_spec(
    name="add_userinfo",
    description="Store user information for future reference. Always specify which user.",
    parameters={
        "info": "The information to store",
        "username": "Username to save info about"
    },
    required=["info", "username"]
)

READ_USERINFO_SPEC = create_tool_spec(
    name="read_userinfo",
    description="Recall stored information about a user.",
    parameters={
        "username": "Username to read info about"
    },
    required=["username"]
)


def add_userinfo(username: str, info: str) -> str:
    """
    Store information about a user by appending it to their user info file.
    
    Args:
        username: The username of the person providing the information
        info: The information to store about the user
        
    Returns:
        Success or error message
    """
    try:
        # Get the current working directory
        cwd = Path.cwd().resolve()
        
        # Create user_info directory if it doesn't exist
        user_info_dir = cwd / "user_info"
        user_info_dir.mkdir(exist_ok=True)
        
        # Sanitize username to be filesystem-safe
        safe_username = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in username)
        if not safe_username or safe_username.replace('_', '') == '':
            safe_username = "unknown_user"
        safe_username = safe_username[:50]  # Limit length
        
        # Create filepath
        filepath = user_info_dir / f"{safe_username}.txt"
        
        # Append the information
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(info + '\n')
        
        return f"Information stored successfully for user {username}."
        
    except Exception as e:
        logger.error(f"Error storing user info for {username}: {str(e)}")
        return f"Error storing user info: {str(e)}"


def read_userinfo(username: str) -> str:
    """
    Read stored information about a user from their user info file.
    
    Args:
        username: The username to read information for
        
    Returns:
        The user's stored information or an error message
    """
    try:
        # Get the current working directory
        cwd = Path.cwd().resolve()
        
        # Get user_info directory
        user_info_dir = cwd / "user_info"
        
        # Sanitize username to be filesystem-safe (same logic as add_userinfo)
        safe_username = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in username)
        if not safe_username or safe_username.replace('_', '') == '':
            safe_username = "unknown_user"
        safe_username = safe_username[:50]  # Limit length
        
        # Create filepath - try case-insensitive match
        filepath = user_info_dir / f"{safe_username}.txt"
        
        # If exact match doesn't exist, try case-insensitive search
        if not filepath.exists() and user_info_dir.exists():
            for file in user_info_dir.glob("*.txt"):
                if file.stem.lower() == safe_username.lower():
                    filepath = file
                    break
        
        # Check if file exists
        if not filepath.exists():
            return f"No information stored for user {username}."
        
        # Read the information
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return f"No information stored for user {username}."
        
        return content
        
    except Exception as e:
        logger.error(f"Error reading user info for {username}: {str(e)}")
        return f"Error reading user info: {str(e)}"
