#!/usr/bin/env python3
"""
File reading tool for the agent.
"""

import logging
from pathlib import Path
from utils import create_tool_spec


logger = logging.getLogger(__name__)


# Tool specification for agent registration
TOOL_SPEC = create_tool_spec(
    name="read_file",
    description="Read a file from the current working directory. Maximum file size is 1 MB.",
    parameters={
        "filepath": "File path relative to the current working directory (e.g., 'README.md', 'src/main.py')"
    },
    required=["filepath"]
)


def read_file(filepath: str) -> str:
    """
    Read a file from the current working directory.
    This tool allows the agent to read files that are needed to answer questions.
    
    Args:
        filepath: Path to the file to read (relative to current working directory)
        
    Returns:
        Content of the file or error message
    """
    logger.info(f"Reading file: {filepath}")
    try:
        # Get the current working directory (fully resolved)
        cwd = Path.cwd().resolve()
        
        # Resolve the file path relative to cwd (this also resolves symlinks)
        file_path = (cwd / filepath).resolve()
        
        # Security check: ensure the resolved path is within cwd
        # This prevents directory traversal attacks and symlink-based bypasses
        try:
            file_path.relative_to(cwd)
        except ValueError:
            logger.warning(f"Access denied for file outside cwd: {filepath}")
            return f"Error: Access denied. File path '{filepath}' is outside the current working directory."
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File not found: {filepath}")
            return f"Error: File '{filepath}' not found in current working directory."
        
        # Check if it's a file (not a directory)
        if not file_path.is_file():
            logger.warning(f"Not a file: {filepath}")
            return f"Error: '{filepath}' is not a file."
        
        # Check file size to avoid reading very large files
        max_size = 1024 * 1024  # 1 MB limit
        file_size = file_path.stat().st_size
        if file_size > max_size:
            logger.warning(f"File too large: {filepath} ({file_size} bytes)")
            return f"Error: File '{filepath}' is too large ({file_size} bytes). Maximum size is {max_size} bytes (1 MB)."
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Successfully read file: {filepath} ({file_size} bytes)")
            return content
            
        except UnicodeDecodeError:
            logger.warning(f"File not UTF-8 encoded: {filepath}")
            return f"Error: File '{filepath}' does not appear to be a text file (encoding error)."
            
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}")
        return f"Error reading file '{filepath}': {str(e)}"
