#!/usr/bin/env python3
"""
Test script for utils module.
This script verifies that the common logging utilities work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from io import StringIO
from utils import ColoredFormatter, setup_logging, CHARS_PER_TOKEN
from colorama import Fore, Style


def test_chars_per_token_constant():
    """Test that CHARS_PER_TOKEN constant is defined correctly."""
    print("Testing CHARS_PER_TOKEN constant...")
    print("="*60)
    
    assert CHARS_PER_TOKEN == 4.5, "CHARS_PER_TOKEN should be 4.5"
    assert isinstance(CHARS_PER_TOKEN, float), "CHARS_PER_TOKEN should be a float"
    
    print(f"✓ CHARS_PER_TOKEN = {CHARS_PER_TOKEN}")
    print("\n" + "="*60)
    print("✓ CHARS_PER_TOKEN constant test passed!")
    print("="*60)


def test_colored_formatter_class():
    """Test that ColoredFormatter class exists and has correct structure."""
    print("\nTesting ColoredFormatter class...")
    print("="*60)
    
    # Check that ColoredFormatter is a class
    assert isinstance(ColoredFormatter, type), "ColoredFormatter should be a class"
    
    # Check that it inherits from logging.Formatter
    assert issubclass(ColoredFormatter, logging.Formatter), \
        "ColoredFormatter should inherit from logging.Formatter"
    
    # Check that COLORS dictionary exists and has correct keys
    assert hasattr(ColoredFormatter, 'COLORS'), "ColoredFormatter should have COLORS attribute"
    expected_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    actual_levels = set(ColoredFormatter.COLORS.keys())
    assert actual_levels == expected_levels, \
        f"COLORS should have levels {expected_levels}, got {actual_levels}"
    
    # Check that colors are assigned correctly
    assert ColoredFormatter.COLORS['DEBUG'] == Fore.BLUE
    assert ColoredFormatter.COLORS['INFO'] == Fore.CYAN
    assert ColoredFormatter.COLORS['WARNING'] == Fore.YELLOW
    assert ColoredFormatter.COLORS['ERROR'] == Fore.RED
    assert ColoredFormatter.COLORS['CRITICAL'] == Fore.RED + Style.BRIGHT
    
    print("✓ ColoredFormatter class exists")
    print("✓ Inherits from logging.Formatter")
    print("✓ COLORS dictionary has all required log levels")
    print("✓ Color assignments are correct:")
    print(f"  - DEBUG: {Fore.BLUE}BLUE{Style.RESET_ALL}")
    print(f"  - INFO: {Fore.CYAN}CYAN{Style.RESET_ALL}")
    print(f"  - WARNING: {Fore.YELLOW}YELLOW{Style.RESET_ALL}")
    print(f"  - ERROR: {Fore.RED}RED{Style.RESET_ALL}")
    print(f"  - CRITICAL: {Fore.RED}{Style.BRIGHT}RED+BRIGHT{Style.RESET_ALL}")
    
    print("\n" + "="*60)
    print("✓ ColoredFormatter class test passed!")
    print("="*60)


def test_colored_formatter_format():
    """Test that ColoredFormatter.format() method works correctly."""
    print("\nTesting ColoredFormatter.format() method...")
    print("="*60)
    
    formatter = ColoredFormatter('%(levelname)s - %(message)s')
    
    # Create a test log record
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Test message',
        args=(),
        exc_info=None
    )
    
    # Format the record
    formatted = formatter.format(record)
    
    # Check that the formatted message contains the colored level name
    assert 'INFO' in formatted, "Formatted message should contain level name"
    assert 'Test message' in formatted, "Formatted message should contain the message"
    
    # Check that levelname is reset after formatting
    assert record.levelname == 'INFO', "levelname should be reset after formatting"
    
    print("✓ ColoredFormatter.format() method works correctly")
    print(f"✓ Sample formatted output: {formatted}")
    
    print("\n" + "="*60)
    print("✓ ColoredFormatter.format() test passed!")
    print("="*60)


def test_setup_logging_function():
    """Test that setup_logging() function exists and works correctly."""
    print("\nTesting setup_logging() function...")
    print("="*60)
    
    # Test default parameters
    logger1 = setup_logging()
    assert logger1 is not None, "setup_logging() should return a logger"
    assert isinstance(logger1, logging.Logger), "Returned value should be a Logger instance"
    
    print("✓ setup_logging() returns a logger")
    print(f"✓ Logger name: {logger1.name}")
    
    # Test with custom level
    logger2 = setup_logging(level=logging.DEBUG)
    assert logger2 is not None, "setup_logging() with custom level should return a logger"
    
    print("✓ setup_logging() works with custom log level")
    
    # Test with custom format
    logger3 = setup_logging(format_string='%(levelname)s: %(message)s')
    assert logger3 is not None, "setup_logging() with custom format should return a logger"
    
    print("✓ setup_logging() works with custom format string")
    
    print("\n" + "="*60)
    print("✓ setup_logging() function test passed!")
    print("="*60)


def test_utils_can_be_imported_by_modules():
    """Test that utils module can be imported by discord_bot and react_agent."""
    print("\nTesting utils module imports in other modules...")
    print("="*60)
    
    # Check discord_bot.py
    discord_bot_path = os.path.join(os.path.dirname(__file__), '..', 'discord_bot.py')
    with open(discord_bot_path, 'r') as f:
        content = f.read()
    
    assert 'from utils import setup_logging, CHARS_PER_TOKEN' in content, \
        "discord_bot.py should import from utils"
    assert 'logger = setup_logging()' in content, \
        "discord_bot.py should call setup_logging()"
    
    # Verify that duplicated code is removed
    assert 'class ColoredFormatter(logging.Formatter):' not in content, \
        "ColoredFormatter class should be removed from discord_bot.py"
    assert 'init(autoreset=True)' not in content, \
        "colorama init should be removed from discord_bot.py"
    
    print("✓ discord_bot.py imports from utils correctly")
    print("✓ Duplicated code removed from discord_bot.py")
    
    # Check react_agent.py
    react_agent_path = os.path.join(os.path.dirname(__file__), '..', 'react_agent.py')
    with open(react_agent_path, 'r') as f:
        content = f.read()
    
    assert 'from utils import setup_logging, CHARS_PER_TOKEN' in content, \
        "react_agent.py should import from utils"
    assert 'logger = setup_logging()' in content, \
        "react_agent.py should call setup_logging()"
    
    # Verify that duplicated code is removed
    assert 'class ColoredFormatter(logging.Formatter):' not in content, \
        "ColoredFormatter class should be removed from react_agent.py"
    assert 'init(autoreset=True)' not in content, \
        "colorama init should be removed from react_agent.py"
    
    print("✓ react_agent.py imports from utils correctly")
    print("✓ Duplicated code removed from react_agent.py")
    
    print("\n" + "="*60)
    print("✓ Utils module imports test passed!")
    print("="*60)


if __name__ == "__main__":
    print("Utils Module Tests")
    print("="*60)
    
    # Test 1: CHARS_PER_TOKEN constant
    test_chars_per_token_constant()
    
    # Test 2: ColoredFormatter class
    test_colored_formatter_class()
    
    # Test 3: ColoredFormatter.format() method
    test_colored_formatter_format()
    
    # Test 4: setup_logging() function
    test_setup_logging_function()
    
    # Test 5: Utils module imports
    test_utils_can_be_imported_by_modules()
    
    print("\n" + "="*60)
    print("✓ ALL UTILS MODULE TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Summary:")
    print("The utils module now provides:")
    print("1. ✓ ColoredFormatter class for colored log output")
    print("2. ✓ setup_logging() function for easy logging configuration")
    print("3. ✓ CHARS_PER_TOKEN constant for token calculations")
    print("4. ✓ Centralized logging utilities used by both discord_bot and react_agent")
    print("="*60)
