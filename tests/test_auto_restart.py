#!/usr/bin/env python3
"""
Test script for the auto-restart development mode.
This test verifies that the file watcher can detect changes and trigger restarts.
"""

import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path to import run_discord_bot_dev
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_discord_bot_dev import BotRestartHandler, BotRunner


class TestAutoRestart(unittest.TestCase):
    """Test cases for auto-restart functionality."""
    
    def test_restart_handler_creation(self):
        """Test that restart handler can be created."""
        restart_callback = Mock()
        handler = BotRestartHandler(restart_callback)
        self.assertIsNotNone(handler)
        self.assertEqual(handler.debounce_seconds, 2)
    
    def test_restart_handler_ignores_directories(self):
        """Test that handler ignores directory changes."""
        restart_callback = Mock()
        handler = BotRestartHandler(restart_callback)
        
        # Create mock event for directory
        event = Mock()
        event.is_directory = True
        event.src_path = "/some/directory"
        
        handler.on_modified(event)
        
        # Callback should not have been called
        restart_callback.assert_not_called()
    
    def test_restart_handler_triggers_on_py_file(self):
        """Test that handler triggers restart on .py file modification."""
        restart_callback = Mock()
        handler = BotRestartHandler(restart_callback)
        
        # Create mock event for .py file
        event = Mock()
        event.is_directory = False
        event.src_path = "/some/path/test.py"
        
        handler.on_modified(event)
        
        # Callback should have been called
        restart_callback.assert_called_once()
    
    def test_restart_handler_triggers_on_yaml_file(self):
        """Test that handler triggers restart on .yaml file modification."""
        restart_callback = Mock()
        handler = BotRestartHandler(restart_callback)
        
        # Create mock event for .yaml file
        event = Mock()
        event.is_directory = False
        event.src_path = "/some/path/config.yaml"
        
        handler.on_modified(event)
        
        # Callback should have been called
        restart_callback.assert_called_once()
    
    def test_restart_handler_ignores_other_files(self):
        """Test that handler ignores non-.py/.yaml files."""
        restart_callback = Mock()
        handler = BotRestartHandler(restart_callback)
        
        # Create mock event for .txt file
        event = Mock()
        event.is_directory = False
        event.src_path = "/some/path/test.txt"
        
        handler.on_modified(event)
        
        # Callback should not have been called
        restart_callback.assert_not_called()
    
    def test_restart_handler_debouncing(self):
        """Test that handler debounces multiple rapid changes."""
        restart_callback = Mock()
        handler = BotRestartHandler(restart_callback)
        handler.debounce_seconds = 1  # Set shorter debounce for testing
        
        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = "/some/path/test.py"
        
        # Trigger multiple times rapidly
        handler.on_modified(event)
        handler.on_modified(event)
        handler.on_modified(event)
        
        # Should only trigger once due to debouncing
        self.assertEqual(restart_callback.call_count, 1)
        
        # Wait for debounce period
        time.sleep(1.1)
        
        # Trigger again
        handler.on_modified(event)
        
        # Should have triggered a second time now
        self.assertEqual(restart_callback.call_count, 2)
    
    def test_bot_runner_initialization(self):
        """Test that BotRunner can be initialized."""
        runner = BotRunner()
        self.assertIsNone(runner.process)
        self.assertTrue(runner.should_run)
        self.assertEqual(runner.restart_count, 0)
        self.assertEqual(runner.max_consecutive_restarts, 5)
    
    @patch('subprocess.Popen')
    def test_bot_runner_start_bot(self, mock_popen):
        """Test that BotRunner can start a bot process."""
        # Setup mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdout = None
        mock_popen.return_value = mock_process
        
        runner = BotRunner()
        runner.start_bot()
        
        # Verify subprocess.Popen was called
        mock_popen.assert_called_once()
        self.assertIsNotNone(runner.process)
    
    @patch('subprocess.Popen')
    def test_bot_runner_stop_bot(self, mock_popen):
        """Test that BotRunner can stop a bot process."""
        # Setup mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdout = None
        mock_popen.return_value = mock_process
        
        runner = BotRunner()
        runner.start_bot()
        runner.stop_bot()
        
        # Verify terminate was called
        mock_process.terminate.assert_called_once()
        self.assertIsNone(runner.process)
    
    @patch('subprocess.Popen')
    def test_bot_runner_restart_bot(self, mock_popen):
        """Test that BotRunner can restart a bot process."""
        # Setup mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdout = None
        mock_popen.return_value = mock_process
        
        runner = BotRunner()
        runner.start_bot()
        
        initial_call_count = mock_popen.call_count
        
        runner.restart_bot()
        
        # Verify terminate was called and Popen was called again
        mock_process.terminate.assert_called_once()
        self.assertEqual(mock_popen.call_count, initial_call_count + 1)
    
    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_bot_runner_restart_limit(self, mock_sleep, mock_popen):
        """Test that BotRunner limits rapid consecutive restarts."""
        # Setup mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdout = None
        mock_popen.return_value = mock_process
        
        runner = BotRunner()
        runner.max_consecutive_restarts = 3  # Lower limit for testing
        runner.restart_reset_time = 100  # Long reset time so it doesn't reset during test
        
        # Initial start (should not count as restart)
        runner.start_bot(is_restart=False)
        
        # Restart multiple times rapidly (should count as restarts)
        for i in range(4):
            runner.start_bot(is_restart=True)
            time.sleep(0.01)  # Small delay to simulate rapid restarts
        
        # Should have triggered the restart limit warning (30 second sleep)
        # Check if sleep was called with 30 seconds
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertIn(30, sleep_calls, "Expected 30 second sleep for restart limit")


class TestWatchdogImport(unittest.TestCase):
    """Test that watchdog is properly installed and importable."""
    
    def test_watchdog_import(self):
        """Test that watchdog can be imported."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import watchdog: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
