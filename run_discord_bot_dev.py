#!/usr/bin/env python3
"""
Development mode runner for discord_bot.py with auto-restart on file changes.
This script watches for changes in .py and .yaml files and automatically restarts the bot.

Usage:
    python run_discord_bot_dev.py

Features:
    - Automatically restarts the bot when .py or .yaml files are modified
    - Debounces rapid changes to avoid multiple restarts
    - Graceful shutdown with Ctrl+C
    - Real-time output from the bot
    - Prevents infinite restart loops with fixed cooldown period

Requirements:
    - token.txt file with Discord bot token
    - OPENROUTER_API_KEY environment variable set
    - watchdog package installed (pip install watchdog)
"""

import sys
import time
import subprocess
import threading
from pathlib import Path
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
                print("⚠️  The development runner will pause for 30 seconds before trying again...")
                time.sleep(30)
                self.restart_count = 0
            
            self.last_restart_time = current_time
            self.restart_count += 1
        
        print("🚀 Starting Discord bot...")
        self.process = subprocess.Popen(
            [sys.executable, "discord_bot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
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

def main():
    """Main function to run the bot with auto-restart."""
    print("="*80)
    print("🤖 Discord Bot - Development Mode with Auto-Restart")
    print("="*80)
    print("\nThis script will automatically restart the bot when .py or .yaml files change.")
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

if __name__ == "__main__":
    main()
