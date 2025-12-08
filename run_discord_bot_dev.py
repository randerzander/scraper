#!/usr/bin/env python3
"""
Development mode runner for discord_bot.py with auto-restart on file changes.
This script watches for changes in .py and .yaml files and automatically restarts the bot.
"""

import sys
import time
import subprocess
import os
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
        
    def start_bot(self):
        """Start the Discord bot process."""
        if self.process is not None:
            self.stop_bot()
        
        print("🚀 Starting Discord bot...")
        self.process = subprocess.Popen(
            [sys.executable, "discord_bot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print bot output in real-time
        if self.process.stdout:
            for line in iter(self.process.stdout.readline, ''):
                if line and self.process.poll() is None:
                    print(line, end='')
                elif self.process.poll() is not None:
                    # Process ended
                    break
    
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
            self.start_bot()

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
            if runner.process is not None and runner.process.poll() is not None:
                print("\n⚠️  Bot process ended unexpectedly. Restarting...")
                time.sleep(2)
                runner.restart_bot()
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        runner.should_run = False
        runner.stop_bot()
        observer.stop()
    
    observer.join()
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()
