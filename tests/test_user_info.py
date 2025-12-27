#!/usr/bin/env python3
"""
Test script for user_info module.
Tests both add_userinfo and read_userinfo functions.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import os

# Add parent directory to path to import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.user_info import add_userinfo, read_userinfo


def test_add_and_read():
    """Test adding and reading user info."""
    print("Test 1: Add and read user info...")
    
    # Create a temporary directory for testing
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Change to temp directory
        os.chdir(temp_dir)
        
        # Test adding user info
        username = "test_user"
        info1 = "Likes Python programming"
        result = add_userinfo(username, info1)
        assert "successfully" in result.lower(), f"Failed to add info: {result}"
        print(f"  ✓ Added info: {info1}")
        
        # Test reading user info
        content = read_userinfo(username)
        assert info1 in content, f"Expected '{info1}' in content, got: {content}"
        print(f"  ✓ Read info: {content.strip()}")
        
        # Test adding more info (appending)
        info2 = "Prefers dark mode"
        result = add_userinfo(username, info2)
        assert "successfully" in result.lower(), f"Failed to add second info: {result}"
        print(f"  ✓ Added second info: {info2}")
        
        # Verify both pieces of info are present
        content = read_userinfo(username)
        assert info1 in content, f"First info missing from content: {content}"
        assert info2 in content, f"Second info missing from content: {content}"
        print(f"  ✓ Both pieces of info present")
        
        print("✓ Test 1 passed\n")
        return True
        
    finally:
        # Clean up
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_nonexistent_user():
    """Test reading info for a user that doesn't exist."""
    print("Test 2: Read nonexistent user...")
    
    # Create a temporary directory for testing
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Change to temp directory
        os.chdir(temp_dir)
        
        # Try to read info for a user that doesn't exist
        username = "nonexistent_user"
        content = read_userinfo(username)
        assert "no information" in content.lower(), f"Expected 'no information' message, got: {content}"
        print(f"  ✓ Correctly handles nonexistent user")
        
        print("✓ Test 2 passed\n")
        return True
        
    finally:
        # Clean up
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_username_sanitization():
    """Test that usernames are properly sanitized."""
    print("Test 3: Username sanitization...")
    
    # Create a temporary directory for testing
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Change to temp directory
        os.chdir(temp_dir)
        
        # Test with special characters
        username = "user@#$%^&*()test"
        info = "Test info with special chars in username"
        result = add_userinfo(username, info)
        assert "successfully" in result.lower(), f"Failed to add info with special chars: {result}"
        print(f"  ✓ Added info for user with special chars: {username}")
        
        # Read it back
        content = read_userinfo(username)
        assert info in content, f"Expected info in content, got: {content}"
        print(f"  ✓ Read back info successfully")
        
        # Verify the file was created with sanitized name
        user_info_dir = Path(temp_dir) / "user_info"
        files = list(user_info_dir.glob("*.txt"))
        assert len(files) == 1, f"Expected 1 file, found {len(files)}"
        filename = files[0].name
        assert "@" not in filename and "#" not in filename, f"Special chars not sanitized: {filename}"
        print(f"  ✓ Username properly sanitized to: {filename}")
        
        print("✓ Test 3 passed\n")
        return True
        
    finally:
        # Clean up
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_case_insensitive():
    """Test that username lookup is case-insensitive."""
    print("Test 4: Case-insensitive username lookup...")
    
    # Create a temporary directory for testing
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Change to temp directory
        os.chdir(temp_dir)
        
        # Add info with one case
        username = "TestUser"
        info = "This user likes testing"
        result = add_userinfo(username, info)
        assert "successfully" in result.lower(), f"Failed to add info: {result}"
        print(f"  ✓ Added info for: {username}")
        
        # Read with different case
        content = read_userinfo("testuser")
        assert info in content, f"Case-insensitive read failed, got: {content}"
        print(f"  ✓ Read info with lowercase: testuser")
        
        # Read with another case variation
        content = read_userinfo("TESTUSER")
        assert info in content, f"Case-insensitive read failed, got: {content}"
        print(f"  ✓ Read info with uppercase: TESTUSER")
        
        # Read with mixed case
        content = read_userinfo("tEsTuSeR")
        assert info in content, f"Case-insensitive read failed, got: {content}"
        print(f"  ✓ Read info with mixed case: tEsTuSeR")
        
        print("✓ Test 4 passed\n")
        return True
        
    finally:
        # Clean up
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("="*50)
    print("Running user_info tests")
    print("="*50 + "\n")
    
    tests = [
        test_add_and_read,
        test_nonexistent_user,
        test_username_sanitization,
        test_case_insensitive
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}\n")
            failed += 1
    
    print("="*50)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*50)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
