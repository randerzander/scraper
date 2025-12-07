#!/usr/bin/env python3
"""
Test script for verifying uv virtual environment setup.
This script tests that the "scraper" venv can be created and used correctly.
"""

import os
import subprocess
import sys
import shutil
import platform


def get_venv_python(venv_path):
    """Get the path to the Python executable in the venv."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def get_venv_activate_script(venv_path):
    """Get the path to the activation script in the venv."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "activate.bat")
    else:
        return os.path.join(venv_path, "bin", "activate")


def run_in_venv(venv_path, command):
    """Run a command in the virtual environment using the venv's Python."""
    python_executable = get_venv_python(venv_path)
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    
    # Use the venv's Python executable directly instead of shell activation
    result = subprocess.run(
        [python_executable] + command.split(),
        capture_output=True,
        text=True
    )
    return result


def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    result = subprocess.run(
        command.split(),
        capture_output=capture_output,
        text=True
    )
    return result


def test_uv_installed():
    """Test that uv is installed and available."""
    print("\n" + "="*60)
    print("Test 1: Checking if uv is installed...")
    print("="*60)
    
    result = run_command("uv --version")
    if result.returncode != 0:
        print("❌ uv is not installed")
        print("Install uv with: pip install uv")
        return False
    
    version = result.stdout.strip()
    print(f"✓ uv is installed: {version}")
    return True


def test_create_venv():
    """Test creating a virtual environment named 'scraper'."""
    print("\n" + "="*60)
    print("Test 2: Creating uv venv named 'scraper'...")
    print("="*60)
    
    # Clean up any existing venv
    venv_path = "scraper"
    if os.path.exists(venv_path):
        print(f"Removing existing venv at {venv_path}...")
        shutil.rmtree(venv_path)
    
    result = run_command("uv venv scraper")
    if result.returncode != 0:
        print(f"❌ Failed to create venv: {result.stderr}")
        return False
    
    if not os.path.exists(venv_path):
        print("❌ Virtual environment directory was not created")
        return False
    
    print("✓ Virtual environment 'scraper' created successfully")
    
    # Check for activation script (platform-specific)
    activate_script = get_venv_activate_script(venv_path)
    if not os.path.exists(activate_script):
        print(f"❌ Activation script not found at {activate_script}")
        return False
    
    print(f"✓ Activation script found at {activate_script}")
    return True


def test_install_dependencies():
    """Test installing dependencies in the venv."""
    print("\n" + "="*60)
    print("Test 3: Installing dependencies with uv pip...")
    print("="*60)
    
    venv_path = "scraper"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment does not exist")
        return False
    
    # Install dependencies using uv pip (uv is a system tool)
    result = subprocess.run(
        ["uv", "pip", "install", "-r", "requirements.txt", "--python", get_venv_python(venv_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    
    print("✓ Dependencies installed successfully")
    
    # Verify key packages are installed
    python_executable = get_venv_python(venv_path)
    result = subprocess.run(
        [python_executable, "-c", "import duckduckgo_search; import pyreadability; import requests"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to import required packages: {result.stderr}")
        return False
    
    print("✓ All required packages are importable")
    return True


def test_run_tests():
    """Test running the existing test suite in the venv."""
    print("\n" + "="*60)
    print("Test 4: Running test suite in the venv...")
    print("="*60)
    
    venv_path = "scraper"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment does not exist")
        return False
    
    # Run tests using the venv's Python
    python_executable = get_venv_python(venv_path)
    result = subprocess.run(
        [python_executable, "test_react_agent.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Test suite failed: {result.stderr}")
        return False
    
    print("✓ All tests passed in the venv")
    return True


def cleanup():
    """Clean up the test virtual environment."""
    print("\n" + "="*60)
    print("Cleaning up test environment...")
    print("="*60)
    
    venv_path = "scraper"
    if os.path.exists(venv_path):
        shutil.rmtree(venv_path)
        print(f"✓ Removed test venv at {venv_path}")
    else:
        print("✓ No cleanup needed")


def main():
    """Run all tests."""
    print("="*60)
    print("UV Virtual Environment Test Suite")
    print("="*60)
    print("\nThis script tests the uv venv setup for the scraper project.")
    print("It will create a venv named 'scraper' and verify it works correctly.")
    
    tests = [
        ("uv installation", test_uv_installed),
        ("venv creation", test_create_venv),
        ("dependency installation", test_install_dependencies),
        ("test suite execution", test_run_tests),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n❌ Test '{test_name}' failed")
                # Continue with other tests even if one fails
        except Exception as e:
            failed += 1
            print(f"\n❌ Test '{test_name}' raised an exception: {e}")
    
    # Cleanup
    cleanup()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
