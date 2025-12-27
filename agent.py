#!/usr/bin/env python3
"""
Agent with web search and URL scraping capabilities using OpenAI tools API.
Supports configurable LLM backends and models.
"""

import os
import json
import logging
import time
import base64
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from colorama import Fore, Style
from utils import setup_logging, CHARS_PER_TOKEN
from tools.read_url import read_url, TOOL_SPEC as READ_URL_SPEC
from tools.code import write_code, run_code, WRITE_CODE_SPEC, RUN_CODE_SPEC
from tools.web_search import web_search, TOOL_SPEC as WEB_SEARCH_SPEC
from tools.read_file import read_file, TOOL_SPEC as READ_FILE_SPEC
from tools.list_files import list_files, LIST_FILES_SPEC

# Configure logging with colored formatter
setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_MODEL_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "image_caption_model": "nvidia/nemotron-nano-12b-v2-vl:free"
}

def load_model_config():
    """Load model configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Config loading failed: {e}, using defaults")
        return DEFAULT_MODEL_CONFIG.copy()

MODEL_CONFIG = load_model_config()

def download_image(url: str, save_path: str = None) -> str:
    """
    Download an image from a URL and save it locally.
    
    Args:
        url: URL of the image to download
        save_path: Optional path to save the image. If None, saves to scratch/
        
    Returns:
        Path to the downloaded image
    """
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        response.raise_for_status()
        
        # Determine save path
        if save_path is None:
            # Extract filename from URL or generate one
            filename = url.split('/')[-1].split('?')[0]
            if not filename or '.' not in filename:
                filename = f"DOWNLOAD_{int(time.time())}.png"
            else:
                filename = f"DOWNLOAD_{filename}"
            save_path = f"scratch/{filename}"
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    except Exception as e:
        raise Exception(f"Error downloading image: {str(e)}")


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def caption_image_with_vlm(image_url: str, api_key: str, prompt: str = None, model: str = None, base_url: str = None) -> str:
    """
    Caption an image using a Vision Language Model (VLM) via OpenRouter API.
    
    Args:
        image_url: URL of the image to caption
        api_key: OpenRouter API key
        prompt: Optional custom prompt for captioning. If None, uses default.
        model: VLM model to use for captioning. If None, uses config default.
        base_url: Base URL for OpenAI-compatible API. If None, uses config default.
        
    Returns:
        Caption text from the VLM
    """
    try:
        # Download the image
        image_path = download_image(image_url)
        
        # Convert to base64
        image_base64 = image_to_base64(image_path)
        
        # Determine the image MIME type from extension
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Default prompt if none provided
        if prompt is None:
            prompt = "Describe this image in detail. What do you see?"
        
        # Use config default if model not specified
        if model is None:
            model = MODEL_CONFIG.get("image_caption_model", "nvidia/nemotron-nano-12b-v2-vl:free")
        
        # Use OpenRouter for VLM if base_url not specified
        # VLM models are typically on OpenRouter, not local servers
        if base_url is None:
            # Check if VLM model is different from default model
            default_model = MODEL_CONFIG.get("default_model", "")
            vlm_model = MODEL_CONFIG.get("image_caption_model", "")
            
            if vlm_model != default_model:
                # Different model = likely needs OpenRouter
                base_url = "https://openrouter.ai/api/v1/chat/completions"
                
            else:
                # Same model = use config default
                base_url = MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        
        # Prepare the API request with image
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Add optional parameters if configured
        if "temperature" in MODEL_CONFIG:
            data["temperature"] = MODEL_CONFIG["temperature"]
        if "top_p" in MODEL_CONFIG:
            data["top_p"] = MODEL_CONFIG["top_p"]
        if "max_tokens" in MODEL_CONFIG:
            data["max_tokens"] = MODEL_CONFIG["max_tokens"]
        
        # Add thinking mode for compatible models if enabled
        if MODEL_CONFIG.get("enable_thinking", False):
            # For OpenAI-compatible APIs that support reasoning_effort (o1, deepseek-reasoner, etc.)
            data["reasoning_effort"] = "high"
        
        start_time = time.time()
        
        response = requests.post(
            base_url,
            headers=headers,
            json=data,
            timeout=90
        )
        
        if response.status_code != 200:
            logger.error(f"VLM API error response: {response.text[:500]}")
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        response_time = time.time() - start_time
        logger.info(f"Image caption completed - Model: {model}, Response time: {response_time:.2f}s")
        
        # Clean up the downloaded image
        try:
            os.remove(image_path)
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Failed to clean up temporary image {image_path}: {str(e)}")
        
        return content
    except Exception as e:
        return f"Error captioning image: {str(e)}"


def two_round_image_caption(image_url: str, api_key: str, user_query: str = None, model: str = None, base_url: str = None) -> str:
    """
    Perform two-round image captioning:
    1. First round: Get a basic caption of the image
    2. Second round: Get a detailed caption based on the user's query
    
    Args:
        image_url: URL of the image to caption
        api_key: OpenRouter API key
        user_query: User's query/question about the image
        model: VLM model to use for captioning. If None, uses config default.
        base_url: Base URL for OpenAI-compatible API. If None, uses config default.
        
    Returns:
        Detailed caption from the second round
    """
    try:
        # Use config default if model not specified
        if model is None:
            model = MODEL_CONFIG.get("image_caption_model", "nvidia/nemotron-nano-12b-v2-vl:free")
        
        # Use OpenRouter for VLM if base_url not specified
        # VLM models are typically on OpenRouter, not local servers
        if base_url is None:
            # Check if VLM model is different from default model
            default_model = MODEL_CONFIG.get("default_model", "")
            vlm_model = MODEL_CONFIG.get("image_caption_model", "")
            
            if vlm_model != default_model:
                # Different model = likely needs OpenRouter
                base_url = "https://openrouter.ai/api/v1/chat/completions"
                
            else:
                # Same model = use config default
                base_url = MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        
        # First round: Basic caption
        basic_prompt = "Describe this image in detail. What do you see? Include objects, people, actions, colors, and setting."
        
        try:
            first_caption = caption_image_with_vlm(image_url, api_key, basic_prompt, model, base_url)
        except Exception as e:
            return f"Error in first round of captioning: {str(e)}"
        
        # Second round: Detailed caption based on user query
        
        # Build a more specific prompt for the second round
        if user_query:
            second_prompt = f"""Based on the user's query: "{user_query}"

Previous basic description: {first_caption}

Now, provide a more detailed analysis of the image, paying particular attention to aspects relevant to the user's query. Include any additional details that might help answer their question."""
        else:
            second_prompt = f"""Previous description: {first_caption}

Now, provide additional details about the image that weren't covered in the first description. Focus on fine details, context, and any notable aspects."""
        
        try:
            second_caption = caption_image_with_vlm(image_url, api_key, second_prompt, model, base_url)
        except Exception as e:
            return f"Error in second round of captioning: {str(e)}\n\nFirst round result: {first_caption}"
        
        # Combine both captions
        combined = f"Initial Description:\n{first_caption}\n\nDetailed Analysis:\n{second_caption}"
        
        return combined
    except Exception as e:
        return f"Error in two-round captioning: {str(e)}"


# ReAct Agent implementation
class ReActAgent:
    """
    An agent that can use tools to answer questions via OpenAI tools API.
    Supports web_search, read_url, and read_file tools.
    """
    
    # Constants
    MAX_CONTENT_LENGTH = 4000  # Maximum length of scraped content to avoid context overflow
    
    def __init__(self, api_key: str, model: str = None, base_url: str = None, enable_logging: bool = True):
        """
        Initialize the ReAct agent.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use for reasoning. If None, uses config default.
            base_url: Base URL for OpenAI-compatible API. If None, uses config default.
            enable_logging: Enable INFO level logging (default: True)
        """
        self.api_key = api_key
        self.api_url = base_url if base_url is not None else MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        
        # Auto-detect model from localhost API if base_url is localhost
        if model is None:
            if self._is_localhost(self.api_url):
                detected_model = self._get_model_from_api()
                if detected_model:
                    self.model = detected_model
                    if enable_logging:
                        logger.info(f"Auto-detected model from localhost: {self.model}")
                else:
                    self.model = MODEL_CONFIG.get("default_model")
                    if not self.model:
                        raise ValueError("No model specified and auto-detection failed. Please specify a model in config.yaml")
            else:
                self.model = MODEL_CONFIG.get("default_model")
                if not self.model:
                    raise ValueError("No default_model specified in config.yaml")
        else:
            self.model = model
            
        self.enable_logging = enable_logging
        
        # Configure logger level based on enable_logging
        if not enable_logging:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        
        # Define available tools with function references
        self.tool_functions = {
            "web_search": web_search,
            "read_url": read_url,
            "read_file": read_file,
            "write_code": write_code,
            "run_code": run_code,
            "list_files": list_files
        }
        
        # Define tools in OpenAI tools API format
        self.tools = [
            WEB_SEARCH_SPEC,
            READ_URL_SPEC,
            READ_FILE_SPEC,
            WRITE_CODE_SPEC,
            RUN_CODE_SPEC,
            LIST_FILES_SPEC
        ]
        
        # Initialize call tracking for logging
        self._initialize_tracking()
    
    def _is_localhost(self, url: str) -> bool:
        """Check if URL points to localhost."""
        return "localhost" in url or "127.0.0.1" in url
    
    def _get_model_from_api(self) -> str:
        """
        Get the running model name from the API's /v1/models endpoint.
        
        Returns:
            Model ID/name or None if failed
        """
        try:
            # Convert chat/completions URL to models endpoint
            if "/chat/completions" in self.api_url:
                models_url = self.api_url.replace("/chat/completions", "/models")
            elif self.api_url.endswith("/v1"):
                models_url = f"{self.api_url}/models"
            else:
                models_url = f"{self.api_url}/v1/models"
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(models_url, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # OpenAI API format: {"object": "list", "data": [...]}
            models = data.get("data", []) if "data" in data else [data] if isinstance(data, dict) else data
            
            if models and len(models) > 0:
                # Get first model's ID
                model = models[0]
                if isinstance(model, dict):
                    model_id = model.get("id") or model.get("name") or model.get("model")
                    # Strip .gguf extension if present for cleaner display
                    if model_id and model_id.endswith(".gguf"):
                        model_id = model_id[:-5]
                    return model_id
                return str(model)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to auto-detect model from API: {e}")
            return None
    
    def _initialize_tracking(self):
        """Initialize or reset tracking data structures."""
        self.call_sequence = []
        self.token_stats = {}
    
    def reset_tracking(self):
        """Reset tracking for a new query."""
        self._initialize_tracking()
    
    def get_tracking_data(self) -> Dict[str, Any]:
        """
        Get the tracking data for logging.
        
        Returns:
            Dictionary with call_sequence and token_stats
        """
        return {
            "call_sequence": self.call_sequence,
            "token_stats": self.token_stats
        }
    
    def _call_llm(self, messages: List[Dict[str, str]], use_tools: bool = True) -> Dict:
        """
        Call the LLM API with optional tools.
        
        Args:
            messages: List of message dicts with role and content
            use_tools: Whether to include tools in the API call
            
        Returns:
            API response dict
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages
        }
        
        # Add tools if requested
        if use_tools:
            data["tools"] = self.tools
            data["tool_choice"] = "auto"
        
        # Add optional parameters if configured
        if "temperature" in MODEL_CONFIG:
            data["temperature"] = MODEL_CONFIG["temperature"]
        if "top_p" in MODEL_CONFIG:
            data["top_p"] = MODEL_CONFIG["top_p"]
        if "max_tokens" in MODEL_CONFIG:
            data["max_tokens"] = MODEL_CONFIG["max_tokens"]
        
        # Add thinking mode for compatible models if enabled
        if MODEL_CONFIG.get("enable_thinking", False):
            # For OpenAI-compatible APIs that support reasoning_effort (o1, deepseek-reasoner, etc.)
            data["reasoning_effort"] = "high"
        
        # Calculate input tokens (rough estimate)
        input_text = str(messages)
        input_tokens = int(len(input_text) / CHARS_PER_TOKEN)
        
        start_time = time.time()
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=None)
            response.raise_for_status()
            result = response.json()
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Get message content
            message = result["choices"][0]["message"]
            content = message.get("content", "") or ""
            
            # Calculate output tokens
            output_tokens = int(len(str(message)) / CHARS_PER_TOKEN)
            
            # Log LLM response
            logger.info(f"LLM call completed - Model: {self.model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}, Response time: {response_time:.2f}s")
            
            # Track call in sequence
            call_entry = {
                "type": "llm_call",
                "model": self.model,
                "timestamp": time.time(),
                "input": messages,
                "output": message,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_time_seconds": round(response_time, 2),
                "input_tokens_per_sec": round(input_tokens / response_time, 2) if response_time > 0 else 0,
                "output_tokens_per_sec": round(output_tokens / response_time, 2) if response_time > 0 else 0
            }
            self.call_sequence.append(call_entry)
            
            # Aggregate token stats
            if self.model not in self.token_stats:
                self.token_stats[self.model] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_calls": 0
                }
            self.token_stats[self.model]["total_input_tokens"] += input_tokens
            self.token_stats[self.model]["total_output_tokens"] += output_tokens
            self.token_stats[self.model]["total_calls"] += 1
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM call failed - Model: {self.model}, Response time: {response_time:.2f}s, Error: {str(e)}")
            
            # Track failed call in sequence
            call_entry = {
                "type": "llm_call",
                "model": self.model,
                "timestamp": time.time(),
                "input": messages,
                "error": str(e),
                "response_time_seconds": round(response_time, 2)
            }
            self.call_sequence.append(call_entry)
            
            raise
    
    def run(self, question: str, max_iterations: int = 30, verbose: bool = True, iteration_callback=None) -> str:
        """
        Run the agent to answer a question using OpenAI tools API.
        
        Args:
            question: The question to answer
            max_iterations: Maximum number of tool calls (default: 30)
            verbose: Whether to print intermediate steps
            iteration_callback: Optional callback function called after each iteration
            
        Returns:
            The final answer
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = MODEL_CONFIG.get("system_prompt", "You are a helpful assistant.")
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt} Current date and time: {current_time}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        for iteration in range(max_iterations):
            if iteration_callback:
                iteration_callback(iteration)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")
            
            try:
                # Call LLM with tools
                result = self._call_llm(messages, use_tools=True)
                
                # Debug: log the result structure
                if verbose:
                    print(f"\nDEBUG: result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"DEBUG: result keys: {result.keys()}")
                
                logger.debug(f"LLM result type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                
                if not isinstance(result, dict):
                    logger.error(f"Result is not a dict, it's: {type(result)}")
                    return f"Error: API returned unexpected type {type(result)}"
                
                if "choices" not in result:
                    logger.error(f"No 'choices' key in result. Keys: {result.keys()}")
                    return f"Error: Unexpected API response format - missing 'choices'"
                
                choices = result.get("choices")
                if not isinstance(choices, list):
                    logger.error(f"'choices' is not a list, it's: {type(choices)}")
                    return f"Error: Unexpected API response format - 'choices' is not a list"
                
                if not choices or len(choices) == 0:
                    logger.error("No choices in API response")
                    return "Error: No response from API"
                
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if not message:
                    logger.error(f"No message in first choice. Choice: {choices[0]}")
                    return "Error: No message in API response"
                
                # Check if there are tool calls
                tool_calls = message.get("tool_calls")
                
                if tool_calls:
                    # Add assistant message with tool calls to conversation
                    messages.append(message)
                    
                    # Execute each tool call
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        function_args = json.loads(tool_call["function"]["arguments"])
                        
                        if verbose:
                            print(f"\nTool Call: {function_name}")
                            print(f"Arguments: {json.dumps(function_args, indent=2)}")
                        
                        # Execute the tool
                        start_time = time.time()
                        try:
                            if function_name in self.tool_functions:
                                # Call the function with unpacked arguments
                                result_content = self.tool_functions[function_name](**function_args)
                                
                                # Format output for specific tools and log details
                                execution_time = round(time.time() - start_time, 2)
                                if function_name == "web_search" and isinstance(result_content, list):
                                    num_results = len(result_content)
                                    formatted = []
                                    for i, r in enumerate(result_content[:10], 1):
                                        if "error" in r:
                                            raise Exception(r["error"])
                                        formatted.append(f"{i}. {r.get('title','N/A')}\n   URL: {r.get('href','N/A')}\n   {r.get('body','N/A')}")
                                    result_content = "\n\n".join(formatted)
                                    query = function_args.get('query', 'unknown')
                                    logger.info(f"Tool {function_name} - Query: '{query}', Results: {num_results}, Characters: {len(result_content)}, Response time: {execution_time}s")
                                elif function_name == "read_url":
                                    success = not result_content.startswith("Error")
                                    status = "success" if success else "failed"
                                    url = function_args.get('url', 'unknown')
                                    logger.info(f"Tool {function_name} {status} - URL: {url}, Characters: {len(result_content)}, Response time: {execution_time}s")
                                elif function_name == "write_code":
                                    # write_code now returns just the filename
                                    success = not result_content.startswith("Error")
                                    logger.info(f"Tool {function_name} completed - Saved to: {result_content}, Response time: {execution_time}s")
                                elif function_name == "run_code":
                                    # result_content is a dict for run_code
                                    if isinstance(result_content, dict):
                                        success = result_content.get('success', False)
                                        exit_code = result_content.get('exit_code', -1)
                                        executed_file = result_content.get('filename', 'unknown')
                                        # Convert dict to string for LLM
                                        result_content_str = json.dumps(result_content, indent=2)
                                        char_count = len(result_content_str)
                                        logger.info(f"Tool {function_name} completed - File: {executed_file}, Success: {success}, Exit code: {exit_code}, Characters: {char_count}, Response time: {execution_time}s")
                                        result_content = result_content_str
                                    else:
                                        logger.info(f"Tool {function_name} completed, response time: {execution_time}s")
                                elif function_name == "list_files":
                                    # Count number of files listed
                                    file_count = result_content.count("\n") - 2 if "\n" in result_content else 0
                                    pattern = function_args.get('pattern', 'all files')
                                    logger.info(f"Tool {function_name} completed - Pattern: '{pattern}', Files: {file_count}, Response time: {execution_time}s")
                                elif function_name == "read_userinfo":
                                    # Log username for read_userinfo
                                    username = function_args.get('username', 'unknown')
                                    logger.info(f"Tool {function_name} completed - User: {username}, returned {len(result_content)} characters, response time: {execution_time}s")
                                elif function_name == "add_userinfo":
                                    # Log username and info length for add_userinfo
                                    username = function_args.get('username', 'unknown')
                                    info = function_args.get('info', '')
                                    logger.info(f"Tool {function_name} completed - User: {username}, stored {len(info)} characters, response time: {execution_time}s")
                                elif not isinstance(result_content, str):
                                    result_content = str(result_content)
                                    logger.info(f"Tool {function_name} completed, returned {len(result_content)} characters, response time: {execution_time}s")
                                else:
                                    logger.info(f"Tool {function_name} completed, returned {len(result_content)} characters, response time: {execution_time}s")
                                
                                # Track successful tool call
                                self.call_sequence.append({
                                    "type": "tool_call",
                                    "tool_name": function_name,
                                    "timestamp": time.time(),
                                    "input": function_args,
                                    "output": result_content,
                                    "execution_time_seconds": round(time.time() - start_time, 2)
                                })
                            else:
                                result_content = f"Error: Unknown function {function_name}"
                                logger.error(f"Unknown tool function: {function_name}")
                        except Exception as e:
                            result_content = f"Error executing {function_name}: {str(e)}"
                            logger.error(f"Tool {function_name} failed: {str(e)}")
                            
                            # Track failed tool call
                            self.call_sequence.append({
                                "type": "tool_call",
                                "tool_name": function_name,
                                "timestamp": time.time(),
                                "input": function_args,
                                "error": str(e),
                                "execution_time_seconds": round(time.time() - start_time, 2)
                            })
                        
                        if verbose:
                            print(f"\nTool Result: {result_content[:500]}{'...' if len(result_content) > 500 else ''}")
                        
                        # Add tool response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result_content
                        })
                
                else:
                    # No tool calls, this should be the final answer
                    content = message.get("content", "")
                    if verbose:
                        print(f"\n{'='*60}")
                        print("FINAL ANSWER")
                        print(f"{'='*60}")
                        print(content)
                    return content
                    
            except Exception as e:
                logger.error(f"Error in agent iteration {iteration}: {str(e)}")
                return f"Error: {str(e)}"
        
        return "Maximum iterations reached without a final answer."


def main():
    """
    Main function to demonstrate the ReAct agent.
    """
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key")
        return
    
    # Create agent
    agent = ReActAgent(api_key)
    
    # Example questions
    questions = [
        "What is the latest news about artificial intelligence?",
        # "What are the main features of Python 3.12? Search for information and then scrape the official Python release notes.",
    ]
    
    for question in questions:
        print(f"\n{'#'*80}")
        print(f"Question: {question}")
        print(f"{'#'*80}")
        
        answer = agent.run(question, verbose=True)
        
        print(f"\n{'='*80}")
        print(f"Final Answer: {answer}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
