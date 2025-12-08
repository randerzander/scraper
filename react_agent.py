#!/usr/bin/env python3
"""
Simple ReAct Agent with DuckDuckGo search and URL scraping capabilities.
Uses OpenRouter API with deepseek-r1t2-chimera:free model.
"""

import os
import json
import re
import logging
import time
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
from duckduckgo_search import DDGS
from pyreadability import Readability
import html2text
import requests
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Token calculation constant
CHARS_PER_TOKEN = 4.5


# Tool implementations
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing title, href, and body
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def scrape_url(url: str) -> str:
    """
    Scrape a URL and convert HTML content to markdown using pyreadability.
    
    Args:
        url: URL to scrape
        
    Returns:
        Markdown content extracted from the page
    """
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        response.raise_for_status()
        
        # Use pyreadability to parse HTML and extract main content
        reader = Readability(response.text)
        result = reader.parse()
        
        # Convert HTML content to markdown using html2text
        h = html2text.HTML2Text()
        h.body_width = 0  # Don't wrap lines
        h.ignore_links = False
        markdown_content = h.handle(result['content'])
        
        # Add title if available
        if result.get('title'):
            markdown_content = f"# {result['title']}\n\n{markdown_content}"
        
        return markdown_content
    except Exception as e:
        return f"Error scraping URL: {str(e)}"


def download_image(url: str, save_path: str = None) -> str:
    """
    Download an image from a URL and save it locally.
    
    Args:
        url: URL of the image to download
        save_path: Optional path to save the image. If None, saves to /tmp/
        
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
                filename = f"image_{int(time.time())}.png"
            save_path = f"/tmp/{filename}"
        
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


def caption_image_with_vlm(image_url: str, api_key: str, prompt: str = None, model: str = "nvidia/nemotron-nano-12b-v2-vl:free") -> str:
    """
    Caption an image using a Vision Language Model (VLM) via OpenRouter API.
    
    Args:
        image_url: URL of the image to caption
        api_key: OpenRouter API key
        prompt: Optional custom prompt for captioning. If None, uses default.
        model: VLM model to use for captioning
        
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
        
        logger.info(f"Captioning image with VLM - Model: {model}")
        start_time = time.time()
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
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


def two_round_image_caption(image_url: str, api_key: str, user_query: str = None, model: str = "nvidia/nemotron-nano-12b-v2-vl:free") -> str:
    """
    Perform two-round image captioning:
    1. First round: Get a basic caption of the image
    2. Second round: Get a detailed caption based on the user's query
    
    Args:
        image_url: URL of the image to caption
        api_key: OpenRouter API key
        user_query: User's query/question about the image
        model: VLM model to use for captioning
        
    Returns:
        Detailed caption from the second round
    """
    try:
        # First round: Basic caption
        logger.info("Starting first round of image captioning...")
        basic_prompt = "Describe this image in detail. What do you see? Include objects, people, actions, colors, and setting."
        
        try:
            first_caption = caption_image_with_vlm(image_url, api_key, basic_prompt, model)
        except Exception as e:
            return f"Error in first round of captioning: {str(e)}"
        
        logger.info(f"First round caption: {first_caption[:100]}...")
        
        # Second round: Detailed caption based on user query
        logger.info("Starting second round of image captioning with user context...")
        
        # Build a more specific prompt for the second round
        if user_query:
            second_prompt = f"""Based on the user's query: "{user_query}"

Previous basic description: {first_caption}

Now, provide a more detailed analysis of the image, paying particular attention to aspects relevant to the user's query. Include any additional details that might help answer their question."""
        else:
            second_prompt = f"""Previous description: {first_caption}

Now, provide additional details about the image that weren't covered in the first description. Focus on fine details, context, and any notable aspects."""
        
        try:
            second_caption = caption_image_with_vlm(image_url, api_key, second_prompt, model)
        except Exception as e:
            return f"Error in second round of captioning: {str(e)}\n\nFirst round result: {first_caption}"
        
        logger.info(f"Second round caption: {second_caption[:100]}...")
        
        # Combine both captions
        combined = f"Initial Description:\n{first_caption}\n\nDetailed Analysis:\n{second_caption}"
        
        return combined
    except Exception as e:
        return f"Error in two-round captioning: {str(e)}"


# ReAct Agent implementation
class ReActAgent:
    """
    A simple ReAct (Reasoning + Acting) agent that can use tools to answer questions.
    """
    
    # Constants
    MAX_CONTENT_LENGTH = 4000  # Maximum length of scraped content to avoid context overflow
    API_TIMEOUT = 30  # Timeout for API calls in seconds
    
    def __init__(self, api_key: str, model: str = "x-ai/grok-4.1-fast"):
        """
        Initialize the ReAct agent.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use for reasoning
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Define available tools
        self.tools = {
            "duckduckgo_search": {
                "function": duckduckgo_search,
                "description": "Search the web using DuckDuckGo. Input should be a search query string.",
                "parameters": ["query"]
            },
            "scrape_url": {
                "function": scrape_url,
                "description": "Scrape and parse HTML content from a URL into markdown format. Input should be a URL.",
                "parameters": ["url"]
            }
        }
    
    def _create_prompt(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Create a ReAct prompt for the LLM.
        
        Args:
            question: The user's question
            history: List of previous thoughts, actions, and observations
            
        Returns:
            Formatted prompt string
        """
        tools_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        
        prompt = f"""You are a helpful assistant that can use tools to answer questions. You follow the ReAct (Reasoning + Acting) pattern.

Available tools:
{tools_desc}

Answer the following question by using the ReAct format:
Thought: [Your reasoning about what to do next]
Action: [Tool name to use: duckduckgo_search or scrape_url]
Action Input: [Input for the tool]

After receiving an observation, you can either:
- Continue with another Thought/Action/Action Input if you need more information
- Provide the final answer with: Final Answer: [Your complete answer]

Question: {question}

"""
        
        # Add history
        for entry in history:
            if entry["type"] == "thought":
                prompt += f"Thought: {entry['content']}\n"
            elif entry["type"] == "action":
                prompt += f"Action: {entry['content']}\n"
            elif entry["type"] == "action_input":
                prompt += f"Action Input: {entry['content']}\n"
            elif entry["type"] == "observation":
                prompt += f"Observation: {entry['content']}\n"
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the OpenRouter API to get a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Calculate input tokens
        input_tokens = int(len(prompt) / CHARS_PER_TOKEN)
        
        # Log LLM call
        logger.info(f"LLM call started - Model: {self.model}, Input tokens: {input_tokens}")
        start_time = time.time()
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Calculate output tokens and response time
            output_tokens = int(len(content) / CHARS_PER_TOKEN)
            response_time = time.time() - start_time
            
            # Log LLM response
            logger.info(f"LLM call completed - Model: {self.model}, Response time: {response_time:.2f}s, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            
            return content
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM call failed - Model: {self.model}, Response time: {response_time:.2f}s, Error: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def _parse_response(self, response: str) -> Dict[str, Optional[str]]:
        """
        Parse the LLM response to extract thought, action, and action input.
        
        Args:
            response: The LLM's response text
            
        Returns:
            Dictionary with parsed components
        """
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        # Check for final answer
        final_answer_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
            return result
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|$))", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", response, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # Extract action input
        action_input_match = re.search(r"Action Input:\s*(.+?)(?=\n(?:Thought:|Action:|Final Answer:)|$)", response, re.DOTALL | re.IGNORECASE)
        if action_input_match:
            result["action_input"] = action_input_match.group(1).strip()
        
        return result
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """
        Execute a tool action.
        
        Args:
            action: Tool name
            action_input: Input for the tool
            
        Returns:
            Result from the tool execution
        """
        if action not in self.tools:
            logger.warning(f"Unknown action attempted: {action}")
            return f"Error: Unknown action '{action}'. Available actions: {', '.join(self.tools.keys())}"
        
        # Log tool usage in yellow
        print(f"{Fore.YELLOW}[TOOL CALL] {action}: {action_input}{Style.RESET_ALL}")
        logger.info(f"Tool used: {action}, Arguments: {action_input}")
        
        tool_info = self.tools[action]
        tool_function = tool_info["function"]
        
        try:
            # Call the tool function with the action input
            if action == "duckduckgo_search":
                result = tool_function(action_input)
                # Format results nicely
                formatted_results = []
                for i, r in enumerate(result[:5], 1):
                    if "error" in r:
                        logger.error(f"Tool {action} failed: {r['error']}")
                        return r["error"]
                    formatted_results.append(
                        f"{i}. {r.get('title', 'N/A')}\n   URL: {r.get('href', 'N/A')}\n   {r.get('body', 'N/A')[:200]}..."
                    )
                logger.info(f"Tool {action} completed successfully, returned {len(formatted_results)} results")
                return "\n\n".join(formatted_results)
            elif action == "scrape_url":
                result = tool_function(action_input)
                # Limit result length to avoid overwhelming the context
                if len(result) > self.MAX_CONTENT_LENGTH:
                    result = result[:self.MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"
                logger.info(f"Tool {action} completed successfully, scraped {len(result)} characters")
                return result
            else:
                result = str(tool_function(action_input))
                logger.info(f"Tool {action} completed successfully")
                return result
        except Exception as e:
            logger.error(f"Tool {action} failed with exception: {str(e)}")
            return f"Error executing action: {str(e)}"
    
    def run(self, question: str, max_iterations: int = 5, verbose: bool = True) -> str:
        """
        Run the ReAct agent to answer a question.
        
        Args:
            question: The question to answer
            max_iterations: Maximum number of reasoning iterations
            verbose: Whether to print intermediate steps
            
        Returns:
            The final answer
        """
        history = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")
            
            # Create prompt and call LLM
            prompt = self._create_prompt(question, history)
            if verbose and iteration == 0:
                print(f"\nInitial Prompt:\n{prompt}")
            
            response = self._call_llm(prompt)
            if verbose:
                print(f"\nLLM Response:\n{response}")
            
            # Parse response
            parsed = self._parse_response(response)
            
            # Check if we have a final answer
            if parsed["final_answer"]:
                if verbose:
                    print(f"\n{'='*60}")
                    print("FINAL ANSWER")
                    print(f"{'='*60}")
                    print(parsed["final_answer"])
                return parsed["final_answer"]
            
            # Add thought to history
            if parsed["thought"]:
                history.append({"type": "thought", "content": parsed["thought"]})
                if verbose:
                    print(f"\nThought: {parsed['thought']}")
            
            # Execute action if present
            if parsed["action"] and parsed["action_input"]:
                history.append({"type": "action", "content": parsed["action"]})
                history.append({"type": "action_input", "content": parsed["action_input"]})
                
                if verbose:
                    print(f"\nAction: {parsed['action']}")
                    print(f"Action Input: {parsed['action_input']}")
                
                # Execute the action
                observation = self._execute_action(parsed["action"], parsed["action_input"])
                history.append({"type": "observation", "content": observation})
                
                if verbose:
                    print(f"\nObservation: {observation[:500]}{'...' if len(observation) > 500 else ''}")
            else:
                # If no valid action, prompt for final answer
                if verbose:
                    print("\nNo valid action found. Prompting for final answer...")
                history.append({
                    "type": "observation",
                    "content": "Please provide your final answer based on the information gathered."
                })
        
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
