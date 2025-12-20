import utils
import json
import asyncio
import time
import os
import yaml
from openai import OpenAI
from pathlib import Path
from agent import ReActAgent
from colorama import Fore, Style

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

def judge_answer(question: str, expected_answer: str, agent_response: str) -> dict:
    """Use LLM to judge if the agent's response contains the expected answer."""
    
    # Get API settings from config
    api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "sk-fake")
    base_url = CONFIG.get("base_url", "http://localhost:8080/v1")
    
    # Strip /chat/completions from base_url if present (OpenAI client adds it)
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    
    model = CONFIG.get("default_model", "gpt-3.5-turbo")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    judge_prompt = f"""
EXPECTED: {expected_answer}
ACTUAL: {agent_response}

Do NOT assess the truth of ACTUAL or Expected. Only assess whether ACTUAL contains the information in EXPECTED.

You may provide reasoning, but you MUST end your response with a final decision on its own line.
Your final line must be ONLY one of: PASS or FAIL

if ACTUAL contains EXPECTED, your final line should be: PASS
if ACTUAL does not contain EXPECTED, your final line should be: FAIL
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    judgment_text = response.choices[0].message.content.strip()
    
    # Get the last non-empty line for the final decision
    lines = [line.strip() for line in judgment_text.split('\n') if line.strip()]
    final_decision = lines[-1] if lines else ""
    is_pass = final_decision.upper() == 'PASS'
    
    # Remove the final PASS/FAIL line from the judgment text for display
    reason_lines = lines[:-1] if lines and final_decision.upper() in ['PASS', 'FAIL'] else lines
    reason_text = '\n'.join(reason_lines)
    
    return {
        "passed": is_pass,
        "judgment": reason_text
    }

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run agent tests')
    parser.add_argument('-qid', type=int, help='Run only the question with this QID')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize agent
    api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    base_url = CONFIG.get("base_url", "http://localhost:8080/v1")
    model = CONFIG.get("default_model", "gpt-3.5-turbo")
    
    # Don't strip base_url for agent - it expects the full path
    agent = ReActAgent(api_key=api_key, model=model, base_url=base_url)
    
    with open("data/qa.jsonl", "r") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]
    
    # Filter out questions with blank answers
    qa_pairs = [qa for qa in qa_pairs if qa.get("answer", "").strip()]
    
    # Filter by qid if specified
    if args.qid:
        qa_pairs = [qa for qa in qa_pairs if qa.get("qid") == args.qid]
        if not qa_pairs:
            print(f"No question found with qid={args.qid}")
            return
    
    print(f"\nRunning evaluation on {len(qa_pairs)} questions with answers...")
    
    results = []
    for qa in qa_pairs:
        question = qa["question"]
        expected = qa["answer"]
        qid = qa.get("qid", "?")
        
        print(f"\n{'='*60}")
        print(f"QID: {qid}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        q_start_time = time.time()
        agent_response = agent.run(question, verbose=False)
        q_elapsed = time.time() - q_start_time
        
        print(f"Agent Response: {agent_response}")
        print(f"\nExpected Answer: {expected}")
        print(f"Time: {q_elapsed:.2f}s")
        
        judgment = judge_answer(question, expected, agent_response)
        
        # Colored output for judgment
        if judgment['passed']:
            print(f"\n{Fore.GREEN}✓ PASS{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}✗ FAIL{Style.RESET_ALL}")
        
        print(f"Reason: {judgment['judgment']}")
        
        results.append({
            "qid": qid,
            "question": question,
            "expected": expected,
            "response": agent_response,
            "passed": judgment["passed"],
            "judgment": judgment["judgment"],
            "time": q_elapsed
        })
    
    total_time = time.time() - start_time
    
    passed_count = sum(r['passed'] for r in results)
    failed_count = len(results) - passed_count
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: ", end="")
    
    # Colored summary
    if passed_count > 0:
        print(f"{Fore.GREEN}{passed_count} passed{Style.RESET_ALL}", end="")
    else:
        print(f"{passed_count} passed", end="")
    
    print(" / ", end="")
    
    if failed_count > 0:
        print(f"{Fore.RED}{failed_count} failed{Style.RESET_ALL}", end="")
    else:
        print(f"{failed_count} failed", end="")
    
    print(f" / {len(results)} total")
    
    print(f"Total Runtime: {total_time:.2f}s")
    if len(results) > 0:
        print(f"Average Time per Question: {total_time/len(results):.2f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
