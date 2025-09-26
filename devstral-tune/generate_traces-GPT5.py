#!/usr/bin/env python3
"""
generate_traces.py
------------------

Given a JSON file of prompts and associated datasets, produce multi-turn
traces that match the specific format of the Go-based Devstral agent.
This version is designed to force a multi-step, agentic workflow and
uses a remote Python executor.
"""

import argparse
import json
import os
import re
import sys
import socket
import textwrap
import time
import uuid
import shutil
from typing import List, Dict, Optional, Tuple, Tuple

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install dependency: pip install openai")

# ---------- Constants ----------
EOM_TOKEN = "<|EOM|>"

# ---------- Devstral-style system prompt (aligned with Go app) ----------
DEVSTRAL_SYSTEM = """You are an expert statistical data analyst using Python. Rigor is mandatory; do not speculate or hallucinate.

If CSV or Excel files are uploaded, treat the first uploaded file as the primary dataset. Always load files by their exact provided names.

---

## CRITICAL: Multi-Step Workflow (MANDATORY)
You MUST work iteratively through multiple steps. A typical analysis requires AT LEAST 5-6 steps.

## Workflow Loop (repeat until complete)
1.  **Observe**: Inspect the latest `<execution_results>`. This is provided to you by the system. **NEVER write `<execution_results>` in your response.**
2.  **Plan**: In 1-2 sentences, state the single next step toward the user's goal.
3.  **Act**: Immediately after your plan, write a short `<python></python>` block to execute the plan.

After EVERY execution result, you MUST continue with another plan and `<python>` block unless the user's request is fully complete.

---

## Best Practices
-   On first load, report: shape, column names, and `df.head(3)`.
-   Check and address missing data, impossible values (e.g., negatives), and data types before analysis.
-   **Never call `display()`**. Use `print()` for all output.
-   Save plots to PNG files and close them: `plt.savefig("plot.png"); plt.close()`.
-   When the analysis is complete, provide a final summary interpreting the results and include generated plots as `<image>plot_name.png</image>. Do not write `<python>` tags in the final summary.
"""

# ---------- Helpers to parse model output ----------
TAG_RE = {
    "python": re.compile(r"<python>(.*?)</python>", re.DOTALL | re.IGNORECASE),
}

def extract_tag(s: str, tag: str) -> Optional[str]:
    m = TAG_RE.get(tag, re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)).search(s or "")
    return m.group(1).strip() if m else None

def sanitize_for_display(s: str, limit: int = 4000) -> str:
    s = s.strip()
    if len(s) > limit:
        return s[:limit] + "\n...[truncated]..."
    return s

# ---------- Executor Interaction ----------
def execute_code_via_socket(code: str, session_id: str, host: str = 'localhost', port: int = 9999) -> Tuple[bool, str]:
    """
    Execute Python code using the remote Dockerized executor via socket.
    Returns (success, output) tuple.
    """
    try:
        # Create a persistent connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        
        # Connect to the executor
        sock.connect((host, port))
        
        # Send the message in the expected format: session_id|code
        message = f"{session_id}|{code}"
        sock.sendall(message.encode('utf-8'))
        
        # Read the response until we get the EOM token
        response_chunks = []
        buffer_size = 4096
        
        while True:
            try:
                chunk = sock.recv(buffer_size)
                if not chunk:
                    # Connection closed by server
                    break
                
                decoded_chunk = chunk.decode('utf-8', errors='replace')
                response_chunks.append(decoded_chunk)
                
                # Check if we've received the complete response with EOM token
                full_response = ''.join(response_chunks)
                if EOM_TOKEN in full_response:
                    break
                    
            except socket.timeout:
                print(f"Socket timeout while waiting for response", file=sys.stderr)
                break
        
        sock.close()
        
        # Combine all chunks and remove the EOM token
        full_response = ''.join(response_chunks)
        
        # Handle case where executor sends empty result with success message
        if not full_response:
            return False, "No response from executor"
            
        output = full_response.replace(EOM_TOKEN, '').strip()
        
        # The executor sends "Success: Code executed with no output." for empty results
        if not output or output == "Success: Code executed with no output.":
            output = "Success: Code executed with no output."
            
        # Check if it's an error
        if output.startswith("Error:"):
            return False, output
            
        return True, output
            
    except socket.error as e:
        error_msg = f"Socket error: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return False, error_msg
    except Exception as e:
        error_msg = f"Executor connection failed: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return False, error_msg

def run_python_in_executor(code: str, session_id: str, workdir: str) -> Dict:
    """Execute Python code using the remote Dockerized executor."""
    # Ensure the workspace directory exists on the host (matching Go agent behavior)
    # This is important because the Docker container mounts ../workspaces:/app/workspaces
    os.makedirs(workdir, exist_ok=True)
    
    # The session_id is used by the executor to cd to /app/workspaces/{session_id}
    success, output = execute_code_via_socket(code, session_id)
    
    # Determine exit code based on success
    exit_code = 0 if success else 1
    
    # For stderr, only use it if there was an error
    stderr = ""
    stdout = output
    if output.startswith("Error:"):
        stderr = output
        stdout = ""
    
    # Check for any artifacts created (plots, CSVs, etc.)
    artifacts = []
    if os.path.exists(workdir):
        artifacts = [f for f in os.listdir(workdir) 
                    if f.endswith(('.png', '.jpg', '.csv', '.xlsx'))]
    
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "artifacts": artifacts
    }

# ---------- OpenAI call ----------
def chat_step(client: OpenAI,
              model: str,
              messages: List[Dict[str, str]],
              max_out_tokens: int = 2000) -> str:
    """Make a single chat completion call with temperature=1.0 as required."""
    params = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_out_tokens,  # Using max_completion_tokens as required
        "temperature": 1.0,  # Fixed at 1.0 as required
    }
    
    print(f"  Calling {model} with temperature=1.0, max_tokens={max_out_tokens}", file=sys.stderr)
    
    try:
        resp = client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr)
        raise

# ---------- Main generation loop ----------
def build_initialization_code(dataset_filename: str) -> str:
    """Creates the Python code to initialize the session, mimicking the Go app."""
    return f"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

uploaded_files = ['{os.path.basename(dataset_filename)}']
print("="*50)
print("POCKET STATISTICIAN SESSION INITIALIZED")
print("="*50)
print(f"Uploaded files detected: {{len(uploaded_files)}}")
for f in uploaded_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024  # Size in KB
        print(f"  ✓ {{f}} ({{size:.1f}} KB)")
    else:
        print(f"  ✗ {{f}} (not found)")
print("="*50)
print(f"Primary file for analysis: {{uploaded_files[0]}}")
print("Ready for statistical analysis!")
print("="*50)
"""

def generate_trace_for_prompt(client: OpenAI,
                              model: str,
                              prompt_data: Dict,
                              max_steps: int = 15,
                              work_root: str = "../workspaces") -> Dict:
    """Generate a complete trace for a single prompt."""
    trace_id = prompt_data.get('id', uuid.uuid4().hex[:8])
    session_id = f"{trace_id}"  # Simplified session ID to match Go agent pattern
    workdir = os.path.join(work_root, session_id)
    
    print(f"  Creating workspace: {workdir}", file=sys.stderr)
    os.makedirs(workdir, exist_ok=True)

    # Copy dataset to workspace (matching Go agent behavior)
    dataset_src_path = prompt_data['dataset_filename']
    dataset_basename = os.path.basename(dataset_src_path)
    if not os.path.exists(dataset_src_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_src_path}")
    
    dataset_dest_path = os.path.join(workdir, dataset_basename)
    shutil.copy(dataset_src_path, dataset_dest_path)
    print(f"  Copied dataset to workspace: {dataset_basename}", file=sys.stderr)

    # --- Initialize Session (matching Go agent) ---
    init_code = build_initialization_code(dataset_basename)
    print(f"  Initializing Python session for {session_id}...", file=sys.stderr)
    
    success, init_output = execute_code_via_socket(init_code, session_id)
    if not success:
        print(f"  WARNING: Session initialization had issues: {init_output}", file=sys.stderr)
    else:
        print(f"  Session initialized successfully", file=sys.stderr)
    
    init_obs_text = f"Exit Code: {0 if success else 1}\nOUTPUT:\n{sanitize_for_display(init_output)}\n"

    # --- Build Initial Messages ---
    # For OpenAI API, we need to use "user" role for execution results
    messages = [
        {"role": "system", "content": DEVSTRAL_SYSTEM},
        {"role": "user", "content": prompt_data['prompt']},
        {"role": "user", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"}
    ]
    
    # For the trace (final output), use the correct "tool" role to match Go agent format
    trace = [
        {"role": "system", "content": DEVSTRAL_SYSTEM},
        {"role": "user", "content": prompt_data['prompt']},
        {"role": "tool", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"}
    ]
    all_artifacts = []
    code_executions = 0

    print(f"  Starting conversation loop (max {max_steps} steps)...", file=sys.stderr)
    
    for step in range(max_steps):
        print(f"  Step {step + 1}/{max_steps} for {trace_id}", file=sys.stderr)
        
        try:
            assistant_response = chat_step(client, model, messages)
        except Exception as e:
            print(f"  ERROR: API call failed at step {step}: {e}", file=sys.stderr)
            break

        cleaned_response = re.sub(r"<execution_results>.*?</execution_results>\n?", "", assistant_response, flags=re.DOTALL).strip()
        if cleaned_response != assistant_response.strip():
            print("    Cleaned invalid <execution_results> tag from model output.", file=sys.stderr)
            assistant_response = cleaned_response

        # Add assistant response to both lists
        messages.append({"role": "assistant", "content": assistant_response})
        trace.append({"role": "assistant", "content": assistant_response})

        code = extract_tag(assistant_response, "python")
        if not code:
            if code_executions < 7: # Allow for more steps
                continuation = "Please continue with the next logical step of the analysis."
                
                # --- FINAL IMPROVEMENT: Context-Aware Nudging ---
                # Check the content of the last tool message to see if it was a data quality check.
                last_tool_message = trace[-1].get("content", "") if trace[-1].get("role") == "tool" else ""
                if "Dtypes:" in last_tool_message and "Missing values" in last_tool_message:
                     continuation = (
                         "Excellent, the data quality issues have been identified. "
                         "Now, please **clean the data** based on these findings. Handle the missing values, "
                         "correct any impossible data points, and then proceed with the analysis."
                     )

                messages.append({"role": "user", "content": continuation})
                trace.append({"role": "user", "content": continuation})
                continue
            else:
                break # Stop if it's still stalling after many steps
        
        # Execute the code
        obs = run_python_in_executor(code, session_id, workdir)
        
        obs_text = (
            f"Exit Code: {obs.get('exit_code')}\n"
            f"OUTPUT:\n{sanitize_for_display(obs.get('stdout',''))}\n"
        )
        
        if obs.get('stderr'):
            obs_text += f"ERROR:\n{obs.get('stderr')}\n"
        
        # For the OpenAI API, use "user" role to avoid the tool_calls requirement
        api_msg = {
            "role": "user",
            "content": f"<execution_results>\n{obs_text.strip()}\n</execution_results>"
        }
        
        # For the trace (final output), use "tool" role to match Go agent format
        tool_msg_for_trace = {
            "role": "tool",
            "content": f"<execution_results>\n{obs_text.strip()}\n</execution_results>"
        }
        
        messages.append(api_msg)
        trace.append(tool_msg_for_trace)

        # Track artifacts
        for artifact in obs.get("artifacts", []):
            if artifact not in all_artifacts:
                all_artifacts.append(artifact)
                print(f"    New artifact created: {artifact}", file=sys.stderr)
    
    # Add image tags to final assistant message if artifacts were created
    if trace and trace[-1].get("role") == "assistant" and all_artifacts:
        final_content = trace[-1].get("content", "")
        if "<image>" not in final_content:
            image_tags = "".join([f"\n<image>{os.path.basename(a)}</image>" 
                                 for a in all_artifacts if a.endswith(".png")])
            if image_tags:
                trace[-1]["content"] += image_tags
                print(f"  Added {len(all_artifacts)} image tags to final response", file=sys.stderr)
    
    print(f"  Trace generation complete: {code_executions} code blocks executed", file=sys.stderr)
    
    return {
        "messages": trace,
        "artifacts": all_artifacts,
        "focus_area": prompt_data.get("focus_area"),
        "data_context": prompt_data.get("data_context"),
        "prompt_id": prompt_data.get("id"),
    }

# ---------- CLI ----------
def load_prompts(path: str) -> List[Dict]:
    """Load prompts from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])

def test_executor_connection(workspaces_root: str = "../workspaces") -> bool:
    """Test if we can connect to the Python executor."""
    try:
        test_session = f"test-{uuid.uuid4().hex[:8]}"
        test_workspace = os.path.join(workspaces_root, test_session)
        
        print(f"Testing executor connection with session: {test_session}", file=sys.stderr)
        
        # CRITICAL: Create the workspace directory first (the executor expects it to exist)
        os.makedirs(test_workspace, exist_ok=True)
        print(f"Created test workspace: {test_workspace}", file=sys.stderr)
        
        # Try a simple test
        success, output = execute_code_via_socket("print('Connection test successful')", test_session)
        
        print(f"Test response - Success: {success}, Output length: {len(output)}", file=sys.stderr)
        print(f"Test output: '{output}'", file=sys.stderr)
        
        if success:
            if "Connection test successful" in output or "Success:" in output:
                print("✓ Executor connection test passed", file=sys.stderr)
                
                # Try a second test to verify session persistence
                success2, output2 = execute_code_via_socket("x = 42; print(f'x = {x}')", test_session)
                if success2 and "x = 42" in output2:
                    print("✓ Session persistence verified", file=sys.stderr)
                
                # Clean up test workspace
                try:
                    shutil.rmtree(test_workspace)
                    print(f"Cleaned up test workspace", file=sys.stderr)
                except:
                    pass
                
                return True
            else:
                print(f"✗ Unexpected output: {output}", file=sys.stderr)
                return False
        else:
            print(f"✗ Executor test failed: {output}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"✗ Cannot connect to executor: {e}", file=sys.stderr)
        return False

def main():
    ap = argparse.ArgumentParser(description="Generate agentic traces for fine-tuning Devstral.")
    ap.add_argument("--prompts", required=True, help="Path to prompts JSON file.")
    ap.add_argument("--output", default="traces.jsonl", help="Output JSONL file.")
    ap.add_argument("--max-steps", type=int, default=15, help="Max iterations per prompt.")
    ap.add_argument("--limit", type=int, default=0, help="Max number of prompts to process.")
    ap.add_argument("--workdir", default="../workspaces", help="Directory for artifacts (matches Docker mount ../workspaces:/app/workspaces).")
    ap.add_argument("--test-only", action="store_true", help="Only test executor connection.")
    args = ap.parse_args()
    
    # Test executor connection first
    if not test_executor_connection(args.workdir):
        print("ERROR: Cannot connect to Python executor. Is Docker container running?", file=sys.stderr)
        print("Run: cd docker && docker-compose up python-executor", file=sys.stderr)
        sys.exit(1)
    
    if args.test_only:
        print("Test completed successfully.", file=sys.stderr)
        sys.exit(0)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    model = "gpt-5"  # Using GPT-5 as required
    
    print(f"Using model: {model}", file=sys.stderr)
    
    # Load prompts
    all_prompts = load_prompts(args.prompts)
    
    # Check for existing traces to resume
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data.get('prompt_id'))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming. Found {len(processed_ids)} completed traces.", file=sys.stderr)
    
    # Filter out already processed prompts
    prompts_to_process = [p for p in all_prompts if p['id'] not in processed_ids]
    if args.limit > 0:
        prompts_to_process = prompts_to_process[:args.limit]
    
    print(f"Processing {len(prompts_to_process)} prompts with model {model}", file=sys.stderr)
    print(f"Temperature: 1.0 (fixed)", file=sys.stderr)
    print(f"Max completion tokens: 2000", file=sys.stderr)
    
    # Ensure workdir exists
    os.makedirs(args.workdir, exist_ok=True)
    
    # Process prompts
    with open(args.output, "a") as out:
        for i, p_data in enumerate(prompts_to_process, 1):
            prompt_id = p_data.get('id', 'unknown')
            try:
                print(f"\n[{i}/{len(prompts_to_process)}] Processing: {prompt_id}", file=sys.stderr)
                
                rec = generate_trace_for_prompt(
                    client,
                    model,
                    p_data,
                    args.max_steps,
                    args.workdir,
                )
                
                out.write(json.dumps(rec) + "\n")
                out.flush()
                
                print(f"[{i}/{len(prompts_to_process)}] ✓ Complete: {prompt_id}", file=sys.stderr)
                
                # Small delay between prompts to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[{i}/{len(prompts_to_process)}] ✗ ERROR on {prompt_id}: {e}", file=sys.stderr)
                error_rec = {"prompt_id": prompt_id, "error": str(e), "messages": []}
                out.write(json.dumps(error_rec) + "\n")
                out.flush()
    
    print(f"\n✓ Processing complete. Output saved to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()