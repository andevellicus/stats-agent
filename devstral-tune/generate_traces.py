#!/usr/bin/env python3
"""
generate_traces.py
------------------

Given a JSON file of prompts and associated datasets, produce multi-turn
traces that match the specific format of the Go-based Devstral agent.

Each output line is a JSON object formatted for fine-tuning.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import uuid
import shutil
import glob
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install dependency: pip install openai")

# ---------- Devstral-style system prompt (aligned with Go app) ----------
DEVSTRAL_SYSTEM = """You are an expert statistical data analyst using Python. Rigor is mandatory; do not speculate or hallucinate.

If CSV or Excel files are uploaded, treat the first uploaded file as the primary dataset. Always load files by their exact provided names.

---

## CRITICAL: Multi-Step Workflow (MANDATORY)
You MUST work iteratively through multiple steps. DO NOT stop after loading data.
MINIMUM REQUIRED STEPS for any analysis:
1. Load and inspect the data
2. Check and clean data issues  
3. Perform initial analysis
4. Iterate/refine based on findings
5. Create visualizations
6. Provide final interpretation

## Workflow Loop (repeat until complete)
1. **Observe**: Inspect the latest <execution_results></execution_results>. If there is an error, briefly explain it.
2. **Plan**: In 1-2 sentences, state the single next step toward the user's goal.
3. **Act**: Execute that step in a short <python></python> block (≤15 lines, one logical step).

After EVERY execution result, you MUST continue with another <python> block unless you have:
- Completed all analysis requested
- Created all necessary visualizations  
- Saved all plots to PNG files

---

## Best Practices
- On first load, report: shape, column names, and df.head(3).
- Check and address missing data before analysis.
- If you see data issues (negative values where impossible, wrong types), fix them before proceeding.
- **Never call display().** Use print() or df.head().round(3).to_string(index=False) for tabular output.
- Save plots to PNG files and close them: plt.savefig("plot.png"); plt.close().
- For statistical modeling, always verify required libraries are imported (statsmodels, scipy, etc.)
- When the analysis is complete, provide a final summary interpreting the results in plain language and include generated plots as <image>plot_name.png</image>.
"""

# ---------- Helpers to parse model output ----------
TAG_RE = {
    "python": re.compile(r"<python>(.*?)</python>", re.DOTALL | re.IGNORECASE),
}

def extract_tag(s: str, tag: str) -> Optional[str]:
    m = TAG_RE.get(tag, re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)).search(s or "")
    return m.group(1).strip() if m else None

def has_tag(s: str, tag: str) -> bool:
    return TAG_RE.get(tag, re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)).search(s or "") is not None

def sanitize_for_display(s: str, limit: int = 4000) -> str:
    s = s.strip()
    if len(s) > limit:
        return s[:limit] + "\n...[truncated]..."
    return s

# ---------- Sandboxed execution ----------
SANDBOX_PREAMBLE = r"""
# --- sandbox preamble: setup and imports ---
import builtins, os, sys, types, warnings

# Set matplotlib backend before any matplotlib import
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings('ignore')

# Import all data science libraries FIRST
# This ensures they can import their dependencies (including network modules) cleanly
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import statistical libraries
try:
    import scipy
    import scipy.stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    pass  # Not all traces may need these

# NOW apply security restrictions after all imports are complete

# Block system/process operations
def _blocked(*a, **k): 
    raise RuntimeError("system/subprocess operations disabled for security")

os.system = _blocked
os.popen = _blocked
os.spawn = _blocked
os.spawnl = _blocked
os.spawnle = _blocked
os.spawnlp = _blocked
os.spawnlpe = _blocked
os.spawnv = _blocked
os.spawnve = _blocked
os.spawnvp = _blocked
os.spawnvpe = _blocked

try:
    import subprocess
    subprocess.run = _blocked
    subprocess.Popen = _blocked
    subprocess.call = _blocked
    subprocess.check_call = _blocked
    subprocess.check_output = _blocked
    subprocess.getoutput = _blocked
    subprocess.getstatusoutput = _blocked
except ImportError:
    pass

# Block network operations
def _no_network(*a, **k):
    raise RuntimeError("network operations disabled for security")

import socket
# Save original for any already-imported modules that might need it internally
_original_socket = socket.socket
# But prevent new connections
socket.socket = _no_network
socket.create_connection = _no_network
socket.getaddrinfo = _no_network
socket.gethostbyname = _no_network
socket.gethostbyaddr = _no_network
socket.gethostname = lambda: 'localhost'  # Allow this harmless operation

# Block file operations outside working directory (optional, commented out for now)
# _original_open = open
# def _restricted_open(file, mode='r', *args, **kwargs):
#     # Could add path checks here
#     return _original_open(file, mode, *args, **kwargs)
# builtins.open = _restricted_open
"""

RUNNER_TEMPLATE = r"""
import sys, io, json, os, resource, signal, traceback, glob

# resource limits (Linux/Unix only)
try:
    # CPU seconds
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
    # Address space / memory (bytes)
    resource.setrlimit(resource.RLIMIT_AS, ({mem_limit}, {mem_limit}))
except Exception:
    pass

stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
_old_out, _old_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = stdout_buf, stderr_buf

exit_code = 0
try:
    # user code starts here
{code}
except Exception:
    exit_code = 1
    traceback.print_exc()

sys.stdout.flush(); sys.stderr.flush()
sys.stdout, sys.stderr = _old_out, _old_err

artifacts = []
for ext in ("*.png","*.csv","*.tsv","*.txt","*.json"):
    artifacts.extend(glob.glob(ext))

print(json.dumps({{
    "exit_code": exit_code,
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "artifacts": artifacts
}}))
"""

def run_python_safely(code: str,
                      workdir: str,
                      timeout_sec: int = 60,
                      cpu_limit: int = 30,
                      mem_limit_mb: int = 4096) -> Dict:
    """Execute Python code in a sandboxed environment."""
    os.makedirs(workdir, exist_ok=True)
    runner_py = os.path.join(workdir, "runner.py")
    mem_bytes = mem_limit_mb * 1024 * 1024
    
    # Combine sandbox preamble with user code
    code_block = SANDBOX_PREAMBLE + "\n\n" + code.strip() + "\n"
    
    # Fixed: Use proper indentation (4 spaces)
    runner = RUNNER_TEMPLATE.format(
        cpu_limit=cpu_limit,
        mem_limit=mem_bytes,
        code=textwrap.indent(code_block, "    ")  # Fixed indentation
    )
    
    with open(runner_py, "w", encoding="utf-8") as f:
        f.write(runner)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    try:
        res = subprocess.run(
            [sys.executable, "runner.py"],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "timeout": True,
            "stdout": "",
            "stderr": "Execution timed out",
            "artifacts": [],
            "exit_code": 124
        }
    
    # Parse the JSON output from runner
    try:
        # Get the last line which should be JSON
        lines = res.stdout.strip().splitlines()
        if lines:
            return json.loads(lines[-1])
        else:
            return {
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr or "No output produced",
                "artifacts": []
            }
    except (json.JSONDecodeError, IndexError) as e:
        return {
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": f"Failed to parse runner output: {e}\n{res.stderr}",
            "artifacts": []
        }

# ---------- OpenAI call ----------
def chat_step(client: OpenAI,
              model: str,
              messages: List[Dict[str, str]],
              temperature: float = 1,  # Increased default
              max_out_tokens: int = 2000) -> str:  # Increased default
    """Make a single chat completion call."""
    params = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_out_tokens,
        "temperature": temperature,
    }
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content or ""

# ---------- Main generation loop ----------
def build_initial_messages(system_prompt: str,
                           user_prompt: str,
                           dataset_filename: str) -> List[Dict[str, str]]:
    """Build initial message list for trace."""
    user_prompt_with_data = (
        f"Here is the task: {user_prompt}\n\n"
        f"The data for this task is in the file '{os.path.basename(dataset_filename)}'. "
        f"Please load and analyze this file."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_with_data},
    ]

def generate_trace_for_prompt(client: OpenAI,
                              model: str,
                              prompt_data: Dict,
                              max_steps: int = 15,  # Increased
                              work_root: str = "./runs",
                              python_timeout: int = 60,
                              min_steps: int = 5,  # New: minimum steps required
                              force_multi_step: bool = True) -> Dict:
    """Generate a complete trace for a single prompt."""
    # Create unique working directory
    trace_id = prompt_data.get('id', uuid.uuid4().hex[:8])
    workdir = os.path.join(work_root, f"trace_{trace_id}")
    os.makedirs(workdir, exist_ok=True)
    
    # Copy dataset to working directory
    dataset_src_path = prompt_data['dataset_filename']
    dataset_basename = os.path.basename(dataset_src_path)
    dataset_dest_path = os.path.join(workdir, dataset_basename)
    
    if not os.path.exists(dataset_src_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_src_path}")
    
    shutil.copy(dataset_src_path, dataset_dest_path)
    
    # Initialize messages and trace
    messages = build_initial_messages(
        DEVSTRAL_SYSTEM,
        prompt_data['prompt'],
        dataset_src_path
    )
    trace = list(messages)
    all_artifacts = []
    
    # Main generation loop
    code_executions = 0
    for step in range(max_steps):
        # Get assistant response
        try:
            # Use higher temperature for more varied responses
            temp = 1 if force_multi_step else 0.5
            assistant_response = chat_step(client, model, messages, temperature=temp, max_out_tokens=2000)
        except Exception as e:
            print(f"API error at step {step} for {trace_id}: {e}", file=sys.stderr)
            break
        
        # Add to messages and trace
        messages.append({"role": "assistant", "content": assistant_response})
        trace.append({"role": "assistant", "content": assistant_response})
        
        # Check for Python code to execute
        code = extract_tag(assistant_response, "python")
        if not code:
            # If we haven't done enough steps and force_multi_step is True
            if code_executions < min_steps and force_multi_step:
                # Add a continuation prompt
                continuation = {
                    "role": "user",
                    "content": "Please continue with the next step of the analysis. Remember to clean the data if needed, then proceed with the statistical modeling."
                }
                messages.append(continuation)
                trace.append(continuation)
                continue
            else:
                # No code and we've done enough steps, probably finished
                break
        
        code_executions += 1
        
        # Execute code in sandbox
        obs = run_python_safely(
            code,
            workdir=workdir,
            timeout_sec=python_timeout
        )
        
        # Format execution results
        obs_text = (
            f"Exit Code: {obs.get('exit_code')}\n"
            f"STDOUT:\n{sanitize_for_display(obs.get('stdout',''))}\n"
        )
        
        # Only add stderr if present
        stderr = obs.get('stderr', '').strip()
        if stderr:
            obs_text += f"STDERR:\n{sanitize_for_display(stderr)}"
        
        obs_text = obs_text.strip()
        
        # Add tool response
        tool_msg = {
            "role": "tool",
            "content": f"<execution_results>\n{obs_text}\n</execution_results>"
        }
        messages.append(tool_msg)
        trace.append(tool_msg)
        
        # Track artifacts
        for artifact in obs.get("artifacts", []):
            if artifact not in all_artifacts:
                all_artifacts.append(artifact)
        
        # Stop if execution failed
        if obs.get('exit_code', 0) != 0 and step > 0:
            # Allow one retry for errors
            pass
    
    # Add image tags if needed
    if trace and trace[-1].get("role") == "assistant":
        final_content = trace[-1].get("content", "")
        
        # Check if images exist but aren't referenced
        if "<image>" not in final_content:
            image_tags = ""
            for artifact in all_artifacts:
                if artifact.endswith(".png"):
                    image_tags += f"\n<image>{os.path.basename(artifact)}</image>"
            
            if image_tags:
                # Update both trace and messages
                final_content_with_images = final_content + image_tags
                trace[-1]["content"] = final_content_with_images
                # Also update messages for consistency
                if messages[-1].get("role") == "assistant":
                    messages[-1]["content"] = final_content_with_images
    
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

def main():
    ap = argparse.ArgumentParser(
        description="Generate agentic traces for fine-tuning Devstral."
    )
    ap.add_argument("--prompts", required=True, help="Path to prompts JSON file.")
    ap.add_argument("--output", default="traces.jsonl", help="Output JSONL file.")
    ap.add_argument("--model", default="gpt-5", help="Model to use for generation.")
    ap.add_argument("--max-steps", type=int, default=10, help="Max iterations per prompt.")
    ap.add_argument("--limit", type=int, default=0, help="Max number of prompts to process.")
    ap.add_argument("--workdir", default="./runs", help="Directory for artifacts.")
    ap.add_argument("--timeout", type=int, default=60, help="Python execution timeout.")
    args = ap.parse_args()
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Load prompts
    all_prompts = load_prompts(args.prompts)
    
    # Check for existing traces (for resuming)
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
    
    # Filter prompts to process
    prompts_to_process = [p for p in all_prompts if p['id'] not in processed_ids]
    if args.limit > 0:
        prompts_to_process = prompts_to_process[:args.limit]
    
    print(f"Processing {len(prompts_to_process)} prompts", file=sys.stderr)
    
    # Ensure work directory exists
    os.makedirs(args.workdir, exist_ok=True)
    
    # Process each prompt
    with open(args.output, "a") as out:
        for i, p_data in enumerate(prompts_to_process, 1):
            prompt_id = p_data.get('id', 'unknown')
            
            try:
                print(f"[{i}/{len(prompts_to_process)}] Processing: {prompt_id}", 
                      file=sys.stderr)
                
                # Generate trace
                rec = generate_trace_for_prompt(
                    client,
                    args.model,
                    p_data,
                    args.max_steps,
                    args.workdir,
                    args.timeout,
                    min_steps=5,  # Require at least 5 steps
                    force_multi_step=True  # Force continuation
                )
                
                # Write to output
                out.write(json.dumps(rec) + "\n")
                out.flush()
                
                print(f"[{i}/{len(prompts_to_process)}] ✓ Complete: {prompt_id}",
                      file=sys.stderr)
                
            except Exception as e:
                print(f"[{i}/{len(prompts_to_process)}] ✗ ERROR on {prompt_id}: {e}",
                      file=sys.stderr)
                # Optionally write error record
                error_rec = {
                    "prompt_id": prompt_id,
                    "error": str(e),
                    "messages": []
                }
                out.write(json.dumps(error_rec) + "\n")
                out.flush()

if __name__ == "__main__":
    main()