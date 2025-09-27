#!/usr/bin/env python3
"""
generate_traces.py
------------------

Given a JSON file of prompts and associated datasets, produce multi-turn
traces for fine-tuning a statistical agent. This version includes a
probabilistic simulation of RAG, generating both "Fact" and "Summary"
memory types, and instructs the model on how to use this memory.
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
import random
from typing import List, Dict, Optional, Tuple

# ---------- Google Gemini client ----------
try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit("Install dependency: pip install google-generativeai")

# ---------- Constants ----------
EOM_TOKEN = "<|EOM|>"
DEFAULT_EXECUTOR_ADDRESS = "localhost:9999"
DEFAULT_EXECUTOR_PORT = 9999
EXECUTOR_HOST = "localhost"
EXECUTOR_PORT = DEFAULT_EXECUTOR_PORT


def configure_executor_target(cli_address: Optional[str] = None) -> Tuple[str, int]:
    """Resolve the Python executor host/port using CLI args or env vars."""

    global EXECUTOR_HOST, EXECUTOR_PORT

    address = (cli_address or os.environ.get("PYTHON_EXECUTOR_ADDRESS") or "").strip()

    if not address:
        env_host = os.environ.get("PYTHON_EXECUTOR_HOST", "").strip()
        env_port = os.environ.get("PYTHON_EXECUTOR_PORT", "").strip()
        if env_host:
            fallback_port = env_port or str(DEFAULT_EXECUTOR_PORT)
            address = f"{env_host}:{fallback_port}"

    if not address:
        address = DEFAULT_EXECUTOR_ADDRESS

    if ":" not in address:
        port_candidate = os.environ.get("PYTHON_EXECUTOR_PORT", str(DEFAULT_EXECUTOR_PORT))
        address = f"{address}:{port_candidate}"

    host_part, port_part = address.rsplit(":", 1)
    host_part = host_part.strip() or "localhost"

    try:
        port_val = int(port_part)
    except ValueError as exc:
        raise ValueError(f"Invalid executor port '{port_part}' in address '{address}'") from exc

    EXECUTOR_HOST = host_part
    EXECUTOR_PORT = port_val
    print(f"Using Python executor at {EXECUTOR_HOST}:{EXECUTOR_PORT}", file=sys.stderr)
    return EXECUTOR_HOST, EXECUTOR_PORT

DEVSTRAL_SYSTEM = """You are an expert statistical data analyst using Python. Rigor is mandatory; do not speculate or hallucinate.

If CSV or Excel files are uploaded, treat the first uploaded file as the primary dataset. Always load files by their exact provided names.

---

## Using Memory
If a <memory></memory> block is provided with facts or summaries from a past analysis, use this information to inform your plan.

---

## Workflow Loop (repeat until complete)
After receiving <execution_results></execution_results>, your response must follow this sequence:
1.  First, state your observation from the execution results. If there was an error, explain it.
2.  Next, in 1-2 sentences, state your plan for the single next step.
3.  Finally, provide a short <python></python> block to execute that plan (≤15 lines, one logical step).

**Do not explicitly write "Observe:", "Plan:", or "Act:" in your response.**

**Critical enforcement**:
- If you intend to run a statistical test, you must first run and report assumption checks in a separate Act step. Do not run the test until you have printed the assumption results and justified the test choice.

---
## Best Practices

### Data Handling
- Import once per session: pandas, numpy, matplotlib, seaborn, scipy. The initialization code has already imported these.
- List available files and load datasets explicitly.
- On first load, report: shape, column names, and df.head(3); round to 3 decimals.
- Check and address missing data before analysis.
- Never invent column names or values.
- **Never call display().** Use print() or df.head().round(3).to_string(index=False) for tabular output.

### Statistical Rigor (Mandatory Assumptions)
Never run a test without verifying assumptions and reporting the results first.

**Parametric tests (t-test, ANOVA, linear regression):**
- Normality: Shapiro-Wilk on residuals (or KS if N > 200); also produce a histogram or QQ-check if plotting is part of the plan.
- Homoscedasticity: Levene's (or Bartlett's when normality is satisfied).
- Independence: justify based on study design; for regression, examine residual patterns.
- Only proceed with the parametric test if assumptions are satisfied; otherwise choose a nonparametric/exact alternative and justify.

**Nonparametric alternatives:**
- Two groups: Mann-Whitney U
- >2 groups: Kruskal-Wallis (+post-hoc with correction)

**Categorical tests:**
- Chi-square requires ≥80% of expected cells ≥5 and no cell <1. If violated, use Fisher's exact (or Monte Carlo).

**Time-to-event:**
- Use Kaplan-Meier/log-rank; check proportional hazards before Cox (e.g., Schoenfeld residuals).

**All tests—reporting requirements:**
- N, counts, and percentages where relevant
- Test statistic and exact p-value
- Effect size with 95% CI (e.g., Cohen's d/Hedges' g; OR/RR with CI; η²; r; Cramér's V)
- Explicit statement of assumption-check outcomes

If assumptions fail and no valid alternative exists, stop and explain why.

### Visualization
- You may use seaborn to construct plots, but always save/close with matplotlib.
- Never call plt.show().
- Save/close pattern:
  plt.savefig("plot_name.png")
  plt.close()

---

## Output Guidelines
- Before each <python></python> block, write 1-2 sentences explaining what and why.
- Use <python></python> for code only.
- Final summary (outside <python>) must:
  - Interpret results in plain language
  - State assumption checks and limitations
  - Include generated plots as <image>plot_name.png</image>
- Do not emit <image></image> tags before the final summary.
- Stop when sufficient evidence answers the question.
- **CRITICAL**: Your response must not contain <thought></thought> or <action></action> tags.
"""

# ---------- Helper functions ----------
def extract_tag(s: str, tag: str) -> Optional[str]:
    m = re.search(f"<{tag}>(.*?)</{tag}>", s or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def sanitize_for_display(s: str, limit: int = 4000) -> str:
    s = s.strip()
    if len(s) > limit:
        return s[:limit] + "\n...[truncated]..."
    return s

def execute_code_via_socket(code: str, session_id: str) -> Tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(60)
            sock.connect((EXECUTOR_HOST, EXECUTOR_PORT))
            message = f"{session_id}|{code}{EOM_TOKEN}"
            sock.sendall(message.encode('utf-8'))
            response_chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk: break
                decoded_chunk = chunk.decode('utf-8', errors='replace')
                response_chunks.append(decoded_chunk)
                if EOM_TOKEN in decoded_chunk: break
            full_response = ''.join(response_chunks).replace(EOM_TOKEN, '').strip()
            if full_response.startswith("Error:"):
                return False, full_response
            return True, full_response or "Success: Code executed with no output."
    except (socket.error, Exception) as e:
        return False, f"Executor connection failed: {e}"

def run_python_in_executor(code: str, session_id: str, workdir: str) -> Dict:
    os.makedirs(workdir, exist_ok=True)
    success, output = execute_code_via_socket(code, session_id)
    stderr = output if output.startswith("Error:") else ""
    stdout = "" if stderr else output
    artifacts = [os.path.join(workdir, f) for f in os.listdir(workdir) if f.endswith(('.png', '.csv'))]
    return {"exit_code": 0 if success else 1, "stdout": stdout, "stderr": stderr, "artifacts": artifacts}

def chat_step(model_name: str, messages: List[Dict[str, str]], system_prompt: str) -> str:
    gemini_history = []
    for msg in messages:
        if msg["role"] == "system": continue
        role = "model" if msg["role"] == "assistant" else msg["role"]
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    try:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
        config = genai.GenerationConfig(temperature=1.0)
        resp = model.generate_content(gemini_history, generation_config=config)
        return resp.text or ""
    except Exception as e:
        print(f"  Gemini API error: {e}", file=sys.stderr)
        raise

def generate_rag_memory(model_name: str, focus_area: str, data_context: str) -> str:
    """Uses the Gemini API to generate a plausible RAG snippet, summary, or fact."""
    
    rand_val = random.random()
    
    if rand_val < 0.33:
        # Generate a concise, one-sentence Fact
        prompt = (
            f"Based on the following keywords, create a concise, one-sentence summary of a past analysis. "
            f"The summary MUST start with 'Fact:'.\n\n"
            f"**CRITICAL**: DO NOT include <thought> or <action> tags. "
            f"Keywords:\n- Focus Area: {focus_area}\n- Data Context: {data_context}\n\n"
            f"Example Output: Fact: A previous analysis on customer behavior showed that checking for data outliers was a critical step."
        )
        fallback = f"- fact: A prior analysis focused on {focus_area}."
    elif rand_val < 0.66:
        # Generate a slightly more narrative, multi-sentence Summary
        prompt = (
            f"Based on the following keywords, create a short, 2-3 sentence summary of a past analysis. "
            f"The summary MUST start with 'Summary:'.\n\n"
            f"**CRITICAL**: DO NOT include <thought> or <action> tags. "
            f"Keywords:\n- Focus Area: {focus_area}\n- Data Context: {data_context}\n\n"
            f"Example Output: Summary: A previous analysis on customer behavior explored churn prediction. The key finding was that tenure and support ticket volume were the most significant predictors."
        )
        fallback = f"- summary: A prior analysis on {data_context} focused on {focus_area}."
    else:
        # Generate a realistic, multi-turn snippet
        prompt = (
            f"Based on the following keywords, create a realistic, multi-turn conversational snippet from a past analysis. "
            f"The snippet should include one 'assistant' turn with a thought and a python block, and one 'tool' turn with a plausible execution result. "
            f"It MUST be formatted with a leading dash and role, like '- assistant:' and '- tool:'.\n\n"
            f"**CRITICAL**: DO NOT include <thought> or <action> tags. "
            f"Keywords:\n- Focus Area: {focus_area}\n- Data Context: {data_context}\n\n"
            f"Example Output:\n"
            f"- assistant: The data for {data_context} has been loaded. I will now check for missing values. <python>df.isnull().sum()</python>\n"
            f"- tool: <execution_results>\nSTDOUT:\n{data_context}_id      0\n{data_context}_value    12\ndtype: int64\n</execution_results>"
        )
        fallback = (
            f"- assistant: A prior analysis on {data_context} involved {focus_area}. I will check for missing data first. <python>df.isnull().sum()</python>\n"
            f"- tool: <execution_results>STDOUT:\ncolumn_a    0\ncolumn_b    5\ndtype: int64</execution_results>"
        )

    try:
        model = genai.GenerativeModel(model_name=model_name)
        config = genai.GenerationConfig(temperature=0.9)
        resp = model.generate_content(prompt, generation_config=config)
        # Check for content before accessing .text
        if resp.parts:
            memory = resp.text.strip()
            # Ensure the output starts with the correct prefix
            if memory.startswith("Fact:") or memory.startswith("Summary:") or memory.startswith("- assistant:"):
                 # Add the leading dash for facts and summaries
                return f"- {memory}" if not memory.startswith("-") else memory
            return fallback
        else:
            print(f"  Could not generate RAG memory, no content in response. Finish Reason: {resp.candidates[0].finish_reason}", file=sys.stderr)
            return fallback

    except Exception as e:
        print(f"  Could not generate RAG memory: {e}", file=sys.stderr)
        return fallback

def build_initialization_code(dataset_filename: str) -> str:
    return f"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, scipy.stats as stats, warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
print(f"Session Initialized. Primary dataset: '{os.path.basename(dataset_filename)}'")
"""

def generate_trace_for_prompt(model_name: str,
                              prompt_data: Dict,
                              max_steps: int = 15,
                              work_root: str = "../workspaces") -> Dict:
    trace_id = prompt_data.get('id', uuid.uuid4().hex[:8])
    session_id = f"{trace_id}"
    workdir = os.path.join(work_root, session_id)

    os.makedirs(workdir, exist_ok=True)
    dataset_src_path = prompt_data['dataset_filename']
    dataset_basename = os.path.basename(dataset_src_path)
    shutil.copy(dataset_src_path, os.path.join(workdir, dataset_basename))

    init_code = build_initialization_code(dataset_basename)
    success, init_output = execute_code_via_socket(init_code, session_id)
    init_obs_text = f"Exit Code: {0 if success else 1}\nOUTPUT:\n{sanitize_for_display(init_output)}\n"

    trace = [{"role": "system", "content": DEVSTRAL_SYSTEM}, {"role": "user", "content": prompt_data['prompt']}]
    messages = [{"role": "user", "content": prompt_data['prompt']}]
    system_prompt = DEVSTRAL_SYSTEM

    if random.random() < 0.25:
        print("    Injecting simulated RAG memory for this trace.", file=sys.stderr)
        focus = prompt_data.get("focus_area", "data analysis")
        context = prompt_data.get("data_context", "a dataset")
        rag_memory = generate_rag_memory(model_name, focus, context)
        rag_memory_block = f"<memory>\n{rag_memory}\n</memory>"
        
        system_prompt += "\n\n" + rag_memory_block
        trace.insert(1, {"role": "system", "content": rag_memory_block})

    messages.append({"role": "user", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"})
    trace.append({"role": "tool", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"})
    
    all_artifacts = []

    for step in range(max_steps):
        print(f"    Step {step+1}/{max_steps}...", file=sys.stderr)
        assistant_response = chat_step(model_name, messages, system_prompt)
        
        # Before any other processing, replace markdown fences with XML tags.
        markdown_match = re.search(r"```python\n(.*?)\n```", assistant_response, re.DOTALL)
        if markdown_match:
            code_content = markdown_match.group(1)
            # Rebuild the response with the correct tags
            assistant_response = (
                assistant_response[:markdown_match.start()] +
                f"<python>{code_content}</python>" +
                assistant_response[markdown_match.end():]
            )
            print("    Replaced markdown code block with <python> tags.", file=sys.stderr)

        cleaned_response = re.sub(r"<execution_results>.*?</execution_results>\n?", "", assistant_response, flags=re.DOTALL).strip()
        if cleaned_response != assistant_response.strip():
            print("    Cleaned invalid <execution_results> tag from model output.", file=sys.stderr)
            assistant_response = cleaned_response

        if "<python>" in assistant_response and "<image>" in assistant_response:
            print("    Warning: Stripping premature <image> tags from model output.", file=sys.stderr)
            assistant_response = re.sub(r"<image>.*?</image>\n?", "", assistant_response, flags=re.DOTALL).strip()

        messages.append({"role": "assistant", "content": assistant_response})
        trace.append({"role": "assistant", "content": assistant_response})

        code = extract_tag(assistant_response, "python")
        if not code:
            print("    No Python code found in response. Ending trace.", file=sys.stderr)
            break
        
        obs = run_python_in_executor(code, session_id, workdir)
        obs_text = f"Exit Code: {obs['exit_code']}\nOUTPUT:\n{sanitize_for_display(obs['stdout'])}\n"
        if obs['stderr']:
            obs_text += f"ERROR:\n{obs['stderr']}\n"

        api_msg = {"role": "user", "content": f"<execution_results>\n{obs_text.strip()}\n</execution_results>"}
        tool_msg_for_trace = {"role": "tool", "content": f"<execution_results>\n{obs_text.strip()}\n</execution_results>"}
        messages.append(api_msg)
        trace.append(tool_msg_for_trace)

        for artifact in obs.get("artifacts", []):
            if artifact not in all_artifacts: all_artifacts.append(artifact)
    
    if trace and trace[-1].get("role") == "assistant" and all_artifacts:
        image_tags = "".join([f'\n<image>{os.path.basename(a)}</image>' for a in all_artifacts if a.endswith(".png")])
        if image_tags and "<image>" not in trace[-1]["content"]:
            trace[-1]["content"] += image_tags
    
    return {"messages": trace, "artifacts": all_artifacts, "prompt_id": prompt_data.get("id")}

def test_executor_connection(workspaces_root: str = "../workspaces") -> bool:
    """Test if we can connect to the Python executor."""
    try:
        test_session = f"test-{uuid.uuid4().hex[:8]}"
        test_workspace = os.path.join(workspaces_root, test_session)
        
        print(
            f"Testing executor connection to {EXECUTOR_HOST}:{EXECUTOR_PORT} with session: {test_session}",
            file=sys.stderr,
        )
        
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

def load_prompts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])

def main():
    ap = argparse.ArgumentParser(description="Generate agentic traces for fine-tuning.")
    ap.add_argument("--prompts", required=True, help="Path to prompts JSON file.")
    ap.add_argument("--output", default="traces.jsonl", help="Output JSONL file.")
    ap.add_argument("--model", default="models/gemini-2.5-pro", help="Gemini model to use.")
    ap.add_argument("--max-steps", type=int, default=15, help="Max iterations per prompt.")
    ap.add_argument("--limit", type=int, default=0, help="Max number of prompts to process.")
    ap.add_argument("--workdir", default="../workspaces", help="Directory for artifacts.")
    ap.add_argument("--executor-address", default=None, help="Override PYTHON_EXECUTOR_ADDRESS (host:port).")
    ap.add_argument("--test-only", action="store_true", help="Only test executor connection.")
    args = ap.parse_args()

    configure_executor_target(args.executor_address)

    # Test executor connection first
    if not test_executor_connection(args.workdir):
        print("ERROR: Cannot connect to Python executor. Is Docker container running?", file=sys.stderr)
        print("Run: cd docker && docker-compose up python-executor", file=sys.stderr)
        sys.exit(1)
    
    if args.test_only:
        print("Test completed successfully.", file=sys.stderr)
        sys.exit(0)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Please set GOOGLE_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    model = args.model
    
    print(f"Using model: {model}", file=sys.stderr)
    all_prompts = load_prompts(args.prompts)
    
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line).get('prompt_id'))
                except (json.JSONDecodeError, KeyError):
                    continue
        if processed_ids:
            print(f"Resuming. Found {len(processed_ids)} completed traces.", file=sys.stderr)

    prompts_to_process = [p for p in all_prompts if p['id'] not in processed_ids]
    if args.limit > 0:
        prompts_to_process = prompts_to_process[:args.limit]
    
    print(f"Processing {len(prompts_to_process)} prompts...", file=sys.stderr)
    os.makedirs(args.workdir, exist_ok=True)
    
    with open(args.output, "a") as out:
        for i, p_data in enumerate(prompts_to_process, 1):
            prompt_id = p_data.get('id', 'unknown')
            try:
                print(f"\n[{i}/{len(prompts_to_process)}] Processing: {prompt_id}", file=sys.stderr)
                rec = generate_trace_for_prompt(
                    model,
                    p_data,
                    args.max_steps,
                    args.workdir
                )
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"[{i}/{len(prompts_to_process)}] ✓ Complete: {prompt_id}", file=sys.stderr)
                time.sleep(1) # Rate limit
            except Exception as e:
                print(f"[{i}/{len(prompts_to_process)}] ✗ ERROR on {prompt_id}: {e}", file=sys.stderr)
                out.write(json.dumps({"prompt_id": prompt_id, "error": str(e), "messages": []}) + "\n")
                out.flush()
    
    print(f"\n✓ Processing complete. Output saved to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
