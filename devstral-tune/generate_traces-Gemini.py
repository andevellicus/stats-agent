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
from typing import List, Dict, Optional, Tuple

# ---------- Google Gemini client ----------
try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit("Install dependency: pip install google-generativeai")

# ---------- Constants ----------
EOM_TOKEN = "<|EOM|>"

# ---------- Devstral-style system prompt (aligned with Go app) ----------
DEVSTRAL_SYSTEM = """You are an expert statistical data analyst using Python. Rigor is mandatory; do not speculate or hallucinate.

If CSV or Excel files are uploaded, treat the first uploaded file as the primary dataset. Always load files by their exact provided names.

---

## Workflow Loop (repeat until complete)
1. **Observe**: Inspect the latest <execution_results></execution_results>. If there is an error, briefly explain it.
2. **Plan**: In 1-2 sentences, state the single next step toward the user's goal.
3. **Act**: Execute that step in a short <python></python> block (≤15 lines, one logical step).

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
- Before each <python> block, write 1-2 sentences explaining what and why.
- Use <python></python> for code only.
- Final summary (outside <python>) must:
  - Interpret results in plain language
  - State assumption checks and limitations
  - Include generated plots as <image>plot_name.png</image>
- Do not emit <image></image> tags before the final summary.
- Stop when sufficient evidence answers the question.
"""

# (All helper functions like extract_tag, execute_code_via_socket, run_python_in_executor, etc. are unchanged)
def extract_tag(s: str, tag: str) -> Optional[str]:
    m = re.search(f"<{tag}>(.*?)</{tag}>", s or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def sanitize_for_display(s: str, limit: int = 4000) -> str:
    s = s.strip()
    if len(s) > limit:
        return s[:limit] + "\n...[truncated]..."
    return s

def execute_code_via_socket(code: str, session_id: str, host: str = 'localhost', port: int = 9999) -> Tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(60)
            sock.connect((host, port))
            message = f"{session_id}|{code}"
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

# ---------- Gemini API call ----------
def chat_step(model_name: str,
              messages: List[Dict[str, str]]) -> str:
    """Make a single chat completion call using the Gemini API."""
    system_prompt = ""
    if messages and messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        history = messages[1:]
    else:
        history = messages

    gemini_history = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    print(f"  Calling {model_name} with temperature=1.0...", file=sys.stderr)

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        # --- CHANGE: Removed max_output_tokens ---
        config = genai.GenerationConfig(
            temperature=1.0
        )
        resp = model.generate_content(gemini_history, generation_config=config)
        return resp.text or ""
    except Exception as e:
        print(f"  Gemini API error: {e}", file=sys.stderr)
        raise

# (Main generation loop and helpers are unchanged)
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

    messages = [
        {"role": "system", "content": DEVSTRAL_SYSTEM},
        {"role": "user", "content": prompt_data['prompt']},
        {"role": "user", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"}
    ]
    trace = [
        {"role": "system", "content": DEVSTRAL_SYSTEM},
        {"role": "user", "content": prompt_data['prompt']},
        {"role": "tool", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"}
    ]
    all_artifacts = []
    code_executions = 0
    consecutive_errors = 0

    for step in range(max_steps):
        assistant_response = chat_step(model_name, messages)

        cleaned_response = re.sub(r"<execution_results>.*?</execution_results>\n?", "", assistant_response, flags=re.DOTALL).strip()
        if cleaned_response != assistant_response.strip():
            print("    Cleaned invalid <execution_results> tag from model output.", file=sys.stderr)
            assistant_response = cleaned_response

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
        
        # --- IMPROVEMENT 3: Removed explicit error intervention ---
        # The script now relies on the model to see the error and self-correct.

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

# (CLI logic is unchanged, except for model name and API key)
def load_prompts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])

def main():
    ap = argparse.ArgumentParser(description="Generate agentic traces for fine-tuning Devstral.")
    ap.add_argument("--prompts", required=True, help="Path to prompts JSON file.")
    ap.add_argument("--output", default="traces.jsonl", help="Output JSONL file.")
    ap.add_argument("--model", default="models/gemini-2.5-pro", help="Gemini model to use.")
    ap.add_argument("--max-steps", type=int, default=15, help="Max iterations per prompt.")
    ap.add_argument("--limit", type=int, default=0, help="Max number of prompts to process.")
    ap.add_argument("--workdir", default="../workspaces", help="Directory for artifacts.")
    args = ap.parse_args()
    
    # Check for Gemini API key
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