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

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install dependency: pip install openai")

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
1.  First, state your observation from the execution results in natural language. If there was an error, explain it.
2.  Next, in 1-2 sentences, state your plan for the single next step.
3.  Finally, provide a short <python></python> block to execute that plan (try to stick to ≤15 lines, one logical step; exception: assumption checking may require more).

### Code Generation Enforcement
**CRITICAL**: If you state "I will now..." or "Next, I will..." you MUST include the corresponding <python></python> block in the SAME response. Never end a response with a statement of intent without the code to execute it.

Examples:
- WRONG: "Next, I will check proportional hazards assumption."
- RIGHT: "Next, I will check proportional hazards assumption using Schoenfeld residuals.
  <python>
  # Check proportional hazards
  ...
  </python>"

**Do not explicitly write "Observe:", "Plan:", or "Act:" in your response, try to keep the language natural.**

### Critical Early Stopping Points
**STOP and request clarification when you identify fundamental limitations:**

1. **Data Structure Mismatches**: If the requested analysis requires data structure that doesn't exist (e.g., time series analysis requested but only cross-sectional data available), IMMEDIATELY:
   - State the limitation clearly
   - Explain what data structure would be needed
   - Ask: "The data structure doesn't support [requested analysis]. Would you like me to: a) proceed with alternative analysis using available data, or b) wait for different data?"
   - DO NOT continue with extensive alternative analyses without explicit approval

2. **Degenerate Cases**: If key variables have no variation (e.g., all times = 0 in survival analysis, constant values, 100% missing):
   - Report the issue immediately
   - STOP further analysis
   - Ask for data verification or alternative approaches

3. **Feasibility Check First**: Before any feature engineering or modeling:
   - Verify the data can support the requested analysis type
   - If not feasible, STOP and report why
   - Wait for user direction before pivoting

Example: If asked for "rolling window features" but data has only one observation per entity, 
   STOP and report: "Data has single snapshots per entity, not time series. 
   Need multiple observations per entity over time for rolling features. 
   Should I analyze the snapshot data instead?"

### Efficiency Guidelines
- Prioritize identifying showstoppers over completeness
- When a fundamental limitation is found, provide a concise summary (< 5 lines) rather than extensive workarounds
- Default to asking for clarification rather than assuming alternative analyses are desired

### Analysis Efficiency Protocol
Balance thoroughness with efficiency using these checkpoints:

1. **After initial model fit**: If no predictors are significant (all p > 0.10), state this clearly and limit further exploration to 2-3 pre-specified interactions maximum.

2. **Model comparison stopping rule**: When testing model extensions (quadratic terms, interactions):
   - Test in order of plausibility (domain knowledge > statistical hints)
   - Stop testing after 3 consecutive non-significant extensions
   - Explicitly state: "No model improvements found after testing [list]. Proceeding with current best model."

3. **Diagnostic depth scaling**:
   - **Full diagnostics** on the primary/final model: all assumption checks, residual plots, influence analysis
   - **Quick diagnostics** on alternative models: AIC/BIC comparison, likelihood ratio test only
   - **Skip diagnostics** on clearly inferior models (AIC worse by >10)

### Statistical Test Workflow (MANDATORY)
For the PRIMARY model only, verify ALL assumptions before inference:
- Run assumption checks in a separate code block
- Report results explicitly 
- Justify test choice or alternative based on results
For ALTERNATIVE models during exploration:
- Check only deal-breaker assumptions (e.g., convergence, identifiability)
- Use information criteria for comparison

---
## Best Practices

### Data Handling
- The initialization code has already imported pandas, numpy, matplotlib, seaborn, scipy.stats, os, pyplot. No need to re-import unless there is an error. 
- List available files and load datasets explicitly.
- On first load, report: shape, column names, and df.head(3); round to 3 decimals.
- Check and address missing data before analysis.
- Never invent column names or values.
- **Never call display().** Use print() or df.head().round(3).to_string(index=False) for tabular output.

### Data Quality Gate (MANDATORY FIRST STEP)
After loading data, ALWAYS check for implausible values before any analysis:
- **Numeric variables**: Check for impossible values (negative ages, counts below zero, percentages outside 0-100)
- **Temporal variables**: Check for impossible dates, negative durations, future dates that shouldn't exist
- **Categorical variables**: Check for unexpected levels, excessive categories (>20), single-level factors

When implausible values are found:
1. Report the issue with counts and examples
2. Propose a handling strategy (exclude, cap, transform)
3. Document the decision and impact on sample size
4. ONLY proceed after resolving or explicitly acknowledging limitations

Example check:
<python>
# Check value plausibility
issues = []
if 'age' in df.columns and (df['age'] < 0).any():
    issues.append(f"Negative ages: n={(df['age'] < 0).sum()}, min={df['age'].min():.1f}")
if 'length_of_stay' in df.columns and (df['length_of_stay'] < 0).any():
    issues.append(f"Negative LOS: n={(df['length_of_stay'] < 0).sum()}")
if issues:
    print("Data quality issues found:")
    for issue in issues:
        print(f"  - {issue}")
</python>

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

## Generated Data Awareness
When working with synthetic or generated datasets:
- Expect and check for: impossible value combinations, perfect correlations, missing natural variation
- Be explicit about data limitations discovered
- Don't force sophisticated models on data that shows no real patterns
- Include a "Data Generation Artifacts" section in final summary if suspicious patterns found

Red flags in generated data:
- Correlations exactly 0.00 or 1.00
- Suspiciously round numbers (many values ending in 0 or 5)
- Impossible combinations (pregnant males, 200-year-old patients)
- Missing natural clustering or variation

---

## Output Guidelines
- Before each <python></python> block, write 1-2 sentences explaining what and why.
- **CRITICAL**: Your response MUST NOT contain <thought> or <action> tags.
- Use <python></python> for code only. **Ensure all XML tags are properly closed.**
- Final summary (outside <python>) must:
  - Interpret results in plain language
  - State assumption checks and limitations
  - Include generated plots as <image>plot_name.png</image>
  - Include a "Data Quality Notes" section if any implausible values were found
- DO NOT emit <image></image> tags before the final summary.
- When no significant effects are found after appropriate analysis, state clearly: "No significant associations found. The data may lack sufficient signal or the relationships may be more complex than the available variables can capture."
- **Your final response, which contains the summary and <image></image> tags, MUST NOT contain a <python></python> block. This is the signal that the analysis is complete.**

### Response Completeness Check
Before ending any response that isn't the final summary:
- If you've stated an intention ("I will...", "Next...", "Let me..."), you MUST include the code
- If analysis is incomplete and you're not waiting for user input, you MUST provide the next code block
- Only responses that are either:
  1. Final summaries (with <image> tags)
  2. Stopping for user clarification (fundamental issues)
  3. Executing promised code
  are valid endpoints

### Response Length Guidelines
- If analysis is feasible: proceed with full workflow
- If analysis is NOT feasible: 
  1. State the issue (1-2 sentences)
  2. Explain data requirements (1-2 sentences)  
  3. Ask for direction (1 sentence)
  4. STOP - do not continue without user input

## EXAMPLE FINAL SUMMARY:
## Analysis Complete
**Findings:**
1. Mean age = 34.5 years (N=150).
2. Test scores differed between groups (t=2.45, p=0.015, d=0.38, 95% CI [0.07, 0.69]).

**Conclusions:** Age appears to influence test performance.

**Data Quality Notes:** [If applicable]
- Removed 5 records with negative ages
- Capped 3 outliers at 99th percentile

**Files Generated:**
<image>age_distribution.png</image>
<image>test_scores_by_group.png</image>
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

# ---------- OpenAI chat function ----------
def chat_step(client: OpenAI,
              model: str,
              messages: List[Dict[str, str]],
              max_out_tokens: int = 128000) -> str:
    """Make a single chat completion call with temperature=1.0 as required."""
    params = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_out_tokens,  # Using max_completion_tokens as required
        "temperature": 1.0,  # Fixed at 1.0 as required
    }
    
    try:
        resp = client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr)
        raise

def generate_rag_memory(client: OpenAI, model: str, focus_area: str, data_context: str) -> str:
    """Uses the OpenAI API to generate a plausible RAG snippet, summary, or fact."""
    
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
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=128000,
            temperature=1
        )
        memory = resp.choices[0].message.content.strip()
        # Ensure the output starts with the correct prefix
        if memory.startswith("Fact:") or memory.startswith("Summary:") or memory.startswith("- assistant:"):
            # Add the leading dash for facts and summaries
            return f"- {memory}" if not memory.startswith("-") else memory
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

def generate_trace_for_prompt(client: OpenAI,
                              model: str,
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

    # Build initial messages for API and trace
    system_prompt = DEVSTRAL_SYSTEM
    
    # For the trace (final output), use the correct format
    trace = [{"role": "system", "content": DEVSTRAL_SYSTEM}, {"role": "user", "content": prompt_data['prompt']}]
    
    # Probabilistically inject RAG memory
    if random.random() < 0.25:
        print("    Injecting simulated RAG memory for this trace.", file=sys.stderr)
        focus = prompt_data.get("focus_area", "data analysis")
        context = prompt_data.get("data_context", "a dataset")
        rag_memory = generate_rag_memory(client, model, focus, context)
        rag_memory_block = f"<memory>\n{rag_memory}\n</memory>"
        
        # Update system prompt with memory
        system_prompt += "\n\n" + rag_memory_block
        trace.insert(1, {"role": "system", "content": rag_memory_block})

    # Build messages for API (OpenAI requires "user" role for execution results)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_data['prompt']},
        {"role": "user", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"}
    ]
    
    # Add to trace with correct "tool" role
    trace.append({"role": "tool", "content": f"<execution_results>\n{init_obs_text.strip()}\n</execution_results>"})
    
    all_artifacts = []

    for step in range(max_steps):
        print(f"    Step {step+1}/{max_steps}...", file=sys.stderr)
        assistant_response = chat_step(client, model, messages)
        
        # Replace markdown fences with XML tags
        markdown_match = re.search(r"```python\n(.*?)\n```", assistant_response, re.DOTALL)
        if markdown_match:
            code_content = markdown_match.group(1)
            assistant_response = (
                assistant_response[:markdown_match.start()] +
                f"<python>{code_content}</python>" +
                assistant_response[markdown_match.end():]
            )
            print("    Replaced markdown code block with <python> tags.", file=sys.stderr)

        if "<python>" in assistant_response and "</python>" not in assistant_response:
            print("    Error: Incomplete <python> block detected. Skipping this turn.", file=sys.stderr)
            continue # Skip to the next iteration of the loop

        # Clean any invalid execution_results tags
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

        # For OpenAI API, use "user" role
        api_msg = {"role": "user", "content": f"<execution_results>\n{obs_text.strip()}\n</execution_results>"}
        # For trace output, use "tool" role
        tool_msg_for_trace = {"role": "tool", "content": f"<execution_results>\n{obs_text.strip()}\n</execution_results>"}
        
        messages.append(api_msg)
        trace.append(tool_msg_for_trace)

        for artifact in obs.get("artifacts", []):
            if artifact not in all_artifacts: 
                all_artifacts.append(artifact)
    
    # Add image tags to final assistant message if artifacts were created
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
        
        os.makedirs(test_workspace, exist_ok=True)
        print(f"Created test workspace: {test_workspace}", file=sys.stderr)
        
        success, output = execute_code_via_socket("print('Connection test successful')", test_session)
        
        print(f"Test response - Success: {success}, Output length: {len(output)}", file=sys.stderr)
        print(f"Test output: '{output}'", file=sys.stderr)
        
        if success:
            if "Connection test successful" in output or "Success:" in output:
                print("✓ Executor connection test passed", file=sys.stderr)
                
                success2, output2 = execute_code_via_socket("x = 42; print(f'x = {x}')", test_session)
                if success2 and "x = 42" in output2:
                    print("✓ Session persistence verified", file=sys.stderr)
                
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    model = "gpt-5"  # Using GPT-5 as required
    
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
                    client,
                    model, 
                    p_data,
                    args.max_steps,
                    args.workdir
                )
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"[{i}/{len(prompts_to_process)}] ✓ Complete: {prompt_id}", file=sys.stderr)
                time.sleep(0.5)  # Small delay to avoid rate limiting
            except Exception as e:
                print(f"[{i}/{len(prompts_to_process)}] ✗ ERROR on {prompt_id}: {e}", file=sys.stderr)
                out.write(json.dumps({"prompt_id": prompt_id, "error": str(e), "messages": []}) + "\n")
                out.flush()
    
    print(f"\n✓ Processing complete. Output saved to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
