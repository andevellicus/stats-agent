#!/usr/bin/env python3
"""
Devstral prompt generator (fixed & hardened)
===========================================

Generates a comprehensive set of fine-tuning prompts for Devstral as a
stats + ML agent. This version uses an advanced two-step process:
1. An LLM generates a prompt AND a matching data schema.
2. The script populates that schema with synthetic data.
This guarantees perfect alignment between the prompt and the dataset.

Usage:
  export OPENAI_API_KEY=sk-...
  python generate_prompts.py \
      --num-prompts 200 \
      --batch-size 10 \
      --model gpt-4-turbo \
      --output prompts_with_data.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
try:
    # Newer SDK
    from openai import OpenAI
except Exception as e:
    OpenAI = None

# --- Catalogs ---
FOCUS_AREAS = [
    "classical_inference", "regression_glm", "mixed_effects", "survival",
    "causal_inference", "time_series", "clustering_dimred", "tree_boosting",
    "model_diagnostics", "feature_engineering", "cross_validation"
]
DATA_CONTEXTS = [
    "clinical_trials", "healthcare_ops", "marketing_campaigns", "customer_behavior",
    "financial_markets", "manufacturing_quality", "educational_assessment",
    "environmental_monitoring", "social_media_analytics", "genomic_data",
    "survey_responses", "sensor_measurements"
]
MAX_ITEMS_PER_CALL = 15

# --- Data Generation ---

def introduce_missing_values(df: pd.DataFrame, missing_fraction: float = 0.05) -> pd.DataFrame:
    df_out = df.copy()
    for col in df_out.select_dtypes(include=np.number).columns:
        if col.endswith('_id') or col in ['timestamp', 'time', 'event']:
            continue
        mask = np.random.rand(len(df_out[col])) < missing_fraction
        df_out.loc[mask, col] = np.nan
    return df_out

def generate_synthetic_data(schema: Dict[str, str], n_samples: int, focus_area: str) -> pd.DataFrame:
    """Generates a synthetic dataset based on a schema provided by an LLM."""
    rng = np.random.default_rng(seed=random.randint(0, 10000))
    data = {}
    
    for col_name, col_type in schema.items():
        if col_type == 'numeric':
            data[col_name] = rng.normal(loc=rng.uniform(0, 100), scale=rng.uniform(5, 50), size=n_samples)
        elif col_type == 'categorical':
            n_cats = random.randint(2, 5)
            cats = [f"{col_name}_{chr(65+i)}" for i in range(n_cats)]
            data[col_name] = rng.choice(cats, n_samples)
        elif col_type == 'binary':
            data[col_name] = rng.integers(0, 2, n_samples)
        elif col_type == 'identifier':
            data[col_name] = [f"ID_{i+1001}" for i in range(n_samples)]
        elif col_type == 'date':
            start_date = datetime(2022, 1, 1)
            data[col_name] = pd.to_datetime([start_date + pd.Timedelta(days=int(d)) for d in range(n_samples)])
        else: # Default to numeric
            data[col_name] = rng.normal(0, 1, n_samples)
            
    df = pd.DataFrame(data)
    
    # Post-processing for specific focus areas
    if focus_area == 'survival' and 'time' in df.columns and 'event' in df.columns:
        df['time'] = np.abs(df['time']) # Ensure time is positive
        df['event'] = df['event'].round().clip(0, 1).astype(int) # Ensure event is 0 or 1

    return introduce_missing_values(df)


# --- OpenAI Interaction ---

def build_system_prompt() -> str:
    return (
        "You are an assistant that generates training prompts for Devstral, an agentic "
        "statistics and machine-learning model. Devstral follows an Observe/Plan/Act loop: "
        "observe execution results, plan briefly, then execute a small <python></python> code block.\n\n"
        "Prompts must require: iterative analysis, assumption checks (normality, "
        "homoscedasticity, proportional hazards as relevant), data prep, model selection, "
        "effect sizes, confidence intervals, diagnostics, and clear reporting."
        "The prompts you create must be realistic user requests that require iterative "
        "analysis, data preparation, assumption checking, modeling, and clear reporting."
    )

def build_user_prompt(k: int, focus_areas: List[str], data_contexts: List[str]) -> str:
    schema_example = json.dumps({
        "prompt": "I've uploaded our latest marketing campaign data. Can you analyze the relationship between ad spend and conversions?",
        "focus_area": "regression_glm",
        "data_context": "marketing_campaigns",
        "schema": {
            "customer_id": "identifier",
            "campaign_spend": "numeric",
            "conversions": "numeric",
            "channel": "categorical"
        }
    })
    
    lines = [
        f"Generate a JSON object with a single key 'prompts'. This key should contain a list of exactly {k} objects.",
        "Each object must contain four keys: `prompt`, `focus_area`, `data_context`, and `schema`.",
        "Follow these rules for each object:",
        f"1. `focus_area` MUST be one of: {', '.join(focus_areas)}.",
        f"2. `data_context` MUST be one of: {', '.join(data_contexts)}.",
        "3. `prompt` MUST be a natural, conversational user request (2-5 sentences) that fits the chosen context and focus. It should NOT mention a filename.",
        "4. `schema` MUST be a JSON object where keys are column names and values are their data type. The schema must be directly relevant to the prompt.",
        "   - Valid data types are: 'numeric', 'categorical', 'binary', 'identifier', 'date'.",
        "   - The schema should contain between 4 and 8 columns.",
        f"\nHere is an example of a single perfect output object:\n{schema_example}",
        "\nReturn ONLY the JSON object with the list of prompts."
    ]
    return "\n".join(lines)


def chat_create(client: OpenAI, messages: List[dict], model: str, temperature: float) -> List[dict]:
    params = dict(model=model, messages=messages, temperature=temperature, response_format={"type": "json_object"})
    resp = client.chat.completions.create(**params)
    content = (resp.choices[0].message.content or "").strip()
    obj = json.loads(content)
    return obj.get("prompts", [])


# --- Main Application Logic ---

def save_checkpoint(all_prompts: List[dict], path: str, meta: Optional[dict] = None) -> None:
    """Saves the current list of prompts to the output file."""
    payload = {"metadata": meta or {}, "prompts": all_prompts}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    print(f"[checkpoint] Saved {len(all_prompts)} prompts to {path}", file=sys.stderr)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate aligned prompts and synthetic data schemas.")
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--model", type=str, default="gpt-5")
    ap.add_argument("--temperature", type=float, default=1)
    ap.add_argument("--output", type=str, default="prompts.json")
    ap.add_argument("--data-dir", type=str, default="synthetic_datasets")
    ap.add_argument("--checkpoint-interval", type=int, default=20, help="Save progress every N prompts")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)
    client = OpenAI()
    
    all_prompt_data: List[Dict[str, Any]] = []

    # --- Checkpoint Loading ---
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if "prompts" in existing_data:
                    all_prompt_data = existing_data["prompts"]
                    print(f"Resuming. Found {len(all_prompt_data)} completed prompts in {args.output}", file=sys.stderr)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint file {args.output}. Starting fresh. Error: {e}", file=sys.stderr)

    last_checkpoint = len(all_prompt_data)

    while len(all_prompt_data) < args.num_prompts:
        want = min(args.batch_size, args.num_prompts - len(all_prompt_data))
        
        focus_areas_sample = random.sample(FOCUS_AREAS, min(5, len(FOCUS_AREAS)))
        data_contexts_sample = random.sample(DATA_CONTEXTS, min(5, len(DATA_CONTEXTS)))
        
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(want, focus_areas_sample, data_contexts_sample)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        try:
            prompt_schemas = chat_create(client, messages, args.model, args.temperature)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}", file=sys.stderr)
            continue

        for prompt_obj in prompt_schemas:
            if len(all_prompt_data) >= args.num_prompts:
                break
            
            schema = prompt_obj.get("schema")
            if not schema or not isinstance(schema, dict):
                continue

            prompt_idx = len(all_prompt_data) + 1
            n_samples = random.randint(150, 500)

            df = generate_synthetic_data(schema, n_samples, prompt_obj.get("focus_area", ""))
            
            dataset_filename = f"dataset_{prompt_idx}.csv"
            dataset_filepath = os.path.join(args.data_dir, dataset_filename)
            df.to_csv(dataset_filepath, index=False)
            
            final_obj = {
                "id": f"prompt_{prompt_idx}",
                "prompt": prompt_obj["prompt"],
                "dataset_filename": dataset_filepath,
                "focus_area": prompt_obj.get("focus_area"),
                "data_context": prompt_obj.get("data_context"),
                "schema": schema,
            }
            all_prompt_data.append(final_obj)

        print(f"… Generated {len(all_prompt_data)}/{args.num_prompts} prompts", file=sys.stderr)

        # --- Checkpoint Saving ---
        if len(all_prompt_data) - last_checkpoint >= args.checkpoint_interval:
            meta = {"generated_at": datetime.now().isoformat(), "model": args.model, "temperature": args.temperature}
            save_checkpoint(all_prompt_data, args.output, meta)
            last_checkpoint = len(all_prompt_data)
    
    # Save the final file
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_prompts": len(all_prompt_data),
            "model": args.model,
            "temperature": args.temperature,
        },
        "prompts": all_prompt_data,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_prompt_data)} prompts and datasets to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()