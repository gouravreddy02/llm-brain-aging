"""
Post-Process LLM Results: JSONL → CSV
======================================
MISSING FROM ORIGINAL GITHUB (but required for the R analysis pipeline).

The original paper's R code expects a CSV with columns:
    eid, llm_overall_age, llm_cardiovascular_age, llm_hepatic_age, etc.

But the Python inference pipeline outputs JSONL with raw LLM text.
This script bridges that gap by:
    1. Reading the JSONL output from ad_aging_generate.py
    2. Parsing predicted ages from the raw LLM text (overall + organ-specific)
    3. Outputting a clean CSV that AD_MainAnalysis.R can load directly

USAGE:
    python post_process_results.py \
        --input_jsonl ./data/result/ukb_ad_overall_result.jsonl \
        --output_csv ./Data/Models/llama3_70b/llama3-70b-ad-result_only_age.csv \
        --prediction_type overall

    # For organ-specific results (run inference once with organ prompt):
    python post_process_results.py \
        --input_jsonl ./data/result/ukb_ad_organ_result.jsonl \
        --output_csv ./Data/Models/llama3_70b/llama3-70b-ad-result_only_age.csv \
        --prediction_type organ_specific
"""

import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process LLM JSONL → CSV")
    parser.add_argument('--input_jsonl', required=True, help='JSONL file from inference')
    parser.add_argument('--output_csv', required=True, help='Output CSV for R analysis')
    parser.add_argument('--prediction_type', default='overall',
                        choices=['overall', 'organ_specific', 'brain'],
                        help='Type of predictions to extract')
    parser.add_argument('--output_raw', default=None,
                        help='Optional: also save raw CoT reasoning to separate CSV')
    return parser.parse_args()


def extract_overall_age(text):
    """Extract a single overall biological age prediction."""
    if not text:
        return None

    patterns = [
        r'[Oo]verall\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
        r'[Bb]iological\s+age\s*(?:is|:)\s*(\d+)',
        r'[Pp]redicted\s+age\s*(?:is|:)\s*(\d+)',
        r'\*\*(\d+)\*\*',
        r'(\d+)\s*years?\s*old',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 30 <= age <= 120:
                return age

    # Fallback: last reasonable number
    numbers = re.findall(r'\b(\d+)\b', text)
    for n in reversed(numbers):
        val = int(n)
        if 30 <= val <= 120:
            return val
    return None


def extract_organ_specific_ages(text):
    """
    Extract organ-specific biological ages from LLM output.
    
    The original paper's prompt asks the LLM to predict ages for:
    overall, cardiovascular, hepatic, pulmonary, renal, metabolic, musculoskeletal
    
    The LLM typically outputs them in a structured way like:
        Overall biological age: 68
        Cardiovascular age: 72
        Hepatic age: 65
        ...
    """
    if not text:
        return {}

    organ_map = {
        'overall': r'[Oo]verall\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
        'cardiovascular': r'[Cc]ardiovascular\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
        'hepatic': r'[Hh]epatic\s+(?:biological\s+)?(?:liver\s+)?age\s*(?:is|:)\s*(\d+)',
        'pulmonary': r'[Pp]ulmonary\s+(?:biological\s+)?(?:lung\s+)?age\s*(?:is|:)\s*(\d+)',
        'renal': r'[Rr]enal\s+(?:biological\s+)?(?:kidney\s+)?age\s*(?:is|:)\s*(\d+)',
        'metabolic': r'[Mm]etabolic\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
        'musculoskeletal': r'[Mm]usculoskeletal\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
    }

    ages = {}
    for organ, pattern in organ_map.items():
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 30 <= age <= 120:
                ages[organ] = age
    return ages


def extract_brain_age(text):
    """Extract brain-specific biological age (NEW for AD project)."""
    if not text:
        return None

    patterns = [
        r'[Bb]rain\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
        r'[Nn]eurological\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
        r'[Cc]ognitive\s+(?:biological\s+)?age\s*(?:is|:)\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 30 <= age <= 120:
                return age
    return None


def process_jsonl(input_path, prediction_type):
    """Read JSONL and extract structured age predictions."""
    records = []
    failed = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Parsing predictions"):
        try:
            case = json.loads(line.strip())
        except json.JSONDecodeError:
            failed += 1
            continue

        eid = case.get("eid", None)
        raw_output = case.get("model_generated_aging_prediction", [""])[0]

        record = {"eid": eid}

        if prediction_type == "overall":
            record["llm_overall_age"] = extract_overall_age(raw_output)
            brain_age = extract_brain_age(raw_output)
            if brain_age is not None:
                record["llm_brain_age"] = brain_age

        elif prediction_type == "organ_specific":
            organ_ages = extract_organ_specific_ages(raw_output)
            for organ, age in organ_ages.items():
                record[f"llm_{organ}_age"] = age

        elif prediction_type == "brain":
            record["llm_brain_age"] = extract_brain_age(raw_output)
            record["llm_overall_age"] = extract_overall_age(raw_output)

        record["raw_output"] = raw_output
        records.append(record)

    if failed > 0:
        print(f"WARNING: {failed} lines could not be parsed as JSON")

    return records


def main():
    args = parse_args()

    print(f"Processing {args.input_jsonl}...")
    records = process_jsonl(args.input_jsonl, args.prediction_type)

    df = pd.DataFrame(records)

    # Report parsing success rate
    age_cols = [c for c in df.columns if c.startswith("llm_") and c.endswith("_age")]
    for col in age_cols:
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100 if len(df) > 0 else 0
        print(f"  {col}: {n_valid}/{len(df)} ({pct:.1f}%) successfully parsed")

    # Save CSV for R (without raw_output column)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_for_r = df.drop(columns=["raw_output"], errors="ignore")
    df_for_r.to_csv(args.output_csv, index=False)
    print(f"Saved age predictions to {args.output_csv}")

    # Optionally save raw output + reasoning
    if args.output_raw:
        df[["eid", "raw_output"]].to_csv(args.output_raw, index=False)
        print(f"Saved raw CoT reasoning to {args.output_raw}")


if __name__ == "__main__":
    main()
