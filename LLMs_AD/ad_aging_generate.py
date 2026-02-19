"""
AD-Specific Aging Prediction Pipeline
======================================
Adapted from Paper 1's aging_generate.py for Alzheimer's Disease prediction.

PIPELINE OVERVIEW (mapping to original code):
1. Load participant health data (JSONL format) -- same as original
2. Apply AD-specific prompt template          -- MODIFIED: AD-focused prompts
3. Run LLM inference via vLLM                 -- same engine as original
4. Parse outputs (biological age + AD risk)   -- NEW: extract AD-specific predictions
5. Save results                               -- same format as original

WHAT CHANGED FROM THE ORIGINAL:
- process_prompt(): Same template substitution logic, but expects AD-specific prompts
- extract_age_from_output(): NEW function to parse LLM output robustly
- Main pipeline: Added AD-specific fields and validation

HOW THE ORIGINAL WORKS:
- Each participant's data is stored as a JSON object in a JSONL file
- Each JSON has keys like "demographics", "lab_results", "medical_history" etc.
- The prompt template has {key} placeholders that get replaced with actual values
- The LLM receives the filled prompt and predicts biological age
"""

import os
import re
import sys
import json
import argparse
from ad_model_processor import ADModelProcessor
from tqdm import tqdm


def parse_args():
    """
    Command-line arguments.
    Same structure as original, with additions for AD-specific options.
    """
    parser = argparse.ArgumentParser(description="AD Prediction Pipeline")
    parser.add_argument('--model', required=True, 
                        help='Path to the model directory (e.g., meta-llama/Meta-Llama-3-70B-Instruct)')
    parser.add_argument('--data_file', required=True, 
                        help='Path to input JSONL file with participant health data')
    parser.add_argument('--cache_file', required=True, 
                        help='Path to output JSONL file for results')
    parser.add_argument('--prompt', default=None, 
                        help='Path to the prompt template file')
    parser.add_argument('--prediction_type', default='overall', 
                        choices=['overall', 'brain', 'cardiovascular', 'metabolic'],
                        help='Type of age prediction (NEW: brain-specific for AD)')
    parser.add_argument('--do_sample', action='store_true', 
                        help='Whether to use sampling (default: False for reproducibility)')
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0, 
                        help='Temperature=0 for deterministic outputs (same as original)')
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help='Max tokens (increased from 1024 for AD-specific CoT)')
    return parser.parse_args()


def load_data(file_path):
    """
    Load JSONL data. IDENTICAL to original.
    
    Each line is a JSON object representing one participant:
    {
        "eid": "1234567",
        "demographics": "Age: 65, Sex: Female, Ethnicity: White British...",
        "lifestyle": "Current smoker: No, Alcohol: Occasional...",
        "physical_exam": "BMI: 27.3, SBP: 138, DBP: 82...",
        "lab_results": "HbA1c: 5.8%, Total cholesterol: 5.2 mmol/L...",
        "medical_history": "Hypertension: Yes, Diabetes: No...",
        "family_history": "Mother: Dementia, Father: Heart disease...",
        "cognitive_data": "Reaction time: 620ms, Pairs matching: 4 errors...",
        "genetic_data": "APOE status: e3/e4, PRS_AD: 1.2 SD above mean..."
    }
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data


def process_prompt(case, prompt):
    """
    Apply prompt template to a participant's data. IDENTICAL logic to original.
    
    HOW IT WORKS:
    - The prompt template contains placeholders like {demographics}, {lab_results}
    - This function finds all {key} patterns and replaces them with actual values
    - If a key exists in the participant's data, it gets substituted
    
    Example:
        prompt = "Patient info: {demographics}. Labs: {lab_results}. Predict age."
        case = {"demographics": "Age 65, Female", "lab_results": "HbA1c 5.8%"}
        result = "Patient info: Age 65, Female. Labs: HbA1c 5.8%. Predict age."
    """
    input_text = prompt
    for key in re.findall(r"\{(.+?)\}", input_text):
        if key in case:
            input_text = input_text.replace(f"{{{key}}}", case[key])
    return input_text


def extract_age_from_output(output_text):
    """
    NEW FUNCTION: Extract predicted age from LLM output.
    
    The original paper doesn't include this parsing step in their public code,
    but it's necessary. The LLM output is free-form text with CoT reasoning.
    We need to extract the numerical age prediction.
    
    Parsing strategy:
    1. Look for explicit age statements: "predicted age: 72" or "biological age is 72"
    2. Look for the last number in the output (often the final answer)
    3. Validate: age should be between 30 and 120
    """
    if not output_text:
        return None
    
    # Strategy 1: Look for explicit age patterns
    patterns = [
        r'(?:predicted|biological|estimated)\s*(?:age|BA)\s*(?:is|:)\s*(\d+)',
        r'(?:overall|brain)\s*(?:age|BA)\s*(?:is|:)\s*(\d+)',
        r'(?:age|BA)\s*(?:prediction|estimate)\s*(?:is|:)\s*(\d+)',
        r'\*\*(\d+)\*\*\s*(?:years?\s*old)?',
        r'(\d+)\s*years?\s*old',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 30 <= age <= 120:
                return age
    
    # Strategy 2: Find all numbers, take the last reasonable one
    numbers = re.findall(r'\b(\d+)\b', output_text)
    for num_str in reversed(numbers):
        num = int(num_str)
        if 30 <= num <= 120:
            return num
    
    return None


def process_file(data_file, cache_file, model_processor, prompt, args):
    """
    Main processing function. ADAPTED from original.
    
    ORIGINAL FLOW:
    1. Load all participant data
    2. Set generation params
    3. Apply prompt template to all cases
    4. Batch generate with vLLM
    5. Write results to JSONL
    
    ADDITIONS FOR AD:
    - Extract numerical age from LLM output
    - Add prediction_type field
    - Add parsed age field alongside raw output
    """
    data = load_data(data_file)
    
    # Set generation parameters (same as original)
    model_processor.set_generation_params(
        do_sample=args.do_sample,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Apply prompt to all cases (same as original)
    input_texts = [process_prompt(case, prompt) for case in data]
    
    # Batch generate (same as original -- vLLM handles batching efficiently)
    generated_outputs = model_processor.generate_ad_prediction(input_texts)
    
    # Write results (ENHANCED: includes parsed age)
    with open(cache_file, 'w', encoding='utf-8') as output_file:
        for case, generated_output in zip(data, generated_outputs):
            # Store raw LLM output (same as original)
            case["model_generated_aging_prediction"] = generated_output
            
            # NEW: Parse the predicted age from the output
            if generated_output and len(generated_output) > 0:
                parsed_age = extract_age_from_output(generated_output[0])
                case["predicted_biological_age"] = parsed_age
            
            # NEW: Store prediction type
            case["prediction_type"] = args.prediction_type
            
            json.dump(case, output_file, ensure_ascii=False)
            output_file.write('\n')


def main():
    args = parse_args()
    
    # Load prompt template (same as original)
    if args.prompt:
        with open(args.prompt, 'r', encoding='utf8') as f:
            prompt = f.read()
    else:
        raise ValueError("Prompt file is required. Use --prompt <path>")

    # Initialize model (same as original)
    model_processor = ADModelProcessor(model_dir=args.model)
    
    # Process (same as original)
    process_file(args.data_file, args.cache_file, model_processor, prompt, args)
    
    print(f"Processing completed for {args.data_file}. Output saved to {args.cache_file}")


if __name__ == "__main__":
    main()
