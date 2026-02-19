#!/bin/bash
# ============================================================================
# AD-Specific LLM Inference Pipeline
# ============================================================================
# Adapted from Paper 1's model_vllm_inference.sh
#
# WHAT THE ORIGINAL DOES:
#   1. Sets model path, data path, cache path, prompt template
#   2. Checks if output already exists (skip if yes)
#   3. Runs aging_generate.py with vLLM
#   4. Removes empty output files
#
# WHAT WE CHANGED:
#   - Multiple prediction types (overall, brain, cardiovascular, metabolic)
#   - AD-specific prompt templates
#   - Support for multiple LLM models (ensemble)
#   - UKB and All of Us data paths
# ============================================================================

# ---- CONFIGURATION ----
# Choose prediction type: overall, brain, cardiovascular, metabolic
PREDICTION_TYPE="overall"

# Model path (same format as original)
# Original used: meta-llama/Meta-Llama-3-8B-Instruct
# We recommend Llama3-70B for best performance (as shown in Paper 1's results)
MODEL_PATH="meta-llama/Meta-Llama-3-70B-Instruct"

# Data paths
# Original format: JSONL with one participant per line
DATA_DIR="./data/source_data"
DATA_FILE="${DATA_DIR}/ukb_health_info.jsonl"

# Output paths
RESULT_DIR="./data/result"
CACHE_FILE="${RESULT_DIR}/ukb_ad_${PREDICTION_TYPE}_result.jsonl"

# Prompt template
# Original had: main_analysis, sensitivity_analysis, chinese_prompt, additional_analysis
# We add: ad_overall, ad_brain_specific
PROMPT_DIR="./prompts"
PROMPT_TEMPLATE="${PROMPT_DIR}/ad_${PREDICTION_TYPE}.txt"

# ---- CREATE DIRECTORIES ----
mkdir -p "${RESULT_DIR}"
mkdir -p "${PROMPT_DIR}"

# ---- CHECK CACHE (same logic as original) ----
if [ -f "$CACHE_FILE" ] && [ -s "$CACHE_FILE" ]; then
    echo "Output already exists: ${CACHE_FILE}. Skipping."
    exit 0
else
    echo "Running inference for prediction_type=${PREDICTION_TYPE}..."
    touch "$CACHE_FILE"
fi

# ---- RUN INFERENCE ----
python3 ad_aging_generate.py \
    --model $MODEL_PATH \
    --prompt $PROMPT_TEMPLATE \
    --data_file $DATA_FILE \
    --cache_file $CACHE_FILE \
    --prediction_type $PREDICTION_TYPE \
    --temperature 0 \
    --max_tokens 2048

# ---- VALIDATE OUTPUT (same as original) ----
if [ ! -s "$CACHE_FILE" ]; then
    echo "Output file is empty. Removing and exiting."
    rm "$CACHE_FILE"
    exit 1
fi

echo "Inference complete. Results saved to: ${CACHE_FILE}"
echo "Number of predictions: $(wc -l < ${CACHE_FILE})"


# ============================================================================
# ENSEMBLE MODE: Run multiple models (NEW - not in original)
# ============================================================================
# Uncomment below to run ensemble predictions across multiple LLMs
# This mirrors Paper 1's comparison of Llama3-8b, Llama3-70b, Qwen models

# MODELS=(
#     "meta-llama/Meta-Llama-3-8B-Instruct"
#     "meta-llama/Meta-Llama-3-70B-Instruct"
#     "Qwen/Qwen2-72B-Instruct"
# )
#
# for MODEL in "${MODELS[@]}"; do
#     MODEL_NAME=$(basename $MODEL)
#     CACHE="${RESULT_DIR}/ukb_ad_${PREDICTION_TYPE}_${MODEL_NAME}.jsonl"
#     
#     if [ -f "$CACHE" ] && [ -s "$CACHE" ]; then
#         echo "Skipping ${MODEL_NAME} (already exists)"
#         continue
#     fi
#     
#     echo "Running ${MODEL_NAME}..."
#     python3 ad_aging_generate.py \
#         --model $MODEL \
#         --prompt $PROMPT_TEMPLATE \
#         --data_file $DATA_FILE \
#         --cache_file $CACHE \
#         --prediction_type $PREDICTION_TYPE \
#         --temperature 0 \
#         --max_tokens 2048
# done


# ============================================================================
# ALL OF US VALIDATION (NEW)
# ============================================================================
# After running UKB, validate with All of Us data
# Same pipeline, different data file

# AOU_DATA="${DATA_DIR}/aou_health_info.jsonl"
# AOU_CACHE="${RESULT_DIR}/aou_ad_${PREDICTION_TYPE}_result.jsonl"
#
# python3 ad_aging_generate.py \
#     --model $MODEL_PATH \
#     --prompt $PROMPT_TEMPLATE \
#     --data_file $AOU_DATA \
#     --cache_file $AOU_CACHE \
#     --prediction_type $PREDICTION_TYPE \
#     --temperature 0 \
#     --max_tokens 2048
