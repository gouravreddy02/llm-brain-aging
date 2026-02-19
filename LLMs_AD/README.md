# LLM-Based Alzheimer's Disease Prediction Pipeline

## Overview

This pipeline adapts two published approaches for AD prediction in UK Biobank with validation in All of Us:

1. **Paper 1** (Nature Medicine 2025): LLM-based biological age prediction from health reports
2. **Paper 2** (npj Digital Medicine 2024): ChatGPT for AD drug repurposing validated in EHR

## How the Original Code Works

### Python Pipeline (LLM Inference)

The original codebase has three Python/shell components that form the inference pipeline:

**`model_processor.py` → `ad_model_processor.py`**
- Wraps vLLM for efficient batch inference on large language models
- Uses `tensor_parallel_size=torch.cuda.device_count()` to split the model across all available GPUs
- Uses `temperature=0` for deterministic, reproducible outputs
- The `generate_aging()` method takes a list of text prompts and returns batch predictions

**`aging_generate.py` → `ad_aging_generate.py`**
- The main orchestration script that connects data, prompts, and model
- `load_data()`: Reads JSONL files where each line = one participant's health data
- `process_prompt()`: Template substitution — finds `{key}` placeholders in the prompt and replaces them with actual participant data
- `process_file()`: The main loop — applies prompts to all participants, runs batch inference, saves results

**`model_vllm_inference.sh` → `run_ad_inference.sh`**
- Shell wrapper that sets paths and runs the Python script
- Checks if output already exists (skip if yes — useful for restarts)
- Validates output file is not empty

### R Analysis Pipeline (`MainAnalysis.R` → `AD_MainAnalysis.R`)

The R script is ~2000 lines covering the complete analysis. Key sections:

| Original Section | What it Does | Our Adaptation |
|---|---|---|
| Section 1-1 | 10-fold CV C-index for 8 diseases × 13 aging proxies | Same method, but for AD/Dementia outcomes |
| Section 1-2 | NHANES validation with 8 epigenetic clocks | All of Us validation with subgroup analyses |
| Section 1-3 | Organ-specific age predictions | Same + brain-specific ages |
| Section 1-4 | Beta coefficients on 12 aging phenotypes | Beta on cognitive phenotypes |
| Section 2-2 | KM curves by age gap tertiles | KM curves for AD incidence |
| Section 2-3 | Adjusted HRs from Cox models | HRs for organ-specific gaps → AD |
| Section 4 | Proteomics (differential expression, GSEA, Venn) | Same but AD-focused pathways |
| Section 5 | SHAP, global surrogate, counterfactual | Same interpretation methods |

### Key Statistical Methods Used

**Age Gap Calculation (critical concept):**
```r
# LLM age gap (NO adjustment needed — key advantage)
llm_overall_acc = llm_overall_age - Age

# ML model age gap (NEEDS adjustment for regression-to-mean)
model <- lm(ml_age_gap ~ Age, data = dat)
adj_ml_acc = ml_age_gap - predict(model)
```

**C-index via 10-fold CV:**
```r
folds <- createFolds(dat$event, k = 10)
for each fold:
  train Cox model: coxph(Surv(time, event) ~ aging_proxy)
  predict risk on test set
  calculate concordance.index()
mean ± t-distribution CI across 10 folds
```

**Adjusted Hazard Ratios:**
```r
coxph(Surv(time, event) ~ age_gap + Age + Sex + Income + Education + 
      Ethnicity + Smoking + Alcohol + BMI + Hypertension)
```

## Pipeline Execution Order

```
Step 1: Data Preparation
    prepare_ukb_data.py        →  data/source_data/ukb_health_info.jsonl
    prepare_ad_outcomes.py     →  Data/covariates_outcomes/ad_outcomes.csv

Step 2: LLM Inference  
    run_ad_inference.sh        →  data/result/ukb_ad_overall_result.jsonl
    (calls ad_aging_generate.py which uses ad_model_processor.py)

Step 3: Post-Processing
    post_process_results.py    →  Data/Models/llama3_70b/llama3-70b-ad-result_only_age.csv

Step 4: Statistical Analysis
    AD_MainAnalysis.R          →  C-index tables, KM curves, HR forest plots, etc.

Step 5: Validation in All of Us
    Repeat Steps 1-4 with All of Us data
```

## File Structure

```
ad_prediction_scripts/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── STEP 1: Data Preparation
│   ├── prepare_ukb_data.py            # UKB tabular data → JSONL health reports
│   └── prepare_ad_outcomes.py         # UKB ICD-10 codes → AD outcome variables
│
├── STEP 2: LLM Inference
│   ├── ad_model_processor.py          # vLLM engine wrapper (from model_processor.py)
│   ├── ad_aging_generate.py           # Main inference pipeline (from aging_generate.py)
│   └── run_ad_inference.sh            # Shell runner (from model_vllm_inference.sh)
│
├── STEP 3: Post-Processing
│   └── post_process_results.py        # Parse JSONL output → CSV for R analysis
│
├── STEP 4: Statistical Analysis
│   └── AD_MainAnalysis.R              # Full analysis pipeline (from MainAnalysis.R)
│
├── prompts/
│   ├── ad_overall.txt                 # Overall + brain biological age prompt
│   └── ad_brain.txt                   # Brain-specific biological age prompt
│
└── data/
    └── source_data/
        └── health_info_example.jsonl  # 3 example participants showing JSONL format
```

## Hardware Requirements

- GPU: NVIDIA A100 (80GB) or H100 recommended for Llama3-70B
- For Llama3-8B: A single V100 (32GB) is sufficient
- RAM: 64GB+ for large UKB datasets
- Storage: ~500GB for model weights + data

## Key Differences from Original Papers

1. **AD-specific outcomes** instead of general aging/mortality
2. **Cognitive and genetic features** added to health reports
3. **All of Us validation** instead of NHANES (more diverse)
4. **Fairness assessment** across demographic subgroups
5. **Drug repurposing integration** with biological age framework
6. **Brain-specific biological age** prediction
