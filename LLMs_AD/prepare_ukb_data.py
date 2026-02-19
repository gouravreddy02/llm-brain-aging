"""
UKB Data Preparation for AD Prediction Pipeline
================================================
This script converts raw UK Biobank data into the JSONL format
expected by the LLM inference pipeline.

ORIGINAL APPROACH (Paper 1):
- Constructed textual health reports from 152 routine health indicators
- Organized into: demographics, lifestyle, physical exam, lab tests, 
  medical history, family history
- Output: JSONL file with one participant per line

OUR ADDITIONS FOR AD:
- Cognitive assessment data (reaction time, pairs matching, etc.)
- Genetic data (APOE status, polygenic risk scores)
- Neuroimaging data (if available: brain volumes, WMH)
- Social determinants (education level, social isolation)

USAGE:
    python prepare_ukb_data.py \
        --input_dir /path/to/ukb_data/ \
        --output_file ./data/source_data/ukb_health_info.jsonl \
        --include_cognitive \
        --include_genetic
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare UKB data for LLM inference")
    parser.add_argument('--input_dir', required=True, help='Directory with UKB data files')
    parser.add_argument('--output_file', required=True, help='Output JSONL file path')
    parser.add_argument('--include_cognitive', action='store_true',
                        help='Include cognitive assessment data')
    parser.add_argument('--include_genetic', action='store_true',
                        help='Include genetic data (APOE, PRS)')
    parser.add_argument('--include_imaging', action='store_true',
                        help='Include neuroimaging data')
    parser.add_argument('--age_min', type=int, default=40,
                        help='Minimum age for inclusion')
    parser.add_argument('--age_max', type=int, default=75,
                        help='Maximum age for inclusion')
    return parser.parse_args()


# ============================================================================
# UKB FIELD MAPPINGS
# ============================================================================
# These map UKB field IDs to the textual descriptions used in health reports.
# The original paper used 152 indicators organized into categories.
# Below is the structure -- you'll need to fill in your specific UKB field IDs.

DEMOGRAPHICS_FIELDS = {
    # UKB Field ID -> Description
    "31": "Sex",
    "21003": "Age at assessment",
    "21000": "Ethnicity",
    "738": "Household income",
    "6138": "Education/qualifications",
    "6142": "Employment status",
}

LIFESTYLE_FIELDS = {
    "20116": "Smoking status",
    "1558": "Alcohol intake frequency",
    "1289": "Cooked vegetable intake",
    "1299": "Salad/raw vegetable intake",
    "1309": "Fresh fruit intake",
    "1329": "Processed meat intake",
    "1349": "Red meat intake",
    "1478": "Salt added to food",
    "894": "Moderate physical activity duration (min/week)",
    "914": "Vigorous physical activity duration (min/week)",
    "1160": "Sleep duration (hours)",
}

PHYSICAL_EXAM_FIELDS = {
    "21001": "BMI",
    "4080": "Systolic blood pressure",
    "4079": "Diastolic blood pressure",
    "50": "Standing height (cm)",
    "21002": "Weight (kg)",
    "48": "Waist circumference (cm)",
    "49": "Hip circumference (cm)",
    "20015": "Sitting height (cm)",
    "4082": "Pulse wave arterial stiffness index",
    "78": "Heel bone mineral density",
}

LAB_RESULTS_FIELDS = {
    "30000": "White blood cell count",
    "30010": "Red blood cell count",
    "30020": "Haemoglobin concentration",
    "30040": "Mean corpuscular volume",
    "30050": "Mean corpuscular haemoglobin",
    "30060": "Mean corpuscular haemoglobin concentration",
    "30080": "Platelet count",
    "30620": "Alanine aminotransferase (ALT)",
    "30600": "Albumin",
    "30610": "Alkaline phosphatase",
    "30630": "Apolipoprotein A",
    "30640": "Apolipoprotein B",
    "30650": "Aspartate aminotransferase (AST)",
    "30680": "Calcium",
    "30690": "Cholesterol",
    "30700": "Creatinine",
    "30710": "C-reactive protein",
    "30720": "Cystatin C",
    "30730": "Direct bilirubin",
    "30740": "Gamma glutamyltransferase",
    "30750": "Glucose",
    "30760": "HbA1c",
    "30770": "HDL cholesterol",
    "30780": "IGF-1",
    "30790": "LDL cholesterol",
    "30800": "Lipoprotein A",
    "30810": "Phosphate",
    "30830": "SHBG",
    "30840": "Total bilirubin",
    "30850": "Total protein",
    "30860": "Triglycerides",
    "30870": "Urate",
    "30880": "Urea",
    "30890": "Vitamin D",
}

MEDICAL_HISTORY_FIELDS = {
    "6150": "Vascular/heart problems diagnosed by doctor",
    "2443": "Diabetes diagnosed by doctor",
    "6177": "Medication for cholesterol/blood pressure/diabetes",
    "20002": "Non-cancer illness code (self-reported)",
}

FAMILY_HISTORY_FIELDS = {
    "20107": "Illnesses of father",
    "20110": "Illnesses of mother",
    "20111": "Illnesses of siblings",
}

# NEW: AD-specific fields
COGNITIVE_FIELDS = {
    "20023": "Mean reaction time (ms)",
    "399": "Pairs matching: number of incorrect matches",
    "4282": "Numeric memory: maximum digits remembered",
    "20016": "Fluid intelligence score",
    "6350": "Duration to complete alphanumeric path (Trail Making)",
    "6373": "Duration to complete numeric path (Trail Making)",
    "20018": "Prospective memory result",
}

GENETIC_FIELDS = {
    # These would come from imputed genotype data
    "APOE": "APOE genotype (e2/e3/e4 status)",
    "PRS_AD": "Polygenic risk score for AD",
}


def format_demographics(row, fields):
    """Format demographic information into natural language text."""
    parts = []
    if "21003" in row and pd.notna(row["21003"]):
        parts.append(f"Age: {int(row['21003'])} years")
    if "31" in row and pd.notna(row["31"]):
        sex = "Male" if row["31"] == 1 else "Female"
        parts.append(f"Sex: {sex}")
    if "21000" in row and pd.notna(row["21000"]):
        parts.append(f"Ethnicity code: {int(row['21000'])}")
    if "738" in row and pd.notna(row["738"]):
        parts.append(f"Household income: {row['738']}")
    if "6138" in row and pd.notna(row["6138"]):
        parts.append(f"Education: {row['6138']}")
    return "; ".join(parts) if parts else "Not available"


def format_lifestyle(row, fields):
    """Format lifestyle factors into natural language text."""
    parts = []
    if "20116" in row and pd.notna(row["20116"]):
        smoke_map = {0: "Never", 1: "Previous", 2: "Current"}
        parts.append(f"Smoking status: {smoke_map.get(int(row['20116']), 'Unknown')}")
    if "1558" in row and pd.notna(row["1558"]):
        parts.append(f"Alcohol frequency code: {int(row['1558'])}")
    if "894" in row and pd.notna(row["894"]):
        parts.append(f"Moderate physical activity: {int(row['894'])} min/week")
    if "914" in row and pd.notna(row["914"]):
        parts.append(f"Vigorous physical activity: {int(row['914'])} min/week")
    if "1160" in row and pd.notna(row["1160"]):
        parts.append(f"Sleep duration: {row['1160']} hours")
    return "; ".join(parts) if parts else "Not available"


def format_physical_exam(row, fields):
    """Format physical exam data into natural language text."""
    parts = []
    if "21001" in row and pd.notna(row["21001"]):
        parts.append(f"BMI: {round(row['21001'], 1)} kg/m2")
    if "4080" in row and pd.notna(row["4080"]):
        parts.append(f"Systolic BP: {int(row['4080'])} mmHg")
    if "4079" in row and pd.notna(row["4079"]):
        parts.append(f"Diastolic BP: {int(row['4079'])} mmHg")
    if "48" in row and pd.notna(row["48"]):
        parts.append(f"Waist circumference: {round(row['48'], 1)} cm")
    return "; ".join(parts) if parts else "Not available"


def format_lab_results(row, fields):
    """Format lab results into natural language text."""
    parts = []
    lab_units = {
        "30760": ("HbA1c", "mmol/mol"),
        "30690": ("Total cholesterol", "mmol/L"),
        "30770": ("HDL cholesterol", "mmol/L"),
        "30790": ("LDL cholesterol", "mmol/L"),
        "30750": ("Glucose", "mmol/L"),
        "30700": ("Creatinine", "umol/L"),
        "30710": ("C-reactive protein", "mg/L"),
        "30620": ("ALT", "U/L"),
        "30860": ("Triglycerides", "mmol/L"),
    }
    for field_id, (name, unit) in lab_units.items():
        if field_id in row and pd.notna(row[field_id]):
            parts.append(f"{name}: {round(row[field_id], 2)} {unit}")
    return "; ".join(parts) if parts else "Not available"


def format_cognitive_data(row, fields):
    """NEW: Format cognitive assessment data for AD prediction."""
    parts = []
    if "20023" in row and pd.notna(row["20023"]):
        parts.append(f"Mean reaction time: {int(row['20023'])} ms")
    if "399" in row and pd.notna(row["399"]):
        parts.append(f"Pairs matching errors: {int(row['399'])}")
    if "4282" in row and pd.notna(row["4282"]):
        parts.append(f"Numeric memory (max digits): {int(row['4282'])}")
    if "20016" in row and pd.notna(row["20016"]):
        parts.append(f"Fluid intelligence score: {int(row['20016'])}")
    if "20018" in row and pd.notna(row["20018"]):
        result_map = {1: "Correct on first attempt", 2: "Correct on second attempt", 0: "Incorrect"}
        parts.append(f"Prospective memory: {result_map.get(int(row['20018']), 'Unknown')}")
    return "; ".join(parts) if parts else "Not available"


def format_medical_history(row, fields):
    """Format medical history into natural language text."""
    parts = []
    # Vascular/heart problems (field 6150)
    if "6150" in row and pd.notna(row["6150"]):
        vascular_map = {
            1: "Heart attack", 2: "Angina", 3: "Stroke",
            4: "High blood pressure", -7: "None of the above"
        }
        val = int(row["6150"])
        parts.append(f"Vascular problems: {vascular_map.get(val, f'Code {val}')}")
    # Diabetes (field 2443)
    if "2443" in row and pd.notna(row["2443"]):
        diabetes_map = {0: "No", 1: "Yes"}
        parts.append(f"Diabetes: {diabetes_map.get(int(row['2443']), 'Unknown')}")
    # Medications (field 6177)
    if "6177" in row and pd.notna(row["6177"]):
        med_map = {
            1: "Cholesterol-lowering medication", 2: "Blood pressure medication",
            3: "Insulin", -7: "None of the above"
        }
        val = int(row["6177"])
        parts.append(f"Medication: {med_map.get(val, f'Code {val}')}")
    return "; ".join(parts) if parts else "Not available"


def format_family_history(row, fields):
    """Format family history into natural language text."""
    parts = []
    illness_map = {
        1: "Heart disease", 2: "Stroke", 3: "Lung cancer",
        4: "Bowel cancer", 5: "Breast cancer", 6: "Chronic bronchitis/emphysema",
        8: "High blood pressure", 9: "Diabetes",
        10: "Alzheimer's disease/dementia", 11: "Parkinson's disease",
        12: "Severe depression", 13: "Prostate cancer",
        -11: "Do not know", -13: "Prefer not to answer",
        -17: "None of the above"
    }
    # Father's illnesses (field 20107)
    if "20107" in row and pd.notna(row["20107"]):
        val = int(row["20107"])
        parts.append(f"Father: {illness_map.get(val, f'Code {val}')}")
    # Mother's illnesses (field 20110)
    if "20110" in row and pd.notna(row["20110"]):
        val = int(row["20110"])
        parts.append(f"Mother: {illness_map.get(val, f'Code {val}')}")
    # Siblings' illnesses (field 20111)
    if "20111" in row and pd.notna(row["20111"]):
        val = int(row["20111"])
        parts.append(f"Siblings: {illness_map.get(val, f'Code {val}')}")
    return "; ".join(parts) if parts else "Not available"


def format_genetic_data(row, fields):
    """NEW: Format genetic information for AD prediction."""
    parts = []
    if "APOE" in row and pd.notna(row["APOE"]):
        parts.append(f"APOE genotype: {row['APOE']}")
    if "PRS_AD" in row and pd.notna(row["PRS_AD"]):
        parts.append(f"AD polygenic risk score: {round(row['PRS_AD'], 2)} SD")
    return "; ".join(parts) if parts else "Not available"


def create_health_report(row, include_cognitive=True, include_genetic=True):
    """
    Create a complete health report for one participant.
    
    This is the KEY function that converts tabular UKB data into the
    textual format that the LLM can process. The original paper did
    exactly this but for 152 general health indicators.
    
    Output format matches what process_prompt() expects:
    {
        "eid": "1234567",
        "demographics": "Age: 65, Sex: Female...",
        "lifestyle": "Smoking: Never, Alcohol: Occasional...",
        ...
    }
    """
    report = {
        "eid": str(int(row.get("eid", 0))),
        "demographics": format_demographics(row, DEMOGRAPHICS_FIELDS),
        "lifestyle": format_lifestyle(row, LIFESTYLE_FIELDS),
        "physical_exam": format_physical_exam(row, PHYSICAL_EXAM_FIELDS),
        "lab_results": format_lab_results(row, LAB_RESULTS_FIELDS),
        "medical_history": format_medical_history(row, MEDICAL_HISTORY_FIELDS),
        "family_history": format_family_history(row, FAMILY_HISTORY_FIELDS),
    }
    
    if include_cognitive:
        report["cognitive_data"] = format_cognitive_data(row, COGNITIVE_FIELDS)
    else:
        report["cognitive_data"] = "Not available"
    
    if include_genetic:
        report["genetic_data"] = format_genetic_data(row, GENETIC_FIELDS)
    else:
        report["genetic_data"] = "Not available"
    
    return report


def main():
    args = parse_args()
    
    # Load your UKB data
    # This will depend on your specific UKB data extract format
    # Common formats: .csv, .tab, .parquet
    print(f"Loading data from {args.input_dir}...")
    
    # Example: load from CSV
    # df = pd.read_csv(os.path.join(args.input_dir, "ukb_data.csv"))
    
    # Example: load from multiple files
    # df_demo = pd.read_csv(os.path.join(args.input_dir, "demographics.csv"))
    # df_labs = pd.read_csv(os.path.join(args.input_dir, "lab_results.csv"))
    # df = df_demo.merge(df_labs, on="eid")
    
    # For demonstration, create a placeholder
    print("NOTE: Replace this section with your actual UKB data loading code")
    print(f"Output will be written to: {args.output_file}")
    
    # Filter by age range
    # df = df[(df["21003"] >= args.age_min) & (df["21003"] <= args.age_max)]
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Convert each row to a health report and write as JSONL
    # with open(args.output_file, 'w', encoding='utf-8') as f:
    #     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating reports"):
    #         report = create_health_report(
    #             row,
    #             include_cognitive=args.include_cognitive,
    #             include_genetic=args.include_genetic
    #         )
    #         json.dump(report, f, ensure_ascii=False)
    #         f.write('\n')
    
    # print(f"Created {len(df)} health reports in {args.output_file}")


if __name__ == "__main__":
    main()
