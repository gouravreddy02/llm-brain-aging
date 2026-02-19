"""
Prepare AD-Specific Outcomes from UKB
======================================
MISSING FROM ORIGINAL: The original paper's R code loads pre-built outcome files
(e.g., "overall_aging_outcomes.rds") but doesn't show how they were created.

This script creates the AD-specific outcome file (ad_outcomes.rds equivalent)
from UKB hospital episode statistics (HES), death registry, and primary care data.

AD/Dementia ICD-10 codes used:
    G30.0 - Alzheimer's disease with early onset
    G30.1 - Alzheimer's disease with late onset
    G30.8 - Other Alzheimer's disease
    G30.9 - Alzheimer's disease, unspecified
    F00   - Dementia in Alzheimer's disease
    F01   - Vascular dementia
    F02   - Dementia in other diseases
    F03   - Unspecified dementia

MCI ICD-10 code:
    F06.7 - Mild cognitive disorder

USAGE:
    python prepare_ad_outcomes.py \
        --ukb_dir /path/to/ukb_data/ \
        --output_file ./Data/covariates_outcomes/ad_outcomes.csv \
        --baseline_date 2010-01-01
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AD outcomes from UKB")
    parser.add_argument('--ukb_dir', required=True, help='Directory with UKB data')
    parser.add_argument('--output_file', required=True, help='Output CSV/RDS path')
    parser.add_argument('--baseline_date', default='2010-01-01',
                        help='Baseline assessment date (approx)')
    parser.add_argument('--censor_date', default='2024-12-31',
                        help='Right-censoring date')
    return parser.parse_args()


# ICD-10 code definitions
AD_CODES = ['G300', 'G301', 'G308', 'G309']
DEMENTIA_AD_CODES = ['F000', 'F001', 'F002', 'F009']
VASCULAR_DEMENTIA_CODES = ['F010', 'F011', 'F012', 'F013', 'F018', 'F019']
OTHER_DEMENTIA_CODES = ['F020', 'F021', 'F022', 'F023', 'F024', 'F028', 'F03']
ALL_DEMENTIA_CODES = AD_CODES + DEMENTIA_AD_CODES + VASCULAR_DEMENTIA_CODES + OTHER_DEMENTIA_CODES
MCI_CODES = ['F067']

# Broader AD-related codes (for sensitivity analysis)
AD_BROAD_CODES = AD_CODES + DEMENTIA_AD_CODES


def match_icd_codes(diagnosis_col, code_list):
    """
    Check if any ICD-10 code in diagnosis_col matches the code_list.
    UKB stores ICD codes without dots (e.g., 'G309' not 'G30.9').
    Handles prefix matching (e.g., 'G30' matches 'G300', 'G301', etc.).
    """
    if pd.isna(diagnosis_col):
        return False
    diag = str(diagnosis_col).strip().upper()
    for code in code_list:
        if diag.startswith(code.upper()):
            return True
    return False


def create_outcome_variables(df_hes, df_death, df_baseline, 
                              baseline_date, censor_date):
    """
    Create time-to-event outcome variables.
    
    This replicates the structure of the original paper's outcome files:
        - "{Disease} diagnose": 0/1 event indicator
        - "{Disease} duration": time from baseline to event/censor (years)
    
    The original paper did this for: All-cause death, CHD, Stroke, COPD, 
    Liver diseases, Renal failure, T2D, Arthritis.
    
    We create equivalent columns for AD-specific outcomes.
    """
    baseline_dt = pd.to_datetime(baseline_date)
    censor_dt = pd.to_datetime(censor_date)
    
    outcomes = pd.DataFrame({'eid': df_baseline['eid'].unique()})
    
    # --- AD (G30.x codes) ---
    # Find first AD diagnosis date for each participant
    ad_events = df_hes[df_hes['diag_icd10'].apply(
        lambda x: match_icd_codes(x, AD_BROAD_CODES)
    )].copy()
    ad_first = ad_events.groupby('eid')['epistart'].min().reset_index()
    ad_first.columns = ['eid', 'ad_date']
    
    outcomes = outcomes.merge(ad_first, on='eid', how='left')
    outcomes['ad_date'] = pd.to_datetime(outcomes['ad_date'])
    
    # Event: AD diagnosed after baseline
    outcomes['AD diagnose'] = np.where(
        (outcomes['ad_date'].notna()) & (outcomes['ad_date'] > baseline_dt), 1, 0
    )
    # Duration: time from baseline to AD or censor
    outcomes['AD duration'] = np.where(
        outcomes['AD diagnose'] == 1,
        (outcomes['ad_date'] - baseline_dt).dt.days / 365.25,
        (censor_dt - baseline_dt).days / 365.25
    )
    
    # --- All Dementia (G30 + F00-F03) ---
    dem_events = df_hes[df_hes['diag_icd10'].apply(
        lambda x: match_icd_codes(x, ALL_DEMENTIA_CODES)
    )].copy()
    dem_first = dem_events.groupby('eid')['epistart'].min().reset_index()
    dem_first.columns = ['eid', 'dementia_date']
    
    outcomes = outcomes.merge(dem_first, on='eid', how='left')
    outcomes['dementia_date'] = pd.to_datetime(outcomes['dementia_date'])
    
    outcomes['Dementia diagnose'] = np.where(
        (outcomes['dementia_date'].notna()) & (outcomes['dementia_date'] > baseline_dt), 1, 0
    )
    outcomes['Dementia duration'] = np.where(
        outcomes['Dementia diagnose'] == 1,
        (outcomes['dementia_date'] - baseline_dt).dt.days / 365.25,
        (censor_dt - baseline_dt).days / 365.25
    )
    
    # --- MCI (F06.7) ---
    mci_events = df_hes[df_hes['diag_icd10'].apply(
        lambda x: match_icd_codes(x, MCI_CODES)
    )].copy()
    mci_first = mci_events.groupby('eid')['epistart'].min().reset_index()
    mci_first.columns = ['eid', 'mci_date']
    
    outcomes = outcomes.merge(mci_first, on='eid', how='left')
    outcomes['mci_date'] = pd.to_datetime(outcomes['mci_date'])
    
    outcomes['MCI diagnose'] = np.where(
        (outcomes['mci_date'].notna()) & (outcomes['mci_date'] > baseline_dt), 1, 0
    )
    outcomes['MCI duration'] = np.where(
        outcomes['MCI diagnose'] == 1,
        (outcomes['mci_date'] - baseline_dt).dt.days / 365.25,
        (censor_dt - baseline_dt).days / 365.25
    )
    
    # --- All-cause death (keep for competing risks) ---
    if df_death is not None:
        death_first = df_death.groupby('eid')['date_of_death'].min().reset_index()
        outcomes = outcomes.merge(death_first, on='eid', how='left')
        outcomes['date_of_death'] = pd.to_datetime(outcomes['date_of_death'])
        
        outcomes['All-cause death diagnose'] = np.where(
            (outcomes['date_of_death'].notna()) & (outcomes['date_of_death'] > baseline_dt), 1, 0
        )
        outcomes['All-cause death duration'] = np.where(
            outcomes['All-cause death diagnose'] == 1,
            (outcomes['date_of_death'] - baseline_dt).dt.days / 365.25,
            (censor_dt - baseline_dt).days / 365.25
        )
        
        # Censor AD/Dementia at death if no diagnosis
        for prefix in ['AD', 'Dementia', 'MCI']:
            diag_col = f'{prefix} diagnose'
            dur_col = f'{prefix} duration'
            outcomes.loc[
                (outcomes[diag_col] == 0) & (outcomes['All-cause death diagnose'] == 1),
                dur_col
            ] = outcomes.loc[
                (outcomes[diag_col] == 0) & (outcomes['All-cause death diagnose'] == 1),
                'All-cause death duration'
            ]
    
    # --- Exclude prevalent cases ---
    # Remove participants diagnosed before baseline
    outcomes['prevalent_dementia'] = np.where(
        (outcomes['dementia_date'].notna()) & (outcomes['dementia_date'] <= baseline_dt), 1, 0
    )
    
    # Clean up date columns
    drop_cols = ['ad_date', 'dementia_date', 'mci_date', 'date_of_death']
    outcomes = outcomes.drop(columns=[c for c in drop_cols if c in outcomes.columns])
    
    return outcomes


def main():
    args = parse_args()
    
    print("=" * 60)
    print("AD Outcomes Preparation for UKB")
    print("=" * 60)
    print(f"Data directory: {args.ukb_dir}")
    print(f"Baseline date: {args.baseline_date}")
    print(f"Censor date:   {args.censor_date}")
    print()
    
    # --- Load UKB data ---
    # NOTE: Adjust file paths below to match your UKB data extract
    
    # Hospital Episode Statistics (HES) - contains ICD-10 diagnoses
    # UKB fields: 41270 (diagnoses), 41280 (dates), 41271 (OPCS4)
    print("Loading HES data...")
    # df_hes = pd.read_csv(os.path.join(args.ukb_dir, "hesin_diag.csv"))
    # Expected columns: eid, diag_icd10, epistart
    
    # Death registry
    print("Loading death registry...")
    # df_death = pd.read_csv(os.path.join(args.ukb_dir, "death.csv"))
    # Expected columns: eid, date_of_death, cause_icd10
    
    # Baseline data (for participant list)
    print("Loading baseline data...")
    # df_baseline = pd.read_csv(os.path.join(args.ukb_dir, "baseline.csv"))
    # Expected columns: eid, assessment_date
    
    # --- Create outcomes ---
    # outcomes = create_outcome_variables(
    #     df_hes, df_death, df_baseline,
    #     args.baseline_date, args.censor_date
    # )
    
    # --- Summary statistics ---
    # print(f"\nTotal participants: {len(outcomes)}")
    # print(f"Prevalent dementia (excluded): {outcomes['prevalent_dementia'].sum()}")
    # outcomes_clean = outcomes[outcomes['prevalent_dementia'] == 0].copy()
    # outcomes_clean = outcomes_clean.drop(columns=['prevalent_dementia'])
    # print(f"Analysis cohort: {len(outcomes_clean)}")
    # print(f"Incident AD: {outcomes_clean['AD diagnose'].sum()}")
    # print(f"Incident Dementia: {outcomes_clean['Dementia diagnose'].sum()}")
    # print(f"Incident MCI: {outcomes_clean['MCI diagnose'].sum()}")
    # print(f"Deaths: {outcomes_clean['All-cause death diagnose'].sum()}")
    
    # --- Save ---
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # outcomes_clean.to_csv(args.output_file, index=False)
    # print(f"\nSaved to {args.output_file}")
    
    print("\nNOTE: Uncomment the data loading sections above and adjust")
    print("file paths to match your specific UKB data extract format.")


if __name__ == "__main__":
    main()
