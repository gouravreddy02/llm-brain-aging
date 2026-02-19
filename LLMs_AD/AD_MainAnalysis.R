################################################################################
# AD-Specific Analysis Pipeline
# Adapted from Paper 1's MainAnalysis.R
################################################################################
#
# This script mirrors the EXACT structure of the original MainAnalysis.R
# but adapts it for Alzheimer's Disease prediction in UKB with All of Us
# validation. Each section below maps to a section in the original code.
#
# ORIGINAL STRUCTURE -> OUR ADAPTATION:
#   1-1. Compare C-index on aging outcomes  -> Compare C-index on AD outcomes
#   1-2. NHANES validation                  -> All of Us validation
#   1-3. Organ-specific ages                -> Brain/organ-specific ages for AD
#   1-4. Compare Beta on phenotypes         -> Compare Beta on cognitive phenotypes
#   2-1. Age gap predictors                 -> AD-specific age gap as predictor
#   2-2. KM curves                          -> KM curves for AD incidence
#   2-3. Organ-specific HRs                 -> Organ-specific HRs for AD
#   2-4. Organ-overall age gaps             -> Brain-overall age gap for AD
#   4.   Proteomics                         -> Proteomics for AD biomarkers
#   5.   SHAP interpretation               -> SHAP for AD risk factors
################################################################################

# ==============================================================================
# REQUIRED LIBRARIES (same as original)
# ==============================================================================
library(arrow)
library(jsonlite)
library(survival)
library(survminer)
library(survcomp)
library(gridExtra)
library(pROC)
library(Hmisc)
library(rms)
library(hexbin)
library(caret)
library(scales)
library(grid)
library(ppcor)
library(tidyverse)
library(lubridate)
library(svglite)
library(broom)
library(ggpubr)  # For ggarrange() and annotate_figure()


# ==============================================================================
# SECTION 1: DATA PREPARATION (UKB)
# ==============================================================================
# ORIGINAL: Loads LLM predictions, covariates, and outcomes, then calculates age gaps
# OUR VERSION: Same structure, but includes AD-specific outcomes and features

# --- 1a. Load LLM-predicted ages ---
# Original: dat_age <- read_csv("Data/Models/llama3_70b/llama3-70b-result_only_age.csv")
# This CSV was extracted from the JSONL output of aging_generate.py
# Columns: eid, llm_overall_age, llm_cardiovascular_age, llm_hepatic_age, etc.
dat_age <- read_csv("Data/Models/llama3_70b/llama3-70b-ad-result_only_age.csv")

# --- 1b. Load covariates ---
# Original: dat_cov <- read_rds("Data/covariates_outcomes/panel_indicators.rds")
# This contains the 152 health indicators used as features
dat_cov <- read_rds("Data/covariates_outcomes/panel_indicators.rds")

# --- 1c. Load outcomes (AD-SPECIFIC - THIS IS THE KEY CHANGE) ---
# Original loaded general aging outcomes (death, CHD, stroke, COPD, etc.)
# We need AD-specific outcomes
dat_outcome <- read_rds("Data/covariates_outcomes/ad_outcomes.rds")
# Expected columns:
#   - "AD diagnose": 0/1 indicator for incident AD (ICD-10: G30.x)
#   - "AD duration": time to AD diagnosis or censoring (years)
#   - "Dementia diagnose": 0/1 for any dementia (broader definition)
#   - "Dementia duration": time to any dementia
#   - "MCI diagnose": 0/1 for mild cognitive impairment
#   - "MCI duration": time to MCI
#   - "Cognitive decline diagnose": significant decline on cognitive tests
#   - "Cognitive decline duration": time to significant decline
#   - "All-cause death diagnose": 0/1 (keep for competing risks)
#   - "All-cause death duration": time to death

# --- 1d. Merge data (IDENTICAL to original) ---
dat_age <- dat_age %>% inner_join(dat_cov, by = "eid")
dat_age <- dat_age %>% inner_join(dat_outcome, by = "eid")

# --- 1e. Calculate age gaps (IDENTICAL logic to original) ---
# KEY CONCEPT: age_gap = LLM_predicted_age - chronological_age
# Positive gap = accelerated aging; Negative = decelerated
# The original does NOT adjust for chronological age (unlike ML models)
# because the LLM approach is unsupervised and avoids regression-to-mean
dat_age <- dat_age %>%
  mutate(llm_overall_acc = llm_overall_age - Age) %>%
  mutate(llm_cardiovascular_acc = llm_cardiovascular_age - Age) %>%
  mutate(llm_hepatic_acc = llm_hepatic_age - Age) %>%
  mutate(llm_pulmonary_acc = llm_pulmonary_age - Age) %>%
  mutate(llm_renal_acc = llm_renal_age - Age) %>%
  mutate(llm_metabolic_acc = llm_metabolic_age - Age) %>%
  mutate(llm_musculoskeletal_acc = llm_musculoskeletal_age - Age)

dat_age <- na.omit(dat_age)


# ==============================================================================
# SECTION 2: C-INDEX COMPARISON FOR AD (adapted from original Section 1-1)
# ==============================================================================
# ORIGINAL: Compared LLM vs telomere, frailty, ML models for 8 diseases
# OUR VERSION: Compare for AD, dementia, MCI, cognitive decline

# --- 2a. Load comparison models ---
# Load traditional aging proxies (same as original)
dat_telomere <- read_csv("Data/covariates_outcomes/telomere.csv")
dat_fi <- read_rds("Data/covariates_outcomes/frailty_index_52.rds")
dat_age <- dat_age %>% inner_join(dat_fi, by = "eid")
dat_telomere <- dplyr::select(dat_telomere, 1:2, 5)
names(dat_telomere)[c(2, 3)] <- c("telomere_adjusted", "z_adjusted_telomere")
dat_telomere <- na.omit(dat_telomere)
dat_age <- dat_age %>% inner_join(dat_telomere, by = "eid")

# Load ML model predictions (same structure as original)
dat_svm <- read_csv("Data/Models/svm_res/test_svm_overall_res.csv")
dat_rf <- read_csv("Data/Models/rf_res/test_rf_overall_res.csv")
dat_xgboost <- read_csv("Data/Models/xgboost_res/test_xgboost_overall_res.csv")
dat_dnn <- read_csv("Data/Models/dnn_res/test_dnn_overall_res.csv")

# Round ML predictions (same as original)
dat_svm$svm_overall_age <- round(dat_svm$svm_overall_age, digits = 0)
dat_rf$rf_overall_age <- round(dat_rf$rf_overall_age, digits = 0)
dat_xgboost$xgboost_overall_age <- round(dat_xgboost$xgboost_overall_age, digits = 0)
dat_dnn$dnn_overall_age <- round(dat_dnn$dnn_overall_age, digits = 0)

# Calculate ML age gaps (same as original)
dat_svm$svm_age_gap <- dat_svm$svm_overall_age - dat_svm$Age
dat_rf$rf_age_gap <- dat_rf$rf_overall_age - dat_rf$Age
dat_xgboost$xgboost_age_gap <- dat_xgboost$xgboost_overall_age - dat_xgboost$Age
dat_dnn$dnn_age_gap <- dat_dnn$dnn_overall_age - dat_dnn$Age

# IMPORTANT: Adjust ML age gaps for chronological age (SAME as original)
# The original paper adjusts ML model age gaps because ML models suffer from
# regression-to-mean bias. The LLM age gap does NOT need this adjustment.
# This is a key advantage of the LLM approach.
adjust_age_gap <- function(dat, gap_col, age_col = "Age") {
  model <- lm(as.formula(paste(gap_col, "~", age_col)), data = dat)
  predicted_gap <- predict(model, newdata = dat)
  adjusted <- dat[[gap_col]] - predicted_gap
  return(round(adjusted, digits = 0))
}

dat_svm$adj_svm_overall_acc <- adjust_age_gap(dat_svm, "svm_age_gap")
dat_rf$adj_rf_overall_acc <- adjust_age_gap(dat_rf, "rf_age_gap")
dat_xgboost$adj_xgboost_overall_acc <- adjust_age_gap(dat_xgboost, "xgboost_age_gap")
dat_dnn$adj_dnn_overall_acc <- adjust_age_gap(dat_dnn, "dnn_age_gap")

# Clean up and merge (same as original)
dat_svm$Age <- NULL; dat_svm$svm_age_gap <- NULL
dat_rf$Age <- NULL; dat_rf$rf_age_gap <- NULL
dat_xgboost$Age <- NULL; dat_xgboost$xgboost_age_gap <- NULL
dat_dnn$Age <- NULL; dat_dnn$dnn_age_gap <- NULL

dat_age <- dat_age %>%
  inner_join(dat_svm, by = "eid") %>%
  inner_join(dat_rf, by = "eid") %>%
  inner_join(dat_xgboost, by = "eid") %>%
  inner_join(dat_dnn, by = "eid")

# --- 2b. AD-Specific Additional Comparisons (NEW) ---
# Load AD-specific risk scores for benchmarking
# dat_caide <- read_csv("Data/Models/caide_risk_score.csv")  # CAIDE Dementia Risk Score
# dat_anuadri <- read_csv("Data/Models/anu_adri_score.csv")  # ANU-ADRI Score
# dat_prs <- read_csv("Data/Models/polygenic_risk_score.csv") # AD Polygenic Risk Score

# --- 2c. Define variables and outcomes ---
# ORIGINAL diseases: All-cause death, CHD, Stroke, COPD, Liver, Renal, T2D, Arthritis
# OUR diseases: AD, Dementia, MCI, Cognitive decline + keep All-cause death
disease <- c("AD", "Dementia", "Cognitive decline", "All-cause death")

# Variables to compare (same structure as original)
var_ls <- c("telomere_adjusted", "frailty_index", "Age", "svm_overall_age",
            "rf_overall_age", "xgboost_overall_age",
            "dnn_overall_age", "llm_overall_age",
            "adj_svm_overall_acc", "adj_rf_overall_acc",
            "adj_xgboost_overall_acc",
            "adj_dnn_overall_acc", "llm_overall_acc")

# --- 2d. 10-fold CV C-index calculation (IDENTICAL LOGIC to original) ---
# This is the CORE evaluation loop from the original paper.
# It trains Cox models on each aging proxy and evaluates C-index via 10-fold CV.
var_mean_c_index <- c()
var_mean_c_index_lower <- c()
var_mean_c_index_upper <- c()
outcome_ls <- c()
var_name_ls <- c()  # BUG FIX: accumulate var names in parallel with outcomes

set.seed(2024)  # Same seed as original for reproducibility
for (i in 1:length(disease)) {
  item <- disease[i]
  item_diagnose <- paste0(item, " diagnose")
  item_duration <- paste0(item, " duration")
  dat_age$event <- dat_age[[item_diagnose]]
  dat_age$time <- dat_age[[item_duration]]
  
  # Exclude prevalent cases (same logic as original)
  # Original had disease-specific exclusions (e.g., MACE for CHD/Stroke)
  # For AD: exclude anyone with pre-existing dementia
  if (item %in% c("AD", "Dementia", "MCI", "Cognitive decline")) {
    dat_cox <- subset(dat_age, `Dementia duration` > 0)  # Dementia-free at baseline
  } else {
    dat_cox <- subset(dat_age, time > 0)
  }
  
  # 10-fold CV (IDENTICAL to original)
  folds <- createFolds(dat_cox$event, k = 10)
  
  for (j in 1:length(var_ls)) {
    var <- var_ls[j]
    c_index_values <- c()
    
    for (k in 1:10) {
      test_indices <- folds[[k]]
      train_data <- dat_cox[-test_indices, ]
      test_data <- dat_cox[test_indices, ]
      
      # Univariate Cox model (same as original)
      formula_covariates <- paste0("survobj ~ ", var)
      f <- as.formula(formula_covariates)
      survobj <- with(train_data, Surv(time, event))
      cox_fit <- coxph(formula = f, data = train_data, na.action = na.omit)
      
      # Predict risk (same as original)
      test_data$predicted_risk <- predict(cox_fit, newdata = test_data, type = "risk")
      
      # Calculate C-index (same as original)
      concordance_result <- concordance.index(
        x = test_data$predicted_risk,
        surv.time = test_data$time,
        surv.event = test_data$event
      )
      c_index_values <- c(c_index_values, concordance_result$c.index)
    }
    
    # Calculate mean and CI using t-distribution (same as original)
    mean_c_index <- mean(c_index_values)
    n_folds <- length(c_index_values)
    se_c_index <- sd(c_index_values) / sqrt(n_folds)
    t_value <- qt(0.975, df = n_folds - 1)
    mean_c_index_lower <- mean_c_index - t_value * se_c_index
    mean_c_index_upper <- mean_c_index + t_value * se_c_index
    
    var_mean_c_index <- c(var_mean_c_index, round(mean_c_index, 3))
    var_mean_c_index_lower <- c(var_mean_c_index_lower, round(mean_c_index_lower, 3))
    var_mean_c_index_upper <- c(var_mean_c_index_upper, round(mean_c_index_upper, 3))
    outcome_ls <- c(outcome_ls, item)
    var_name_ls <- c(var_name_ls, var)  # Accumulate var name for each row
    
    print(paste0(item, " -- ", var, " -- C-index: ", round(mean_c_index, 3)))
  }
}

# Create results dataframe (same as original)
# NOTE: var_name_ls has length(disease) * length(var_ls) entries,
# one per disease-variable combination, matching outcome_ls
dat_plot <- data.frame(
  outcome = outcome_ls,
  var_name = var_name_ls,
  c_index = var_mean_c_index,
  c_index_lower = var_mean_c_index_lower,
  c_index_upper = var_mean_c_index_upper
)

# Rename for plotting (same approach as original)
dat_plot <- dat_plot %>%
  mutate(var_name = case_when(
    var_name == "telomere_adjusted" ~ "Telomere",
    var_name == "frailty_index" ~ "Frailty index",
    var_name == "Age" ~ "Chronological age",
    var_name == "svm_overall_age" ~ "SVM overall age",
    var_name == "rf_overall_age" ~ "RF overall age",
    var_name == "xgboost_overall_age" ~ "XGBoost overall age",
    var_name == "dnn_overall_age" ~ "DNN overall age",
    var_name == "llm_overall_age" ~ "LLM overall age",
    var_name == "adj_svm_overall_acc" ~ "SVM overall age gap",
    var_name == "adj_rf_overall_acc" ~ "RF overall age gap",
    var_name == "adj_xgboost_overall_acc" ~ "XGBoost overall age gap",
    var_name == "adj_dnn_overall_acc" ~ "DNN overall age gap",
    var_name == "llm_overall_acc" ~ "LLM overall age gap"
  ))

# Save C-index results
write_csv(dat_plot, "ad_c_index_comparison_results.csv")
cat("C-index results saved to ad_c_index_comparison_results.csv\n")


# ==============================================================================
# SECTION 3: KAPLAN-MEIER CURVES FOR AD (adapted from original Section 2-2)
# ==============================================================================
# ORIGINAL: KM curves stratifying by top/median/bottom 10% of overall age gap
# for 8 diseases. We do the same but for AD incidence.

# Rank by overall age gap (IDENTICAL to original)
dat_age <- dat_age[order(dat_age$llm_overall_acc), ]
n <- nrow(dat_age)
dat_age$group <- "Other"
dat_age$group[1:(n * 0.1)] <- "Bottom 10%"
dat_age$group[(n * 0.9 + 1):n] <- "Top 10%"
dat_age$group[(n * 0.45 + 1):(n * 0.55)] <- "Median 10%"

# AD-specific KM (adapted from original)
disease <- c("AD", "Dementia", "Cognitive decline", "All-cause death")

plots <- list()
for (i in 1:length(disease)) {
  item <- disease[i]
  item_diagnose <- paste0(item, " diagnose")
  item_duration <- paste0(item, " duration")
  dat_age$event <- dat_age[[item_diagnose]]
  dat_age$time <- dat_age[[item_duration]]
  
  # Exclude prevalent cases (same approach as original)
  if (item %in% c("AD", "Dementia")) {
    dat_cox <- subset(dat_age, `Dementia duration` > 0)
  } else {
    dat_cox <- subset(dat_age, time > 0)
  }
  
  dat_cox <- subset(dat_cox, group %in% c("Bottom 10%", "Top 10%", "Median 10%"))
  
  # KM fit (IDENTICAL to original)
  fit <- survfit(Surv(time, event) ~ group, data = dat_cox)
  
  ggsurv <- ggsurvplot(fit,
    data = dat_cox,
    conf.int = TRUE,
    fun = "event",
    xlab = "",
    ylab = "",
    xlim = c(0, 15),
    palette = c("#90D3C7", "#80B1D3", "#ca0020"),
    legend.title = "Overall age gap group",
    legend.labs = c("Bottom 10%", "Median 10%", "Top 10%"),
    legend = "bottom",
    title = item,
    ggtheme = theme_minimal()
  )
  
  ggsurv$plot <- ggsurv$plot +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.border = element_blank(),
          axis.line = element_line(),
          axis.title = element_text(size = 22),
          axis.text = element_text(size = 22, color = "black"),
          legend.title = element_text(size = 22),
          legend.text = element_text(size = 22),
          plot.title = element_text(size = 24, hjust = 0.5, vjust = 2)) +
    scale_x_continuous(breaks = c(5, 10)) +
    scale_y_continuous(labels = function(x) x * 100)
  
  plots[[i]] <- ggsurv$plot
}

combined_plot <- ggarrange(plotlist = plots, ncol = 2, nrow = 2,
                           common.legend = TRUE, legend = "bottom")
combined_plot <- annotate_figure(combined_plot,
  bottom = text_grob("Time (years)", size = 22),
  left = text_grob("Cumulative event (%)", size = 22, rot = 90)
)
ggsave("ad_km_curves.pdf", plot = combined_plot, width = 14, height = 12)


# ==============================================================================
# SECTION 4: ADJUSTED HAZARD RATIOS FOR AD (adapted from original Section 2-3)
# ==============================================================================
# ORIGINAL: Cox models with covariates for organ-specific age gaps -> 8 diseases
# OUR VERSION: Same Cox models but outcome is AD incidence

Cox_analysis_AD <- function(dat_baseline, disease_ls, var_ls) {
  # This function is IDENTICAL to the original Cox_analysis()
  # except we focus on AD-related outcomes
  
  disease_name_ls <- c()
  res_name_ls <- c()
  hr_ls <- c()
  conf_lower_ls <- c()
  conf_upper_ls <- c()
  pvalue_ls <- c()
  
  for (item in disease_ls) {
    item_diagnose <- paste0(item, " diagnose")
    item_duration <- paste0(item, " duration")
    dat_baseline$event <- dat_baseline[[item_diagnose]]
    dat_baseline$time <- dat_baseline[[item_duration]]
    
    # Exclude prevalent cases
    dat_cox <- subset(dat_baseline, `Dementia duration` > 0 & time > 0)
    
    for (i in 1:length(var_ls)) {
      var_name <- var_ls[i]
      
      # SAME Cox formula structure as original
      # Covariates: Age, Sex, Income, Employment, Education, Center, Ethnicity,
      #             Smoking, Alcohol, BMI, Hypertension
      formula_covariates <- paste0(
        "survobj ~ Age + Sex + Income + Employment + ",
        "Education + UKB_assessment_center + Ethnicity + ",
        "Current_smoker + Daily_alcohol_intake + ",
        "BMI + Hypertension_history + ", var_name
      )
      f <- as.formula(formula_covariates)
      survobj <- with(dat_cox, Surv(time, event == 1))
      
      cox_fit <- coxph(formula = f, data = dat_cox, na.action = na.omit)
      
      hr <- round(summary(cox_fit)$coefficients[var_name, "exp(coef)"], 3)
      conf_interval <- exp(confint(cox_fit)[var_name, ])
      conf_lower <- round(conf_interval[1], 3)
      conf_upper <- round(conf_interval[2], 3)
      p_value <- summary(cox_fit)$coefficients[var_name, "Pr(>|z|)"]
      
      disease_name_ls <- c(disease_name_ls, item)
      res_name_ls <- c(res_name_ls, var_name)
      hr_ls <- c(hr_ls, hr)
      conf_lower_ls <- c(conf_lower_ls, conf_lower)
      conf_upper_ls <- c(conf_upper_ls, conf_upper)
      pvalue_ls <- c(pvalue_ls, p_value)
      
      print(paste0(item, ": ", var_name, " HR=", hr, " p=", round(p_value, 4)))
    }
  }
  
  res <- data.frame(disease = disease_name_ls, var = res_name_ls,
                    HR = hr_ls, Lower = conf_lower_ls,
                    Upper = conf_upper_ls, p_value = pvalue_ls)
  return(res)
}

# Run for AD outcomes
disease_ad <- c("AD", "Dementia")
var_ls_organs <- c("llm_cardiovascular_acc", "llm_hepatic_acc", "llm_pulmonary_acc",
                   "llm_renal_acc", "llm_metabolic_acc", "llm_musculoskeletal_acc")

ad_results_hr <- Cox_analysis_AD(dat_baseline = dat_age,
                                  disease_ls = disease_ad,
                                  var_ls = var_ls_organs)

# Save results
write_csv(ad_results_hr, "ad_organ_specific_hr_results.csv")


# ==============================================================================
# SECTION 5: ALL OF US VALIDATION (NEW - replaces original NHANES section)
# ==============================================================================
# ORIGINAL: Validated in NHANES by comparing LLM vs 8 epigenetic clocks
# OUR VERSION: Validate in All of Us (more diverse, larger)

# --- 5a. Load All of Us data ---
# dat_aou <- read_csv("Data/AllOfUs/aou_merged_data.csv")
# dat_aou_llm <- read_csv("Data/AllOfUs/aou_llm_predictions.csv")
# dat_aou <- dat_aou %>% inner_join(dat_aou_llm, by = "person_id")

# --- 5b. Calculate age gaps (same as UKB) ---
# dat_aou <- dat_aou %>%
#   mutate(llm_overall_acc = llm_overall_age - Age)

# --- 5c. Run same C-index analysis ---
# (Use identical code from Section 2d above, just with dat_aou instead of dat_age)

# --- 5d. Subgroup analyses (NEW - fairness assessment) ---
# Test prediction performance across demographic groups
# This is critical for equity and not done in either original paper
#
# subgroups <- list(
#   "White" = subset(dat_aou, race == "White"),
#   "Black" = subset(dat_aou, race == "Black or African American"),
#   "Hispanic" = subset(dat_aou, ethnicity == "Hispanic"),
#   "Asian" = subset(dat_aou, race == "Asian"),
#   "Female" = subset(dat_aou, sex == "Female"),
#   "Male" = subset(dat_aou, sex == "Male"),
#   "Age 65-74" = subset(dat_aou, Age >= 65 & Age < 75),
#   "Age 75-84" = subset(dat_aou, Age >= 75 & Age < 85),
#   "Age 85+" = subset(dat_aou, Age >= 85)
# )
#
# Run C-index for each subgroup and compare


# ==============================================================================
# SECTION 6: DRUG REPURPOSING VALIDATION (NEW - from Paper 2)
# ==============================================================================
# This section integrates Paper 2's drug repurposing approach with our
# biological age framework. We test whether drugs that reduce the LLM age gap
# also reduce AD incidence.

# --- 6a. Define drug exposure ---
# Top candidates from Paper 2: Metformin, Simvastatin, Losartan
# drug_exposure <- read_csv("Data/DrugExposure/ukb_medication_history.csv")
# 
# dat_drug <- dat_age %>%
#   inner_join(drug_exposure, by = "eid") %>%
#   filter(age_at_first_exposure <= 65)  # Exposure before age 65

# --- 6b. Propensity score matching (adapted from Paper 2) ---
# Paper 2 used 2:1 PS matching with basic covariates
# We use high-dimensional PS with more covariates (improvement)
#
# library(MatchIt)
#
# For each drug:
# ps_model <- matchit(
#   drug_exposed ~ Age + Sex + race + education + BMI + smoking +
#     alcohol + hypertension + diabetes + hyperlipidemia +
#     cardiovascular_disease + num_comorbidities + num_medications,
#   data = dat_drug,
#   method = "nearest",
#   ratio = 2,
#   caliper = 0.2
# )
#
# matched_data <- match.data(ps_model)
# cox_fit <- coxph(Surv(ad_time, ad_event) ~ drug_exposed + Age + Sex,
#                  data = matched_data)


# ==============================================================================
# SECTION 7: INTERPRETABILITY (adapted from original Section 5)
# ==============================================================================
# ORIGINAL: SHAP values, global surrogate model, counterfactual simulation
# OUR VERSION: Same methods but interpret AD-specific risk factors

# --- 7a. SHAP analysis (IDENTICAL method to original) ---
library(iml)

# Load data for interpretability
# dat_shap <- read_csv("Data/Models/llama3_70b_interpretation/ad_shap_analysis.csv")
# dat_shap <- dat_shap %>% inner_join(dat_cov, by = "eid")
# dat_shap$llm_overall_acc <- dat_shap$llm_overall_age - dat_shap$Age

# Build linear surrogate model (same as original)
# model <- lm(llm_overall_acc ~ ., data = dat_shap_features)

# Create predictor object (same as original)
# predictor <- Predictor$new(
#   model = model,
#   data = dat_shap_features[, -ncol(dat_shap_features)],
#   y = dat_shap_features$llm_overall_acc,
#   predict.function = function(model, newdata) predict(model, newdata)
# )

# Individual SHAP (same as original)
# shapley <- Shapley$new(predictor, x.interest = dat_shap_features[1, -ncol(.)])

# Global feature importance (same as original)
# feature_imp <- FeatureImp$new(predictor, loss = "mae")


# ==============================================================================
# SECTION 8: SUMMARY OUTPUTS
# ==============================================================================

cat("\n========================================\n")
cat("AD Prediction Pipeline Complete\n")
cat("========================================\n")
cat("Key outputs generated:\n")
cat("  - C-index comparison table\n")
cat("  - KM curves for AD incidence\n")
cat("  - Organ-specific HR results\n")
cat("  - Drug repurposing analysis (if enabled)\n")
cat("  - SHAP interpretability plots (if enabled)\n")
cat("========================================\n")
