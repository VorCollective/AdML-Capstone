# Capstone Project: Early Prediction of Grade 3+ Chemotherapy Toxicity and Unplanned Hospitalization in Breast Cancer Patients

**Version**: Draft 1.0  
**Date**: January 2026  
**Author**: L.L  
**Location**: Nairobi, Kenya

## 1. Statement of the Problem

Breast cancer patients undergoing chemotherapy face a high risk of severe adverse events, particularly in the early cycles (1–3), where toxicities often peak. Common grade 3+ toxicities include:

- Neutropenia / febrile neutropenia (infection risk)
- Neuropathy (taxane-related)
- Mucositis / gastrointestinal toxicity
- Anemia, thrombocytopenia, fatigue

These events lead to:
- Dose reductions, delays, or treatment discontinuation
- Unplanned hospitalizations or emergency visits (often 15–30% in the first cycles)
- Increased morbidity, reduced quality of life, and higher healthcare costs
- In low-resource settings (e.g., Kenya), limited access to growth factors (G-CSF) or supportive care amplifies the impact

Current risk assessment relies on clinical judgment, basic risk scores (e.g., MASCC or CISNE for febrile neutropenia), or trial-derived rules — which are not personalized, miss dynamic factors (serial labs, cumulative dose), and perform modestly in real-world diverse populations.

**Core Problem**  
There is a need for **accurate, early, personalized risk prediction models** that use routinely collected clinical data (baseline characteristics, serial labs, treatment details) to forecast severe toxicity or unplanned admission **before or during early cycles**, enabling oncologists to:
- Adjust doses or schedules proactively
- Initiate prophylactic G-CSF or supportive care
- Intensify monitoring or switch regimens
- Optimize resource use in constrained settings

This is more actionable and urgent than 30-day readmission prediction post-discharge, as it intervenes **during active treatment**.

## 2. Project Objectives

Primary:
- Build and validate ML models to predict the composite outcome of **grade 3+ toxicity or unplanned hospitalization** within the first 2–3 chemotherapy cycles in breast cancer patients.

Secondary:
- Identify key predictive features (e.g., serial neutrophil trends, cumulative dose, baseline frailty, comorbidities)
- Provide interpretable risk scores and personalized intervention recommendations
- Evaluate model fairness across age, socioeconomic proxies, and treatment settings
- Assess feasibility for low-resource deployment (e.g., simple features, minimal compute)

## 3. Why This Is a Strong Capstone Idea

- **Higher clinical urgency & actionability** — Predictions occur during treatment (not after discharge), allowing real-time adjustments
- **Novelty** — Less saturated than general readmission models; focuses on early-cycle dynamics and serial data
- **Technical depth** — Opportunity for time-series modeling (lab trends), multi-task learning (toxicity + admission), SHAP explainability, fairness audits
- **Relevance to Kenya/LMICs** — Chemotherapy toxicity is a major barrier to completing curative treatment in resource-limited settings; models can highlight when G-CSF or supportive care would be most cost-effective
- **Publishable potential** — Aligns with recent trends in precision oncology toxicity prediction (2023–2025 papers on ML for febrile neutropenia, etc.)

## 4. Data Sources & Feasibility

Primary public option:
- **MIMIC-IV** (PhysioNet) — Filter to breast cancer admissions (ICD-10 C50.*, D05.*) with chemotherapy administration (HCPCS/CPT codes or medication tables)
  - Strengths: Rich labs (serial CBC, neutrophils), vitals, medications, procedures, outcomes
  - Challenge: Chemotherapy cycles may need reconstruction from timestamps/meds; toxicity grades inferred from labs/admissions (e.g., ANC <1.0 × 10⁹/L for severe neutropenia)
  - https://huggingface.co/datasets

Augmentation strategies:
- Synthetic data generation (e.g., CTGAN or Gaussian Copula) based on MIMIC patterns or literature-reported toxicity rates in breast cancer cohorts
- Public breast cancer datasets with treatment outcomes (limited):  
  - TCGA-BRCA (genomic/clinical) — limited toxicity data, but useful for baseline features
  - METABRIC or other curated cohorts (cBioPortal) — some treatment/response data
  - Literature-derived synthetic cohorts (simulate based on known risk factors from papers)

Target cohort size goal: 1,000–5,000 breast cancer chemotherapy encounters (feasible after filtering MIMIC-IV + augmentation)

Outcome definition (binary or multi-label):
- Grade 3+ toxicity: Inferred from labs (e.g., neutropenia ANC <1.0), admissions for fever/infection, or medication changes
- Unplanned hospitalization: New admission within 21–30 days after cycle start (exclude planned)

## 5. Modeling Approach

| Model Type                | Why Suitable                                      | Expected Performance (Literature) |
|---------------------------|---------------------------------------------------|-----------------------------------|
| Logistic Regression       | Baseline, interpretable odds ratios               | AUC 0.65–0.75                     |
| XGBoost / LightGBM        | Handles imbalance, tabular data, feature importance | AUC **0.75–0.85** (common top)  |
| Random Forest             | Robust to missing, non-linear interactions        | AUC 0.72–0.82                     |
| LSTM / Time-Series Models | Captures serial lab trends (neutrophil trajectories) | AUC 0.78–0.88 (if time-series strong) |

Key techniques:
- Class imbalance handling: scale_pos_weight, SMOTE, focal loss
- Feature engineering: Cumulative dose, lab deltas, frailty proxies (age + comorbidities)
- Explainability: SHAP values → highlight actionable factors (e.g., "baseline ANC <3.5 → high risk")
- Fairness: Demographic parity checks, re-weighting

Output format:
- Risk probability + risk category (low/medium/high)
- Personalized recommendations (e.g., "Consider G-CSF prophylaxis", "Increase monitoring frequency")

## 6. Expected Challenges & Mitigations

- Data sparsity (toxicity grading not explicit) → Use proxy outcomes from labs/admissions
- Limited breast cancer specificity in MIMIC → Augment with synthetic data + literature validation
- Generalizability → Stratify models by regimen (taxane vs anthracycline) if possible

## 7. Next Steps

1. Obtain MIMIC-IV access/https://huggingface.co/datasets and filter breast cancer + chemotherapy cohort
2. Define exact outcome labels and perform exploratory analysis
3. Prototype baseline XGBoost model
4. Add time-series features and compare performance
5. Implement SHAP + fairness evaluation



